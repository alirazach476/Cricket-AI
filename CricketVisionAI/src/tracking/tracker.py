"""
ByteTrack-based person tracker for CricketVisionAI.

People from cricket detection/best.pt (data.yaml):
  model 0 batsman → pipeline 1 Batsman
  model 1 fielder → pipeline 2 Fielder
  model 2 umpire  → pipeline 3 Umpire

Bowler ONLY from Cricket Player Detection/best.pt:
  model class 2 Bowler → pipeline 4 Bowler
  (Ball / Batsman / ScoreCards from that model are ignored)

Ball from dedicated cricket_ball_detection weights (pipeline class 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from ultralytics import YOLO

from src.config.settings import (
    BALL_CLASS_ID,
    BALL_CONFIDENCE,
    BOWLER_CLASS_ID,
    BOWLER_DETECTION_WEIGHTS,
    BOWLER_MODEL_CLASS_ID,
    CRICKET_MODEL_TO_PIPELINE,
    FIELDER_CLASS_ID,
    PERSON_CLASS_IDS,
    PLAYER_CLASS_NAMES,
    PLAYER_CONFIDENCE,
    PLAYER_DETECTION_WEIGHTS,
    PLAYER_IMGSZ,
    PLAYER_IOU,
    PROJECT_ROOT,
)
from src.tracking.models import TrackedObject

# YOLO classes in cricket detection/best.pt
TRACKABLE_CLASS_IDS = (0, 1, 2)  # batsman, fielder, umpire

DEFAULT_BYTETRACK_CFG = PROJECT_ROOT / "src" / "tracking" / "bytetrack.yaml"


class PlayerTracker:
    """Cricket detection people + bowler-only player model + ball YOLO."""

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        tracker_cfg: Optional[Union[str, Path]] = None,
        conf: float = PLAYER_CONFIDENCE,
        ball_conf: float = BALL_CONFIDENCE,
        iou: float = PLAYER_IOU,
        imgsz: int = PLAYER_IMGSZ,
        class_ids: Sequence[int] = TRACKABLE_CLASS_IDS,
        device: Optional[str] = None,
        persist: bool = True,
    ) -> None:
        from src.detection.ball_detector import DedicatedBallDetector
        from src.detection.ball_hybrid import HybridBallDetector

        self.weights = Path(weights) if weights else PLAYER_DETECTION_WEIGHTS
        self.bowler_weights = BOWLER_DETECTION_WEIGHTS
        self.tracker_cfg = Path(tracker_cfg) if tracker_cfg else DEFAULT_BYTETRACK_CFG
        self.conf = conf
        self.ball_conf = ball_conf
        self.iou = iou
        self.imgsz = int(imgsz)
        self.class_ids = tuple(class_ids)
        self.device = device
        self.persist = persist
        self._next_temp_id = 900_000
        self._next_bowler_id = 800_000

        if not self.weights.exists():
            raise FileNotFoundError(f"Player detection weights not found: {self.weights}")
        if not self.bowler_weights.exists():
            raise FileNotFoundError(
                f"Bowler detection weights not found: {self.bowler_weights}"
            )
        if not self.tracker_cfg.exists():
            raise FileNotFoundError(f"ByteTrack config not found: {self.tracker_cfg}")

        self.model = YOLO(str(self.weights))
        self.bowler_model = YOLO(str(self.bowler_weights))
        self.ball_detector = DedicatedBallDetector(
            conf=ball_conf, device=device, ball_track_id=1
        )
        self.ball_hybrid = HybridBallDetector(
            ball_track_id=1,
            enable_classical=False,
            coast_frames=0,
            min_conf=ball_conf,
            max_jump_px=95.0,
        )

    def update(
        self,
        frame: np.ndarray,
        pitch_box: Optional[tuple] = None,
    ) -> List[TrackedObject]:
        """ByteTrack batsman/fielder/umpire + bowler-only + ball."""
        results = self.model.track(
            source=frame,
            persist=self.persist,
            tracker=str(self.tracker_cfg),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=list(self.class_ids),
            device=self.device,
            verbose=False,
            max_det=50,
        )

        tracked = self._parse_results(results[0])
        bowlers = self._detect_bowlers(frame)
        tracked = self._merge_bowlers(tracked, bowlers)
        tracked = self._dedupe_players(tracked)
        others = [o for o in tracked if o.class_id != BALL_CLASS_ID]

        candidates = self.ball_detector.detect_candidates(
            frame, pitch_box=pitch_box, conf=self.ball_conf
        )
        tracked = self.ball_hybrid.fuse(
            frame, others, pitch_box=pitch_box, candidates=candidates
        )
        return tracked

    def _detect_bowlers(self, frame: np.ndarray) -> List[TrackedObject]:
        """Use Cricket Player Detection best.pt — Bowler class only."""
        results = self.bowler_model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=min(self.imgsz, 960),
            classes=[BOWLER_MODEL_CLASS_ID],  # Bowler only
            device=self.device,
            verbose=False,
            max_det=5,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        out: List[TrackedObject] = []
        for i in range(len(xyxy)):
            conf = float(confs[i])
            if conf < self.conf:
                continue
            x1, y1, x2, y2 = map(float, xyxy[i])
            area = max(1.0, (x2 - x1) * (y2 - y1))
            if area < 2500:
                continue
            self._next_bowler_id += 1
            out.append(
                TrackedObject(
                    object_id=int(self._next_bowler_id),
                    class_id=BOWLER_CLASS_ID,
                    class_name=PLAYER_CLASS_NAMES.get(BOWLER_CLASS_ID, "Bowler"),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                )
            )
        # Keep best 1–2 bowlers
        out.sort(key=lambda o: o.confidence, reverse=True)
        return out[:2]

    def _merge_bowlers(
        self,
        tracked: List[TrackedObject],
        bowlers: List[TrackedObject],
    ) -> List[TrackedObject]:
        """Prefer Bowler over overlapping Fielder from cricket detection."""
        if not bowlers:
            return tracked
        kept: List[TrackedObject] = []
        for o in tracked:
            if o.class_id != FIELDER_CLASS_ID:
                kept.append(o)
                continue
            # Drop fielder if same person as a detected bowler
            if any(self._iou(o, b) >= 0.25 for b in bowlers):
                continue
            kept.append(o)
        return kept + bowlers

    @staticmethod
    def _iou(a: TrackedObject, b: TrackedObject) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            return 0.0
        area_a = max(1.0, (a.x2 - a.x1) * (a.y2 - a.y1))
        area_b = max(1.0, (b.x2 - b.x1) * (b.y2 - b.y1))
        return inter / (area_a + area_b - inter)

    def _dedupe_players(self, tracked: List[TrackedObject]) -> List[TrackedObject]:
        balls = [o for o in tracked if o.class_id == BALL_CLASS_ID]
        kept: List[TrackedObject] = list(balls)
        caps = {1: 2, 2: 8, 3: 2, 4: 2}
        for class_id in PERSON_CLASS_IDS:
            group = sorted(
                [o for o in tracked if o.class_id == class_id],
                key=lambda o: (o.x2 - o.x1) * (o.y2 - o.y1) * o.confidence,
                reverse=True,
            )
            group = [
                o
                for o in group
                if (o.x2 - o.x1) * (o.y2 - o.y1) >= 2500 and o.confidence >= 0.25
            ]
            selected: List[TrackedObject] = []
            limit = caps.get(class_id, 4)
            for o in group:
                if all(self._iou(o, s) < 0.45 for s in selected):
                    selected.append(o)
                if len(selected) >= limit:
                    break
            kept.extend(selected)
        return kept

    def _parse_results(self, result) -> List[TrackedObject]:
        tracked: List[TrackedObject] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return tracked

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        has_ids = boxes.id is not None
        track_ids = (
            boxes.id.cpu().numpy().astype(int) if has_ids else [None] * len(xyxy)
        )

        for i in range(len(xyxy)):
            model_class_id = int(cls_ids[i])
            if model_class_id not in self.class_ids:
                continue
            conf = float(confs[i])
            if conf < self.conf:
                continue
            pipeline_class_id = int(
                CRICKET_MODEL_TO_PIPELINE.get(model_class_id, model_class_id)
            )
            tid = track_ids[i]
            if tid is None:
                self._next_temp_id += 1
                tid = self._next_temp_id
            x1, y1, x2, y2 = map(float, xyxy[i])
            tracked.append(
                TrackedObject(
                    object_id=int(tid),
                    class_id=pipeline_class_id,
                    class_name=PLAYER_CLASS_NAMES.get(
                        pipeline_class_id, str(pipeline_class_id)
                    ),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                )
            )
        return tracked

    def reset(self) -> None:
        self.model = YOLO(str(self.weights))
        self.bowler_model = YOLO(str(self.bowler_weights))
        self._next_temp_id = 900_000
        self._next_bowler_id = 800_000
        self.ball_hybrid.reset()
