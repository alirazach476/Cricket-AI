"""
Shot detection / classification from Cricket Actions YOLO pose weights.

Runs the pose model on a batsman crop (higher accuracy), reports label to
analytics / HUD only — no shot bounding boxes.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from ultralytics import YOLO

from src.config.settings import (
    BALL_CLASS_ID,
    SHOT_ACTIONS_CONFIDENCE,
    SHOT_ACTIONS_IMGSZ,
    SHOT_ACTIONS_WEIGHTS,
    SHOT_BALL_NEAR_FACTOR,
    SHOT_CLASSIFY_EVERY_N_FRAMES,
    SHOT_DISPLAY_NAMES,
    SHOT_MIN_VOTES,
)
from src.tracking.models import TrackedObject

BATSMAN_CLASS_ID = 1


@dataclass
class ShotPrediction:
    frame_number: int
    timestamp_sec: float
    object_id: int
    shot_name: str
    class_index: int
    confidence: float
    bbox: Tuple[float, float, float, float]
    all_probs: Optional[List[float]] = None


@dataclass
class ShotActionDetection:
    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


_RAW_TO_DISPLAY = {
    "Cover-Drive": "Cover Drive",
    "Front-foot": "Front Foot",
    "Square-short": "Square Short",
    "helicopter-shot": "Helicopter Shot",
    "pull-shorts": "Pull Shot",
}


class ShotClassifier:
    """YOLO pose/action model → shot label for analytics (no boxes drawn)."""

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        class_names: Optional[Sequence[str]] = None,
        input_size=None,
        min_confidence: float = SHOT_ACTIONS_CONFIDENCE,
        classify_every_n: int = SHOT_CLASSIFY_EVERY_N_FRAMES,
        fps: float = 30.0,
        pad_ratio: float = 0.45,
        vote_window: int = 4,
        impact_distance_factor: float = SHOT_BALL_NEAR_FACTOR,
        history: int = 60,
        device: Optional[str] = None,
        imgsz: int = SHOT_ACTIONS_IMGSZ,
        min_votes: int = SHOT_MIN_VOTES,
    ) -> None:
        del input_size, history
        self.weights = Path(weights) if weights else SHOT_ACTIONS_WEIGHTS
        self.min_confidence = float(min_confidence)
        self.classify_every_n = max(1, int(classify_every_n))
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.pad_ratio = float(pad_ratio)
        self.vote_window = max(int(min_votes), int(vote_window))
        self.min_votes = max(2, int(min_votes))
        self.impact_distance_factor = float(impact_distance_factor)
        self.device = device
        self.imgsz = int(imgsz)

        if not self.weights.exists():
            raise FileNotFoundError(f"Shot action weights not found: {self.weights}")

        self.model = YOLO(str(self.weights))
        raw_names = self.model.names or {}
        self.class_names: List[str] = []
        for i in range(len(raw_names)):
            raw = str(raw_names.get(i, f"class_{i}"))
            self.class_names.append(
                _RAW_TO_DISPLAY.get(raw, raw.replace("-", " ").title())
            )
        if class_names:
            self.class_names = list(class_names)
        elif not self.class_names:
            self.class_names = list(SHOT_DISPLAY_NAMES)

        self._last: Optional[ShotPrediction] = None
        self._last_dets: List[ShotActionDetection] = []
        self._recent: Deque[Tuple[int, float]] = deque(maxlen=self.vote_window)
        self._committed_until = -1
        self._hold_frames = int(self.fps * 2.5)
        self._no_batsman = 0
        self._cooldown_until = -1

    def update(
        self,
        frame: np.ndarray,
        frame_number: int,
        tracked_objects: Iterable[TrackedObject],
        extra_boxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
        pitch_box: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[ShotPrediction]:
        del pitch_box
        objs = list(tracked_objects)
        batsman = self._select_batsman(objs)

        if batsman is None:
            self._no_batsman += 1
            if self._no_batsman > int(self.fps * 0.8):
                self.reset()
            return None
        self._no_batsman = 0

        # Hold last confirmed label for HUD / dashboard continuity
        if self._last is not None and frame_number <= self._committed_until:
            return self._last

        if frame_number < self._cooldown_until:
            return self._last if self._last and frame_number <= self._committed_until else None

        if frame_number % self.classify_every_n != 0:
            return None

        # Soft gate: prefer moments when ball is near OR bat/pad visible
        near_ball = self._ball_near_batsman(objs, batsman)
        has_bat = bool(extra_boxes)
        # Always try on batsman crop; boost threshold if neither cue present
        hard_conf = self.min_confidence if (near_ball or has_bat) else (
            self.min_confidence + 0.12
        )

        dets = self._predict_on_batsman(frame, batsman)
        self._last_dets = dets
        chosen = max(dets, key=lambda d: d.confidence) if dets else None
        if chosen is None or chosen.confidence < hard_conf:
            return None

        self._recent.append((chosen.class_id, chosen.confidence))
        voted = self._majority_vote()
        if voted is None:
            return None
        class_index, confidence = voted
        if confidence < self.min_confidence:
            return None

        shot_name = (
            self.class_names[class_index]
            if class_index < len(self.class_names)
            else chosen.class_name
        )
        pred = ShotPrediction(
            frame_number=frame_number,
            timestamp_sec=frame_number / self.fps,
            object_id=batsman.object_id,
            shot_name=shot_name,
            class_index=class_index,
            confidence=confidence,
            bbox=(chosen.x1, chosen.y1, chosen.x2, chosen.y2),
            all_probs=None,
        )
        self._last = pred
        self._committed_until = frame_number + self._hold_frames
        self._cooldown_until = frame_number + max(
            self._hold_frames, int(self.fps * 1.8)
        )
        self._recent.clear()
        return pred

    def _predict_on_batsman(
        self, frame: np.ndarray, batsman: TrackedObject
    ) -> List[ShotActionDetection]:
        h, w = frame.shape[:2]
        bw = max(1.0, batsman.x2 - batsman.x1)
        bh = max(1.0, batsman.y2 - batsman.y1)
        pad_x = bw * self.pad_ratio
        pad_y = bh * self.pad_ratio
        x1 = max(0, int(batsman.x1 - pad_x))
        y1 = max(0, int(batsman.y1 - pad_y))
        x2 = min(w, int(batsman.x2 + pad_x))
        y2 = min(h, int(batsman.y2 + pad_y))
        if x2 - x1 < 40 or y2 - y1 < 40:
            return self._predict_full(frame)

        crop = frame[y1:y2, x1:x2]
        results = self.model.predict(
            source=crop,
            conf=max(0.15, self.min_confidence * 0.75),
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            max_det=5,
        )
        return self._parse_boxes(results[0].boxes, ox=x1, oy=y1)

    def _predict_full(self, frame: np.ndarray) -> List[ShotActionDetection]:
        results = self.model.predict(
            source=frame,
            conf=self.min_confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            max_det=5,
        )
        return self._parse_boxes(results[0].boxes, ox=0, oy=0)

    def _parse_boxes(self, boxes, ox: int, oy: int) -> List[ShotActionDetection]:
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        out: List[ShotActionDetection] = []
        for i in range(len(xyxy)):
            cid = int(cls_ids[i])
            conf = float(confs[i])
            name = (
                self.class_names[cid]
                if 0 <= cid < len(self.class_names)
                else str(self.model.names.get(cid, cid))
            )
            x1, y1, x2, y2 = map(float, xyxy[i])
            out.append(
                ShotActionDetection(
                    class_id=cid,
                    class_name=name,
                    x1=x1 + ox,
                    y1=y1 + oy,
                    x2=x2 + ox,
                    y2=y2 + oy,
                    confidence=conf,
                )
            )
        return out

    def _ball_near_batsman(
        self, tracked: Sequence[TrackedObject], batsman: TrackedObject
    ) -> bool:
        balls = [o for o in tracked if o.class_id == BALL_CLASS_ID]
        if not balls:
            return False
        bx, by = batsman.center
        diag = max(
            40.0,
            ((batsman.x2 - batsman.x1) ** 2 + (batsman.y2 - batsman.y1) ** 2) ** 0.5,
        )
        limit = diag * self.impact_distance_factor
        for ball in balls:
            cx, cy = ball.center
            if ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5 <= limit:
                return True
        return False

    def _majority_vote(self) -> Optional[Tuple[int, float]]:
        if len(self._recent) < self.min_votes:
            return None
        counts = Counter(idx for idx, _ in self._recent)
        best_idx, n = counts.most_common(1)[0]
        if n < self.min_votes:
            return None
        if n < (len(self._recent) + 1) // 2:
            return None
        confs = [c for i, c in self._recent if i == best_idx]
        return best_idx, float(sum(confs) / len(confs))

    def _select_batsman(
        self, tracked_objects: Iterable[TrackedObject]
    ) -> Optional[TrackedObject]:
        batsmen = [o for o in tracked_objects if o.class_id == BATSMAN_CLASS_ID]
        if not batsmen:
            return None
        # Prefer larger boxes (real players, not logos)
        def score(b: TrackedObject) -> float:
            area = max(1.0, (b.x2 - b.x1) * (b.y2 - b.y1))
            return float(b.confidence) * (area ** 0.5)

        if self._last is not None:
            for b in batsmen:
                if b.object_id == self._last.object_id:
                    return b
        return max(batsmen, key=score)

    @property
    def last_prediction(self) -> Optional[ShotPrediction]:
        return self._last

    @property
    def last_detections(self) -> List[ShotActionDetection]:
        return list(self._last_dets)

    def reset(self) -> None:
        self._last = None
        self._last_dets = []
        self._recent.clear()
        self._committed_until = -1
        self._cooldown_until = -1
        self._no_batsman = 0
