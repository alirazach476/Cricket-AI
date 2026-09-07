"""
Dedicated cricket-ball detector from cricket_ball_detection/best.pt.

Returns candidate boxes; association / velocity gating happens in the tracker
so grass spots and logos are not chased as the ball.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from ultralytics import YOLO

from src.config.settings import (
    BALL_CLASS_ID,
    BALL_CONFIDENCE,
    BALL_DETECTION_WEIGHTS,
    BALL_IMGSZ,
    BALL_MAX_ASPECT,
    BALL_MAX_SIDE_PX,
    BALL_MIN_SIDE_PX,
    PLAYER_CLASS_NAMES,
)
from src.tracking.models import TrackedObject

BBox = Tuple[float, float, float, float]


class DedicatedBallDetector:
    """YOLO ball model — produce filtered candidates for association."""

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        conf: float = BALL_CONFIDENCE,
        imgsz: int = BALL_IMGSZ,
        device: Optional[str] = None,
        ball_track_id: int = 1,
    ) -> None:
        self.weights = Path(weights) if weights else BALL_DETECTION_WEIGHTS
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        self.ball_track_id = int(ball_track_id)

        if not self.weights.exists():
            raise FileNotFoundError(f"Ball weights not found: {self.weights}")

        self.model = YOLO(str(self.weights))
        names = self.model.names or {0: "ball"}
        self._ball_class_ids = tuple(int(k) for k in names.keys())
        self._preferred_ids = {
            int(k)
            for k, v in names.items()
            if str(v).strip().lower() in ("ball", "cricket_ball", "cricket-ball")
        }

    def detect_candidates(
        self,
        frame: np.ndarray,
        pitch_box: Optional[BBox] = None,
        conf: Optional[float] = None,
    ) -> List[TrackedObject]:
        """All shape-filtered ball candidates (may be empty)."""
        h, w = frame.shape[:2]
        source = frame
        ox = oy = 0
        thr = float(conf) if conf is not None else self.conf

        if pitch_box is not None:
            x1, y1, x2, y2 = map(int, pitch_box)
            # Delivery often needs air above / beside pitch — moderate pad
            pad = int(max(50, 0.22 * max(x2 - x1, y2 - y1)))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            if x2 - x1 > 40 and y2 - y1 > 40:
                source = frame[y1:y2, x1:x2]
                ox, oy = x1, y1

        results = self.model.predict(
            source=source,
            conf=max(0.15, thr * 0.7),  # keep a few extras; score later
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            max_det=15,
            classes=list(self._ball_class_ids),
        )
        boxes = results[0].boxes
        # If ROI crop misses sky/ground travel, retry full frame.
        if (boxes is None or len(boxes) == 0) and source is not frame:
            results = self.model.predict(
                source=frame,
                conf=max(0.15, thr * 0.7),
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                max_det=20,
                classes=list(self._ball_class_ids),
            )
            boxes = results[0].boxes
            ox, oy = 0, 0
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        out: List[TrackedObject] = []
        for i in range(len(xyxy)):
            conf_i = float(confs[i])
            x1, y1, x2, y2 = map(float, xyxy[i])
            x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
            if not self._looks_like_ball(x1, y1, x2, y2):
                continue
            cid = int(cls_ids[i])
            boost = 0.06 if cid in self._preferred_ids else 0.0
            out.append(
                TrackedObject(
                    object_id=self.ball_track_id,
                    class_id=BALL_CLASS_ID,
                    class_name=PLAYER_CLASS_NAMES.get(BALL_CLASS_ID, "Ball"),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=min(0.99, conf_i + boost),
                )
            )
        return out

    def detect(
        self,
        frame: np.ndarray,
        pitch_box: Optional[BBox] = None,
    ) -> List[TrackedObject]:
        """Legacy: best single candidate by confidence (association preferred)."""
        cands = self.detect_candidates(frame, pitch_box=pitch_box)
        if not cands:
            return []
        best = max(cands, key=lambda o: o.confidence)
        if best.confidence < self.conf:
            return []
        return [best]

    @staticmethod
    def _near_pitch(cx: float, cy: float, pitch_box: BBox, pad_ratio: float) -> bool:
        px1, py1, px2, py2 = pitch_box
        pw, ph = px2 - px1, py2 - py1
        pad = max(40.0, pad_ratio * max(pw, ph))
        return px1 - pad <= cx <= px2 + pad and py1 - pad <= cy <= py2 + pad

    @staticmethod
    def _looks_like_ball(x1: float, y1: float, x2: float, y2: float) -> bool:
        bw, bh = x2 - x1, y2 - y1
        if bw < BALL_MIN_SIDE_PX or bh < BALL_MIN_SIDE_PX:
            return False
        if bw > BALL_MAX_SIDE_PX or bh > BALL_MAX_SIDE_PX:
            return False
        aspect = max(bw, bh) / max(1e-3, min(bw, bh))
        return aspect <= BALL_MAX_ASPECT
