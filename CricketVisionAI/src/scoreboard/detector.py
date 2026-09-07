"""
Scoreboard detector for CricketVisionAI.

Uses weights/player_detection.pt class 3 (Scoreboard / ScoreCards).
Does not train — inference only. Returns the best scoreboard crop/box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from src.config.settings import (
    PLAYER_CONFIDENCE,
    PLAYER_DETECTION_WEIGHTS,
    PLAYER_IOU,
)

SCOREBOARD_CLASS_ID = 3


@dataclass
class ScoreboardBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class ScoreboardDetector:
    """Detect Scoreboard with the existing player_detection.pt model."""

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        conf: float = PLAYER_CONFIDENCE,
        iou: float = PLAYER_IOU,
        device: Optional[str] = None,
        pad_ratio: float = 0.02,
    ) -> None:
        self.weights = Path(weights) if weights else PLAYER_DETECTION_WEIGHTS
        self.conf = conf
        self.iou = iou
        self.device = device
        self.pad_ratio = float(pad_ratio)

        if not self.weights.exists():
            raise FileNotFoundError(f"Player weights not found: {self.weights}")

        self.model = YOLO(str(self.weights))

    def detect(self, frame: np.ndarray) -> List[ScoreboardBox]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=[SCOREBOARD_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        out: List[ScoreboardBox] = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            out.append(
                ScoreboardBox(
                    x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(confs[i])
                )
            )
        return out

    def best(self, frame: np.ndarray) -> Optional[ScoreboardBox]:
        boards = self.detect(frame)
        if not boards:
            return None
        return max(boards, key=lambda b: b.confidence)

    def crop(
        self,
        frame: np.ndarray,
        box: ScoreboardBox,
    ) -> Optional[np.ndarray]:
        """Crop only the scoreboard region (with light padding)."""
        h, w = frame.shape[:2]
        bw, bh = box.x2 - box.x1, box.y2 - box.y1
        px, py = bw * self.pad_ratio, bh * self.pad_ratio
        x1 = max(0, int(box.x1 - px))
        y1 = max(0, int(box.y1 - py))
        x2 = min(w, int(box.x2 + px))
        y2 = min(h, int(box.y2 + py))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None
