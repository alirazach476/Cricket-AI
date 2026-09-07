"""
DRS object detector for CricketVisionAI.

Loads weights/drs_detection.pt (no training).
Detects: Bat (0), Pad (1), Pitch (2).
Stumps (3) are intentionally disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from ultralytics import YOLO

from src.config.settings import (
    DRS_CLASS_NAMES,
    DRS_CONFIDENCE,
    DRS_DETECTION_WEIGHTS,
    DRS_IMGSZ,
    DRS_IOU,
    STUMPS_CLASS_ID,
)

PITCH_CLASS_ID = 2
# Bat, Pad, Pitch only — no stumps
DEFAULT_DRS_CLASSES = (0, 1, 2)


@dataclass
class DrsDetection:
    """One DRS detection on a single frame."""

    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class DrsDetector:
    """Load drs_detection.pt once; detect Bat / Pad / Pitch (no stumps)."""

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        conf: float = DRS_CONFIDENCE,
        stumps_conf: float = 0.0,  # kept for API compatibility; unused
        iou: float = DRS_IOU,
        imgsz: int = DRS_IMGSZ,
        class_ids: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
        detect_stumps: bool = False,
    ) -> None:
        del stumps_conf
        self.weights = Path(weights) if weights else DRS_DETECTION_WEIGHTS
        self.conf = conf
        self.iou = iou
        self.imgsz = int(imgsz)
        self.detect_stumps = bool(detect_stumps)
        if class_ids is not None:
            ids = tuple(int(c) for c in class_ids)
        else:
            ids = DEFAULT_DRS_CLASSES
        if not self.detect_stumps:
            ids = tuple(c for c in ids if c != STUMPS_CLASS_ID)
        self.class_ids = ids
        self.device = device

        if not self.weights.exists():
            raise FileNotFoundError(f"DRS weights not found: {self.weights}")

        self.model = YOLO(str(self.weights))

    def detect(self, frame: np.ndarray) -> List[DrsDetection]:
        """Run DRS detection for Bat / Pad / Pitch only."""
        kwargs = {
            "source": frame,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
            "max_det": 50,
            "classes": list(self.class_ids),
        }
        results = self.model.predict(**kwargs)
        dets = self._parse(results[0])
        # Keep only the best pitch box to avoid drawing fake "Pitch" labels
        best = self.best_pitch(dets)
        others = [d for d in dets if d.class_id != PITCH_CLASS_ID]
        return (others + [best]) if best is not None else others

    def best_pitch(self, detections: Sequence[DrsDetection]) -> Optional[DrsDetection]:
        """Pick largest confident pitch; reject tiny false positives (logos/UI)."""
        pitches = []
        for d in detections:
            if d.class_id != PITCH_CLASS_ID:
                continue
            if d.confidence < max(self.conf, 0.35):
                continue
            w = d.x2 - d.x1
            h = d.y2 - d.y1
            area = w * h
            if area < 8000:  # ignore tiny "Pitch" on ads / names
                continue
            if w < 80 or h < 80:
                continue
            pitches.append(d)
        if not pitches:
            return None
        # Prefer large + confident
        return max(pitches, key=lambda d: d.confidence * (d.x2 - d.x1) * (d.y2 - d.y1))

    def best_stumps(
        self, detections: Sequence[DrsDetection]
    ) -> Optional[DrsDetection]:
        # Always disabled
        del detections
        return None

    def _parse(self, result) -> List[DrsDetection]:
        out: List[DrsDetection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            class_id = int(cls_ids[i])
            if class_id == STUMPS_CLASS_ID and not self.detect_stumps:
                continue
            if class_id not in self.class_ids:
                continue
            conf = float(confs[i])
            if conf < self.conf:
                continue
            x1, y1, x2, y2 = map(float, xyxy[i])
            out.append(
                DrsDetection(
                    class_id=class_id,
                    class_name=DRS_CLASS_NAMES.get(class_id, str(class_id)),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                )
            )
        return out
