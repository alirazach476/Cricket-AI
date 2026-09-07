"""
Scoreboard pipeline helper: detect → crop → OCR → JSON.
Drawing overlay lives in src/utils/draw.py (draw_scoreboard_reading).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from src.config.settings import OUTPUT_DIR
from src.scoreboard.detector import ScoreboardBox, ScoreboardDetector
from src.scoreboard.ocr_reader import ScoreboardOCR
from src.scoreboard.parser import ScoreboardReading

DEFAULT_SCOREBOARD_JSON = OUTPUT_DIR / "scoreboard.json"


class ScoreboardReader:
    """
    End-to-end scoreboard extraction for one frame (or last good reading).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        ocr_gpu: bool = False,
        every_n_frames: int = 30,
    ) -> None:
        self.detector = ScoreboardDetector(device=device)
        self.ocr = ScoreboardOCR(gpu=ocr_gpu)
        self.every_n_frames = max(1, int(every_n_frames))
        self._last: Optional[ScoreboardReading] = None
        self._last_box: Optional[ScoreboardBox] = None

    def update(
        self,
        frame: np.ndarray,
        frame_number: int,
    ) -> Tuple[Optional[ScoreboardReading], Optional[ScoreboardBox]]:
        """
        Run OCR every N frames; return cached reading between updates.
        """
        if frame_number % self.every_n_frames != 0:
            return self._last, self._last_box

        box = self.detector.best(frame)
        if box is None:
            return self._last, self._last_box

        crop = self.detector.crop(frame, box)
        if crop is None:
            return self._last, self._last_box

        reading = self.ocr.read_crop(crop)
        self._last = reading
        self._last_box = box
        return reading, box

    def save_json(
        self,
        path: Optional[Union[str, Path]] = None,
        frame_number: Optional[int] = None,
    ) -> Optional[Path]:
        if self._last is None:
            return None
        extra = {}
        if frame_number is not None:
            extra["frame_number"] = frame_number
        if self._last_box is not None:
            extra["bbox"] = {
                "x1": self._last_box.x1,
                "y1": self._last_box.y1,
                "x2": self._last_box.x2,
                "y2": self._last_box.y2,
                "confidence": self._last_box.confidence,
            }
        return self.ocr.save_json(
            self._last,
            path=path or DEFAULT_SCOREBOARD_JSON,
            extra=extra,
        )

    @property
    def last_reading(self) -> Optional[ScoreboardReading]:
        return self._last
