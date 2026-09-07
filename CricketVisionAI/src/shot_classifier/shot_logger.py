"""
CSV logger for shot classification results.

Writes output/shots.csv with shot name and confidence per prediction.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Union

from src.config.settings import OUTPUT_DIR, SHOT_CONFIDENCE
from src.shot_classifier.classifier import ShotPrediction

CSV_COLUMNS = [
    "frame_number",
    "timestamp_sec",
    "object_id",
    "shot_name",
    "class_index",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]

DEFAULT_SHOTS_CSV = OUTPUT_DIR / "shots.csv"


class ShotLogger:
    """Buffer shot predictions and flush to shots.csv."""

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        min_confidence: float = SHOT_CONFIDENCE,
        log_only_on_update: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_SHOTS_CSV
        self.min_confidence = float(min_confidence)
        self.log_only_on_update = log_only_on_update
        self._rows: List[dict] = []
        self._last_logged_frame: Optional[int] = None

    def log(self, prediction: Optional[ShotPrediction]) -> None:
        """Append a row when prediction is fresh enough and above threshold."""
        if prediction is None:
            return
        if prediction.confidence < self.min_confidence:
            return
        if (
            self.log_only_on_update
            and self._last_logged_frame is not None
            and prediction.frame_number == self._last_logged_frame
        ):
            return

        x1, y1, x2, y2 = prediction.bbox
        self._rows.append(
            {
                "frame_number": int(prediction.frame_number),
                "timestamp_sec": round(prediction.timestamp_sec, 6),
                "object_id": int(prediction.object_id),
                "shot_name": prediction.shot_name,
                "class_index": int(prediction.class_index),
                "confidence": round(prediction.confidence, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        )
        self._last_logged_frame = prediction.frame_number

    def save(self) -> Path:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)
        return self.csv_path

    def clear(self) -> None:
        self._rows.clear()
        self._last_logged_frame = None

    @property
    def row_count(self) -> int:
        return len(self._rows)
