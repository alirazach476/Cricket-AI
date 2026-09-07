"""
CSV logger for ByteTrack player tracking results.

Writes one row per tracked object per frame to output/tracking.csv.
Columns: frame_number, timestamp_sec, object_id, class_id, class_name,
         x1, y1, x2, y2, center_x, center_y, confidence
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Union

from src.config.settings import OUTPUT_DIR
from src.tracking.models import TrackedObject

CSV_COLUMNS = [
    "frame_number",
    "timestamp_sec",
    "object_id",
    "class_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "confidence",
]

DEFAULT_TRACKING_CSV = OUTPUT_DIR / "tracking.csv"


class TrackingLogger:
    """
    Collect tracking rows in memory and flush them to tracking.csv.

    Frame numbering is 0-based.
    timestamp_sec = frame_number / fps
    """

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        fps: float = 30.0,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_TRACKING_CSV
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._rows: List[dict] = []

    def log_frame(
        self,
        frame_number: int,
        tracked_objects: Iterable[TrackedObject],
    ) -> None:
        """Append one CSV row for each tracked object on this frame."""
        timestamp_sec = frame_number / self.fps

        for obj in tracked_objects:
            cx, cy = obj.center
            self._rows.append(
                {
                    "frame_number": int(frame_number),
                    "timestamp_sec": round(timestamp_sec, 6),
                    "object_id": int(obj.object_id),
                    "class_id": int(obj.class_id),
                    "class_name": obj.class_name,
                    "x1": round(obj.x1, 2),
                    "y1": round(obj.y1, 2),
                    "x2": round(obj.x2, 2),
                    "y2": round(obj.y2, 2),
                    "center_x": round(cx, 2),
                    "center_y": round(cy, 2),
                    "confidence": round(obj.confidence, 4),
                }
            )

    def save(self) -> Path:
        """Write all collected rows to tracking.csv. Returns the file path."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)

        return self.csv_path

    def clear(self) -> None:
        """Drop buffered rows (e.g. before a new video)."""
        self._rows.clear()

    @property
    def row_count(self) -> int:
        return len(self._rows)
