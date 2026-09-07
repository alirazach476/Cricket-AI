"""
CSV logger for ball trajectory and speed analytics.

Writes output/analytics.csv with per-frame ball path and speed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Union

from src.analytics.ball_speed import BallSpeedSample
from src.analytics.ball_trajectory import TrajectoryPoint
from src.config.settings import OUTPUT_DIR

CSV_COLUMNS = [
    "frame_number",
    "timestamp_sec",
    "object_id",
    "detected",
    "raw_x",
    "raw_y",
    "smooth_x",
    "smooth_y",
    "speed_px_per_sec",
    "instant_px_per_sec",
    "speed_kmh",
]

DEFAULT_ANALYTICS_CSV = OUTPUT_DIR / "analytics.csv"


class AnalyticsLogger:
    """Buffer analytics rows and flush to analytics.csv."""

    def __init__(self, csv_path: Optional[Union[str, Path]] = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_ANALYTICS_CSV
        self._rows: List[dict] = []

    def log(
        self,
        point: Optional[TrajectoryPoint],
        speed: Optional[BallSpeedSample],
    ) -> None:
        """Append one row when a trajectory point exists."""
        if point is None:
            return

        speed_px = speed.speed_px_per_sec if speed is not None else 0.0
        instant_px = speed.instant_px_per_sec if speed is not None else 0.0
        speed_kmh = speed.speed_kmh if speed is not None else None

        self._rows.append(
            {
                "frame_number": int(point.frame_number),
                "timestamp_sec": round(point.timestamp_sec, 6),
                "object_id": int(point.object_id),
                "detected": int(bool(point.detected)),
                "raw_x": round(point.raw_x, 2),
                "raw_y": round(point.raw_y, 2),
                "smooth_x": round(point.smooth_x, 2),
                "smooth_y": round(point.smooth_y, 2),
                "speed_px_per_sec": round(speed_px, 2),
                "instant_px_per_sec": round(instant_px, 2),
                "speed_kmh": "" if speed_kmh is None else round(speed_kmh, 2),
            }
        )

    def save(self) -> Path:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)
        return self.csv_path

    def clear(self) -> None:
        self._rows.clear()

    @property
    def row_count(self) -> int:
        return len(self._rows)
