"""GET /trajectory — ball path points synced to video frames."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from app.config import ANALYTICS_CSV, OUTPUT_DIR

router = APIRouter(tags=["trajectory"])

TRAJECTORY_JSON = OUTPUT_DIR / "trajectory.json"


@router.get("/trajectory")
def get_trajectory() -> Dict[str, Any]:
    points: List[Dict[str, Any]] = []

    if TRAJECTORY_JSON.exists():
        with TRAJECTORY_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "success", **data}

    if ANALYTICS_CSV.exists():
        with ANALYTICS_CSV.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    points.append(
                        {
                            "frame_number": int(float(row.get("frame_number", 0))),
                            "timestamp_sec": float(row.get("timestamp_sec", 0) or 0),
                            "x": float(row.get("smooth_x", 0) or 0),
                            "y": float(row.get("smooth_y", 0) or 0),
                            "speed_px_per_sec": float(
                                row.get("speed_px_per_sec", 0) or 0
                            ),
                        }
                    )
                except (TypeError, ValueError):
                    continue

    return {
        "status": "success",
        "count": len(points),
        "points": points,
    }
