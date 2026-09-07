"""GET /analytics — combined live analytics payload for the dashboard."""

from __future__ import annotations

import csv
import json
from collections import Counter
from typing import Any, Dict, List

from fastapi import APIRouter

from app.config import (
    ANALYTICS_CSV,
    SCOREBOARD_JSON,
    TRACKING_CSV,
)
from app.services.data_readers import (
    list_artifacts,
    read_ball_speed,
    read_highlights,
)
from app.services.job_store import job_store

router = APIRouter(tags=["analytics"])


def _load_scoreboard_json() -> Dict[str, Any]:
    if not SCOREBOARD_JSON.exists():
        return {
            "runs": None,
            "wickets": None,
            "overs": None,
            "crr": None,
            "rrr": None,
            "raw_text": "",
        }
    with SCOREBOARD_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def _detection_counts() -> Dict[str, int]:
    counts: Counter = Counter()
    if not TRACKING_CSV.exists():
        return dict(counts)
    with TRACKING_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("class_name") or "Unknown").strip()
            counts[name] += 1
    return dict(counts)


def _latest_speed() -> Dict[str, Any]:
    data = read_ball_speed()
    points = data.get("points") or []
    last = points[-1] if points else None
    return {
        "latest_px_s": last.speed_px_per_sec if last else None,
        "latest_kmh": last.speed_kmh if last else None,
        "max_px_s": data.get("max_speed_px_per_sec"),
        "avg_px_s": data.get("avg_speed_px_per_sec"),
        "series": [
            {
                "frame": p.frame_number,
                "t": p.timestamp_sec,
                "speed_px_s": p.speed_px_per_sec,
                "speed_kmh": p.speed_kmh,
            }
            for p in points[:: max(1, len(points) // 200 or 1)]  # downsample
        ],
    }


@router.get("/analytics")
def get_analytics() -> Dict[str, Any]:
    speed = _latest_speed()
    board = _load_scoreboard_json()
    highs = read_highlights()
    latest = job_store.latest()

    # Tracked player names from last tracking rows (best-effort)
    batsman = "Batsman"
    fielder = "Fielder"
    umpire = "Umpire"
    bowler = "Bowler"

    return {
        "status": "success",
        "job_id": latest.job_id if latest else None,
        "job_status": latest.status if latest else None,
        "live": {
            "ball_speed_px_s": speed["latest_px_s"],
            "ball_speed_kmh": speed["latest_kmh"],
            "shot_type": None,
            "shot_confidence": None,
            "batsman": batsman,
            "fielder": fielder,
            "umpire": umpire,
            "bowler": bowler,
            "runs": board.get("runs"),
            "wickets": board.get("wickets"),
            "overs": board.get("overs"),
            "crr": board.get("crr"),
            "rrr": board.get("rrr"),
        },
        "ball_speed": speed,
        "shot_analysis": {"count": 0, "distribution": {}, "top_shot": None, "shots": []},
        "scoreboard": board,
        "detection_counts": _detection_counts(),
        "highlights": {"count": highs["count"], "items": [i.model_dump() for i in highs["items"]]},
        "artifacts": list_artifacts(),
        "stats": latest.stats if latest else {},
    }
