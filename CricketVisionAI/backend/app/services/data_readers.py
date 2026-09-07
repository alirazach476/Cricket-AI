"""
CSV / artifact readers for JSON API endpoints.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import (
    ANALYTICS_CSV,
    ANNOTATED_VIDEO,
    HIGHLIGHTS_DIR,
    HIGHLIGHTS_INDEX,
    PITCH_MAP_PNG,
    SHOTS_CSV,
    TRACKING_CSV,
    WAGON_WHEEL_PNG,
)
from app.schemas.models import (
    BallSpeedPoint,
    HighlightItem,
    ScoreboardDetection,
    ShotItem,
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def read_ball_speed(analytics_csv: Path = ANALYTICS_CSV) -> Dict[str, Any]:
    rows = _read_csv(analytics_csv)
    points: List[BallSpeedPoint] = []
    for r in rows:
        speed = _f(r, "speed_px_per_sec")
        kmh_raw = r.get("speed_kmh", "")
        speed_kmh = None
        if kmh_raw not in ("", None):
            try:
                speed_kmh = float(kmh_raw)
            except ValueError:
                speed_kmh = None
        points.append(
            BallSpeedPoint(
                frame_number=_i(r, "frame_number"),
                timestamp_sec=_f(r, "timestamp_sec"),
                speed_px_per_sec=speed,
                instant_px_per_sec=_f(r, "instant_px_per_sec"),
                speed_kmh=speed_kmh,
                object_id=_i(r, "object_id") if r.get("object_id") not in ("", None) else None,
            )
        )
    speeds = [p.speed_px_per_sec for p in points]
    return {
        "count": len(points),
        "max_speed_px_per_sec": max(speeds) if speeds else None,
        "avg_speed_px_per_sec": (sum(speeds) / len(speeds)) if speeds else None,
        "points": points,
    }


def read_shots(shots_csv: Path = SHOTS_CSV) -> Dict[str, Any]:
    rows = _read_csv(shots_csv)
    shots: List[ShotItem] = []
    for r in rows:
        shots.append(
            ShotItem(
                frame_number=_i(r, "frame_number"),
                timestamp_sec=_f(r, "timestamp_sec"),
                object_id=_i(r, "object_id"),
                shot_name=r.get("shot_name", "Unknown"),
                class_index=_i(r, "class_index"),
                confidence=_f(r, "confidence"),
                x1=_f(r, "x1"),
                y1=_f(r, "y1"),
                x2=_f(r, "x2"),
                y2=_f(r, "y2"),
            )
        )
    dist = Counter(s.shot_name for s in shots)
    top = max(shots, key=lambda s: s.confidence) if shots else None
    return {
        "count": len(shots),
        "distribution": dict(dist),
        "top_shot": top,
        "shots": shots,
    }


def read_scoreboard(tracking_csv: Path = TRACKING_CSV) -> Dict[str, Any]:
    """
    Scoreboard is class_id 3 in player_detection, but tracking currently
    logs only Ball/Batsman/Bowler. We still scan tracking.csv for class_name
    Scoreboard if present; otherwise return empty with a clear message.
    """
    rows = _read_csv(tracking_csv)
    detections: List[ScoreboardDetection] = []
    for r in rows:
        name = (r.get("class_name") or "").strip().lower()
        class_id = _i(r, "class_id", -1)
        if name in ("scoreboard", "scorecards") or class_id == 3:
            detections.append(
                ScoreboardDetection(
                    frame_number=_i(r, "frame_number"),
                    timestamp_sec=_f(r, "timestamp_sec"),
                    object_id=_i(r, "object_id") if r.get("object_id") else None,
                    confidence=_f(r, "confidence") if r.get("confidence") else None,
                    x1=_f(r, "x1"),
                    y1=_f(r, "y1"),
                    x2=_f(r, "x2"),
                    y2=_f(r, "y2"),
                )
            )
    return {
        "detected": len(detections) > 0,
        "count": len(detections),
        "detections": detections,
        "message": (
            f"Found {len(detections)} scoreboard row(s)."
            if detections
            else "No scoreboard rows in tracking.csv (tracker filters to Ball/Batsman/Bowler)."
        ),
    }


def read_highlights(index_csv: Path = HIGHLIGHTS_INDEX) -> Dict[str, Any]:
    rows = _read_csv(index_csv)
    items: List[HighlightItem] = []
    for r in rows:
        fname = r.get("file") or None
        items.append(
            HighlightItem(
                index=_i(r, "index") if r.get("index") else None,
                type=r.get("type", "Unknown"),
                label=r.get("label", ""),
                frame_number=_i(r, "frame_number"),
                timestamp_sec=_f(r, "timestamp_sec"),
                start_frame=_i(r, "start_frame") if r.get("start_frame") else None,
                end_frame=_i(r, "end_frame") if r.get("end_frame") else None,
                score=_f(r, "score") if r.get("score") not in ("", None) else None,
                file=fname,
                url=f"/static/highlights/{fname}" if fname else None,
            )
        )
    return {
        "count": len(items),
        "highlights_dir": str(HIGHLIGHTS_DIR),
        "items": items,
    }


def list_artifacts() -> Dict[str, Any]:
    def info(path: Path) -> Dict[str, Any]:
        return {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }

    return {
        "annotated_video": info(ANNOTATED_VIDEO),
        "tracking_csv": info(TRACKING_CSV),
        "analytics_csv": info(ANALYTICS_CSV),
        "shots_csv": info(SHOTS_CSV),
        "pitch_map": info(PITCH_MAP_PNG),
        "wagon_wheel": info(WAGON_WHEEL_PNG),
        "highlights_index": info(HIGHLIGHTS_INDEX),
    }
