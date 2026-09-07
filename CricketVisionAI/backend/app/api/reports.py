"""GET /reports — combined JSON report of all analytics."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.models import ReportsResponse
from app.services.data_readers import (
    list_artifacts,
    read_ball_speed,
    read_highlights,
    read_scoreboard,
    read_shots,
)
from app.services.job_store import job_store

router = APIRouter(tags=["reports"])


@router.get("/reports", response_model=ReportsResponse)
def get_reports() -> ReportsResponse:
    speed = read_ball_speed()
    shots = read_shots()
    board = read_scoreboard()
    highs = read_highlights()
    artifacts = list_artifacts()
    latest = job_store.latest()

    ball_speed_summary = {
        "count": speed["count"],
        "max_speed_px_per_sec": speed["max_speed_px_per_sec"],
        "avg_speed_px_per_sec": speed["avg_speed_px_per_sec"],
    }
    shot_summary = {
        "count": shots["count"],
        "distribution": shots["distribution"],
        "top_shot": shots["top_shot"].model_dump() if shots["top_shot"] else None,
    }
    scoreboard_summary = {
        "detected": board["detected"],
        "count": board["count"],
        "message": board["message"],
    }
    highlights_summary = {
        "count": highs["count"],
        "items": [i.model_dump() for i in highs["items"]],
    }

    summary = {
        "job_id": latest.job_id if latest else None,
        "job_status": latest.status if latest else None,
        "frames_processed": (latest.stats or {}).get("frames_processed") if latest else None,
        "input_video": latest.video_path if latest else None,
    }

    return ReportsResponse(
        status="success",
        job_id=latest.job_id if latest else None,
        summary=summary,
        ball_speed=ball_speed_summary,
        shot_analysis=shot_summary,
        scoreboard=scoreboard_summary,
        highlights=highlights_summary,
        artifacts=artifacts,
    )
