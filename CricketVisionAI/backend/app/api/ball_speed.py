"""GET /ball-speed — return ball speed analytics as JSON."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.models import BallSpeedResponse
from app.services.data_readers import read_ball_speed

router = APIRouter(tags=["ball-speed"])


@router.get("/ball-speed", response_model=BallSpeedResponse)
def get_ball_speed() -> BallSpeedResponse:
    data = read_ball_speed()
    return BallSpeedResponse(
        status="success",
        count=data["count"],
        max_speed_px_per_sec=data["max_speed_px_per_sec"],
        avg_speed_px_per_sec=data["avg_speed_px_per_sec"],
        points=data["points"],
    )
