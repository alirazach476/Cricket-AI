"""GET /shot-analysis — return shot classification results as JSON."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.models import ShotAnalysisResponse
from app.services.data_readers import read_shots

router = APIRouter(tags=["shot-analysis"])


@router.get("/shot-analysis", response_model=ShotAnalysisResponse)
def get_shot_analysis() -> ShotAnalysisResponse:
    data = read_shots()
    return ShotAnalysisResponse(
        status="success",
        count=data["count"],
        distribution=data["distribution"],
        top_shot=data["top_shot"],
        shots=data["shots"],
    )
