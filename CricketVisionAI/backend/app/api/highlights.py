"""GET /highlights — return highlight clip index as JSON."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.models import HighlightsResponse
from app.services.data_readers import read_highlights

router = APIRouter(tags=["highlights"])


@router.get("/highlights", response_model=HighlightsResponse)
def get_highlights() -> HighlightsResponse:
    data = read_highlights()
    return HighlightsResponse(
        status="success",
        count=data["count"],
        highlights_dir=data["highlights_dir"],
        items=data["items"],
    )
