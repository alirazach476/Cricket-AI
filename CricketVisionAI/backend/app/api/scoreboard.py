"""GET /scoreboard — OCR JSON + detection summary."""

from __future__ import annotations

import json

from fastapi import APIRouter

from app.config import SCOREBOARD_JSON
from app.schemas.models import ScoreboardResponse
from app.services.data_readers import read_scoreboard

router = APIRouter(tags=["scoreboard"])


@router.get("/scoreboard")
def get_scoreboard():
    ocr = {}
    if SCOREBOARD_JSON.exists():
        with SCOREBOARD_JSON.open("r", encoding="utf-8") as f:
            ocr = json.load(f)

    tracking = read_scoreboard()
    return {
        "status": "success",
        "message": tracking["message"],
        "detected": bool(ocr.get("runs") is not None or tracking["detected"]),
        "count": tracking["count"],
        "ocr": {
            "runs": ocr.get("runs"),
            "wickets": ocr.get("wickets"),
            "overs": ocr.get("overs"),
            "crr": ocr.get("crr"),
            "rrr": ocr.get("rrr"),
            "raw_text": ocr.get("raw_text", ""),
            "confidence": ocr.get("confidence"),
        },
        "detections": [d.model_dump() for d in tracking["detections"]],
        "note": (
            "Scoreboard crop from player_detection.pt class Scoreboard; "
            "text via EasyOCR."
        ),
    }
