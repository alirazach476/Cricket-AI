"""POST /upload — accept a cricket video."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.models import UploadResponse
from app.services.upload_service import save_upload

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)) -> UploadResponse:
    try:
        rec = await save_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    size = Path(rec.video_path).stat().st_size if Path(rec.video_path).exists() else 0
    return UploadResponse(
        status="success",
        message="Video uploaded successfully",
        job_id=rec.job_id,
        filename=rec.filename,
        path=rec.video_path,
        size_bytes=size,
    )
