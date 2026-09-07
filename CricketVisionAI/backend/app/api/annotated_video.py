"""GET /annotated-video — stream the processed match video for web playback."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.config import ANNOTATED_VIDEO

router = APIRouter(tags=["annotated-video"])


@router.get("/annotated-video")
def get_annotated_video(request: Request):
    if not ANNOTATED_VIDEO.exists():
        raise HTTPException(
            status_code=404,
            detail="Annotated video not found. Run POST /process first.",
        )
    # Inline playback in <video> — not forced download
    return FileResponse(
        path=str(ANNOTATED_VIDEO),
        media_type="video/mp4",
        filename="annotated_output.mp4",
        content_disposition_type="inline",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


@router.get("/annotated-video/info")
def annotated_video_info():
    exists = ANNOTATED_VIDEO.exists()
    return {
        "status": "success",
        "exists": exists,
        "path": str(ANNOTATED_VIDEO),
        "url": "/annotated-video" if exists else None,
        "static_url": "/static/output/annotated_output.mp4" if exists else None,
        "size_bytes": ANNOTATED_VIDEO.stat().st_size if exists else 0,
    }
