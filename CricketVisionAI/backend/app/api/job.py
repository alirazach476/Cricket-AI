"""GET /job/{job_id} — poll upload / process progress."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.job_store import job_store

router = APIRouter(tags=["job"])


@router.get("/job/{job_id}")
def get_job(job_id: str):
    rec = job_store.get(job_id)
    if rec is None:
        # One more recovery attempt via latest upload with same id prefix
        rec = job_store.get(job_id)
    if rec is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown job_id: {job_id}. "
                "Upload the video again (server may have restarted)."
            ),
        )
    return {
        "status": "success",
        "job_id": rec.job_id,
        "job_status": rec.status,
        "message": rec.message,
        "progress": rec.progress,
        "stage": rec.stage,
        "filename": rec.filename,
        "artifacts": rec.artifacts,
        "stats": rec.stats,
        "input_video": rec.video_path,
        "output_video": rec.artifacts.get("annotated_video") if rec.artifacts else None,
    }


@router.get("/job")
def get_latest_job():
    rec = job_store.latest()
    if rec is None:
        raise HTTPException(status_code=404, detail="No jobs yet")
    return {
        "status": "success",
        "job_id": rec.job_id,
        "job_status": rec.status,
        "message": rec.message,
        "progress": rec.progress,
        "stage": rec.stage,
        "filename": rec.filename,
        "artifacts": rec.artifacts,
        "stats": rec.stats,
        "input_video": rec.video_path,
        "output_video": rec.artifacts.get("annotated_video") if rec.artifacts else None,
    }
