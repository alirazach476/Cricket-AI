"""POST /process — run the full AI pipeline (background, poll GET /job/{id})."""

from __future__ import annotations

import threading
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.schemas.models import ProcessRequest, ProcessResponse
from app.services.job_store import job_store
from app.services.process_service import resolve_video_path, run_pipeline

router = APIRouter(tags=["process"])

_lock = threading.Lock()
_active: Dict[str, threading.Thread] = {}


def _worker(kwargs: Dict[str, Any]) -> None:
    jid = kwargs.get("job_id") or "unknown"
    try:
        run_pipeline(**kwargs)
    except Exception as exc:  # noqa: BLE001
        job_store.update(
            jid,
            status="failed",
            message=f"Processing failed: {exc}",
            progress=0.0,
            stage="failed",
        )
    finally:
        with _lock:
            _active.pop(str(jid), None)


@router.post("/process", response_model=ProcessResponse)
def process_video(body: ProcessRequest) -> ProcessResponse:
    try:
        jid, path = resolve_video_path(body.job_id, body.video_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    with _lock:
        existing = _active.get(jid)
        if existing is not None and existing.is_alive():
            rec = job_store.get(jid)
            return ProcessResponse(
                status="accepted",
                message=rec.message if rec else "Already processing",
                job_id=jid,
                input_video=str(path),
                output_video=None,
                artifacts={},
                stats={"progress": rec.progress if rec else 0},
            )

        job_store.update(
            jid,
            status="processing",
            message="Queued — loading models…",
            progress=1.0,
            stage="prepare",
        )

        kwargs = {
            "job_id": jid,
            "video_path": str(path),
            "device": body.device,
            "shot_every": body.shot_every,
            "drs_every": body.drs_every,
            "meters_per_pixel": body.meters_per_pixel,
            "generate_highlights": body.generate_highlights,
        }
        t = threading.Thread(target=_worker, args=(kwargs,), daemon=True)
        _active[jid] = t
        t.start()

    return ProcessResponse(
        status="accepted",
        message="Processing started — poll GET /job/{job_id} for progress",
        job_id=jid,
        input_video=str(path),
        output_video=None,
        artifacts={},
        stats={},
    )
