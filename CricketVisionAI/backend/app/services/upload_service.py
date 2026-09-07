"""Upload service — save incoming videos under input/uploads/."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import ALLOWED_VIDEO_EXTENSIONS, MAX_UPLOAD_MB, UPLOADS_DIR
from app.services.job_store import JobRecord, job_store


async def save_upload(file: UploadFile) -> JobRecord:
    if not file.filename:
        raise ValueError("Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    job_id = uuid.uuid4().hex[:12]
    safe_name = f"{job_id}_{Path(file.filename).name}"
    dest = UPLOADS_DIR / safe_name

    size = 0
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise ValueError(f"File exceeds {MAX_UPLOAD_MB} MB limit")
            out.write(chunk)

    return job_store.create(job_id=job_id, filename=file.filename, video_path=str(dest))
