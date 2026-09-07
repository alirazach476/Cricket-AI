"""
Persistent job store for upload / process state.

Survives uvicorn --reload and server restarts by writing JSON to disk.
Also recovers orphaned uploads named ``{job_id}_*`` under input/uploads/.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import OUTPUT_DIR, UPLOADS_DIR

JOBS_FILE = OUTPUT_DIR / "jobs_state.json"


@dataclass
class JobRecord:
    job_id: str
    filename: str
    video_path: str
    created_at: str
    status: str = "uploaded"  # uploaded | processing | completed | failed
    message: str = ""
    progress: float = 0.0  # 0–100
    stage: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


class JobStore:
    def __init__(self, persist_path: Path = JOBS_FILE) -> None:
        self._lock = threading.RLock()
        self._jobs: Dict[str, JobRecord] = {}
        self._latest_id: Optional[str] = None
        self._path = Path(persist_path)
        self._load()

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._latest_id = data.get("latest_id")
            for jid, raw in (data.get("jobs") or {}).items():
                self._jobs[jid] = JobRecord(
                    job_id=raw.get("job_id", jid),
                    filename=raw.get("filename", ""),
                    video_path=raw.get("video_path", ""),
                    created_at=raw.get("created_at", ""),
                    status=raw.get("status", "uploaded"),
                    message=raw.get("message", ""),
                    progress=float(raw.get("progress", 0.0) or 0.0),
                    stage=raw.get("stage", ""),
                    artifacts=raw.get("artifacts") or {},
                    stats=raw.get("stats") or {},
                )
        except Exception:
            self._jobs = {}
            self._latest_id = None

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "latest_id": self._latest_id,
                "jobs": {jid: asdict(rec) for jid, rec in self._jobs.items()},
            }
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            pass

    def create(self, job_id: str, filename: str, video_path: str) -> JobRecord:
        with self._lock:
            rec = JobRecord(
                job_id=job_id,
                filename=filename,
                video_path=video_path,
                created_at=datetime.now(timezone.utc).isoformat(),
                status="uploaded",
                message="Video uploaded",
                progress=0.0,
                stage="uploaded",
            )
            self._jobs[job_id] = rec
            self._latest_id = job_id
            self._save()
            return rec

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                return rec
            # Recover from uploaded file after server restart
            recovered = self._recover_from_upload(job_id)
            return recovered

    def _recover_from_upload(self, job_id: str) -> Optional[JobRecord]:
        try:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            matches = sorted(
                UPLOADS_DIR.glob(f"{job_id}_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not matches:
                return None
            path = matches[0]
            # Strip "{job_id}_" prefix for display name
            name = path.name
            prefix = f"{job_id}_"
            filename = name[len(prefix) :] if name.startswith(prefix) else name
            rec = JobRecord(
                job_id=job_id,
                filename=filename,
                video_path=str(path.resolve()),
                created_at=datetime.now(timezone.utc).isoformat(),
                status="uploaded",
                message="Recovered upload after server restart",
                progress=0.0,
                stage="uploaded",
            )
            self._jobs[job_id] = rec
            self._latest_id = job_id
            self._save()
            return rec
        except Exception:
            return None

    def latest(self) -> Optional[JobRecord]:
        with self._lock:
            if self._latest_id and self._latest_id in self._jobs:
                return self._jobs[self._latest_id]
            if self._jobs:
                # pick newest by created_at
                rec = max(self._jobs.values(), key=lambda r: r.created_at or "")
                self._latest_id = rec.job_id
                return rec
            # Scan uploads folder
            try:
                files = sorted(
                    UPLOADS_DIR.glob("*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for path in files:
                    if not path.is_file():
                        continue
                    name = path.name
                    jid = name.split("_", 1)[0] if "_" in name else None
                    if not jid or len(jid) < 8:
                        continue
                    return self._recover_from_upload(jid)
            except Exception:
                pass
            return None

    def update(self, job_id: str, **kwargs: Any) -> Optional[JobRecord]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                rec = self._recover_from_upload(job_id)
            if rec is None:
                return None
            for k, v in kwargs.items():
                if hasattr(rec, k):
                    setattr(rec, k, v)
            self._save()
            return rec


job_store = JobStore()
