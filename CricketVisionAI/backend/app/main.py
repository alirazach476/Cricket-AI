"""
CricketVisionAI FastAPI application.

Run from backend/:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Or from CricketVisionAI/:
    uvicorn app.main:app --app-dir backend --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import (
    analytics_router,
    annotated_video_router,
    ball_speed_router,
    highlights_router,
    job_router,
    process_router,
    reports_router,
    scoreboard_router,
    shot_analysis_router,
    trajectory_router,
    upload_router,
)
from app.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    HIGHLIGHTS_DIR,
    OUTPUT_DIR,
)

app = FastAPI(title=API_TITLE, version=API_VERSION, description=API_DESCRIPTION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for highlight clips / output assets
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount(
    "/static/highlights",
    StaticFiles(directory=str(HIGHLIGHTS_DIR)),
    name="highlights",
)

app.include_router(upload_router)
app.include_router(process_router)
app.include_router(job_router)
app.include_router(ball_speed_router)
app.include_router(shot_analysis_router)
app.include_router(scoreboard_router)
app.include_router(highlights_router)
app.include_router(reports_router)
app.include_router(annotated_video_router)
app.include_router(trajectory_router)
app.include_router(analytics_router)


@app.get("/")
def root():
    return {
        "status": "success",
        "message": "CricketVisionAI API",
        "version": API_VERSION,
        "endpoints": [
            "POST /upload",
            "POST /process",
            "GET /job/{job_id}",
            "GET /annotated-video",
            "GET /analytics",
            "GET /trajectory",
            "GET /ball-speed",
            "GET /shot-analysis",
            "GET /scoreboard",
            "GET /highlights",
            "GET /reports",
            "GET /docs",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
