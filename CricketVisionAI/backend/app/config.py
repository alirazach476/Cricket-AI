"""
Backend configuration for CricketVisionAI API.

Paths point at the existing project root (weights/, input/, output/).
"""

from __future__ import annotations

from pathlib import Path

# backend/app/config.py → parents[2] = CricketVisionAI/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
HIGHLIGHTS_DIR = OUTPUT_DIR / "highlights"
UPLOADS_DIR = INPUT_DIR / "uploads"

# Ensure runtime dirs exist
for _d in (INPUT_DIR, OUTPUT_DIR, HIGHLIGHTS_DIR, UPLOADS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Default artifacts produced by main.py / pipeline
TRACKING_CSV = OUTPUT_DIR / "tracking.csv"
ANALYTICS_CSV = OUTPUT_DIR / "analytics.csv"
SHOTS_CSV = OUTPUT_DIR / "shots.csv"
ANNOTATED_VIDEO = OUTPUT_DIR / "annotated_output.mp4"
PITCH_MAP_PNG = OUTPUT_DIR / "pitch_map.png"
WAGON_WHEEL_PNG = OUTPUT_DIR / "wagon_wheel.png"
HIGHLIGHTS_INDEX = HIGHLIGHTS_DIR / "highlights_index.csv"
SCOREBOARD_JSON = OUTPUT_DIR / "scoreboard.json"
TRAJECTORY_JSON = OUTPUT_DIR / "trajectory.json"

# Job state (in-memory for single-process API)
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_MB = 500

API_TITLE = "CricketVisionAI API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Modular JSON API for cricket video upload, processing, "
    "ball speed, shot analysis, scoreboard, highlights, and reports."
)
