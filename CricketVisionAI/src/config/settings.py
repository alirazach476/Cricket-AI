"""
Central configuration for the CricketVisionAI inference pipeline.

All paths, class names, confidence thresholds, and drawing colors live here.
No training settings — inference only.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_DIR = PROJECT_ROOT / "weights"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "src" / "config"

# Sibling training folders (workspace root = parent of CricketVisionAI)
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Trained checkpoints — use only each folder's best.pt (not last/epoch/yolov8n base)
# People: cricket detection → batsman, fielder, umpire
# Bowler only: Cricket Player Detection/best.pt (class Bowler only — nothing else)
PLAYER_DETECTION_WEIGHTS = WORKSPACE_ROOT / "cricket detection" / "best.pt"
BOWLER_DETECTION_WEIGHTS = WORKSPACE_ROOT / "Cricket Player Detection" / "best.pt"
DRS_DETECTION_WEIGHTS = WEIGHTS_DIR / "drs_detection.pt"
BALL_DETECTION_WEIGHTS = WORKSPACE_ROOT / "cricket_ball_detection" / "best.pt"
SHOT_ACTIONS_WEIGHTS = WORKSPACE_ROOT / "Cricket Actions" / "best.pt"
SHOT_CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "shot_classifier.keras"  # legacy; unused
SHOT_CLASS_NAMES_FILE = CONFIG_DIR / "class_names.json"

# ---------------------------------------------------------------------------
# Cricket Detection (YOLOv8) — classes from cricket detection/data.yaml
# Model indices: 0 batsman, 1 fielder, 2 umpire
# Pipeline IDs keep Ball=0 from the dedicated ball model, then remap people:
#   model 0 → 1 Batsman, model 1 → 2 Fielder, model 2 → 3 Umpire
# Bowler from Cricket Player Detection only (model class 2 → pipeline 4)
# ---------------------------------------------------------------------------
PLAYER_CLASS_NAMES = {
    0: "Ball",
    1: "Batsman",
    2: "Fielder",
    3: "Umpire",
    4: "Bowler",
}

# Model class id → pipeline class id (Ball stays reserved as 0)
CRICKET_MODEL_TO_PIPELINE = {
    0: 1,  # batsman
    1: 2,  # fielder
    2: 3,  # umpire
}

# Cricket Player Detection/data.yaml: 0 Ball, 1 Batsman, 2 Bowler, 3 ScoreCards
# We only run / keep class 2 (Bowler).
BOWLER_MODEL_CLASS_ID = 2

BATSMAN_CLASS_ID = 1
FIELDER_CLASS_ID = 2
UMPIRE_CLASS_ID = 3
BOWLER_CLASS_ID = 4
PERSON_CLASS_IDS = (
    BATSMAN_CLASS_ID,
    FIELDER_CLASS_ID,
    UMPIRE_CLASS_ID,
    BOWLER_CLASS_ID,
)

BALL_CLASS_ID = 0
# Ball association needs moderate YOLO floor; velocity gate kills false positives
PLAYER_CONFIDENCE = 0.25
BALL_CONFIDENCE = 0.28
BALL_IMGSZ = 640
PLAYER_IOU = 0.45
PLAYER_IMGSZ = 960
BALL_MAX_SIDE_PX = 48
BALL_MIN_SIDE_PX = 3
BALL_MAX_ASPECT = 1.55  # max(w/h, h/w)

# BGR colors for drawing (OpenCV)
PLAYER_COLORS = {
    0: (0, 255, 255),    # Ball — yellow
    1: (0, 255, 0),      # Batsman — green
    2: (255, 160, 0),    # Fielder — orange
    3: (0, 0, 255),      # Umpire — red
    4: (255, 128, 0),    # Bowler — blue-orange
}

# ---------------------------------------------------------------------------
# DRS Detection (YOLOv8) — classes from drs/data.yaml
# Model indices: 0 Bat, 1 Pad, 2 Pitch, 3 Stumps
# ---------------------------------------------------------------------------
DRS_CLASS_NAMES = {
    0: "Bat",
    1: "Pad",
    2: "Pitch",
    3: "Stumps",
}

DRS_CONFIDENCE = 0.12
DRS_STUMPS_CONFIDENCE = 0.08  # unused — stumps detection disabled
DRS_IOU = 0.45
DRS_IMGSZ = 1280
STUMPS_CLASS_ID = 3
DETECT_STUMPS = False

DRS_COLORS = {
    0: (0, 165, 255),    # Bat — orange
    1: (255, 255, 0),    # Pad — cyan
    2: (128, 128, 0),    # Pitch — teal-ish
    3: (0, 0, 255),      # Stumps — red
}

# ---------------------------------------------------------------------------
# Shot detection — Cricket Actions YOLO pose (shot_actions.pt)
# Model: Cover-Drive, Front-foot, Square-short, helicopter-shot, pull-shorts
# ---------------------------------------------------------------------------
SHOT_CLASS_NAMES = [
    "Cover Drive",
    "Front Foot",
    "Square Short",
    "Helicopter Shot",
    "Pull Shot",
]
SHOT_DISPLAY_NAMES = list(SHOT_CLASS_NAMES)

SHOT_INPUT_SIZE = (224, 224)  # unused by YOLO actions model
SHOT_CONFIDENCE = 0.25  # analytics/CSV gate
SHOT_ACTIONS_CONFIDENCE = 0.22
SHOT_ACTIONS_IMGSZ = 640
SHOT_MIN_VOTES = 2
SHOT_BALL_NEAR_FACTOR = 2.2

# How often (in frames) to run shot-action detection
SHOT_CLASSIFY_EVERY_N_FRAMES = 2

SHOT_ACTION_COLORS = {
    0: (0, 200, 100),   # Cover Drive
    1: (255, 200, 0),   # Front Foot
    2: (0, 140, 255),   # Square Short
    3: (255, 0, 255),   # Helicopter
    4: (0, 0, 255),     # Pull
}

# ---------------------------------------------------------------------------
# Video I/O defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT_VIDEO = INPUT_DIR / "input.mp4"
DEFAULT_OUTPUT_VIDEO = OUTPUT_DIR / "annotated_output.mp4"
OUTPUT_CODEC = "mp4v"

# On-the-fly mid-frames (1 = fastest/stable; 2 = denser detection, slower).
ANALYSIS_EXPAND_FACTOR = 1

# EasyOCR hangs a long time on CPU / first download — off by default.
ENABLE_SCOREBOARD_OCR = False
