"""
Shot classification package for CricketVisionAI.

Uses Cricket Actions YOLO pose weights (shot_actions.pt).
"""

from src.shot_classifier.classifier import (
    BATSMAN_CLASS_ID,
    ShotActionDetection,
    ShotClassifier,
    ShotPrediction,
)
from src.config.settings import SHOT_DISPLAY_NAMES
from src.shot_classifier.shot_logger import (
    CSV_COLUMNS,
    DEFAULT_SHOTS_CSV,
    ShotLogger,
)

__all__ = [
    "ShotClassifier",
    "ShotPrediction",
    "ShotActionDetection",
    "ShotLogger",
    "SHOT_DISPLAY_NAMES",
    "BATSMAN_CLASS_ID",
    "CSV_COLUMNS",
    "DEFAULT_SHOTS_CSV",
]
