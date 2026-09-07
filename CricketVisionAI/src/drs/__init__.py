"""
DRS detection package for CricketVisionAI.

Uses weights/drs_detection.pt — Bat, Pad, Pitch, Stumps.
"""

from src.drs.drs_detector import (
    PITCH_CLASS_ID,
    DrsDetection,
    DrsDetector,
)

__all__ = [
    "DrsDetector",
    "DrsDetection",
    "PITCH_CLASS_ID",
]
