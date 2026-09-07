"""Player / ball detection helpers."""

from src.detection.ball_detector import DedicatedBallDetector
from src.detection.ball_hybrid import HybridBallDetector

__all__ = ["DedicatedBallDetector", "HybridBallDetector"]
