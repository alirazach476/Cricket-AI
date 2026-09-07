"""
Highlights package for CricketVisionAI.

Detect Four / Six / Fast Ball / Best Shot from shot classification +
ball speed, then export MP4 clips to output/highlights/.
"""

from src.highlights.clip_exporter import (
    DEFAULT_HIGHLIGHTS_DIR,
    HighlightClipExporter,
)
from src.highlights.events import (
    HighlightDetector,
    HighlightEvent,
    HighlightType,
    ShotPoint,
    SpeedPoint,
)

__all__ = [
    "HighlightDetector",
    "HighlightEvent",
    "HighlightType",
    "SpeedPoint",
    "ShotPoint",
    "HighlightClipExporter",
    "DEFAULT_HIGHLIGHTS_DIR",
]
