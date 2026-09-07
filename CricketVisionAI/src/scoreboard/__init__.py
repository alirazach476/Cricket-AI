"""
Scoreboard OCR package for CricketVisionAI.

Detect Scoreboard with player_detection.pt → crop → preprocess → EasyOCR
→ Runs / Wickets / Overs / CRR / RRR → JSON.
"""

from src.scoreboard.detector import SCOREBOARD_CLASS_ID, ScoreboardBox, ScoreboardDetector
from src.scoreboard.ocr_reader import ScoreboardOCR
from src.scoreboard.parser import ScoreboardReading, parse_scoreboard_text
from src.scoreboard.preprocess import preprocess_scoreboard, preprocess_variants
from src.scoreboard.reader import ScoreboardReader

__all__ = [
    "SCOREBOARD_CLASS_ID",
    "ScoreboardBox",
    "ScoreboardDetector",
    "ScoreboardOCR",
    "ScoreboardReading",
    "parse_scoreboard_text",
    "preprocess_scoreboard",
    "preprocess_variants",
    "ScoreboardReader",
]
