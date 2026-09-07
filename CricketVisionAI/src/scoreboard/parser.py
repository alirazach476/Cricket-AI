"""
Parse EasyOCR text lines into cricket scoreboard fields.

Extracts:
  Runs, Wickets, Overs, Current Run Rate (CRR), Required Run Rate (RRR)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ScoreboardReading:
    runs: Optional[int] = None
    wickets: Optional[int] = None
    overs: Optional[float] = None
    crr: Optional[float] = None
    rrr: Optional[float] = None
    raw_text: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def score_string(self) -> Optional[str]:
        if self.runs is None:
            return None
        if self.wickets is None:
            return str(self.runs)
        return f"{self.runs}/{self.wickets}"


# Patterns commonly seen on broadcast / app scoreboards
_SCORE = re.compile(
    r"(?P<runs>\d{1,3})\s*[/\-]\s*(?P<wickets>\d{1,2})"
)
_OVERS = re.compile(
    r"(?:ovs?|overs?|over)?\s*[:=\-]?\s*(?P<overs>\d{1,2}(?:\.\d)?)\s*(?:ovs?|overs?)?",
    re.I,
)
_OVERS_BARE = re.compile(r"\b(?P<overs>\d{1,2}\.\d)\b")
_CRR = re.compile(
    r"(?:crr|current\s*run\s*rate|run\s*rate)\s*[:=\-]?\s*(?P<crr>\d{1,2}(?:\.\d{1,2})?)",
    re.I,
)
_RRR = re.compile(
    r"(?:rrr|req(?:uired)?\s*run\s*rate|required)\s*[:=\-]?\s*(?P<rrr>\d{1,2}(?:\.\d{1,2})?)",
    re.I,
)


def parse_scoreboard_text(
    texts: Sequence[str],
    confidences: Optional[Sequence[float]] = None,
) -> ScoreboardReading:
    """
    Parse OCR strings into structured fields.

    Tries the joined blob first, then individual lines.
    """
    cleaned = [_normalize_line(t) for t in texts if t and t.strip()]
    blob = " | ".join(cleaned)
    reading = ScoreboardReading(raw_text=blob)

    if confidences:
        vals = [float(c) for c in confidences if c is not None]
        reading.confidence = float(sum(vals) / len(vals)) if vals else 0.0

    _apply_patterns(reading, blob)
    for line in cleaned:
        _apply_patterns(reading, line)

    return reading


def _normalize_line(text: str) -> str:
    t = text.strip()
    # Common OCR confusions on scoreboards
    t = t.replace("O", "0").replace("o", "0") if re.fullmatch(r"[Oo0/\-\.\d\s]+", t) else t
    t = re.sub(r"\s+", " ", t)
    return t


def _apply_patterns(reading: ScoreboardReading, text: str) -> None:
    if reading.runs is None or reading.wickets is None:
        m = _SCORE.search(text)
        if m:
            reading.runs = int(m.group("runs"))
            reading.wickets = int(m.group("wickets"))

    if reading.overs is None:
        m = _OVERS.search(text)
        if m:
            try:
                reading.overs = float(m.group("overs"))
            except ValueError:
                pass
        if reading.overs is None:
            m = _OVERS_BARE.search(text)
            if m:
                try:
                    val = float(m.group("overs"))
                    # Prefer plausible over counts (0.0 – 50.0+)
                    if 0.0 <= val <= 100.0:
                        reading.overs = val
                except ValueError:
                    pass

    if reading.crr is None:
        m = _CRR.search(text)
        if m:
            try:
                reading.crr = float(m.group("crr"))
            except ValueError:
                pass

    if reading.rrr is None:
        m = _RRR.search(text)
        if m:
            try:
                reading.rrr = float(m.group("rrr"))
            except ValueError:
                pass

    # Derive CRR if overs and runs known and CRR missing
    if reading.crr is None and reading.runs is not None and reading.overs and reading.overs > 0:
        reading.crr = round(reading.runs / reading.overs, 2)
