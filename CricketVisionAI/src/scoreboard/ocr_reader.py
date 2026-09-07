"""
EasyOCR-based scoreboard text reader.

Loads EasyOCR once, preprocesses the crop, reads text, parses fields,
and can save JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.config.settings import OUTPUT_DIR
from src.scoreboard.parser import ScoreboardReading, parse_scoreboard_text
from src.scoreboard.preprocess import preprocess_scoreboard, preprocess_variants

DEFAULT_SCOREBOARD_JSON = OUTPUT_DIR / "scoreboard.json"


class ScoreboardOCR:
    """
    Read cricket scoreboard digits/text with EasyOCR.

    Requires: pip install easyocr
    """

    def __init__(
        self,
        languages: Optional[Sequence[str]] = None,
        gpu: bool = False,
        use_dual_preprocess: bool = True,
    ) -> None:
        self.languages = list(languages) if languages else ["en"]
        self.gpu = bool(gpu)
        self.use_dual_preprocess = bool(use_dual_preprocess)
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr  # lazy import

            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def read_crop(self, crop_bgr: np.ndarray) -> ScoreboardReading:
        """Preprocess crop → EasyOCR → structured ScoreboardReading."""
        reader = self._get_reader()
        texts: List[str] = []
        confs: List[float] = []

        if self.use_dual_preprocess:
            variants = preprocess_variants(crop_bgr)
        else:
            variants = (preprocess_scoreboard(crop_bgr),)

        for img_rgb in variants:
            # EasyOCR accepts numpy RGB/BGR; detail=1 → (bbox, text, conf)
            results = reader.readtext(img_rgb, detail=1, paragraph=False)
            for item in results:
                if len(item) < 3:
                    continue
                text = str(item[1]).strip()
                conf = float(item[2])
                if text:
                    texts.append(text)
                    confs.append(conf)

        # Deduplicate while preserving order
        seen = set()
        uniq_texts: List[str] = []
        uniq_confs: List[float] = []
        for t, c in zip(texts, confs):
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq_texts.append(t)
            uniq_confs.append(c)

        return parse_scoreboard_text(uniq_texts, uniq_confs)

    def save_json(
        self,
        reading: ScoreboardReading,
        path: Optional[Union[str, Path]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        out = Path(path) if path else DEFAULT_SCOREBOARD_JSON
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = reading.to_dict()
        if extra:
            payload.update(extra)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out
