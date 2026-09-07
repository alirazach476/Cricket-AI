"""
Export highlight moments as MP4 clips under output/highlights/.

Reads the source cricket video once per clip (or seek+copy frames)
and writes short annotated or plain clips for each HighlightEvent.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import cv2

from src.config.settings import OUTPUT_CODEC, OUTPUT_DIR
from src.highlights.events import HighlightEvent, HighlightType

DEFAULT_HIGHLIGHTS_DIR = OUTPUT_DIR / "highlights"


def _safe_name(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", text.strip())
    return text.strip("_")[:80] or "highlight"


class HighlightClipExporter:
    """
    Write one MP4 per highlight event into output/highlights/.

    Also writes highlights_index.csv listing all exported clips.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        codec: str = OUTPUT_CODEC,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_HIGHLIGHTS_DIR
        self.codec = codec
        self._exported: List[dict] = []

    def export(
        self,
        video_path: Union[str, Path],
        events: Sequence[HighlightEvent],
        overlay_label: bool = True,
    ) -> List[Path]:
        """
        Generate MP4 clips for each event. Returns list of written paths.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Source video not found: {video_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._exported.clear()
        written: List[Path] = []

        if not events:
            self._write_index()
            return written

        for i, ev in enumerate(events, start=1):
            out_path = self._clip_path(i, ev)
            ok = self._write_clip(video_path, ev, out_path, overlay_label)
            if ok:
                written.append(out_path)
                self._exported.append(
                    {
                        "index": i,
                        "type": ev.kind.value,
                        "label": ev.label,
                        "frame_number": ev.frame_number,
                        "timestamp_sec": round(ev.timestamp_sec, 3),
                        "start_frame": ev.start_frame,
                        "end_frame": ev.end_frame,
                        "score": round(ev.score, 3),
                        "file": out_path.name,
                    }
                )

        self._write_index()
        return written

    def _clip_path(self, index: int, ev: HighlightEvent) -> Path:
        kind = _safe_name(ev.kind.value)
        stub = _safe_name(ev.label)[:40]
        name = f"{index:02d}_{kind}_f{ev.frame_number}_{stub}.mp4"
        return self.output_dir / name

    def _write_clip(
        self,
        video_path: Path,
        ev: HighlightEvent,
        out_path: Path,
        overlay_label: bool,
    ) -> bool:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        start = max(0, int(ev.start_frame))
        end = int(ev.end_frame)
        if total > 0:
            end = min(end, total - 1)
        if end < start:
            cap.release()
            return False

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_idx = start
        try:
            while frame_idx <= end:
                ok, frame = cap.read()
                if not ok:
                    break
                if overlay_label:
                    frame = self._draw_banner(frame, ev)
                writer.write(frame)
                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        return out_path.exists() and out_path.stat().st_size > 0

    @staticmethod
    def _draw_banner(frame, ev: HighlightEvent):
        color_map = {
            HighlightType.FOUR: (0, 215, 255),
            HighlightType.SIX: (0, 140, 255),
            HighlightType.FAST_BALL: (0, 0, 255),
            HighlightType.BEST_SHOT: (0, 255, 128),
        }
        color = color_map.get(ev.kind, (255, 255, 255))
        text = ev.label
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = 20, 40
        pad = 10
        cv2.rectangle(
            frame,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline + pad // 2),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA
        )
        return frame

    def _write_index(self) -> Path:
        path = self.output_dir / "highlights_index.csv"
        cols = [
            "index",
            "type",
            "label",
            "frame_number",
            "timestamp_sec",
            "start_frame",
            "end_frame",
            "score",
            "file",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(self._exported)
        return path

    @property
    def exported_count(self) -> int:
        return len(self._exported)
