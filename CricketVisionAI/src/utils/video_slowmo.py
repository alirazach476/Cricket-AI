"""
Slow-motion preprocessing for denser frame analysis.

Expands the timeline by inserting blended mid-frames between consecutive
frames so detectors / trackers see finer motion. Annotated output is written
at original FPS so playback appears slowed.

Speed / timestamp math must use analysis_fps = original_fps * expand_factor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

ProgressCb = Optional[Callable[[float, str], None]]


def create_slow_motion_video(
    source: Path,
    destination: Path,
    expand_factor: int = 2,
    codec: str = "mp4v",
    on_progress: ProgressCb = None,
) -> Tuple[Path, float, float, int, int, int]:
    """
    Write a temporally expanded copy of ``source``.

    Returns
    -------
    dest, analysis_fps, playback_fps, width, height, expanded_frame_count
    """
    expand_factor = max(1, int(expand_factor))
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for slow-mo: {source}")

    try:
        playback_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if expand_factor == 1:
            # No-op: caller can keep using the original file
            if on_progress:
                on_progress(100.0, "Using original speed")
            return source, playback_fps, playback_fps, width, height, total

        analysis_fps = playback_fps * expand_factor
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(destination), fourcc, playback_fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open slow-mo writer: {destination}")

        prev: Optional[np.ndarray] = None
        read_i = 0
        written = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read_i += 1

            if prev is not None:
                # Insert expand_factor - 1 blends between prev and current
                for k in range(1, expand_factor):
                    alpha = k / float(expand_factor)
                    mid = cv2.addWeighted(prev, 1.0 - alpha, frame, alpha, 0)
                    writer.write(mid)
                    written += 1

            writer.write(frame)
            written += 1
            prev = frame

            if on_progress and total > 0 and (read_i % 5 == 0 or read_i == total):
                pct = min(99.0, (read_i / total) * 100.0)
                on_progress(pct, f"Slowing video {read_i}/{total}")

        writer.release()
        if on_progress:
            on_progress(100.0, "Slow-motion ready")
        return (
            destination,
            analysis_fps,
            playback_fps,
            width,
            height,
            written,
        )
    finally:
        cap.release()
