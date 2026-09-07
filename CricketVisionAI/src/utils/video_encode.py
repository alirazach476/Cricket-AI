"""
Re-encode annotated OpenCV output to browser-playable H.264 MP4.

OpenCV's mp4v (MPEG-4 Part 2) often fails in Chrome/Firefox/Safari.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def to_browser_mp4(source: Path, destination: Path | None = None) -> Path:
    """
    Transcode ``source`` to H.264 + yuv420p in ``destination`` (in-place if None).

    Falls back to copying the original when ffmpeg is unavailable.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    dest = Path(destination) if destination else source
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if dest != source:
            shutil.copy2(source, dest)
        return dest

    tmp = dest.with_suffix(".h264tmp.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Replace destination
        if dest.exists() and dest.resolve() == source.resolve():
            source.unlink(missing_ok=True)
        tmp.replace(dest)
        return dest
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        if dest != source:
            shutil.copy2(source, dest)
        return dest
