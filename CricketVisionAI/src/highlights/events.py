"""
Highlight event detection for CricketVisionAI.

Uses shot classification confidence and ball speed / outbound travel
to flag: Four, Six, Fast Ball, Best Shot.

Heuristics (no scoreboard OCR — boundary is approximated from
post-shot ball travel + peak speed):

  Fast Ball  — peak ball speed above threshold
  Best Shot  — highest shot-classifier confidence in the window / clip set
  Four       — strong outbound distance + high post-shot speed (ground hit)
  Six        — even larger outbound distance and/or loft proxy (rise then long flight)

Callers feed timestamped speed samples and shot predictions; this module
only scores events (it does not open video files).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from typing import Iterable, List, Optional, Sequence, Tuple

Point2D = Tuple[float, float]


class HighlightType(str, Enum):
    FOUR = "Four"
    SIX = "Six"
    FAST_BALL = "Fast Ball"
    BEST_SHOT = "Best Shot"


@dataclass
class SpeedPoint:
    frame_number: int
    speed_px_per_sec: float
    timestamp_sec: float = 0.0


@dataclass
class ShotPoint:
    frame_number: int
    shot_name: str
    confidence: float
    timestamp_sec: float = 0.0


@dataclass
class HighlightEvent:
    """One detected highlight with a suggested clip window."""

    kind: HighlightType
    frame_number: int
    timestamp_sec: float
    score: float
    label: str
    start_frame: int
    end_frame: int
    meta: dict = field(default_factory=dict)


class HighlightDetector:
    """
    Detect highlight moments from speed + shot streams.

    Thresholds are in image units (px/s, px). Tune via constructor.
    """

    def __init__(
        self,
        fps: float = 30.0,
        # Peak ball speed (px/s) to count as Fast Ball
        fast_ball_px_s: float = 800.0,
        # Outbound path length after shot for boundary-like events
        four_distance_px: float = 250.0,
        six_distance_px: float = 450.0,
        # Minimum shot confidence to consider Best Shot / boundary
        min_shot_confidence: float = 0.45,
        # Clip padding (seconds) around the event frame
        pre_roll_sec: float = 1.5,
        post_roll_sec: float = 2.5,
        # Require post-shot peak speed at least this high for Four/Six
        boundary_min_speed_px_s: float = 400.0,
        # Loft proxy: upward motion fraction for Six (image y decreases)
        six_loft_ratio: float = 0.35,
    ) -> None:
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.fast_ball_px_s = float(fast_ball_px_s)
        self.four_distance_px = float(four_distance_px)
        self.six_distance_px = float(six_distance_px)
        self.min_shot_confidence = float(min_shot_confidence)
        self.pre_roll_sec = float(pre_roll_sec)
        self.post_roll_sec = float(post_roll_sec)
        self.boundary_min_speed_px_s = float(boundary_min_speed_px_s)
        self.six_loft_ratio = float(six_loft_ratio)

        self._speeds: List[SpeedPoint] = []
        self._shots: List[ShotPoint] = []
        self._ball_xy: List[Point2D] = []
        self._ball_frames: List[int] = []

    def add_speed(
        self,
        frame_number: int,
        speed_px_per_sec: float,
        timestamp_sec: Optional[float] = None,
    ) -> None:
        ts = (
            float(timestamp_sec)
            if timestamp_sec is not None
            else frame_number / self.fps
        )
        self._speeds.append(
            SpeedPoint(
                frame_number=int(frame_number),
                speed_px_per_sec=float(speed_px_per_sec),
                timestamp_sec=ts,
            )
        )

    def add_shot(
        self,
        frame_number: int,
        shot_name: str,
        confidence: float,
        timestamp_sec: Optional[float] = None,
    ) -> None:
        ts = (
            float(timestamp_sec)
            if timestamp_sec is not None
            else frame_number / self.fps
        )
        self._shots.append(
            ShotPoint(
                frame_number=int(frame_number),
                shot_name=str(shot_name),
                confidence=float(confidence),
                timestamp_sec=ts,
            )
        )

    def add_ball_point(self, frame_number: int, xy: Point2D) -> None:
        self._ball_frames.append(int(frame_number))
        self._ball_xy.append((float(xy[0]), float(xy[1])))

    def detect(self, total_frames: Optional[int] = None) -> List[HighlightEvent]:
        """Run all detectors and return deduplicated, scored events."""
        events: List[HighlightEvent] = []
        events.extend(self._detect_fast_balls(total_frames))
        events.extend(self._detect_best_shots(total_frames))
        events.extend(self._detect_boundaries(total_frames))
        events.sort(key=lambda e: (e.frame_number, -e.score))
        return self._dedupe_nearby(events)

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def _window(self, frame: int, total_frames: Optional[int]) -> Tuple[int, int]:
        pre = int(round(self.pre_roll_sec * self.fps))
        post = int(round(self.post_roll_sec * self.fps))
        start = max(0, frame - pre)
        end = frame + post
        if total_frames is not None:
            end = min(max(total_frames - 1, 0), end)
        return start, end

    def _detect_fast_balls(
        self, total_frames: Optional[int]
    ) -> List[HighlightEvent]:
        out: List[HighlightEvent] = []
        if not self._speeds:
            return out

        # Local peaks above threshold
        speeds = self._speeds
        for i, sp in enumerate(speeds):
            if sp.speed_px_per_sec < self.fast_ball_px_s:
                continue
            prev_v = speeds[i - 1].speed_px_per_sec if i > 0 else -1.0
            next_v = speeds[i + 1].speed_px_per_sec if i + 1 < len(speeds) else -1.0
            if sp.speed_px_per_sec < prev_v or sp.speed_px_per_sec < next_v:
                continue
            start, end = self._window(sp.frame_number, total_frames)
            out.append(
                HighlightEvent(
                    kind=HighlightType.FAST_BALL,
                    frame_number=sp.frame_number,
                    timestamp_sec=sp.timestamp_sec,
                    score=sp.speed_px_per_sec,
                    label=f"Fast Ball ({sp.speed_px_per_sec:.0f} px/s)",
                    start_frame=start,
                    end_frame=end,
                    meta={"speed_px_per_sec": sp.speed_px_per_sec},
                )
            )
        return out

    def _detect_best_shots(
        self, total_frames: Optional[int]
    ) -> List[HighlightEvent]:
        out: List[HighlightEvent] = []
        eligible = [
            s for s in self._shots if s.confidence >= self.min_shot_confidence
        ]
        if not eligible:
            return out

        best = max(eligible, key=lambda s: s.confidence)
        # Also keep other near-best shots (≥ 95% of best confidence)
        threshold = best.confidence * 0.95
        seen_frames = set()
        for s in sorted(eligible, key=lambda x: -x.confidence):
            if s.confidence < threshold:
                break
            # Collapse duplicates within 1 second
            if any(abs(s.frame_number - f) < self.fps for f in seen_frames):
                continue
            seen_frames.add(s.frame_number)
            start, end = self._window(s.frame_number, total_frames)
            out.append(
                HighlightEvent(
                    kind=HighlightType.BEST_SHOT,
                    frame_number=s.frame_number,
                    timestamp_sec=s.timestamp_sec,
                    score=s.confidence,
                    label=f"Best Shot: {s.shot_name} ({s.confidence * 100:.1f}%)",
                    start_frame=start,
                    end_frame=end,
                    meta={
                        "shot_name": s.shot_name,
                        "confidence": s.confidence,
                    },
                )
            )
        return out

    def _detect_boundaries(
        self, total_frames: Optional[int]
    ) -> List[HighlightEvent]:
        """
        Approximate Four / Six from post-shot outbound travel + speed.

        For each confident shot, measure ball path length after the shot
        frame and peak speed in that window. Large travel → Four / Six.
        Six also prefers a loft proxy (net upward motion early after impact).
        """
        out: List[HighlightEvent] = []
        if not self._shots or len(self._ball_xy) < 4:
            return out

        for shot in self._shots:
            if shot.confidence < self.min_shot_confidence:
                continue

            # Ball samples after the shot frame
            idxs = [
                i
                for i, f in enumerate(self._ball_frames)
                if f >= shot.frame_number
            ]
            if len(idxs) < 3:
                continue

            path = [self._ball_xy[i] for i in idxs]
            frames = [self._ball_frames[i] for i in idxs]
            distance_px = _path_length(path)
            peak_speed = self._peak_speed_between(frames[0], frames[-1])

            if (
                distance_px < self.four_distance_px
                or peak_speed < self.boundary_min_speed_px_s
            ):
                continue

            loft = _loft_ratio(path)
            is_six = distance_px >= self.six_distance_px or (
                distance_px >= self.four_distance_px * 1.2
                and loft >= self.six_loft_ratio
            )

            kind = HighlightType.SIX if is_six else HighlightType.FOUR
            start, end = self._window(shot.frame_number, total_frames)
            score = distance_px + 0.25 * peak_speed
            out.append(
                HighlightEvent(
                    kind=kind,
                    frame_number=shot.frame_number,
                    timestamp_sec=shot.timestamp_sec,
                    score=score,
                    label=(
                        f"{kind.value}: {shot.shot_name} "
                        f"({distance_px:.0f} px, {peak_speed:.0f} px/s)"
                    ),
                    start_frame=start,
                    end_frame=end,
                    meta={
                        "shot_name": shot.shot_name,
                        "confidence": shot.confidence,
                        "distance_px": distance_px,
                        "peak_speed_px_s": peak_speed,
                        "loft_ratio": loft,
                    },
                )
            )
        return out

    def _peak_speed_between(self, start_f: int, end_f: int) -> float:
        vals = [
            s.speed_px_per_sec
            for s in self._speeds
            if start_f <= s.frame_number <= end_f
        ]
        return max(vals) if vals else 0.0

    def _dedupe_nearby(
        self,
        events: Sequence[HighlightEvent],
        gap_frames: Optional[int] = None,
    ) -> List[HighlightEvent]:
        """Keep highest-score event per kind within a short time gap."""
        gap = int(gap_frames if gap_frames is not None else self.fps * 1.5)
        kept: List[HighlightEvent] = []
        for ev in events:
            clash = False
            for i, k in enumerate(kept):
                if k.kind != ev.kind:
                    continue
                if abs(k.frame_number - ev.frame_number) <= gap:
                    clash = True
                    if ev.score > k.score:
                        kept[i] = ev
                    break
            if not clash:
                kept.append(ev)
        kept.sort(key=lambda e: e.frame_number)
        return kept

    def reset(self) -> None:
        self._speeds.clear()
        self._shots.clear()
        self._ball_xy.clear()
        self._ball_frames.clear()


def _path_length(path: Sequence[Point2D]) -> float:
    total = 0.0
    for i in range(1, len(path)):
        total += hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
    return total


def _loft_ratio(path: Sequence[Point2D]) -> float:
    """
    Fraction of early outbound motion that goes upward (decreasing image y).
    Higher → more lofted (Six-like).
    """
    if len(path) < 2:
        return 0.0
    n = max(2, len(path) // 2)
    up = 0.0
    down = 0.0
    for i in range(1, n):
        dy = path[i][1] - path[i - 1][1]
        if dy < 0:
            up += -dy
        else:
            down += dy
    denom = up + down
    return up / denom if denom > 1e-6 else 0.0
