"""
Ball trajectory builder for CricketVisionAI.

Takes tracked ball centers, smooths noisy detections (EMA + light rolling
average), and keeps a rolling path for drawing and speed estimation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

from src.tracking.models import TrackedObject

BALL_CLASS_ID = 0


@dataclass
class TrajectoryPoint:
    """One smoothed ball position on the timeline."""

    frame_number: int
    timestamp_sec: float
    object_id: int
    raw_x: float
    raw_y: float
    smooth_x: float
    smooth_y: float
    detected: bool  # False if interpolated / coasting when ball was missing
    segment_id: int = 0


def chaikin_smooth(
    points: Sequence[Tuple[float, float]], iterations: int = 2
) -> List[Tuple[float, float]]:
    """Chaikin corner-cutting for a smoother drawn polyline."""
    if len(points) < 3:
        return list(points)
    pts = [(float(x), float(y)) for x, y in points]
    for _ in range(max(0, iterations)):
        new: List[Tuple[float, float]] = [pts[0]]
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            new.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            new.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        new.append(pts[-1])
        pts = new
    return pts


class BallTrajectory:
    """
    Maintain a smoothed trajectory for the primary tracked ball.

    When multiple balls appear, prefers the previously followed object_id;
    otherwise picks the highest-confidence Ball detection.
    """

    def __init__(
        self,
        fps: float = 30.0,
        smooth_alpha: float = 0.18,
        max_points: int = 60,
        max_coast_frames: int = 2,
        max_jump_px: float = 90.0,
    ) -> None:
        """
        smooth_alpha: EMA weight on new detection (0–1). Lower = smoother.
        max_points: max trail length kept for drawing.
        max_coast_frames: keep last ID briefly when detection drops out.
        """
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.smooth_alpha = float(smooth_alpha)
        self.max_points = int(max_points)
        self.max_coast_frames = int(max_coast_frames)
        self.max_jump_px = float(max_jump_px)

        self._points: Deque[TrajectoryPoint] = deque(maxlen=self.max_points)
        self._active_id: Optional[int] = None
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None
        self._missed: int = 0
        self._segment_id: int = 0

    def update(
        self,
        frame_number: int,
        tracked_objects: Iterable[TrackedObject],
    ) -> Optional[TrajectoryPoint]:
        """
        Ingest this frame's tracks; return the new trajectory point or None
        if the ball has been missing too long.
        """
        balls = [o for o in tracked_objects if o.class_id == BALL_CLASS_ID]
        ball = self._select_ball(balls)
        timestamp_sec = frame_number / self.fps

        if ball is None:
            self._missed += 1
            if (
                self._smooth_x is None
                or self._smooth_y is None
                or self._missed > self.max_coast_frames
            ):
                if self._missed > self.max_coast_frames:
                    self._reset_smoother()
                    self._segment_id += 1
                return None
            # Coast for speed continuity only — do not append to draw path
            return TrajectoryPoint(
                frame_number=frame_number,
                timestamp_sec=timestamp_sec,
                object_id=self._active_id if self._active_id is not None else -1,
                raw_x=self._smooth_x,
                raw_y=self._smooth_y,
                smooth_x=self._smooth_x,
                smooth_y=self._smooth_y,
                detected=False,
                segment_id=self._segment_id,
            )

        self._missed = 0
        self._active_id = ball.object_id
        raw_x, raw_y = ball.center

        # Teleport / scene-cut: wipe trail instead of scribbling across the frame
        if self._smooth_x is not None and self._smooth_y is not None:
            jump = ((raw_x - self._smooth_x) ** 2 + (raw_y - self._smooth_y) ** 2) ** 0.5
            if jump > self.max_jump_px:
                self._points.clear()
                self._smooth_x = None
                self._smooth_y = None
                self._segment_id += 1

        if self._smooth_x is None or self._smooth_y is None:
            sx, sy = raw_x, raw_y
        else:
            a = self.smooth_alpha
            sx = a * raw_x + (1.0 - a) * self._smooth_x
            sy = a * raw_y + (1.0 - a) * self._smooth_y

        self._smooth_x, self._smooth_y = sx, sy

        point = TrajectoryPoint(
            frame_number=frame_number,
            timestamp_sec=timestamp_sec,
            object_id=ball.object_id,
            raw_x=raw_x,
            raw_y=raw_y,
            smooth_x=sx,
            smooth_y=sy,
            detected=True,
            segment_id=self._segment_id,
        )
        self._points.append(point)
        return point

    def _select_ball(
        self, balls: Sequence[TrackedObject]
    ) -> Optional[TrackedObject]:
        if not balls:
            return None
        if self._active_id is not None:
            for b in balls:
                if b.object_id == self._active_id:
                    return b
        return max(balls, key=lambda b: b.confidence)

    def _reset_smoother(self) -> None:
        self._active_id = None
        self._smooth_x = None
        self._smooth_y = None

    @property
    def points(self) -> List[TrajectoryPoint]:
        return list(self._points)

    @property
    def path_xy(self) -> List[Tuple[float, float]]:
        """Detected smoothed path for drawing (no coast ghosts)."""
        return [(p.smooth_x, p.smooth_y) for p in self._points if p.detected]

    def draw_path_xy(self, chaikin_iters: int = 2) -> List[Tuple[float, float]]:
        """Extra-smoothed path for UI / panel (detected points only)."""
        raw = self.path_xy
        if len(raw) < 3:
            return raw
        # Light 3-point moving average before Chaikin
        avg: List[Tuple[float, float]] = [raw[0]]
        for i in range(1, len(raw) - 1):
            x = (raw[i - 1][0] + raw[i][0] + raw[i + 1][0]) / 3.0
            y = (raw[i - 1][1] + raw[i][1] + raw[i + 1][1]) / 3.0
            avg.append((x, y))
        avg.append(raw[-1])
        return chaikin_smooth(avg, iterations=chaikin_iters)

    @property
    def latest(self) -> Optional[TrajectoryPoint]:
        return self._points[-1] if self._points else None

    def reset(self) -> None:
        self._points.clear()
        self._reset_smoother()
        self._missed = 0
        self._segment_id += 1


class DeliveryPanelPath:
    """
    Persistent pitch-normalized (u,v) path for the circular trajectory panel.

    Accumulates from the first ball detection of a delivery and keeps the full
    line visible (no rolling window / blink). Resets only on scene cut or
    teleport to a new delivery.
    """

    def __init__(self, min_step_uv: float = 0.006, break_uv: float = 0.32) -> None:
        self.min_step_uv = float(min_step_uv)
        self.break_uv = float(break_uv)
        self._uv: List[Tuple[float, float]] = []

    def reset(self) -> None:
        self._uv.clear()

    def add(
        self,
        x: float,
        y: float,
        pitch_box: Optional[Tuple[float, float, float, float]],
    ) -> None:
        if pitch_box is None:
            return
        px1, py1, px2, py2 = pitch_box
        u = (x - px1) / max(1.0, px2 - px1)
        v = (y - py1) / max(1.0, py2 - py1)
        if self._uv:
            lu, lv = self._uv[-1]
            du, dv = u - lu, v - lv
            dist = (du * du + dv * dv) ** 0.5
            if dist < self.min_step_uv:
                return
            if dist > self.break_uv:
                self._uv = [(u, v)]
                return
        self._uv.append((float(u), float(v)))

    @property
    def path_uv(self) -> List[Tuple[float, float]]:
        return list(self._uv)

    def smooth_uv(self, chaikin_iters: int = 1) -> List[Tuple[float, float]]:
        raw = self.path_uv
        if len(raw) < 3:
            return raw
        return chaikin_smooth(raw, iterations=chaikin_iters)
