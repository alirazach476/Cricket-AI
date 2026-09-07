"""
Ball speed estimation in km/h with pitch-based scale calibration.

Auto-calibrates meters/pixel from the DRS pitch box when available,
and reports cricket-realistic delivery speeds (km/h).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Deque, Optional, Tuple

from src.analytics.ball_trajectory import TrajectoryPoint

# Real pitch dimensions (metres)
PITCH_LENGTH_M = 20.12
PITCH_WIDTH_M = 3.05

# Reasonable bowling-speed bounds for smoothing / display
MIN_DISPLAY_KMH = 50.0
MAX_DISPLAY_KMH = 165.0  # above world-record bowling → drop
MAX_PLAUSIBLE_KMH = 170.0


@dataclass
class BallSpeedSample:
    frame_number: int
    timestamp_sec: float
    object_id: int
    speed_px_per_sec: float
    speed_kmh: Optional[float]
    instant_px_per_sec: float
    detected: bool
    peak_kmh: Optional[float] = None


class BallSpeedEstimator:
    def __init__(
        self,
        fps: float = 30.0,
        meters_per_pixel: Optional[float] = None,
        speed_smooth_alpha: float = 0.28,
        history: int = 9,
        max_speed_px_s: float = 4000.0,
        max_step_px: float = 220.0,
        frame_height: Optional[float] = None,
    ) -> None:
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.meters_per_pixel = meters_per_pixel
        self.speed_smooth_alpha = float(speed_smooth_alpha)
        self.max_speed_px_s = float(max_speed_px_s)
        self.max_step_px = float(max_step_px)
        self._prev: Optional[TrajectoryPoint] = None
        self._smooth_speed: Optional[float] = None
        self._recent: Deque[float] = deque(maxlen=max(3, int(history)))
        self._peak_kmh: Optional[float] = None
        self._peak_hold = 0
        # Fallback scale until pitch is seen
        if self.meters_per_pixel is None and frame_height and frame_height > 0:
            # Assume ~55% of frame height spans ~pitch length (broadcast approx.)
            self.meters_per_pixel = PITCH_LENGTH_M / max(frame_height * 0.55, 1.0)

    def set_meters_per_pixel(self, mpp: float) -> None:
        if mpp and mpp > 0:
            self.meters_per_pixel = float(mpp)

    def calibrate_from_pitch_box(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """
        Estimate m/px from pitch AABB.

        Pitch length (~20.12 m) roughly maps to the longer side of the
        detected pitch box in most broadcast angles; also blend width
        (~3.05 m) for stability.
        """
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w < 10 or h < 10:
            return self.meters_per_pixel or 0.0

        # Longer side ≈ pitch length projection; shorter ≈ width
        long_px = max(w, h)
        mpp_len = PITCH_LENGTH_M / long_px
        mpp_wid = PITCH_WIDTH_M / min(w, h)
        # Pitch length dominates scale for bowling speed
        mpp = 0.82 * mpp_len + 0.18 * mpp_wid

        # Soft update so jittery boxes don't thrash scale
        if self.meters_per_pixel is None:
            self.meters_per_pixel = mpp
        else:
            self.meters_per_pixel = 0.85 * self.meters_per_pixel + 0.15 * mpp
        return self.meters_per_pixel

    def calibrate_from_homography(
        self,
        image_to_map,
        map_height_px: float,
        pitch_length_m: float = PITCH_LENGTH_M,
        sample_xy: Optional[Tuple[float, float]] = None,
    ) -> Optional[float]:
        """m/px near a sample point using pitch-map scale along map Y."""
        if image_to_map is None or map_height_px <= 1:
            return self.meters_per_pixel
        x, y = sample_xy if sample_xy else (0.0, 0.0)
        p0 = image_to_map(x, y)
        p1 = image_to_map(x, y + 40.0)
        if p0 is None or p1 is None:
            return self.meters_per_pixel
        map_dy = abs(p1[1] - p0[1])
        if map_dy < 1e-3:
            return self.meters_per_pixel
        # 40 image pixels → map_dy map pixels → meters
        meters = (map_dy / map_height_px) * pitch_length_m
        mpp = meters / 40.0
        if self.meters_per_pixel is None:
            self.meters_per_pixel = mpp
        else:
            self.meters_per_pixel = 0.8 * self.meters_per_pixel + 0.2 * mpp
        return self.meters_per_pixel

    def update(self, point: Optional[TrajectoryPoint]) -> Optional[BallSpeedSample]:
        if point is None or not point.detected:
            if point is None:
                self._prev = None
            if self._smooth_speed is not None and point is not None:
                kmh = self._to_kmh(float(self._smooth_speed))
                return BallSpeedSample(
                    frame_number=point.frame_number,
                    timestamp_sec=point.timestamp_sec,
                    object_id=point.object_id,
                    speed_px_per_sec=float(self._smooth_speed),
                    speed_kmh=kmh,
                    instant_px_per_sec=0.0,
                    detected=False,
                    peak_kmh=self._peak_kmh,
                )
            return None

        if self._prev is None or point.frame_number <= self._prev.frame_number:
            self._prev = point
            return BallSpeedSample(
                frame_number=point.frame_number,
                timestamp_sec=point.timestamp_sec,
                object_id=point.object_id,
                speed_px_per_sec=0.0,
                speed_kmh=0.0,
                instant_px_per_sec=0.0,
                detected=True,
                peak_kmh=self._peak_kmh,
            )

        frame_delta = max(1, point.frame_number - self._prev.frame_number)
        dt = frame_delta / self.fps
        dx = point.smooth_x - self._prev.smooth_x
        dy = point.smooth_y - self._prev.smooth_y
        distance_px = hypot(dx, dy)

        if distance_px > self.max_step_px * frame_delta:
            self._prev = point
            kmh = self._to_kmh(float(self._smooth_speed or 0.0))
            return BallSpeedSample(
                frame_number=point.frame_number,
                timestamp_sec=point.timestamp_sec,
                object_id=point.object_id,
                speed_px_per_sec=float(self._smooth_speed or 0.0),
                speed_kmh=kmh,
                instant_px_per_sec=0.0,
                detected=True,
                peak_kmh=self._peak_kmh,
            )

        instant = distance_px / dt if dt > 0 else 0.0
        instant = min(instant, self.max_speed_px_s)

        # Ignore near-static "logo/grass" detections posing as a ball
        if distance_px < 1.5 and frame_delta <= 2:
            self._prev = point
            kmh = self._to_kmh(float(self._smooth_speed or 0.0))
            return BallSpeedSample(
                frame_number=point.frame_number,
                timestamp_sec=point.timestamp_sec,
                object_id=point.object_id,
                speed_px_per_sec=float(self._smooth_speed or 0.0),
                speed_kmh=self._clamp_display(kmh),
                instant_px_per_sec=0.0,
                detected=True,
                peak_kmh=self._peak_kmh,
            )

        # Reject physically impossible km/h even in px space
        trial_kmh = self._to_kmh(instant) or 0.0
        if trial_kmh > MAX_PLAUSIBLE_KMH:
            self._prev = point
            kmh = self._to_kmh(float(self._smooth_speed or 0.0))
            return BallSpeedSample(
                frame_number=point.frame_number,
                timestamp_sec=point.timestamp_sec,
                object_id=point.object_id,
                speed_px_per_sec=float(self._smooth_speed or 0.0),
                speed_kmh=kmh,
                instant_px_per_sec=0.0,
                detected=True,
                peak_kmh=self._peak_kmh,
            )

        # Suppress crawl speeds unless we already have a delivery peak
        if trial_kmh < 25.0 and (self._peak_kmh is None or self._peak_hold <= 0):
            self._prev = point
            return BallSpeedSample(
                frame_number=point.frame_number,
                timestamp_sec=point.timestamp_sec,
                object_id=point.object_id,
                speed_px_per_sec=float(self._smooth_speed or 0.0),
                speed_kmh=None if self._smooth_speed is None else self._clamp_display(
                    self._to_kmh(float(self._smooth_speed))
                ),
                instant_px_per_sec=float(instant),
                detected=True,
                peak_kmh=self._peak_kmh,
            )

        self._recent.append(instant)
        ordered = sorted(self._recent)
        median = ordered[len(ordered) // 2]

        if self._smooth_speed is None:
            self._smooth_speed = median
        else:
            a = self.speed_smooth_alpha
            self._smooth_speed = a * median + (1.0 - a) * self._smooth_speed

        kmh = self._to_kmh(float(self._smooth_speed)) or 0.0
        # Track peak delivery-ish speed (hold briefly for readout)
        if kmh >= MIN_DISPLAY_KMH:
            if self._peak_kmh is None or kmh > self._peak_kmh:
                self._peak_kmh = kmh
                self._peak_hold = int(self.fps * 2.0)
            elif self._peak_hold > 0:
                self._peak_hold -= 1
            else:
                # decay peak slowly toward current
                self._peak_kmh = 0.92 * self._peak_kmh + 0.08 * kmh

        display_kmh = self._clamp_display(
            self._peak_kmh if self._peak_hold > 0 and self._peak_kmh else kmh
        )

        self._prev = point
        return BallSpeedSample(
            frame_number=point.frame_number,
            timestamp_sec=point.timestamp_sec,
            object_id=point.object_id,
            speed_px_per_sec=float(self._smooth_speed),
            speed_kmh=display_kmh,
            instant_px_per_sec=float(instant),
            detected=True,
            peak_kmh=self._peak_kmh,
        )

    def _to_kmh(self, speed_px_per_sec: float) -> Optional[float]:
        if self.meters_per_pixel is None or self.meters_per_pixel <= 0:
            return None
        return float(speed_px_per_sec * self.meters_per_pixel * 3.6)

    @staticmethod
    def _clamp_display(kmh: Optional[float]) -> Optional[float]:
        if kmh is None:
            return None
        if kmh < 5:
            return 0.0
        return float(max(0.0, min(MAX_DISPLAY_KMH, kmh)))

    @property
    def current_speed_px_per_sec(self) -> Optional[float]:
        return self._smooth_speed

    def reset(self) -> None:
        self._prev = None
        self._smooth_speed = None
        self._recent.clear()
        self._peak_kmh = None
        self._peak_hold = 0
