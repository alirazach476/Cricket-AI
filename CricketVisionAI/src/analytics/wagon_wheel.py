"""
Wagon Wheel analytics for CricketVisionAI.

Uses:
  - Smoothed ball trajectory (ball_detection.pt + tracking)
  - Shot classification label (from shot_actions.pt)

Estimates:
  - Shot angle (degrees) — direction the ball leaves the batsman
  - Shot distance (pixels, optional metres if scale is known)

Generates a classic wagon-wheel diagram and saves it as an image.

This module does not load models itself — callers pass trajectory points,
batsman position, and the shot prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees, hypot, radians
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.config.settings import OUTPUT_DIR
from src.shot_classifier.classifier import SHOT_DISPLAY_NAMES

Point2D = Tuple[float, float]

# Canvas: batsman at centre; 0° = straight down the ground (toward bowler / top)
WHEEL_SIZE = 720
WHEEL_MARGIN = 40

# Colours per shot type (BGR)
SHOT_COLORS = {
    "Cover Drive": (0, 200, 100),
    "Front Foot": (255, 200, 0),
    "Square Short": (0, 140, 255),
    "Helicopter Shot": (255, 0, 255),
    "Pull Shot": (0, 0, 255),
    # legacy labels kept for older CSVs
    "Cut Shot": (0, 140, 255),
    "Straight Drive": (0, 255, 255),
    "Leg Glance": (255, 128, 0),
    "Scoop Shot": (255, 0, 255),
}


@dataclass
class ShotGeometry:
    """Estimated direction and range of one scored shot."""

    shot_name: str
    confidence: float
    # Angle in degrees: 0 = straight (toward bowler / top of wheel),
    # positive = off side (right for RH batter in standard broadcast view),
    # negative = leg side (left). Range roughly [-180, 180].
    angle_deg: float
    # Path length after impact in image pixels
    distance_px: float
    # Optional real-world distance if meters_per_pixel is set
    distance_m: Optional[float]
    origin_xy: Point2D  # batsman / impact origin in image
    end_xy: Point2D  # last ball point used for the ray
    impact_frame: int
    end_frame: int


@dataclass
class WagonWheelResult:
    geometry: Optional[ShotGeometry]
    wheel_image: np.ndarray
    spokes: List[ShotGeometry] = field(default_factory=list)


class WagonWheelBuilder:
    """
    Build a wagon wheel from post-shot ball flight relative to the batsman.

    Angle / distance math is documented in `estimate_geometry` and in the
    project reply that accompanies this file.
    """

    def __init__(
        self,
        wheel_size: int = WHEEL_SIZE,
        meters_per_pixel: Optional[float] = None,
        output_dir: Optional[Union[str, Path]] = None,
        # Minimum frames of outbound ball motion after impact
        min_outbound_points: int = 3,
        # Fraction of max observed distance mapped to full wheel radius
        max_distance_px: Optional[float] = None,
    ) -> None:
        self.wheel_size = int(wheel_size)
        self.meters_per_pixel = meters_per_pixel
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.min_outbound_points = int(min_outbound_points)
        self.max_distance_px = max_distance_px
        self._history: List[ShotGeometry] = []

    # ------------------------------------------------------------------
    # Calculations
    # ------------------------------------------------------------------

    def estimate_geometry(
        self,
        ball_xy: Sequence[Point2D],
        frame_numbers: Sequence[int],
        batsman_xy: Point2D,
        shot_name: str,
        confidence: float = 1.0,
        pitch_axis_xy: Optional[Tuple[Point2D, Point2D]] = None,
    ) -> Optional[ShotGeometry]:
        """
        Estimate shot angle and distance from the ball path after impact.

        Impact proxy
        ------------
        We treat the path point nearest the batsman centre as the start of
        the outbound stroke (ball nearest batsman ≈ contact / just after).

        Angle (degrees)
        ---------------
        Let O = batsman (origin), E = a later outbound ball position
        (we use a robust average direction of the outbound segment).

            dx = E_x - O_x
            dy = E_y - O_y

        Image coords: x right, y down.

        Camera/cricket convention for the wheel:
            0°   = straight down the wicket toward the bowler
            +90° = off side (cover / point for a right-hander in many feeds)
            -90° = leg side (square leg / fine leg)

        If `pitch_axis_xy` is given as (bowler_end_point, batsman_end_point)
        in image space, "straight" is that axis. Otherwise we assume the
        bowler is toward the TOP of the frame (decreasing y):

            angle_math = atan2(dx, -dy)   # 0 when moving straight up
            angle_deg  = degrees(angle_math)

        Distance
        --------
            distance_px = sum of segment lengths along the outbound path
                          from impact index to the end
                        = Σ hypot(x[i]-x[i-1], y[i]-y[i-1])

            distance_m  = distance_px * meters_per_pixel   (if calibrated)
        """
        if len(ball_xy) < self.min_outbound_points + 1:
            return None
        if len(frame_numbers) != len(ball_xy):
            frame_numbers = list(range(len(ball_xy)))

        ox, oy = batsman_xy
        # Index of ball sample closest to batsman → impact proxy
        dists = [hypot(p[0] - ox, p[1] - oy) for p in ball_xy]
        impact_idx = int(np.argmin(dists))

        outbound = list(ball_xy[impact_idx:])
        outbound_frames = list(frame_numbers[impact_idx:])
        if len(outbound) < self.min_outbound_points:
            # Fall back: use entire path from the middle onward
            impact_idx = max(0, len(ball_xy) // 3)
            outbound = list(ball_xy[impact_idx:])
            outbound_frames = list(frame_numbers[impact_idx:])
        if len(outbound) < 2:
            return None

        # Path length (shot distance in pixels)
        distance_px = 0.0
        for i in range(1, len(outbound)):
            distance_px += hypot(
                outbound[i][0] - outbound[i - 1][0],
                outbound[i][1] - outbound[i - 1][1],
            )

        # Direction: average unit vectors along outbound segments
        # (more stable than endpoint alone)
        dir_x, dir_y = 0.0, 0.0
        for i in range(1, len(outbound)):
            sx = outbound[i][0] - outbound[i - 1][0]
            sy = outbound[i][1] - outbound[i - 1][1]
            seg = hypot(sx, sy)
            if seg < 1e-6:
                continue
            dir_x += sx / seg
            dir_y += sy / seg

        if hypot(dir_x, dir_y) < 1e-6:
            # Endpoint fallback relative to batsman
            dir_x = outbound[-1][0] - ox
            dir_y = outbound[-1][1] - oy
        if hypot(dir_x, dir_y) < 1e-6:
            return None

        angle_deg = self._direction_to_angle(dir_x, dir_y, pitch_axis_xy)

        distance_m = None
        if self.meters_per_pixel is not None and self.meters_per_pixel > 0:
            distance_m = distance_px * self.meters_per_pixel

        return ShotGeometry(
            shot_name=shot_name,
            confidence=float(confidence),
            angle_deg=float(angle_deg),
            distance_px=float(distance_px),
            distance_m=distance_m,
            origin_xy=(float(ox), float(oy)),
            end_xy=(float(outbound[-1][0]), float(outbound[-1][1])),
            impact_frame=int(outbound_frames[0]),
            end_frame=int(outbound_frames[-1]),
        )

    def _direction_to_angle(
        self,
        dir_x: float,
        dir_y: float,
        pitch_axis_xy: Optional[Tuple[Point2D, Point2D]],
    ) -> float:
        """
        Convert an image-space direction into wagon-wheel degrees.

        Default (no pitch axis): 0° = toward top of frame (-Y).
        With pitch axis (bowler_pt, batsman_pt): 0° = batsman → bowler.
        """
        if pitch_axis_xy is not None:
            bowler_pt, batsman_pt = pitch_axis_xy
            # Straight unit vector: batsman → bowler
            ax = bowler_pt[0] - batsman_pt[0]
            ay = bowler_pt[1] - batsman_pt[1]
            a_len = hypot(ax, ay)
            if a_len > 1e-6:
                ax, ay = ax / a_len, ay / a_len
                # Angle of shot dir relative to straight axis
                # rotate so axis maps to 0°
                # straight angle in image:
                axis_ang = atan2(ax, -ay)
                shot_ang = atan2(dir_x, -dir_y)
                return (degrees(shot_ang - axis_ang) + 180) % 360 - 180

        # atan2(x, -y): 0 when moving up (toward bowler if bowler is at top)
        return degrees(atan2(dir_x, -dir_y))

    # ------------------------------------------------------------------
    # Wagon wheel render
    # ------------------------------------------------------------------

    def add_shot(self, geometry: ShotGeometry) -> None:
        self._history.append(geometry)

    def build(
        self,
        ball_xy: Sequence[Point2D],
        frame_numbers: Sequence[int],
        batsman_xy: Point2D,
        shot_name: str,
        confidence: float = 1.0,
        pitch_axis_xy: Optional[Tuple[Point2D, Point2D]] = None,
        include_history: bool = True,
    ) -> WagonWheelResult:
        geometry = self.estimate_geometry(
            ball_xy=ball_xy,
            frame_numbers=frame_numbers,
            batsman_xy=batsman_xy,
            shot_name=shot_name,
            confidence=confidence,
            pitch_axis_xy=pitch_axis_xy,
        )
        spokes = list(self._history) if include_history else []
        if geometry is not None:
            spokes = spokes + [geometry]
            self.add_shot(geometry)

        wheel = self.render_wheel(spokes)
        return WagonWheelResult(geometry=geometry, wheel_image=wheel, spokes=spokes)

    def render_wheel(self, spokes: Sequence[ShotGeometry]) -> np.ndarray:
        """Draw circular wagon wheel with angle rays scaled by distance."""
        s = self.wheel_size
        img = np.zeros((s, s, 3), dtype=np.uint8)
        img[:] = (28, 28, 28)
        cx = cy = s // 2
        radius = s // 2 - WHEEL_MARGIN

        # Ground (inner fill)
        cv2.circle(img, (cx, cy), radius, (36, 90, 40), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, (220, 220, 220), 2, cv2.LINE_AA)

        # Radial grid every 30°
        for deg in range(0, 360, 30):
            ang = radians(deg)
            # Our 0° is up → screen: x = sin, y = -cos
            x2 = int(cx + radius * np.sin(ang))
            y2 = int(cy - radius * np.cos(ang))
            cv2.line(img, (cx, cy), (x2, y2), (55, 55, 55), 1, cv2.LINE_AA)

        # Concentric rings (25/50/75/100%)
        for frac in (0.25, 0.5, 0.75, 1.0):
            cv2.circle(
                img,
                (cx, cy),
                int(radius * frac),
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )

        # Labels
        self._put_centered(img, "Straight", cx, cy - radius - 8)
        self._put_centered(img, "Off", cx + radius + 8, cy, verticalish=True)
        self._put_centered(img, "Leg", cx - radius - 8, cy, verticalish=True)
        cv2.circle(img, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)

        # Scale spokes to max distance
        max_d = self.max_distance_px
        if max_d is None or max_d <= 0:
            max_d = max((sp.distance_px for sp in spokes), default=1.0)
            max_d = max(max_d, 1.0)

        for sp in spokes:
            color = SHOT_COLORS.get(sp.shot_name, (200, 200, 200))
            r = radius * min(sp.distance_px / max_d, 1.0)
            # Clamp tiny rays so they stay visible
            r = max(r, radius * 0.08)
            ang = radians(sp.angle_deg)
            x2 = int(cx + r * np.sin(ang))
            y2 = int(cy - r * np.cos(ang))
            cv2.line(img, (cx, cy), (x2, y2), color, 3, cv2.LINE_AA)
            cv2.circle(img, (x2, y2), 5, color, -1, cv2.LINE_AA)

        # Legend + latest stats
        self._draw_legend(img, spokes)

        if spokes:
            latest = spokes[-1]
            dist_txt = f"{latest.distance_px:.0f} px"
            if latest.distance_m is not None:
                dist_txt = f"{latest.distance_m:.1f} m ({latest.distance_px:.0f} px)"
            stats = (
                f"{latest.shot_name}  |  "
                f"angle {latest.angle_deg:+.1f} deg  |  "
                f"dist {dist_txt}"
            )
            cv2.putText(
                img,
                stats,
                (16, s - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        title = "Wagon Wheel"
        cv2.putText(
            img,
            title,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    def _draw_legend(self, img: np.ndarray, spokes: Sequence[ShotGeometry]) -> None:
        names = list(SHOT_DISPLAY_NAMES)
        x0, y0 = 16, 48
        for i, name in enumerate(names):
            color = SHOT_COLORS.get(name, (180, 180, 180))
            y = y0 + i * 18
            cv2.rectangle(img, (x0, y - 8), (x0 + 14, y + 4), color, -1)
            cv2.putText(
                img,
                name,
                (x0 + 20, y + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _put_centered(
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        verticalish: bool = False,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        if verticalish:
            # Keep label near left/right edge without rotation complexity
            tx = x - tw // 2 if x > img.shape[1] // 2 else x - tw // 2
            tx = int(np.clip(tx, 4, img.shape[1] - tw - 4))
            ty = y + th // 2
        else:
            tx = x - tw // 2
            ty = y
        cv2.putText(
            img, text, (tx, ty), font, scale, (220, 220, 220), 1, cv2.LINE_AA
        )

    def save(self, wheel_image: np.ndarray, filename: str = "wagon_wheel.png") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        cv2.imwrite(str(path), wheel_image)
        return path

    def reset(self) -> None:
        self._history.clear()
