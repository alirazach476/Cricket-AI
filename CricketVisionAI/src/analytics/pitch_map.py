"""
Pitch map analytics for CricketVisionAI.

Uses:
  - Pitch bounding box from the DRS model (class 2: Pitch)
  - Smoothed ball centers from ByteTrack

Estimates the bounce point in image space, projects it onto a
top-down pitch diagram via homography, categorizes delivery length
(Yorker / Full Length / Good Length / Short Ball), and saves a
visualization image.

This module does not run the DRS model itself — callers pass the
detected pitch box and ball coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.config.settings import OUTPUT_DIR

# Real cricket pitch length (m). Width used for schematic map.
PITCH_LENGTH_M = 20.12
PITCH_WIDTH_M = 3.05

# Map canvas size (pixels) — bowling end at top, batting end at bottom
MAP_WIDTH = 240
MAP_HEIGHT = 640

Point2D = Tuple[float, float]
Quad = np.ndarray  # shape (4, 2) — TL, TR, BR, BL in image coords


def hypot_dist(x0: float, y0: float, x1: float, y1: float) -> float:
    return float(((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5)


class DeliveryLength(str, Enum):
    YORKER = "Yorker"
    FULL_LENGTH = "Full Length"
    GOOD_LENGTH = "Good Length"
    SHORT_BALL = "Short Ball"
    UNKNOWN = "Unknown"


# Distance of bounce from BATSMAN end (metres along pitch)
# Yorker near the toes; Short farther toward the bowler.
LENGTH_BINS_M = (
    (0.0, 2.0, DeliveryLength.YORKER),
    (2.0, 4.5, DeliveryLength.FULL_LENGTH),
    (4.5, 8.0, DeliveryLength.GOOD_LENGTH),
    (8.0, PITCH_LENGTH_M, DeliveryLength.SHORT_BALL),
)


@dataclass
class BounceEstimate:
    """Bounce location in image and pitch-map coordinates."""

    frame_number: int
    image_xy: Point2D
    map_xy: Point2D
    # Metres from batting crease toward bowler (0 = at batsman end)
    distance_from_batsman_m: float
    # Normalised [0, 1] along pitch (0 = batting end, 1 = bowling end)
    pitch_t: float
    length: DeliveryLength


@dataclass
class PitchMapResult:
    bounce: Optional[BounceEstimate]
    map_image: np.ndarray
    ball_map_path: List[Point2D] = field(default_factory=list)
    pitch_quad_image: Optional[Quad] = None
    homography: Optional[np.ndarray] = None  # 3x3 image -> map


class PitchMapBuilder:
    """
    Build a top-down pitch map from a DRS pitch box + ball track.

    Homography:
        H maps points from the camera image onto a rectangular pitch
        diagram so perspective foreshortening is removed. See module
        docstring / project notes for the conceptual explanation.
    """

    def __init__(
        self,
        map_size: Tuple[int, int] = (MAP_WIDTH, MAP_HEIGHT),
        pitch_length_m: float = PITCH_LENGTH_M,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.map_w, self.map_h = map_size
        self.pitch_length_m = float(pitch_length_m)
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR

        # Destination rectangle corners: TL, TR, BR, BL
        # Top = bowling end, bottom = batting end (common pitch-map view)
        self._dst_quad = np.float32(
            [
                [0, 0],
                [self.map_w - 1, 0],
                [self.map_w - 1, self.map_h - 1],
                [0, self.map_h - 1],
            ]
        )

        self._H: Optional[np.ndarray] = None  # image -> map
        self._H_inv: Optional[np.ndarray] = None
        self._src_quad: Optional[Quad] = None

    # ------------------------------------------------------------------
    # Pitch geometry / homography
    # ------------------------------------------------------------------

    def set_pitch_bbox(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        perspective_skew: float = 0.08,
    ) -> np.ndarray:
        """
        Approximate a pitch quadrilateral from a YOLO axis-aligned box.

        The top edge is slightly narrowed to simulate camera perspective
        (bowling end farther from camera in many broadcast angles).
        For best results, replace this with true pitch corner points
        when available.
        """
        w = x2 - x1
        inset = w * perspective_skew
        # TL, TR, BR, BL
        quad = np.float32(
            [
                [x1 + inset, y1],
                [x2 - inset, y1],
                [x2, y2],
                [x1, y2],
            ]
        )
        return self.set_pitch_quad(quad)

    def set_pitch_quad(self, quad: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Set four pitch corners (TL, TR, BR, BL) in image coordinates
        and compute the homography to the rectangular pitch map.
        """
        src = np.float32(quad).reshape(4, 2)
        self._src_quad = src
        self._H = cv2.getPerspectiveTransform(src, self._dst_quad)
        self._H_inv = cv2.getPerspectiveTransform(self._dst_quad, src)
        return self._H

    @property
    def homography(self) -> Optional[np.ndarray]:
        return self._H

    def image_to_map(self, x: float, y: float) -> Optional[Point2D]:
        if self._H is None:
            return None
        pts = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pts, self._H)[0, 0]
        return float(out[0]), float(out[1])

    def map_to_image(self, x: float, y: float) -> Optional[Point2D]:
        if self._H_inv is None:
            return None
        pts = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pts, self._H_inv)[0, 0]
        return float(out[0]), float(out[1])

    # ------------------------------------------------------------------
    # Bounce estimation
    # ------------------------------------------------------------------

    def estimate_bounce(
        self,
        ball_xy: Sequence[Point2D],
        frame_numbers: Optional[Sequence[int]] = None,
    ) -> Optional[BounceEstimate]:
        """
        Estimate bounce as the frame where the ball's downward motion
        (increasing image-y) reverses while near/on the pitch strip.

        Uses smoothed centres; picks the strongest local peak in image-y
        whose map projection lands inside the pitch rectangle.
        """
        if self._H is None or len(ball_xy) < 3:
            return None

        ys = np.array([p[1] for p in ball_xy], dtype=np.float64)
        xs = np.array([p[0] for p in ball_xy], dtype=np.float64)

        # Drop teleport outliers before bounce search
        keep = [0]
        for i in range(1, len(xs)):
            if hypot_dist(xs[i], ys[i], xs[keep[-1]], ys[keep[-1]]) <= 90:
                keep.append(i)
        if len(keep) < 3:
            return None
        xs = xs[keep]
        ys = ys[keep]
        if frame_numbers is not None:
            frame_numbers = [frame_numbers[i] for i in keep]

        # Local maxima in image-y ≈ lowest points on screen (bounce candidates)
        candidates: List[int] = []
        for i in range(1, len(ys) - 1):
            if ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1]:
                candidates.append(i)

        if not candidates:
            candidates = [int(np.argmax(ys))]

        best_idx: Optional[int] = None
        best_score = -1.0

        for i in candidates:
            mapped = self.image_to_map(float(xs[i]), float(ys[i]))
            if mapped is None:
                continue
            mx, my = mapped
            # Must land well inside pitch map (reject edge noise)
            if not (0.08 * self.map_w <= mx < 0.92 * self.map_w):
                continue
            if not (0.05 * self.map_h <= my < 0.98 * self.map_h):
                continue
            centre_pen = abs(mx - self.map_w / 2.0) / max(self.map_w / 2.0, 1.0)
            # Prefer mid-pitch lengths over extreme (stumps / very short noise)
            length_pref = 1.0 - abs((my / max(self.map_h - 1, 1)) - 0.55)
            score = float(ys[i]) * (1.0 - 0.4 * centre_pen) * (0.6 + 0.4 * length_pref)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            return None

        ix, iy = float(xs[best_idx]), float(ys[best_idx])
        mapped = self.image_to_map(ix, iy)
        if mapped is None:
            return None
        mx, my = mapped

        pitch_t_from_bowler = my / max(self.map_h - 1, 1)
        distance_from_batsman_m = (1.0 - pitch_t_from_bowler) * self.pitch_length_m
        distance_from_batsman_m = float(
            np.clip(distance_from_batsman_m, 0.0, self.pitch_length_m)
        )
        pitch_t = distance_from_batsman_m / self.pitch_length_m

        length = categorize_length(distance_from_batsman_m)
        frame_number = (
            int(frame_numbers[best_idx])
            if frame_numbers is not None and best_idx < len(frame_numbers)
            else best_idx
        )

        return BounceEstimate(
            frame_number=frame_number,
            image_xy=(ix, iy),
            map_xy=(mx, my),
            distance_from_batsman_m=distance_from_batsman_m,
            pitch_t=pitch_t,
            length=length,
        )

    # ------------------------------------------------------------------
    # Visualization + pipeline helper
    # ------------------------------------------------------------------

    def build(
        self,
        ball_xy: Sequence[Point2D],
        frame_numbers: Optional[Sequence[int]] = None,
    ) -> PitchMapResult:
        """Project ball path, estimate bounce, render map image."""
        map_path: List[Point2D] = []
        for x, y in ball_xy:
            p = self.image_to_map(x, y)
            if p is not None:
                map_path.append(p)

        bounce = self.estimate_bounce(ball_xy, frame_numbers)
        map_image = self.render_map(map_path, bounce)

        return PitchMapResult(
            bounce=bounce,
            map_image=map_image,
            ball_map_path=map_path,
            pitch_quad_image=None if self._src_quad is None else self._src_quad.copy(),
            homography=None if self._H is None else self._H.copy(),
        )

    def render_map(
        self,
        ball_map_path: Sequence[Point2D],
        bounce: Optional[BounceEstimate],
    ) -> np.ndarray:
        """
        Circular cricket ground with a rectangular pitch strip in the centre.
        Internal bounce math still uses the rectangular map coords.
        """
        ground = 720
        img = np.zeros((ground, ground, 3), dtype=np.uint8)
        img[:] = (18, 28, 22)

        cx = cy = ground // 2
        radius = ground // 2 - 18
        cv2.circle(img, (cx, cy), radius, (42, 128, 52), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, (230, 230, 230), 3, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(radius * 0.72), (55, 145, 65), 2, cv2.LINE_AA)

        pitch_w = max(48, int(self.map_w * 0.42))
        pitch_h = max(220, int(self.map_h * 0.55))
        ox = cx - pitch_w // 2
        oy = cy - pitch_h // 2

        pitch = np.zeros((pitch_h, pitch_w, 3), dtype=np.uint8)
        pitch[:] = (34, 110, 45)
        self._draw_length_bands_on(pitch, pitch_w, pitch_h)
        cv2.rectangle(pitch, (0, 0), (pitch_w - 1, pitch_h - 1), (220, 220, 220), 2)
        mid = pitch_w // 2
        cv2.line(pitch, (mid, 0), (mid, pitch_h - 1), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(pitch, (6, 14), (pitch_w - 6, 14), (255, 255, 255), 2)
        cv2.line(pitch, (6, pitch_h - 14), (pitch_w - 6, pitch_h - 14), (255, 255, 255), 2)

        roi = img[oy : oy + pitch_h, ox : ox + pitch_w]
        if roi.shape[:2] == pitch.shape[:2]:
            img[oy : oy + pitch_h, ox : ox + pitch_w] = pitch

        def to_ground(mx: float, my: float) -> Tuple[int, int]:
            gx = ox + int(mx / max(self.map_w - 1, 1) * (pitch_w - 1))
            gy = oy + int(my / max(self.map_h - 1, 1) * (pitch_h - 1))
            return gx, gy

        if len(ball_map_path) >= 2:
            mapped = [to_ground(p[0], p[1]) for p in ball_map_path]
            for i in range(1, len(mapped)):
                if abs(mapped[i][0] - mapped[i - 1][0]) + abs(
                    mapped[i][1] - mapped[i - 1][1]
                ) > 80:
                    continue
                cv2.line(img, mapped[i - 1], mapped[i], (0, 255, 255), 2, cv2.LINE_AA)

        if bounce is not None:
            bx, by = to_ground(bounce.map_xy[0], bounce.map_xy[1])
            cv2.circle(img, (bx, by), 9, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (bx, by), 13, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                img,
                bounce.length.value,
                (cx - 70, ground - 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"{bounce.distance_from_batsman_m:.1f}m",
                (cx - 40, ground - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                img,
                "No bounce",
                (cx - 50, ground - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            img,
            "GROUND",
            (cx - 42, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Bowler",
            (cx - 28, oy + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Batsman",
            (cx - 32, oy + pitch_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return img

    def _draw_length_bands(self, img: np.ndarray) -> None:
        self._draw_length_bands_on(img, self.map_w, self.map_h)

    def _draw_length_bands_on(self, img: np.ndarray, w: int, h: int) -> None:
        """Shade length zones measured from the batting end (bottom)."""
        colors = {
            DeliveryLength.YORKER: (40, 40, 140),
            DeliveryLength.FULL_LENGTH: (40, 90, 140),
            DeliveryLength.GOOD_LENGTH: (40, 120, 60),
            DeliveryLength.SHORT_BALL: (40, 100, 40),
        }
        for start_m, end_m, length in LENGTH_BINS_M:
            y_bottom = int(h * (1.0 - start_m / self.pitch_length_m))
            y_top = int(h * (1.0 - end_m / self.pitch_length_m))
            y0, y1 = min(y_top, y_bottom), max(y_top, y_bottom)
            overlay = img.copy()
            cv2.rectangle(overlay, (0, y0), (w - 1, y1), colors[length], -1)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    def save(self, map_image: np.ndarray, filename: str = "pitch_map.png") -> Path:
        """Save pitch map visualization under output/."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        cv2.imwrite(str(path), map_image)
        return path


def categorize_length(distance_from_batsman_m: float) -> DeliveryLength:
    """Map bounce distance from batsman end to a length category."""
    d = float(distance_from_batsman_m)
    for start, end, label in LENGTH_BINS_M:
        if start <= d < end:
            return label
    if d >= PITCH_LENGTH_M:
        return DeliveryLength.SHORT_BALL
    return DeliveryLength.UNKNOWN
