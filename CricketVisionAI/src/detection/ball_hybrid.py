"""
Hybrid ball association for CricketVisionAI.

YOLO may return multiple false positives (grass patches, logos). This module
picks at most one ball using a constant-velocity gate + pitch corridor score.
Classical CV is NOT used to invent detections.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.config.settings import BALL_CLASS_ID, BALL_CONFIDENCE, PLAYER_CLASS_NAMES
from src.tracking.models import TrackedObject

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def _expand_box(box: BBox, pad: float, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = box
    return (
        max(0.0, x1 - pad),
        max(0.0, y1 - pad),
        min(float(w), x2 + pad),
        min(float(h), y2 + pad),
    )


def point_in_box(x: float, y: float, box: BBox) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


class HybridBallDetector:
    """Associate YOLO ball candidates with a short velocity memory."""

    def __init__(
        self,
        max_jump_px: float = 75.0,
        ball_track_id: int = 1,
        coast_frames: int = 0,
        enable_classical: bool = False,
        min_conf: float = BALL_CONFIDENCE,
    ) -> None:
        self.max_jump_px = float(max_jump_px)
        self.ball_track_id = int(ball_track_id)
        self.coast_frames = int(coast_frames)
        self.enable_classical = bool(enable_classical)
        self.min_conf = float(min_conf)
        self._last_center: Optional[Point] = None
        self._last_bbox: Optional[BBox] = None
        self._velocity: Optional[Point] = None  # px / frame
        self._missed = 0
        self._static_hits = 0
        self._last_accepted_raw: Optional[Point] = None

    def fuse(
        self,
        frame: np.ndarray,
        tracked: List[TrackedObject],
        pitch_box: Optional[BBox] = None,
        candidates: Optional[List[TrackedObject]] = None,
    ) -> List[TrackedObject]:
        h, w = frame.shape[:2]
        others = [o for o in tracked if o.class_id != BALL_CLASS_ID]
        yolo_balls = list(candidates) if candidates is not None else [
            o for o in tracked if o.class_id == BALL_CLASS_ID
        ]

        search_box: Optional[BBox] = None
        if pitch_box is not None:
            pw = pitch_box[2] - pitch_box[0]
            ph = pitch_box[3] - pitch_box[1]
            search_box = _expand_box(
                pitch_box, pad=max(90.0, 0.62 * max(pw, ph)), w=w, h=h
            )

        def corridor_score(cx: float, cy: float) -> float:
            if pitch_box is None:
                return 0.5
            px1, py1, px2, py2 = pitch_box
            # Prefer centre strip of the pitch (bowling line)
            mid_x = 0.5 * (px1 + px2)
            mid_pen = abs(cx - mid_x) / max(1.0, 0.5 * (px2 - px1))
            # Prefer inside pitch y-range
            in_y = 1.0 if py1 <= cy <= py2 else 0.25
            return max(0.0, 1.0 - 0.7 * mid_pen) * in_y

        predicted: Optional[Point] = None
        if self._last_center is not None:
            if self._velocity is not None:
                predicted = (
                    self._last_center[0] + self._velocity[0],
                    self._last_center[1] + self._velocity[1],
                )
            else:
                predicted = self._last_center

        scored: List[Tuple[float, TrackedObject]] = []
        for o in yolo_balls:
            cx, cy = o.center
            bw, bh = o.x2 - o.x1, o.y2 - o.y1
            if bw < 3 or bh < 3 or bw > 55 or bh > 55:
                continue
            aspect = max(bw, bh) / max(1e-3, min(bw, bh))
            if aspect > 1.65:
                continue
            if o.confidence < self.min_conf * 0.85:
                continue
            if (
                search_box is not None
                and self._missed < 2
                and not point_in_box(cx, cy, search_box)
            ):
                continue

            gate = self.max_jump_px
            if self._missed > 0:
                gate = self.max_jump_px * (1.0 + 0.35 * self._missed)
            prox = 1.0
            if predicted is not None:
                dist = ((cx - predicted[0]) ** 2 + (cy - predicted[1]) ** 2) ** 0.5
                if dist > gate:
                    continue
                prox = max(0.0, 1.0 - dist / max(gate, 1.0))

            score = (
                0.45 * float(o.confidence)
                + 0.40 * prox
                + 0.15 * corridor_score(cx, cy)
            )
            scored.append((score, o))

        chosen: Optional[TrackedObject] = None
        if scored:
            scored.sort(key=lambda t: t[0], reverse=True)
            best_score, best = scored[0]
            # Need a reasonable overall score; cold-start needs higher conf
            if self._last_center is None:
                if best.confidence >= self.min_conf and best_score >= 0.35:
                    chosen = best
            elif best_score >= 0.32:
                chosen = best

        # Reject nearly-static noise (logo/grass glued in place)
        if chosen is not None and self._last_accepted_raw is not None:
            cx, cy = chosen.center
            lx, ly = self._last_accepted_raw
            moved = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            if moved < 1.2:
                self._static_hits += 1
                if self._static_hits >= 8 and chosen.confidence < 0.65:
                    chosen = None  # stuck on a paint spot
            else:
                self._static_hits = 0

        if chosen is None:
            self._missed += 1
            if self._missed > 12:
                self.reset()
            return others

        cx, cy = chosen.center
        if self._last_center is not None:
            vx = cx - self._last_center[0]
            vy = cy - self._last_center[1]
            if self._velocity is None:
                self._velocity = (vx, vy)
            else:
                self._velocity = (
                    0.55 * vx + 0.45 * self._velocity[0],
                    0.55 * vy + 0.45 * self._velocity[1],
                )
            # Cap insane velocity (teleport residue)
            speed = (self._velocity[0] ** 2 + self._velocity[1] ** 2) ** 0.5
            if speed > self.max_jump_px:
                scale = self.max_jump_px / speed
                self._velocity = (self._velocity[0] * scale, self._velocity[1] * scale)

        self._last_center = (cx, cy)
        self._last_bbox = chosen.bbox
        self._last_accepted_raw = (cx, cy)
        self._missed = 0

        ball = TrackedObject(
            object_id=self.ball_track_id,
            class_id=BALL_CLASS_ID,
            class_name=PLAYER_CLASS_NAMES.get(BALL_CLASS_ID, "Ball"),
            x1=chosen.x1,
            y1=chosen.y1,
            x2=chosen.x2,
            y2=chosen.y2,
            confidence=chosen.confidence,
        )
        return others + [ball]

    def reset(self) -> None:
        self._last_center = None
        self._last_bbox = None
        self._velocity = None
        self._missed = 0
        self._static_hits = 0
        self._last_accepted_raw = None
