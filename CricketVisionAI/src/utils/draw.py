"""
Drawing helpers for CricketVisionAI annotated video frames.

Draws bounding boxes and labels including ByteTrack object IDs
(e.g. "Batsman #3") for Ball, Batsman, and Bowler.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config.settings import DRS_COLORS, PLAYER_COLORS, SHOT_ACTION_COLORS
from src.tracking.models import TrackedObject

Color = Tuple[int, int, int]

BROADCAST_ROLE_COLORS: Dict[str, Color] = {
    "BATSMAN": (0, 255, 0),
    "FIELDER": (255, 160, 0),
    "BOWLER": (255, 128, 0),
    "UMPIRE": (0, 0, 255),
    "BALL": (0, 255, 255),
}
BROADCAST_ROLE_SHORT = {
    "BATSMAN": "B",
    "FIELDER": "F",
    "BOWLER": "Bo",
    "UMPIRE": "U",
    "BALL": "o",
}

# Ball trail color (BGR) — matches Ball class yellow family
TRAJECTORY_COLOR: Color = (0, 220, 255)
SPEED_PANEL_BG: Color = (20, 20, 20)
SPEED_PANEL_FG: Color = (0, 255, 255)

# Compact HUD text
HUD_FONT_SCALE = 0.42
HUD_THICKNESS = 1
HUD_PAD = 4
HUD_LINE_H = 16


def _draw_hud_line(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    fg: Color,
) -> np.ndarray:
    """Small single-line HUD badge (much less screen coverage)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, HUD_FONT_SCALE, HUD_THICKNESS)
    x, y = origin
    pad = HUD_PAD
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline),
        SPEED_PANEL_BG,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y - 1),
        font,
        HUD_FONT_SCALE,
        fg,
        HUD_THICKNESS,
        cv2.LINE_AA,
    )
    return frame


def draw_ball_speed(
    frame: np.ndarray,
    speed_px_per_sec: Optional[float],
    speed_kmh: Optional[float] = None,
    origin: Tuple[int, int] = (12, 18),
) -> np.ndarray:
    if speed_kmh is not None and speed_kmh > 0:
        text = f"Speed: {speed_kmh:.0f} km/h"
    elif speed_px_per_sec is not None and speed_px_per_sec > 0:
        text = f"Speed: {speed_px_per_sec:.0f} px/s"
    else:
        text = "Speed: -- km/h"
    return _draw_hud_line(frame, text, origin, SPEED_PANEL_FG)


def draw_shot_prediction(
    frame: np.ndarray,
    shot_name: Optional[str],
    confidence: Optional[float],
    origin: Tuple[int, int] = (12, 36),
) -> np.ndarray:
    if shot_name is None or confidence is None:
        text = "Shot: --"
    else:
        text = f"Shot: {shot_name} ({confidence * 100:.0f}%)"
    return _draw_hud_line(frame, text, origin, (0, 255, 128))


def draw_shot_actions(
    frame: np.ndarray,
    detections: Sequence,
    thickness: int = 2,
    font_scale: float = 0.45,
) -> np.ndarray:
    """Draw Cricket Actions YOLO boxes (Cover Drive, Pull Shot, …)."""
    for det in detections:
        class_id = int(getattr(det, "class_id", -1))
        color: Color = SHOT_ACTION_COLORS.get(class_id, (0, 255, 128))
        x1, y1, x2, y2 = map(
            int,
            (
                getattr(det, "x1", 0),
                getattr(det, "y1", 0),
                getattr(det, "x2", 0),
                getattr(det, "y2", 0),
            ),
        )
        name = str(getattr(det, "class_name", "Shot"))
        conf = float(getattr(det, "confidence", 0.0))
        label = f"{name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        _draw_label(frame, label, x1, y1, color, font_scale)
    return frame


def _broadcast_role(obj: TrackedObject, frame: np.ndarray) -> str:
    """Map pipeline class ids from cricket detection + bowler + ball models."""
    if obj.class_id == 0:
        return "BALL"
    if obj.class_id == 4:
        return "BOWLER"
    if obj.class_id == 3:
        return "UMPIRE"
    if obj.class_id == 2:
        return "FIELDER"
    return "BATSMAN"


def _xy_to_pitch_uv(
    x: float,
    y: float,
    pitch_box: Optional[Tuple[float, float, float, float]],
    frame_wh: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[float, float]]:
    if pitch_box is not None:
        px1, py1, px2, py2 = pitch_box
        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)
        return (
            float(np.clip((x - px1) / pw, -0.15, 1.15)),
            float(np.clip((y - py1) / ph, -0.15, 1.15)),
        )
    if frame_wh is not None:
        fw, fh = frame_wh
        return (
            float(np.clip(x / max(1.0, fw), 0.0, 1.0)),
            float(np.clip(y / max(1.0, fh), 0.0, 1.0)),
        )
    return None


def draw_tracked_objects(
    frame: np.ndarray,
    tracked_objects: Iterable[TrackedObject],
    colors: Optional[dict] = None,
    thickness: int = 2,
    font_scale: float = 0.45,
    broadcast_style: bool = False,
) -> np.ndarray:
    """
    Draw boxes and labels on a copy of the frame.

    broadcast_style=True matches aaa.mp4: BATSMAN: 0.82, white center dot.
    """
    out = frame.copy()
    palette = colors if colors is not None else PLAYER_COLORS

    for obj in tracked_objects:
        if broadcast_style:
            role = _broadcast_role(obj, frame)
            color = BROADCAST_ROLE_COLORS.get(role, (255, 255, 255))
            label = f"{role}: {obj.confidence:.2f}"
        else:
            color = palette.get(obj.class_id, (255, 255, 255))
            label = f"{obj.class_name} #{obj.object_id}"

        x1, y1, x2, y2 = map(int, (obj.x1, obj.y1, obj.x2, obj.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        _draw_label(out, label, x1, y1, color, font_scale)

        cx, cy = map(int, obj.center)
        if broadcast_style:
            cv2.circle(out, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(out, (cx, cy), 3, color, -1)
        else:
            cv2.circle(out, (cx, cy), 3, color, -1)

    return out


def _panel_glass(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fill: Color = (16, 22, 28),
    alpha: float = 0.82,
) -> None:
    """Soft translucent panel base."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _put(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Color,
    scale: float = 0.4,
    thick: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thick,
        cv2.LINE_AA,
    )


def draw_cricket_analysis_panel(
    frame: np.ndarray,
    frame_number: int,
    ball_speed_kmh: Optional[float] = None,
    max_speed_kmh: Optional[float] = None,
    avg_speed_kmh: Optional[float] = None,
    detections: int = 0,
    collisions: int = 0,
    status: str = "Tracking",
    origin: Tuple[int, int] = (10, 10),
    panel_w: int = 236,
    panel_h: int = 162,
) -> np.ndarray:
    """Top-left live analysis HUD — broadcast glass card."""
    x0, y0 = origin
    x1, y1 = x0 + panel_w, y0 + panel_h
    accent = (48, 196, 210)  # warm teal (BGR)
    ink = (236, 238, 240)
    mute = (160, 168, 176)

    _panel_glass(frame, x0, y0, x1, y1, fill=(18, 24, 30), alpha=0.86)
    # Left accent rail
    cv2.rectangle(frame, (x0, y0), (x0 + 4, y1), accent, -1)
    # Top hairline
    cv2.line(frame, (x0 + 4, y0), (x1, y0), accent, 1, cv2.LINE_AA)
    cv2.line(frame, (x0 + 4, y1), (x1, y1), (50, 58, 66), 1, cv2.LINE_AA)

    # Header band
    cv2.rectangle(frame, (x0 + 4, y0), (x1, y0 + 28), (28, 36, 44), -1)
    _put(frame, "LIVE ANALYSIS", (x0 + 14, y0 + 19), accent, 0.45, 1)
    _put(frame, f"#{frame_number:04d}", (x1 - 58, y0 + 19), mute, 0.38, 1)

    def _fmt(v: Optional[float]) -> str:
        if v is None or v <= 0:
            return "--"
        return f"{v:.0f}"

    # Hero speed block
    speed_y = y0 + 52
    _put(frame, "BALL SPEED", (x0 + 14, speed_y), mute, 0.34, 1)
    speed_txt = f"{_fmt(ball_speed_kmh)} km/h"
    _put(frame, speed_txt, (x0 + 14, speed_y + 22), ink, 0.62, 2)

    # Divider
    dy = speed_y + 34
    cv2.line(frame, (x0 + 14, dy), (x1 - 12, dy), (48, 56, 64), 1, cv2.LINE_AA)

    # Metric grid 2x2
    rows = [
        ("MAX", f"{_fmt(max_speed_kmh)}", "AVG", f"{_fmt(avg_speed_kmh)}"),
        ("DETECT", str(detections), "CLASH", str(collisions)),
    ]
    my = dy + 18
    col_w = (panel_w - 28) // 2
    for r0_l, r0_v, r1_l, r1_v in rows:
        _put(frame, r0_l, (x0 + 14, my), mute, 0.32, 1)
        _put(frame, r0_v, (x0 + 14, my + 16), ink, 0.48, 1)
        _put(frame, r1_l, (x0 + 14 + col_w, my), mute, 0.32, 1)
        _put(frame, r1_v, (x0 + 14 + col_w, my + 16), ink, 0.48, 1)
        my += 38

    return frame


def draw_field_view_panel(
    frame: np.ndarray,
    tracked_objects: Iterable[TrackedObject],
    pitch_box: Optional[Tuple[float, float, float, float]] = None,
    ball_uv: Optional[Tuple[float, float]] = None,
    player_trails: Optional[Dict[int, Deque[Tuple[float, float]]]] = None,
    panel_w: int = 176,
    panel_h: int = 196,
    margin: int = 10,
) -> np.ndarray:
    """
    Top-right oval ground schematic with role markers and player trails.
    """
    h, w = frame.shape[:2]
    x0 = w - panel_w - margin
    y0 = margin
    x1, y1 = x0 + panel_w, y0 + panel_h
    accent = (72, 180, 96)
    ink = (236, 238, 240)
    mute = (150, 160, 168)

    _panel_glass(frame, x0, y0, x1, y1, fill=(14, 20, 18), alpha=0.88)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (40, 70, 50), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x1, y0 + 26), (24, 40, 30), -1)
    _put(frame, "FIELD MAP", (x0 + 12, y0 + 18), accent, 0.42, 1)

    # Ground oval
    map_top = y0 + 34
    map_bot = y1 - 36
    map_cx = (x0 + x1) // 2
    map_cy = (map_top + map_bot) // 2
    rx = max(40, (panel_w - 28) // 2)
    ry = max(42, (map_bot - map_top) // 2)

    ground = frame.copy()
    cv2.ellipse(ground, (map_cx, map_cy), (rx, ry), 0, 0, 360, (36, 98, 52), -1, cv2.LINE_AA)
    cv2.addWeighted(ground, 0.92, frame, 0.08, 0, frame)
    cv2.ellipse(frame, (map_cx, map_cy), (rx, ry), 0, 0, 360, (90, 170, 110), 1, cv2.LINE_AA)
    # Inner circle
    cv2.ellipse(
        frame,
        (map_cx, map_cy),
        (int(rx * 0.58), int(ry * 0.58)),
        0,
        0,
        360,
        (70, 140, 90),
        1,
        cv2.LINE_AA,
    )

    # Pitch strip (center)
    pitch_w = max(12, rx // 5)
    pitch_h = max(44, int(ry * 1.15))
    px0 = map_cx - pitch_w // 2
    py0 = map_cy - pitch_h // 2
    cv2.rectangle(
        frame,
        (px0, py0),
        (px0 + pitch_w, py0 + pitch_h),
        (210, 220, 200),
        -1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        frame,
        (px0, py0),
        (px0 + pitch_w, py0 + pitch_h),
        (180, 190, 170),
        1,
        cv2.LINE_AA,
    )
    # Creases
    cv2.line(
        frame,
        (px0 - 2, py0 + 6),
        (px0 + pitch_w + 2, py0 + 6),
        (120, 130, 110),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (px0 - 2, py0 + pitch_h - 6),
        (px0 + pitch_w + 2, py0 + pitch_h - 6),
        (120, 130, 110),
        1,
        cv2.LINE_AA,
    )

    players: List[Tuple[str, float, float, Color]] = []
    ball_found = ball_uv is not None
    frame_wh = (w, h)
    for obj in tracked_objects:
        role = _broadcast_role(obj, frame)
        uv = _xy_to_pitch_uv(
            obj.center[0], obj.center[1], pitch_box, frame_wh=frame_wh
        )
        if uv is None:
            continue
        u, v = uv
        if role == "BALL":
            ball_uv = (u, v)
            ball_found = True
            continue
        players.append((role, u, v, BROADCAST_ROLE_COLORS.get(role, (200, 200, 200))))

    # Player movement trails (not ball)
    if player_trails:
        for trail in player_trails.values():
            pts = list(trail)
            for i in range(1, len(pts)):
                u0, v0 = pts[i - 1]
                u1, v1 = pts[i]
                p0 = (int(px0 + u0 * pitch_w), int(py0 + v0 * pitch_h))
                p1 = (int(px0 + u1 * pitch_w), int(py0 + v1 * pitch_h))
                t = i / max(1, len(pts) - 1)
                col = (int(60 + 80 * t), int(140 + 60 * t), int(200 + 40 * t))
                cv2.line(frame, p0, p1, col, 1, cv2.LINE_AA)

    player_count = 0
    for role, u, v, color in players:
        # Map slightly beyond pitch into oval for distant fielders
        mx = int(px0 + float(np.clip(u, -0.2, 1.2)) * pitch_w)
        my = int(py0 + float(np.clip(v, -0.2, 1.2)) * pitch_h)
        cv2.circle(frame, (mx, my), 7, (20, 24, 28), -1, cv2.LINE_AA)
        cv2.circle(frame, (mx, my), 5, color, -1, cv2.LINE_AA)
        short = BROADCAST_ROLE_SHORT.get(role, role[:1])
        _put(frame, short, (mx - 4, my + 3), (12, 12, 12), 0.28, 1)
        player_count += 1

    if ball_uv is not None:
        bu, bv = ball_uv
        bx = int(px0 + bu * pitch_w)
        by = int(py0 + bv * pitch_h)
        cv2.circle(frame, (bx, by), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 7, (0, 220, 255), 1, cv2.LINE_AA)

    # Footer status
    _put(frame, f"{player_count} players", (x0 + 12, y1 - 18), ink, 0.36, 1)
    ball_txt = "Ball Found" if ball_found else "Ball Not Found"
    ball_col = accent if ball_found else (70, 70, 210)
    (btw, _), _ = cv2.getTextSize(ball_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
    bx0 = x1 - btw - 18
    cv2.rectangle(frame, (bx0, y1 - 28), (x1 - 8, y1 - 10), (30, 36, 40), -1)
    _put(frame, ball_txt, (bx0 + 5, y1 - 14), ball_col, 0.32, 1)
    return frame


def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Color,
    font_scale: float,
) -> None:
    """Draw a filled label background above the box top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Place label above the box; clamp inside the frame if near the top
    text_x = max(0, x)
    text_y = y - 8
    if text_y - th - 4 < 0:
        text_y = y + th + 8

    pad = 4
    cv2.rectangle(
        frame,
        (text_x, text_y - th - pad),
        (text_x + tw + pad, text_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (text_x + 2, text_y - 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_ball_trajectory(
    frame: np.ndarray,
    path_xy: Sequence[Tuple[float, float]],
    color: Color = TRAJECTORY_COLOR,
    thickness: int = 2,
    max_gap_px: float = 70.0,
) -> np.ndarray:
    """Draw smoothed ball path; break the polyline on large jumps."""
    if path_xy is None or len(path_xy) < 1:
        return frame

    pts = [(int(x), int(y)) for x, y in path_xy]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 > max_gap_px:
            continue
        cv2.line(frame, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

    n = len(pts)
    for i, (x, y) in enumerate(pts):
        radius = 2 if i < n - 1 else 5
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)

    return frame


def draw_drs_detections(
    frame: np.ndarray,
    detections: Sequence,
    thickness: int = 1,
    font_scale: float = 0.4,
) -> np.ndarray:
    """
    Draw DRS boxes (Bat, Pad, Pitch) onto an existing frame.
    Stumps are skipped even if present.
    """
    for det in detections:
        class_id = int(det.class_id)
        if class_id == 3:  # Stumps — disabled
            continue
        color: Color = DRS_COLORS.get(class_id, (200, 200, 200))
        x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
        label = f"{det.class_name} {float(det.confidence):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        _draw_label(frame, label, x1, y1, color, font_scale)
    return frame


def draw_delivery_length(
    frame: np.ndarray,
    length_name: Optional[str],
    distance_m: Optional[float] = None,
    origin: Tuple[int, int] = (12, 54),
) -> np.ndarray:
    if length_name is None:
        text = "Length: --"
    elif distance_m is not None:
        text = f"Length: {length_name} ({distance_m:.1f}m)"
    else:
        text = f"Length: {length_name}"
    return _draw_hud_line(frame, text, origin, (0, 200, 255))


def draw_shot_geometry(
    frame: np.ndarray,
    angle_deg: Optional[float],
    distance_px: Optional[float],
    distance_m: Optional[float] = None,
    origin: Tuple[int, int] = (12, 72),
) -> np.ndarray:
    if angle_deg is None:
        text = "Wagon: --"
    else:
        # Suppress absurd accumulated path lengths after scene cuts
        if distance_px is not None and distance_px > 2500:
            dist = "--"
        elif distance_m is not None:
            dist = f"{distance_m:.1f}m"
        elif distance_px is not None:
            dist = f"{distance_px:.0f}px"
        else:
            dist = "--"
        text = f"Wagon: {angle_deg:+.0f}deg {dist}"
    return _draw_hud_line(frame, text, origin, (255, 180, 100))


def draw_scoreboard_reading(
    frame: np.ndarray,
    reading: Optional[object],
    box: Optional[object] = None,
    origin: Tuple[int, int] = (12, 90),
) -> np.ndarray:
    if box is not None and hasattr(box, "x1"):
        x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

    if reading is None:
        return frame  # don't paint empty "Score: --" over the broadcast graphics

    score = getattr(reading, "score_string", None)
    if not score or score in ("--", "---"):
        return frame
    overs = getattr(reading, "overs", None)
    crr = getattr(reading, "crr", None)
    text = f"Score: {score}  Ov:{overs if overs is not None else '--'}  CRR:{crr if crr is not None else '--'}"
    return _draw_hud_line(frame, text, origin, (255, 0, 255))


def draw_trajectory_panel(
    frame: np.ndarray,
    path_xy: Optional[Sequence[Tuple[float, float]]] = None,
    pitch_box: Optional[Tuple[float, float, float, float]] = None,
    path_uv: Optional[Sequence[Tuple[float, float]]] = None,
    panel_size: int = 148,
    margin: int = 10,
) -> np.ndarray:
    """
    Circular cricket-ground mini-map (TOP RIGHT).

    Prefer path_uv (pitch-normalized 0–1) for a stable bowler→batsman line that
    grows from the first ball detection and does not blink.
    """
    h, w = frame.shape[:2]
    x0 = w - panel_size - margin
    y0 = margin
    cx = x0 + panel_size // 2
    cy = y0 + panel_size // 2
    radius = panel_size // 2 - 2

    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius, (36, 110, 48), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.circle(frame, (cx, cy), radius, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), int(radius * 0.55), (55, 140, 65), 1, cv2.LINE_AA)

    pw, ph = max(10, radius // 6), max(36, int(radius * 0.7))
    pitch_x0, pitch_y0 = cx - pw // 2, cy - ph // 2
    cv2.rectangle(
        frame,
        (pitch_x0, pitch_y0),
        (pitch_x0 + pw, pitch_y0 + ph),
        (200, 220, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "TRAJECTORY",
        (cx - 42, y0 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    mapped: List[Optional[Tuple[int, int]]] = []
    if path_uv is not None and len(path_uv) >= 2:
        for u, v in path_uv:
            nu = float(np.clip(u, -0.12, 1.12))
            nv = float(np.clip(v, -0.12, 1.12))
            mx = int(pitch_x0 + nu * pw)
            my = int(pitch_y0 + nv * ph)
            mapped.append((mx, my))
    elif path_xy is not None and len(path_xy) >= 2:
        if pitch_box is not None:
            px1, py1, px2, py2 = pitch_box
            for x, y in path_xy:
                nu = float(np.clip((x - px1) / max(1.0, px2 - px1), -0.12, 1.12))
                nv = float(np.clip((y - py1) / max(1.0, py2 - py1), -0.12, 1.12))
                mx = int(pitch_x0 + nu * pw)
                my = int(pitch_y0 + nv * ph)
                mapped.append((mx, my))
        else:
            # Fallback when pitch is missed: map by local motion direction only
            fx, fy = path_xy[0]
            lx, ly = path_xy[-1]
            dx, dy = lx - fx, ly - fy
            n = max(1e-3, float((dx * dx + dy * dy) ** 0.5))
            ux, uy = dx / n, dy / n
            mapped = [
                (cx, cy),
                (int(cx + ux * (radius - 8)), int(cy + uy * (radius - 8))),
            ]
    else:
        return frame

    valid = [p for p in mapped if p is not None]
    if len(valid) >= 2:
        # Single stable direction line: pitch center -> boundary toward ball direction
        start = valid[0]
        end = valid[-1]
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        n = (dx * dx + dy * dy) ** 0.5
        if n > 1e-3:
            ux, uy = dx / n, dy / n
            # Start at pitch center; extend to boundary of ground circle
            p0 = (cx, cy)
            p1 = (int(cx + ux * (radius - 6)), int(cy + uy * (radius - 6)))
            cv2.line(frame, p0, p1, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, p1, 5, (0, 210, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, p1, 7, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_detections(
    frame: np.ndarray,
    boxes_xyxy: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    class_names: dict,
    colors: Optional[dict] = None,
    confidences: Optional[Sequence[float]] = None,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Draw generic detections (e.g. DRS / Scoreboard) without track IDs.

    boxes_xyxy: list of (x1, y1, x2, y2)
    """
    out = frame.copy()
    palette = colors if colors is not None else PLAYER_COLORS

    for i, box in enumerate(boxes_xyxy):
        class_id = int(class_ids[i])
        color: Color = palette.get(class_id, (200, 200, 200))
        x1, y1, x2, y2 = map(int, box)
        name = class_names.get(class_id, str(class_id))

        label = name
        if confidences is not None:
            label = f"{name} {float(confidences[i]):.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        _draw_label(out, label, x1, y1, color, font_scale)

    return out
