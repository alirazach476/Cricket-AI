"""
Process service — run the CricketVisionAI inference pipeline on a video.

Reuses existing src/ modules and weights (no training).
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import (
    ANALYTICS_CSV,
    ANNOTATED_VIDEO,
    HIGHLIGHTS_DIR,
    OUTPUT_DIR,
    PITCH_MAP_PNG,
    PROJECT_ROOT,
    SCOREBOARD_JSON,
    TRACKING_CSV,
    TRAJECTORY_JSON,
)
from app.services.job_store import job_store

# Ensure project root is importable for `src.*`
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

Point2D = Tuple[float, float]
BATSMAN_CLASS_ID = 1


def _best_batsman_center(tracked) -> Optional[Point2D]:
    batsmen = [o for o in tracked if o.class_id == BATSMAN_CLASS_ID]
    if not batsmen:
        return None
    return max(batsmen, key=lambda o: o.confidence).center


def resolve_video_path(
    job_id: Optional[str] = None,
    video_path: Optional[str] = None,
) -> Tuple[str, Path]:
    """Return (job_id, path). Creates a synthetic job_id if needed."""
    if video_path:
        path = Path(video_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        jid = job_id or "manual"
        if job_store.get(jid) is None:
            job_store.create(jid, path.name, str(path))
        return jid, path

    if job_id:
        rec = job_store.get(job_id)
        if rec is None:
            raise FileNotFoundError(f"Unknown job_id: {job_id}")
        path = Path(rec.video_path)
        if not path.exists():
            raise FileNotFoundError(f"Uploaded video missing: {path}")
        return job_id, path

    latest = job_store.latest()
    if latest is not None and Path(latest.video_path).exists():
        return latest.job_id, Path(latest.video_path)

    default = PROJECT_ROOT / "input" / "input.mp4"
    if default.exists():
        jid = "default"
        if job_store.get(jid) is None:
            job_store.create(jid, default.name, str(default))
        return jid, default

    raise FileNotFoundError(
        "No video available. Upload via /upload or place input/input.mp4"
    )


def run_pipeline(
    job_id: Optional[str] = None,
    video_path: Optional[str] = None,
    device: Optional[str] = None,
    shot_every: int = 5,
    drs_every: int = 3,
    meters_per_pixel: Optional[float] = None,
    generate_highlights: bool = True,
) -> Dict[str, Any]:
    """
    Full inference pass. Returns stats + artifact paths as a plain dict.

    Mid-frames are generated on the fly (no separate slow-mo rewrite),
    so progress updates immediately and the UI does not freeze.
    """
    jid, input_path = resolve_video_path(job_id, video_path)

    def _progress(pct: float, message: str, stage: str = "processing") -> None:
        job_store.update(
            jid,
            status="processing",
            message=message,
            progress=float(max(0.0, min(100.0, pct))),
            stage=stage,
        )

    _progress(3, "Starting pipeline…", "prepare")

    from src.config.settings import (
        ANALYSIS_EXPAND_FACTOR,
        ENABLE_SCOREBOARD_OCR,
        OUTPUT_CODEC,
        PLAYER_IMGSZ,
    )

    # Reduce first-run stalls: smaller imgsz is much faster on CPU
    player_imgsz = min(int(PLAYER_IMGSZ), 960)

    _progress(6, "Importing tracking modules…", "prepare")
    from src.analytics import (
        AnalyticsLogger,
        BallSpeedEstimator,
        BallTrajectory,
        PitchMapBuilder,
    )
    from src.drs import DrsDetector
    from src.highlights import HighlightClipExporter, HighlightDetector
    from src.tracking import PlayerTracker, TrackingLogger
    from src.utils.draw import (
        draw_cricket_analysis_panel,
        draw_drs_detections,
        draw_field_view_panel,
        draw_scoreboard_reading,
        draw_tracked_objects,
        _xy_to_pitch_uv,
    )

    # EasyOCR often hangs 5–15+ min on CPU — disabled unless explicitly enabled
    scoreboard_reader = None
    if ENABLE_SCOREBOARD_OCR:
        _progress(8, "Loading scoreboard OCR (slow on CPU)…", "prepare")
        try:
            from src.scoreboard import ScoreboardReader

            scoreboard_reader = ScoreboardReader(device=device, every_n_frames=45)
        except Exception:
            scoreboard_reader = None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANNOTATED_VIDEO

    expand = max(1, int(ANALYSIS_EXPAND_FACTOR))
    _progress(12, "Opening video…", "analyze")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        job_store.update(jid, status="failed", message="Could not open video", progress=0)
        raise RuntimeError(f"Could not open video: {input_path}")

    playback_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    analysis_fps = playback_fps * expand
    fps = float(analysis_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    total_frames = max(1, src_frames * expand)

    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    writer = cv2.VideoWriter(
        str(output_path), fourcc, playback_fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        job_store.update(
            jid, status="failed", message="Could not open video writer", progress=0
        )
        raise RuntimeError(f"Could not open video writer: {output_path}")

    _progress(18, "Loading player YOLO…", "prepare")
    tracker = PlayerTracker(device=device, imgsz=player_imgsz)

    _progress(28, "Loading DRS YOLO…", "prepare")
    drs = DrsDetector(device=device, imgsz=min(player_imgsz, 960))

    _progress(35, "Loading analytics…", "prepare")
    track_logger = TrackingLogger(csv_path=TRACKING_CSV, fps=fps)
    trajectory = BallTrajectory(fps=fps)
    speed_estimator = BallSpeedEstimator(
        fps=fps,
        meters_per_pixel=meters_per_pixel,
        frame_height=float(height),
    )
    analytics_logger = AnalyticsLogger(csv_path=ANALYTICS_CSV)
    pitch_map = PitchMapBuilder(output_dir=OUTPUT_DIR)
    highlight_detector = HighlightDetector(fps=fps)

    _progress(40, "Analyzing frames…", "analyze")

    ball_path: List[Point2D] = []
    ball_frames: List[int] = []
    last_drs = []
    last_batsman_xy: Optional[Point2D] = None
    sb_reading = None
    sb_box = None
    display_speed_kmh: Optional[float] = None
    speed_hold_until = 0
    max_speed_kmh: float = 0.0
    speed_kmh_samples: List[float] = []
    total_collisions = 0
    player_trails: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
        lambda: deque(maxlen=30)
    )
    current_ball_uv: Optional[Tuple[float, float]] = None

    def _ball_on_pitch(
        cx: float, cy: float, box: Tuple[float, float, float, float]
    ) -> bool:
        px1, py1, px2, py2 = box
        pad = 0.2 * max(px2 - px1, py2 - py1)
        return px1 - pad <= cx <= px2 + pad and py1 - pad <= cy <= py2 + pad

    frame_number = 0
    drs_every = max(1, int(drs_every))
    prev_frame: Optional[np.ndarray] = None
    prev_gray_small: Optional[np.ndarray] = None

    def _scene_cut(frame: np.ndarray) -> bool:
        nonlocal prev_gray_small
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90))
        cut = False
        if prev_gray_small is not None:
            diff = float(np.mean(cv2.absdiff(small, prev_gray_small)))
            cut = diff > 28.0
        prev_gray_small = small
        return cut

    def _reset_motion_state() -> None:
        nonlocal ball_path, ball_frames
        nonlocal display_speed_kmh, speed_hold_until
        nonlocal max_speed_kmh, speed_kmh_samples, total_collisions
        nonlocal current_ball_uv
        ball_path = []
        ball_frames = []
        display_speed_kmh = None
        speed_hold_until = 0
        max_speed_kmh = 0.0
        speed_kmh_samples = []
        total_collisions = 0
        current_ball_uv = None
        player_trails.clear()
        trajectory.reset()
        speed_estimator.reset()
        tracker.ball_hybrid.reset()

    def _analyze_frame(frame: np.ndarray) -> None:
        nonlocal frame_number, last_drs
        nonlocal last_batsman_xy
        nonlocal sb_reading, sb_box
        nonlocal ball_path, ball_frames
        nonlocal display_speed_kmh, speed_hold_until
        nonlocal max_speed_kmh, speed_kmh_samples, total_collisions
        nonlocal current_ball_uv

        if _scene_cut(frame):
            _reset_motion_state()

        pitch_box = None
        pitch = None
        if frame_number % drs_every == 0:
            last_drs = drs.detect(frame)
            pitch = drs.best_pitch(last_drs)
        else:
            pitch = drs.best_pitch(last_drs) if last_drs else None
        if pitch is not None:
            pitch_box = (pitch.x1, pitch.y1, pitch.x2, pitch.y2)
            pitch_map.set_pitch_bbox(pitch.x1, pitch.y1, pitch.x2, pitch.y2)
            # Auto-calibrate real-world scale → km/h
            speed_estimator.calibrate_from_pitch_box(
                pitch.x1, pitch.y1, pitch.x2, pitch.y2
            )
            if pitch_map.homography is not None:
                mid = (
                    (pitch.x1 + pitch.x2) * 0.5,
                    (pitch.y1 + pitch.y2) * 0.5,
                )
                speed_estimator.calibrate_from_homography(
                    pitch_map.image_to_map,
                    map_height_px=float(pitch_map.map_h),
                    sample_xy=mid,
                )

        tracked = tracker.update(frame, pitch_box=pitch_box)
        track_logger.log_frame(frame_number, tracked)

        batsman_xy = _best_batsman_center(tracked)
        if batsman_xy is not None:
            last_batsman_xy = batsman_xy

        point = trajectory.update(frame_number, tracked)
        # If trajectory wiped after a teleport, drop stale ball path too
        if point is not None and point.detected and ball_path:
            lx, ly = ball_path[-1]
            if ((point.smooth_x - lx) ** 2 + (point.smooth_y - ly) ** 2) ** 0.5 > 90:
                ball_path = []
                ball_frames = []
                speed_estimator.reset()
                display_speed_kmh = None
                speed_hold_until = 0

        speed = speed_estimator.update(point)
        analytics_logger.log(point, speed)

        if (
            speed is not None
            and pitch_box is not None
            and point is not None
            and point.detected
            and speed.instant_px_per_sec > 0
            and speed_estimator.meters_per_pixel
        ):
            instant_kmh = (
                speed.instant_px_per_sec
                * speed_estimator.meters_per_pixel
                * 3.6
            )
            if _ball_on_pitch(point.smooth_x, point.smooth_y, pitch_box) and (
                instant_kmh >= 50.0
            ):
                if display_speed_kmh is None:
                    display_speed_kmh = float(instant_kmh)
                speed_hold_until = frame_number + int(fps * 6)

        if speed is not None and speed.speed_kmh is not None:
            highlight_detector.add_speed(
                frame_number, speed.speed_px_per_sec, speed.timestamp_sec
            )

        if point is not None and point.detected:
            ball_path.append((point.smooth_x, point.smooth_y))
            ball_frames.append(frame_number)
            if len(ball_path) > 90:
                ball_path = ball_path[-90:]
                ball_frames = ball_frames[-90:]
            highlight_detector.add_ball_point(
                frame_number, (point.smooth_x, point.smooth_y)
            )

        if scoreboard_reader is not None:
            try:
                sb_reading, sb_box = scoreboard_reader.update(frame, frame_number)
            except Exception:
                pass

        hud_speed_kmh = None
        if display_speed_kmh is not None and frame_number <= speed_hold_until:
            hud_speed_kmh = min(165.0, display_speed_kmh)
        elif speed is not None:
            hud_speed_kmh = speed.speed_kmh

        if hud_speed_kmh is not None and hud_speed_kmh > 0:
            max_speed_kmh = max(max_speed_kmh, float(hud_speed_kmh))
            speed_kmh_samples.append(float(hud_speed_kmh))
        avg_speed_kmh = (
            float(sum(speed_kmh_samples) / len(speed_kmh_samples))
            if speed_kmh_samples
            else None
        )

        frame_players = [o for o in tracked if o.class_id in (1, 2, 3, 4)]
        frame_collisions = 0
        for i in range(len(frame_players)):
            for j in range(i + 1, len(frame_players)):
                a, b = frame_players[i], frame_players[j]
                x1 = max(a.x1, b.x1)
                y1 = max(a.y1, b.y1)
                x2 = min(a.x2, b.x2)
                y2 = min(a.y2, b.y2)
                if x2 > x1 and y2 > y1:
                    inter = (x2 - x1) * (y2 - y1)
                    area_a = max(1.0, (a.x2 - a.x1) * (a.y2 - a.y1))
                    area_b = max(1.0, (b.x2 - b.x1) * (b.y2 - b.y1))
                    if inter / (area_a + area_b - inter) > 0.08:
                        frame_collisions += 1
        total_collisions += frame_collisions

        current_ball_uv = None
        for obj in tracked:
            if obj.class_id == 0:
                uv = _xy_to_pitch_uv(
                    obj.center[0],
                    obj.center[1],
                    pitch_box,
                    frame_wh=(width, height),
                )
                if uv is not None:
                    current_ball_uv = uv
            elif obj.class_id in (1, 2, 3, 4):
                uv = _xy_to_pitch_uv(
                    obj.center[0],
                    obj.center[1],
                    pitch_box,
                    frame_wh=(width, height),
                )
                if uv is not None:
                    player_trails[obj.object_id].append(uv)

        annotated = draw_tracked_objects(frame, tracked, broadcast_style=True)
        annotated = draw_drs_detections(annotated, last_drs)
        annotated = draw_field_view_panel(
            annotated,
            tracked_objects=tracked,
            pitch_box=pitch_box,
            ball_uv=current_ball_uv,
            player_trails=player_trails,
        )
        annotated = draw_cricket_analysis_panel(
            annotated,
            frame_number=frame_number,
            ball_speed_kmh=hud_speed_kmh,
            max_speed_kmh=max_speed_kmh if max_speed_kmh > 0 else None,
            avg_speed_kmh=avg_speed_kmh,
            detections=len(tracked),
            collisions=total_collisions,
            status="Tracking",
        )
        annotated = draw_scoreboard_reading(annotated, sb_reading, sb_box)
        writer.write(annotated)
        frame_number += 1

        if frame_number % 2 == 0 or frame_number == 1:
            frac = min(1.0, frame_number / float(total_frames))
            _progress(
                40.0 + frac * 52.0,
                f"Analyzing frame {frame_number}/{total_frames}",
                "analyze",
            )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if prev_frame is not None and expand > 1:
                for k in range(1, expand):
                    alpha = k / float(expand)
                    mid = cv2.addWeighted(prev_frame, 1.0 - alpha, frame, alpha, 0)
                    _analyze_frame(mid)

            _analyze_frame(frame)
            prev_frame = frame
    finally:
        cap.release()
        writer.release()

    _progress(94, "Encoding browser-ready video…", "finalize")
    from src.utils.video_encode import to_browser_mp4

    try:
        to_browser_mp4(output_path)
    except Exception:
        pass

    _progress(95, "Building reports & highlights…", "finalize")

    if pitch_map.homography is not None and len(ball_path) >= 3:
        result = pitch_map.build(ball_path, ball_frames)
        pitch_map.save(result.map_image, filename=PITCH_MAP_PNG.name)

    track_logger.save()
    analytics_logger.save()

    if scoreboard_reader is not None:
        try:
            scoreboard_reader.save_json(path=SCOREBOARD_JSON, frame_number=frame_number)
        except Exception:
            pass

    import json

    TRAJECTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with TRAJECTORY_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "count": 0,
                "fps": playback_fps,
                "analysis_fps": fps,
                "points": [],
                "delivery_path_uv": [],
            },
            f,
        )

    highlight_files: List[str] = []
    if generate_highlights:
        events = highlight_detector.detect(total_frames=frame_number)
        exporter = HighlightClipExporter(output_dir=HIGHLIGHTS_DIR)
        paths = exporter.export(input_path, events, overlay_label=True)
        highlight_files = [p.name for p in paths]

    artifacts = {
        "annotated_video": str(output_path),
        "tracking_csv": str(TRACKING_CSV),
        "analytics_csv": str(ANALYTICS_CSV),
        "pitch_map": str(PITCH_MAP_PNG),
        "scoreboard_json": str(SCOREBOARD_JSON),
        "trajectory_json": str(TRAJECTORY_JSON),
        "highlights_dir": str(HIGHLIGHTS_DIR),
        "highlight_clips": highlight_files,
    }
    stats = {
        "frames_processed": frame_number,
        "fps": fps,
        "playback_fps": playback_fps,
        "expand_factor": expand,
        "width": width,
        "height": height,
        "tracking_rows": track_logger.row_count,
        "analytics_rows": analytics_logger.row_count,
        "highlight_count": len(highlight_files),
    }

    job_store.update(
        jid,
        status="completed",
        message="Processing completed",
        progress=100.0,
        stage="done",
        artifacts=artifacts,
        stats=stats,
    )

    return {
        "job_id": jid,
        "input_video": str(input_path),
        "output_video": str(output_path),
        "artifacts": artifacts,
        "stats": stats,
    }
