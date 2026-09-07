"""
CricketVisionAI — full inference entry point (no training).

Loads existing checkpoints only:
  - weights/player_detection.pt + ByteTrack (Batsman, Bowler)
  - weights/ball_detection.pt (classes 0 & ball → Ball)
  - weights/shot_actions.pt (Cricket Actions pose shots)
  - weights/drs_detection.pt (Bat, Pad, Pitch)

Writes annotated video, tracking.csv, analytics.csv, shots.csv,
pitch_map.png, wagon_wheel.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

# Allow `from src...` when running: python main.py
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import (  # noqa: E402
    AnalyticsLogger,
    BallSpeedEstimator,
    BallTrajectory,
    PitchMapBuilder,
    WagonWheelBuilder,
)
from src.config.settings import (  # noqa: E402
    ANALYSIS_EXPAND_FACTOR,
    DEFAULT_INPUT_VIDEO,
    DEFAULT_OUTPUT_VIDEO,
    OUTPUT_CODEC,
    OUTPUT_DIR,
    SHOT_CLASSIFY_EVERY_N_FRAMES,
)
from src.drs import DrsDetector  # noqa: E402
from src.shot_classifier import ShotClassifier, ShotLogger  # noqa: E402
from src.tracking import PlayerTracker, TrackingLogger  # noqa: E402
from src.utils.draw import (  # noqa: E402
    draw_ball_speed,
    draw_drs_detections,
    draw_shot_geometry,
    draw_shot_prediction,
    draw_tracked_objects,
)

Point2D = Tuple[float, float]
BATSMAN_CLASS_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CricketVisionAI: tracking + pitch map + wagon wheel + shots"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_VIDEO,
        help=f"Input video path (default: {DEFAULT_INPUT_VIDEO})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_VIDEO,
        help=f"Annotated output video (default: {DEFAULT_OUTPUT_VIDEO})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=OUTPUT_DIR / "tracking.csv",
        help="Tracking CSV path (default: output/tracking.csv)",
    )
    parser.add_argument(
        "--analytics-csv",
        type=Path,
        default=OUTPUT_DIR / "analytics.csv",
        help="Analytics CSV path (default: output/analytics.csv)",
    )
    parser.add_argument(
        "--shots-csv",
        type=Path,
        default=OUTPUT_DIR / "shots.csv",
        help="Shot classification CSV path (default: output/shots.csv)",
    )
    parser.add_argument(
        "--pitch-map",
        type=Path,
        default=OUTPUT_DIR / "pitch_map.png",
        help="Pitch map image path (default: output/pitch_map.png)",
    )
    parser.add_argument(
        "--wagon-wheel",
        type=Path,
        default=OUTPUT_DIR / "wagon_wheel.png",
        help="Wagon wheel image path (default: output/wagon_wheel.png)",
    )
    parser.add_argument(
        "--drs-every",
        type=int,
        default=3,
        help="Run DRS detection every N frames (default: 3)",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=None,
        help="Optional scale for real distance / km/h",
    )
    parser.add_argument(
        "--shot-every",
        type=int,
        default=SHOT_CLASSIFY_EVERY_N_FRAMES,
        help="Run shot classifier every N frames (default from settings)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cpu, 0 (default: Ultralytics auto)",
    )
    return parser.parse_args()


def _best_batsman_center(tracked) -> Optional[Point2D]:
    batsmen = [o for o in tracked if o.class_id == BATSMAN_CLASS_ID]
    if not batsmen:
        return None
    best = max(batsmen, key=lambda o: o.confidence)
    return best.center


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output
    csv_path: Path = args.csv
    analytics_csv_path: Path = args.analytics_csv
    shots_csv_path: Path = args.shots_csv
    pitch_map_path: Path = args.pitch_map
    wagon_wheel_path: Path = args.wagon_wheel
    drs_every = max(1, int(args.drs_every))

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_path}\n"
            f"Place a cricket video in '{PROJECT_ROOT / 'input'}' "
            f"or pass --input <path>"
        )

    from src.utils.video_slowmo import create_slow_motion_video

    slow_path = OUTPUT_DIR / "_slow_cli.mp4"
    print(f"Building {ANALYSIS_EXPAND_FACTOR}× slow-mo for denser detection…")
    analysis_path, analysis_fps, playback_fps, width, height, _ = (
        create_slow_motion_video(
            input_path,
            slow_path,
            expand_factor=ANALYSIS_EXPAND_FACTOR,
            codec=OUTPUT_CODEC,
        )
    )

    cap = cv2.VideoCapture(str(analysis_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {analysis_path}")

    fps = float(analysis_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    writer = cv2.VideoWriter(
        str(output_path), fourcc, float(playback_fps), (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {output_path}")

    tracker = PlayerTracker(device=args.device)
    drs = DrsDetector(device=args.device)
    track_logger = TrackingLogger(csv_path=csv_path, fps=fps)
    trajectory = BallTrajectory(fps=fps)
    speed_estimator = BallSpeedEstimator(
        fps=fps,
        meters_per_pixel=args.meters_per_pixel,
        frame_height=float(height),
    )
    analytics_logger = AnalyticsLogger(csv_path=analytics_csv_path)
    shot_classifier = ShotClassifier(
        fps=fps,
        classify_every_n=args.shot_every,
    )
    shot_logger = ShotLogger(csv_path=shots_csv_path)
    pitch_map = PitchMapBuilder(output_dir=pitch_map_path.parent)
    wagon = WagonWheelBuilder(
        meters_per_pixel=args.meters_per_pixel,
        output_dir=wagon_wheel_path.parent,
    )

    ball_path: List[Point2D] = []
    ball_frames: List[int] = []
    last_drs = []
    length_name: Optional[str] = None
    length_dist: Optional[float] = None
    last_batsman_xy: Optional[Point2D] = None
    last_shot_name: Optional[str] = None
    last_shot_conf: float = 0.0
    wagon_angle: Optional[float] = None
    wagon_dist_px: Optional[float] = None
    wagon_dist_m: Optional[float] = None
    display_speed_kmh: Optional[float] = None
    speed_hold_until = 0

    frame_number = 0
    pitch_box = None
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Pitch map    : {pitch_map_path}")
    print(f"Wagon wheel  : {wagon_wheel_path}")
    print(f"Weights: player_detection.pt | ball_detection.pt | shot_actions.pt | drs_detection.pt")
    print(f"FPS   : {fps:.2f}  |  Size: {width}x{height}  |  Frames: {total_frames}")
    print("Running pipeline...")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tracked = tracker.update(frame, pitch_box=pitch_box)
            track_logger.log_frame(frame_number, tracked)

            batsman_xy = _best_batsman_center(tracked)
            if batsman_xy is not None:
                last_batsman_xy = batsman_xy

            pitch_box = None
            if frame_number % drs_every == 0:
                last_drs = drs.detect(frame)
                pitch = drs.best_pitch(last_drs)
            else:
                pitch = drs.best_pitch(last_drs) if last_drs else None
            if pitch is not None:
                pitch_box = (pitch.x1, pitch.y1, pitch.x2, pitch.y2)
                pitch_map.set_pitch_bbox(pitch.x1, pitch.y1, pitch.x2, pitch.y2)
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

            point = trajectory.update(frame_number, tracked)
            speed = speed_estimator.update(point)
            analytics_logger.log(point, speed)

            if point is not None and point.detected:
                ball_path.append((point.smooth_x, point.smooth_y))
                ball_frames.append(frame_number)
                if len(ball_path) > 90:
                    ball_path = ball_path[-90:]
                    ball_frames = ball_frames[-90:]

            if len(ball_path) >= 3 and pitch_map.homography is not None:
                bounce = pitch_map.estimate_bounce(ball_path, ball_frames)
                if bounce is not None:
                    length_name = bounce.length.value
                    length_dist = bounce.distance_from_batsman_m

            bat_boxes = [
                (d.x1, d.y1, d.x2, d.y2)
                for d in last_drs
                if d.class_id in (0, 1)  # Bat, Pad
            ]
            shot = shot_classifier.update(
                frame,
                frame_number,
                tracked,
                extra_boxes=bat_boxes,
                pitch_box=pitch_box,
            )
            shot_logger.log(shot)
            if shot is not None:
                last_shot_name = shot.shot_name
                last_shot_conf = shot.confidence

            # Live wagon-wheel geometry when we have ball + batsman + shot
            if (
                last_batsman_xy is not None
                and last_shot_name is not None
                and len(ball_path) >= 4
            ):
                geom = wagon.estimate_geometry(
                    ball_xy=ball_path,
                    frame_numbers=ball_frames,
                    batsman_xy=last_batsman_xy,
                    shot_name=last_shot_name,
                    confidence=last_shot_conf,
                )
                if geom is not None:
                    wagon_angle = geom.angle_deg
                    wagon_dist_px = geom.distance_px
                    wagon_dist_m = geom.distance_m

            hud_speed_kmh = None
            if display_speed_kmh is not None and frame_number <= speed_hold_until:
                hud_speed_kmh = min(165.0, display_speed_kmh)
            elif speed is not None:
                hud_speed_kmh = speed.speed_kmh

            annotated = draw_tracked_objects(frame, tracked)
            annotated = draw_drs_detections(annotated, last_drs)
            annotated = draw_ball_speed(
                annotated,
                speed_px_per_sec=speed.speed_px_per_sec if speed else None,
                speed_kmh=hud_speed_kmh,
            )
            annotated = draw_shot_prediction(
                annotated,
                shot_name=shot.shot_name if shot else last_shot_name,
                confidence=(
                    shot.confidence
                    if shot
                    else (last_shot_conf if last_shot_name else None)
                ),
            )
            annotated = draw_shot_geometry(
                annotated,
                angle_deg=wagon_angle,
                distance_px=wagon_dist_px,
                distance_m=wagon_dist_m,
            )
            writer.write(annotated)

            frame_number += 1
            if frame_number % 50 == 0:
                print(f"  Processed {frame_number} frames...")

    finally:
        cap.release()
        writer.release()
        try:
            if slow_path.exists() and slow_path.resolve() != input_path.resolve():
                slow_path.unlink(missing_ok=True)
        except Exception:
            pass

    from src.utils.video_encode import to_browser_mp4

    print("Encoding annotated video for web playback (H.264)…")
    try:
        to_browser_mp4(output_path)
    except Exception as exc:
        print(f"  H.264 encode skipped: {exc}")

    # Final pitch map
    saved_pitch = None
    if pitch_map.homography is not None and len(ball_path) >= 3:
        result = pitch_map.build(ball_path, ball_frames)
        saved_pitch = pitch_map.save(result.map_image, filename=pitch_map_path.name)
        if result.bounce is not None:
            print(
                f"Bounce length: {result.bounce.length.value} "
                f"({result.bounce.distance_from_batsman_m:.1f} m from batsman)"
            )
    else:
        print("Pitch map skipped (no Pitch from DRS and/or insufficient ball track).")

    # Final wagon wheel (ball trajectory + shot class + batsman origin)
    saved_wagon = None
    if last_batsman_xy is not None and last_shot_name is not None and len(ball_path) >= 4:
        wagon.reset()
        wheel_result = wagon.build(
            ball_xy=ball_path,
            frame_numbers=ball_frames,
            batsman_xy=last_batsman_xy,
            shot_name=last_shot_name,
            confidence=last_shot_conf,
            include_history=False,
        )
        saved_wagon = wagon.save(wheel_result.wheel_image, filename=wagon_wheel_path.name)
        if wheel_result.geometry is not None:
            g = wheel_result.geometry
            print(
                f"Wagon wheel: {g.shot_name} | "
                f"angle {g.angle_deg:+.1f} deg | "
                f"dist {g.distance_px:.0f} px"
            )
    else:
        print("Wagon wheel skipped (need batsman + shot + ball trajectory).")

    saved_tracking = track_logger.save()
    saved_analytics = analytics_logger.save()
    saved_shots = shot_logger.save()
    print(f"Done. Frames processed: {frame_number}")
    print(f"Tracking rows : {track_logger.row_count} → {saved_tracking}")
    print(f"Analytics rows: {analytics_logger.row_count} → {saved_analytics}")
    print(f"Shot rows     : {shot_logger.row_count} → {saved_shots}")
    if saved_pitch:
        print(f"Pitch map     : {saved_pitch}")
    if saved_wagon:
        print(f"Wagon wheel   : {saved_wagon}")
    print(f"Video saved   : {output_path}")


if __name__ == "__main__":
    main()
