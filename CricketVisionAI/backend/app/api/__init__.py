"""API routers package."""

from app.api.ball_speed import router as ball_speed_router
from app.api.highlights import router as highlights_router
from app.api.job import router as job_router
from app.api.process import router as process_router
from app.api.reports import router as reports_router
from app.api.scoreboard import router as scoreboard_router
from app.api.shot_analysis import router as shot_analysis_router
from app.api.upload import router as upload_router
from app.api.annotated_video import router as annotated_video_router
from app.api.trajectory import router as trajectory_router
from app.api.analytics import router as analytics_router

__all__ = [
    "upload_router",
    "process_router",
    "job_router",
    "ball_speed_router",
    "shot_analysis_router",
    "scoreboard_router",
    "highlights_router",
    "reports_router",
    "annotated_video_router",
    "trajectory_router",
    "analytics_router",
]
