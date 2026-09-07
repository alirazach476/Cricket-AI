"""Pydantic schemas — all API JSON responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    status: str = "success"
    message: str
    job_id: str
    filename: str
    path: str
    size_bytes: int


class ProcessRequest(BaseModel):
    job_id: Optional[str] = Field(
        default=None,
        description="Upload job_id. If omitted, uses latest upload or default input.",
    )
    video_path: Optional[str] = Field(
        default=None,
        description="Optional absolute/relative path override.",
    )
    device: Optional[str] = None
    shot_every: int = 5
    drs_every: int = 3
    meters_per_pixel: Optional[float] = None
    generate_highlights: bool = True


class ProcessResponse(BaseModel):
    status: str
    message: str
    job_id: str
    input_video: str
    output_video: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    stats: Dict[str, Any] = Field(default_factory=dict)


class BallSpeedPoint(BaseModel):
    frame_number: int
    timestamp_sec: float
    speed_px_per_sec: float
    instant_px_per_sec: Optional[float] = None
    speed_kmh: Optional[float] = None
    object_id: Optional[int] = None


class BallSpeedResponse(BaseModel):
    status: str = "success"
    count: int
    max_speed_px_per_sec: Optional[float] = None
    avg_speed_px_per_sec: Optional[float] = None
    points: List[BallSpeedPoint]


class ShotItem(BaseModel):
    frame_number: int
    timestamp_sec: float
    object_id: int
    shot_name: str
    class_index: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class ShotAnalysisResponse(BaseModel):
    status: str = "success"
    count: int
    distribution: Dict[str, int] = Field(default_factory=dict)
    top_shot: Optional[ShotItem] = None
    shots: List[ShotItem]


class ScoreboardDetection(BaseModel):
    frame_number: int
    timestamp_sec: float
    object_id: Optional[int] = None
    confidence: Optional[float] = None
    x1: float
    y1: float
    x2: float
    y2: float


class ScoreboardResponse(BaseModel):
    status: str = "success"
    message: str
    detected: bool
    count: int
    detections: List[ScoreboardDetection] = Field(default_factory=list)
    note: str = (
        "Scoreboard boxes come from player_detection.pt class Scoreboard. "
        "OCR digit reading is not enabled in this version."
    )


class HighlightItem(BaseModel):
    index: Optional[int] = None
    type: str
    label: str
    frame_number: int
    timestamp_sec: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    score: Optional[float] = None
    file: Optional[str] = None
    url: Optional[str] = None


class HighlightsResponse(BaseModel):
    status: str = "success"
    count: int
    highlights_dir: str
    items: List[HighlightItem]


class ReportsResponse(BaseModel):
    status: str = "success"
    job_id: Optional[str] = None
    summary: Dict[str, Any] = Field(default_factory=dict)
    ball_speed: Dict[str, Any] = Field(default_factory=dict)
    shot_analysis: Dict[str, Any] = Field(default_factory=dict)
    scoreboard: Dict[str, Any] = Field(default_factory=dict)
    highlights: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
