# backend/app/schemas.py
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class CameraConnectRequest(BaseModel):
    rtsp_url: Optional[str] = None  # if omitted, use default from config


class CameraStatus(BaseModel):
    connected: bool
    rtsp_url: Optional[str] = None
    running: bool


class SessionStartRequest(BaseModel):
    name: Optional[str] = None  # optional label (e.g., "Rat A – baseline")


class SessionInfo(BaseModel):
    session_id: str
    name: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    video_path: Optional[str]


class BehaviorEvent(BaseModel):
    timestamp_s: float
    label: str
    confidence: float


class BehaviorTimeline(BaseModel):
    session_id: str
    events: List[BehaviorEvent]


class OfflineAnalysisRequest(BaseModel):
    dlc_csv_path: str = Field(..., description="Path to a DeepLabCut CSV export.")
    fps: Optional[float] = Field(
        default=None,
        description="Video frames per second. If omitted, timestamps remain frame-based.",
    )
    subject_id: Optional[str] = Field(
        default=None, description="Optional rat identifier for downstream reporting."
    )
    condition: Optional[str] = Field(
        default=None,
        description="Experimental condition such as baseline, meth, or amphetamine.",
    )


class VideoAnalysisRequest(BaseModel):
    video_path: str
    config_path: Optional[str] = None
    fps: Optional[float] = None
    subject_id: Optional[str] = None
    condition: Optional[str] = None
    classifier_type: Optional[str] = None


class SimBAAnalysisRequest(BaseModel):
    dlc_csv_path: str = Field(..., description="Path to a DeepLabCut CSV export.")
    fps: float = Field(default=30.0, description="Video frames per second.")
    subject_id: Optional[str] = Field(
        default=None, description="Optional rat identifier for downstream reporting."
    )
    condition: Optional[str] = Field(
        default=None,
        description="Experimental condition such as baseline, meth, or amphetamine.",
    )


class ConditionAnomaly(BaseModel):
    behavior: str
    observed: float
    expected_range: List[float]
    direction: str
    deviation: float
    severity: str


class ConditionReport(BaseModel):
    condition: str
    condition_description: str
    flag_behaviors: List[str]
    anomalies: List[ConditionAnomaly]
    anomaly_count: int
    severe_anomaly_count: int


class SimBAFramePrediction(BaseModel):
    frame: int
    timestamp_s: float
    label: str
    confidence: float
    logits: List[float]


class SimBAAnalysisResponse(BaseModel):
    subject_id: Optional[str]
    condition: Optional[str]
    source: str
    classifier_mode: str
    predictions: List[SimBAFramePrediction]
    behavior_share: dict[str, float]
    quadrant_time: dict[str, float]
    total_frames: int
    duration_s: float
    condition_report: Optional[ConditionReport] = None
    notes: List[str]


class BehaviorBout(BaseModel):
    label: str
    start_s: float
    end_s: float
    duration_s: float
    mean_speed_px_s: float
    confidence: float


class BehavioralMetrics(BaseModel):
    duration_s: float
    distance_px: float
    mean_speed_px_s: float
    peak_speed_px_s: float
    mean_acceleration_px_s2: float
    movement_ratio: float
    immobility_ratio: float
    exploratory_ratio: float
    stereotypy_candidate_ratio: float
    pace_confined_score: float
    spatial_span_px: float
    valid_pose_ratio: float


class RatBehaviorEvent(BaseModel):
    timestamp_s: float
    label: str
    confidence: float
    note: str


class AnalystMessage(BaseModel):
    timestamp_s: float
    speaker: str
    message: str


class PoseAnalysisResponse(BaseModel):
    subject_id: Optional[str]
    condition: Optional[str]
    source: str
    bodyparts_used: List[str]
    metrics: BehavioralMetrics
    bouts: List[BehaviorBout]
    notes: List[str]


class UploadedAnalysisResponse(BaseModel):
    analysis_id: str
    pipeline_mode: str
    model_stack: List[str]
    selected_models: dict[str, str]
    source_video_url: str
    annotated_video_url: str
    timeline_url: str
    fps: float
    frame_count: int
    duration_s: float
    dominant_behavior: str
    behavior_share: dict[str, float]
    bouts: List[BehaviorBout]
    behavior_events: List[RatBehaviorEvent]
    analyst_messages: List[AnalystMessage]
    analysis_summary: str
    review_priority: str
    classifier_mode: str
    condition: Optional[str] = None
    condition_report: Optional[ConditionReport] = None
    condition_anomalies: List[ConditionAnomaly] = Field(default_factory=list)
    notes: List[str]
