from __future__ import annotations

from typing import Any, Final, TypedDict

from ttflux.pipeline.states import RunStatus


RUN_SCHEMA_VERSION: Final[int] = 1
VIDEO_SNAPSHOT_SCHEMA_VERSION: Final[int] = 1
CLIP_SCHEMA_VERSION: Final[int] = 1
ANALYSIS_SCHEMA_VERSION: Final[int] = 4

PIPELINE_NAME: Final[str] = "motion_tracks_probe"
PIPELINE_VERSION: Final[int] = 1


VIDEO_SNAPSHOT_KEYS: Final[tuple[str, ...]] = (
    "id",
    "filename",
    "relative_path",
    "extension",
    "size_mb",
    "width",
    "height",
    "fps",
    "duration_s",
    "frame_count",
    "codec",
    "probe_error",
)


TRACKING_DESCRIPTOR_REQUIRED_KEYS: Final[
    tuple[str, ...]
] = (
    "schema_version",
    "engine",
    "scorer_id",
    "configuration",
)


class PipelineDescriptor(TypedDict):
    name: str
    version: int


class RunConfiguration(TypedDict):
    clip_start_s: float
    clip_duration_s: float
    candidate_detection_enabled: bool
    track_probe_enabled: bool
    ball_tracking_enabled: bool
    table_context_enabled: bool
    pose_enabled: bool


class RunArtifacts(TypedDict, total=False):
    run: str
    video: str
    source_clip: str
    clip: str
    candidates: str
    candidate_metrics: str
    candidate_overlay: str
    tracks: str
    track_metrics: str
    track_overlay: str
    analysis: str


class RunFailure(TypedDict):
    type: str
    message: str


class RunPayload(TypedDict, total=False):
    schema_version: int
    run_id: str
    video_id: str
    video_filename: str
    video_relative_path: str
    created_at: str
    started_at: str
    completed_at: str
    failed_at: str
    status: RunStatus
    pipeline: PipelineDescriptor
    configuration: RunConfiguration
    artifacts: RunArtifacts
    error: RunFailure
    tracking: dict[str, Any]
    metrics: dict[str, Any]
    track_metrics: dict[str, Any]


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "CLIP_SCHEMA_VERSION",
    "PIPELINE_NAME",
    "PIPELINE_VERSION",
    "RUN_SCHEMA_VERSION",
    "TRACKING_DESCRIPTOR_REQUIRED_KEYS",
    "VIDEO_SNAPSHOT_KEYS",
    "VIDEO_SNAPSHOT_SCHEMA_VERSION",
    "PipelineDescriptor",
    "RunArtifacts",
    "RunConfiguration",
    "RunFailure",
    "RunPayload",
]
