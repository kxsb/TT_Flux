from __future__ import annotations

from datetime import datetime
from typing import Any

from ttflux.pipeline.contracts import (
    ANALYSIS_SCHEMA_VERSION,
    TRACKING_DESCRIPTOR_REQUIRED_KEYS,
    RunPayload,
)


def tracking_descriptor(
    tracking_result: Any,
) -> dict[str, Any]:
    """Extract portable provenance from a tracking result."""

    payload = tracking_result.to_dict()

    if not isinstance(payload, dict):
        raise TypeError(
            "Tracking result serialization must be an object."
        )

    missing_keys = [
        key
        for key in TRACKING_DESCRIPTOR_REQUIRED_KEYS
        if key not in payload
    ]

    if missing_keys:
        joined = ", ".join(missing_keys)
        raise ValueError(
            "Tracking result is missing provenance fields: "
            f"{joined}"
        )

    if not isinstance(payload["engine"], dict):
        raise TypeError(
            "Tracking engine provenance must be an object."
        )

    if not isinstance(
        payload["configuration"],
        dict,
    ):
        raise TypeError(
            "Tracking configuration must be an object."
        )

    return {
        "schema_version": payload["schema_version"],
        "engine": dict(payload["engine"]),
        "scorer_id": payload["scorer_id"],
        "configuration": dict(
            payload["configuration"]
        ),
    }


def build_analysis(
    run_payload: RunPayload,
    video_payload: dict[str, Any],
    clip_payload: dict[str, Any],
    candidate_metrics: dict[str, Any],
    track_metrics: dict[str, Any],
    tracking: dict[str, Any],
    generated_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "run_id": run_payload["run_id"],
        "video_id": run_payload["video_id"],
        "pipeline": run_payload["pipeline"],
        "tracking": tracking,
        "generated_at": generated_at.isoformat(
            timespec="seconds"
        ),
        "source": {
            "filename": run_payload[
                "video_filename"
            ],
            "frame_count": video_payload.get(
                "frame_count"
            ),
            "duration_s": video_payload.get(
                "duration_s"
            ),
            "fps": video_payload.get("fps"),
            "resolution": {
                "width": video_payload.get(
                    "width"
                ),
                "height": video_payload.get(
                    "height"
                ),
            },
        },
        "clip": {
            "artifact": clip_payload["artifact"],
            "requested_start_s": clip_payload[
                "requested_start_s"
            ],
            "requested_duration_s": clip_payload[
                "requested_duration_s"
            ],
            "actual_duration_s": clip_payload.get(
                "duration_s"
            ),
            "frame_count": clip_payload.get(
                "frame_count"
            ),
            "fps": clip_payload.get("fps"),
            "resolution": {
                "width": clip_payload.get(
                    "width"
                ),
                "height": clip_payload.get(
                    "height"
                ),
            },
        },
        "candidates": candidate_metrics["summary"],
        "tracks_probe": track_metrics["summary"],
    }


__all__ = [
    "build_analysis",
    "tracking_descriptor",
]
