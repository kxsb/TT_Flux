from __future__ import annotations

from importlib import import_module

import ttflux.pipeline as pipeline
from ttflux.pipeline.contracts import (
    ANALYSIS_SCHEMA_VERSION,
    CLIP_SCHEMA_VERSION,
    PIPELINE_NAME,
    PIPELINE_VERSION,
    RUN_SCHEMA_VERSION,
    VIDEO_SNAPSHOT_SCHEMA_VERSION,
)
from ttflux.pipeline.errors import (
    InvalidClipRangeError,
    InvalidRunStateError,
    UnknownRunError,
    UnknownVideoError,
)
from ttflux.pipeline.states import (
    RUN_COMPLETED,
    RUN_CREATED,
    RUN_FAILED,
    RUN_RUNNING,
    RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    is_run_status,
)


runs = import_module("ttflux.pipeline.runs")


def test_run_errors_have_canonical_identity() -> None:
    assert runs.UnknownVideoError is UnknownVideoError
    assert runs.UnknownRunError is UnknownRunError
    assert runs.InvalidRunStateError is InvalidRunStateError
    assert runs.InvalidClipRangeError is InvalidClipRangeError

    assert pipeline.UnknownVideoError is UnknownVideoError


def test_run_states_are_stable_serialized_values() -> None:
    assert RUN_CREATED == "created"
    assert RUN_RUNNING == "running"
    assert RUN_COMPLETED == "completed"
    assert RUN_FAILED == "failed"

    assert RUN_STATUSES == {
        "created",
        "running",
        "completed",
        "failed",
    }
    assert TERMINAL_RUN_STATUSES == {
        "completed",
        "failed",
    }

    assert is_run_status("created")
    assert is_run_status("failed")
    assert not is_run_status("unknown")
    assert not is_run_status(None)


def test_run_schema_contracts_are_preserved() -> None:
    assert RUN_SCHEMA_VERSION == 1
    assert VIDEO_SNAPSHOT_SCHEMA_VERSION == 1
    assert CLIP_SCHEMA_VERSION == 1
    assert ANALYSIS_SCHEMA_VERSION == 4

    assert PIPELINE_NAME == "motion_tracks_probe"
    assert PIPELINE_VERSION == 1

    snapshot = runs._video_snapshot(
        {
            "id": "video-1",
            "filename": "match.mp4",
            "absolute_path": "C:/private/match.mp4",
        }
    )

    assert snapshot["schema_version"] == 1
    assert snapshot["id"] == "video-1"
    assert snapshot["filename"] == "match.mp4"
    assert "absolute_path" not in snapshot
