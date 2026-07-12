from __future__ import annotations

from pathlib import Path

import pytest

import ttflux.tracking as tracking
import ttflux.tracking.engine as engine_module
from ttflux.tracking.artifacts import (
    BallTrackingArtifacts as DedicatedArtifacts,
)
from ttflux.tracking.result import (
    BallTrackingResult as DedicatedResult,
)


def test_public_contracts_use_dedicated_modules() -> None:
    assert (
        tracking.BallTrackingArtifacts
        is DedicatedArtifacts
    )
    assert (
        engine_module.BallTrackingArtifacts
        is DedicatedArtifacts
    )
    assert (
        tracking.BallTrackingResult
        is DedicatedResult
    )
    assert (
        engine_module.BallTrackingResult
        is DedicatedResult
    )


def test_artifact_contract_validates_outputs(
    tmp_path: Path,
) -> None:
    artifacts = tracking.BallTrackingArtifacts.in_directory(
        tmp_path
    )

    assert artifacts.as_dict() == {
        "candidates": "candidates.csv",
        "candidate_metrics": "candidates_metrics.json",
        "candidate_overlay": "overlay_candidates.mp4",
        "tracks": "tracks_probe.csv",
        "track_metrics": "tracks_metrics.json",
        "track_overlay": "overlay_tracks_probe.mp4",
    }

    assert artifacts.missing_paths() == (
        artifacts.candidates,
        artifacts.candidate_metrics,
        artifacts.candidate_overlay,
        artifacts.tracks,
        artifacts.track_metrics,
        artifacts.track_overlay,
    )

    with pytest.raises(
        RuntimeError,
        match="canonical artifacts",
    ):
        artifacts.validate_created()

    for path in artifacts.all_paths():
        path.write_bytes(b"artifact")

    assert artifacts.missing_paths() == ()
    artifacts.validate_created()


def test_result_contract_preserves_serialization(
    tmp_path: Path,
) -> None:
    clip_path = tmp_path / "clip.mp4"
    output_dir = tmp_path / "run"

    clip_path.write_bytes(b"clip")
    output_dir.mkdir()

    artifacts = tracking.BallTrackingArtifacts.in_directory(
        output_dir
    )

    result = tracking.BallTrackingResult(
        status="completed",
        clip_path=clip_path,
        output_dir=output_dir,
        scorer_id="heuristic_v1",
        candidate_config=tracking.CandidateConfig(
            max_candidates_per_frame=12,
        ),
        track_config=tracking.TrackConfig(
            max_output_tracks=8,
        ),
        candidate_metrics={
            "summary": {
                "total_candidates": 20,
            },
        },
        track_metrics={
            "summary": {
                "selected_tracks": 4,
            },
        },
        artifacts=artifacts,
    )

    payload = result.to_dict()

    assert payload["schema_version"] == 1
    assert payload["engine"] == {
        "name": "ball_tracking_engine",
        "version": 1,
    }
    assert payload["scorer_id"] == "heuristic_v1"
    assert payload["configuration"]["candidates"][
        "max_candidates_per_frame"
    ] == 12
    assert payload["configuration"]["tracks"][
        "max_output_tracks"
    ] == 8
    assert payload["metrics"]["candidates"] == {
        "total_candidates": 20,
    }
    assert payload["metrics"]["tracks"] == {
        "selected_tracks": 4,
    }
