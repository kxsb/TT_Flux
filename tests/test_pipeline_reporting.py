from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

import ttflux.pipeline.reporting as reporting
import ttflux.pipeline.runs as runs


def make_tracking_result(
    payload: Any,
) -> SimpleNamespace:
    return SimpleNamespace(
        to_dict=lambda: payload,
    )


def test_tracking_descriptor_keeps_portable_provenance() -> None:
    descriptor = reporting.tracking_descriptor(
        make_tracking_result(
            {
                "schema_version": 1,
                "engine": {
                    "name": "ball_tracking_engine",
                    "version": 1,
                },
                "scorer_id": "heuristic_v1",
                "configuration": {
                    "candidates": {
                        "max_candidates_per_frame": 24,
                    },
                },
                "status": "completed",
                "clip": {
                    "filename": "source_clip.mp4",
                },
                "output_dir": "private/path",
            }
        )
    )

    assert descriptor == {
        "schema_version": 1,
        "engine": {
            "name": "ball_tracking_engine",
            "version": 1,
        },
        "scorer_id": "heuristic_v1",
        "configuration": {
            "candidates": {
                "max_candidates_per_frame": 24,
            },
        },
    }


def test_tracking_descriptor_validates_contract() -> None:
    with pytest.raises(
        TypeError,
        match="serialization",
    ):
        reporting.tracking_descriptor(
            make_tracking_result([])
        )

    with pytest.raises(
        ValueError,
        match="scorer_id",
    ):
        reporting.tracking_descriptor(
            make_tracking_result(
                {
                    "schema_version": 1,
                    "engine": {},
                    "configuration": {},
                }
            )
        )

    with pytest.raises(
        TypeError,
        match="engine provenance",
    ):
        reporting.tracking_descriptor(
            make_tracking_result(
                {
                    "schema_version": 1,
                    "engine": "invalid",
                    "scorer_id": "heuristic_v1",
                    "configuration": {},
                }
            )
        )


def test_analysis_payload_preserves_schema() -> None:
    generated_at = datetime(
        2026,
        7,
        12,
        20,
        30,
        tzinfo=timezone.utc,
    )

    payload = reporting.build_analysis(
        {
            "run_id": "run-test",
            "video_id": "video-test",
            "video_filename": "source.mp4",
            "pipeline": {
                "name": "motion_tracks_probe",
                "version": 1,
            },
        },
        {
            "frame_count": 6000,
            "duration_s": 120.0,
            "fps": 50.0,
            "width": 1280,
            "height": 720,
        },
        {
            "artifact": "source_clip.mp4",
            "requested_start_s": 10.0,
            "requested_duration_s": 15.0,
            "duration_s": 15.0,
            "frame_count": 750,
            "fps": 50.0,
            "width": 1280,
            "height": 720,
        },
        {
            "summary": {
                "total_candidates": 1200,
            },
        },
        {
            "summary": {
                "selected_tracks": 3,
            },
        },
        {
            "schema_version": 1,
            "engine": {
                "name": "ball_tracking_engine",
                "version": 1,
            },
            "scorer_id": "heuristic_v1",
            "configuration": {},
        },
        generated_at,
    )

    assert payload["schema_version"] == 4
    assert payload["run_id"] == "run-test"
    assert payload["video_id"] == "video-test"
    assert payload["generated_at"] == (
        "2026-07-12T20:30:00+00:00"
    )
    assert payload["source"]["resolution"] == {
        "width": 1280,
        "height": 720,
    }
    assert payload["clip"]["artifact"] == (
        "source_clip.mp4"
    )
    assert payload["candidates"] == {
        "total_candidates": 1200,
    }
    assert payload["tracks_probe"] == {
        "selected_tracks": 3,
    }


def test_runs_wrappers_use_reporting_module() -> None:
    assert runs._tracking_descriptor_impl is (
        reporting.tracking_descriptor
    )
    assert runs._build_analysis_impl is (
        reporting.build_analysis
    )

    result = make_tracking_result(
        {
            "schema_version": 1,
            "engine": {
                "name": "ball_tracking_engine",
                "version": 1,
            },
            "scorer_id": "heuristic_v1",
            "configuration": {},
        }
    )

    assert runs._tracking_descriptor(
        result
    ) == reporting.tracking_descriptor(
        result
    )
