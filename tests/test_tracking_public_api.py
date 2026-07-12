from __future__ import annotations

import pytest

import ttflux.tracking as tracking


EXPECTED_PUBLIC_NAMES = {
    "BallCandidateScorer",
    "BallTrackingArtifacts",
    "BallTrackingConfig",
    "BallTrackingEngine",
    "BallTrackingResult",
    "CandidateConfig",
    "HeuristicV1BallCandidateScorer",
    "TrackConfig",
}


def test_tracking_package_exposes_stable_public_api() -> None:
    assert set(tracking.__all__) == EXPECTED_PUBLIC_NAMES

    for name in EXPECTED_PUBLIC_NAMES:
        assert getattr(tracking, name) is not None


def test_ball_tracking_config_groups_all_parameters() -> None:
    config = tracking.BallTrackingConfig(
        candidates=tracking.CandidateConfig(
            max_candidates_per_frame=12,
        ),
        tracks=tracking.TrackConfig(
            max_output_tracks=8,
        ),
    )

    config.validate()

    payload = config.to_dict()

    assert payload["candidates"][
        "max_candidates_per_frame"
    ] == 12
    assert payload["tracks"][
        "max_output_tracks"
    ] == 8


def test_engine_accepts_aggregate_configuration() -> None:
    config = tracking.BallTrackingConfig(
        candidates=tracking.CandidateConfig(
            max_candidates_per_frame=10,
        ),
        tracks=tracking.TrackConfig(
            max_output_tracks=6,
        ),
    )

    engine = tracking.BallTrackingEngine(
        config=config,
    )

    assert engine.config is config
    assert engine.candidate_config is (
        config.candidates
    )
    assert engine.track_config is config.tracks
    assert engine.scorer_id == "heuristic_v1"


def test_engine_rejects_mixed_configuration_styles() -> None:
    config = tracking.BallTrackingConfig()

    with pytest.raises(
        ValueError,
        match="aggregate config",
    ):
        tracking.BallTrackingEngine(
            config=config,
            candidate_config=tracking.CandidateConfig(),
        )
