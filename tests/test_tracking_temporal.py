from __future__ import annotations

import ttflux.tracking as tracking
from ttflux.tracking.temporal import (
    TrackConfig as PackageConfig,
)
from ttflux.tracking.temporal.tracklets import (
    CandidatePoint,
    Hypothesis,
    TrackConfig,
    analyze_tracks,
    build_tracklets,
    evaluate_link,
)


def make_point(
    frame: int,
    x: float,
    y: float,
) -> CandidatePoint:
    return CandidatePoint(
        candidate_id=f"c{frame}",
        frame=frame,
        time_s=frame / 50.0,
        x=x,
        y=y,
        score=0.8,
        area=20.0,
        mean_brightness=220.0,
        bbox_w=5.0,
        bbox_h=5.0,
        rank=1,
    )


def test_temporal_package_and_public_api_use_canonical_config() -> None:
    assert PackageConfig is TrackConfig
    assert tracking.TrackConfig is TrackConfig


def test_temporal_exports_are_canonical() -> None:
    expected_module = "ttflux.tracking.temporal.tracklets"

    assert CandidatePoint.__module__ == expected_module
    assert Hypothesis.__module__ == expected_module
    assert TrackConfig.__module__ == expected_module
    assert analyze_tracks.__module__ == expected_module
    assert build_tracklets.__module__ == expected_module
    assert evaluate_link.__module__ == expected_module


def test_temporal_tracklet_behavior_is_preserved() -> None:
    candidates = [
        make_point(
            frame,
            100.0 + 3.0 * frame,
            200.0 + frame,
        )
        for frame in range(20)
    ]

    config = TrackConfig(
        max_track_span_frames=12,
        max_output_tracks=4,
        min_points=5,
        seed_stride_frames=3,
    )

    tracks, counters = build_tracklets(
        candidates,
        config,
    )

    assert counters["eligible_hypotheses"] > 0
    assert tracks
    assert max(
        track.span_frames
        for track in tracks
    ) <= 12
