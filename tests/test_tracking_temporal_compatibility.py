from __future__ import annotations

import ttflux.tracking as tracking

from ttflux.analysis.tracks import (
    CandidatePoint as LegacyPoint,
)
from ttflux.analysis.tracks import (
    Hypothesis as LegacyHypothesis,
)
from ttflux.analysis.tracks import (
    TrackConfig as LegacyConfig,
)
from ttflux.analysis.tracks import (
    analyze_tracks as LegacyAnalyze,
)
from ttflux.analysis.tracks import (
    build_tracklets as LegacyBuild,
)
from ttflux.analysis.tracks import (
    evaluate_link as LegacyEvaluate,
)
from ttflux.tracking.temporal import (
    TrackConfig as PackageConfig,
)
from ttflux.tracking.temporal.tracklets import (
    CandidatePoint as CanonicalPoint,
)
from ttflux.tracking.temporal.tracklets import (
    Hypothesis as CanonicalHypothesis,
)
from ttflux.tracking.temporal.tracklets import (
    TrackConfig as CanonicalConfig,
)
from ttflux.tracking.temporal.tracklets import (
    analyze_tracks as CanonicalAnalyze,
)
from ttflux.tracking.temporal.tracklets import (
    build_tracklets as CanonicalBuild,
)
from ttflux.tracking.temporal.tracklets import (
    evaluate_link as CanonicalEvaluate,
)


def make_point(
    frame: int,
    x: float,
    y: float,
) -> CanonicalPoint:
    return CanonicalPoint(
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


def test_legacy_temporal_imports_are_identity_shims() -> None:
    assert LegacyPoint is CanonicalPoint
    assert LegacyHypothesis is CanonicalHypothesis
    assert LegacyConfig is CanonicalConfig
    assert LegacyAnalyze is CanonicalAnalyze
    assert LegacyBuild is CanonicalBuild
    assert LegacyEvaluate is CanonicalEvaluate


def test_temporal_package_and_public_api_use_canonical_config() -> None:
    assert PackageConfig is CanonicalConfig
    assert tracking.TrackConfig is CanonicalConfig

    assert CanonicalPoint.__module__ == (
        "ttflux.tracking.temporal.tracklets"
    )
    assert CanonicalHypothesis.__module__ == (
        "ttflux.tracking.temporal.tracklets"
    )
    assert CanonicalConfig.__module__ == (
        "ttflux.tracking.temporal.tracklets"
    )
    assert CanonicalBuild.__module__ == (
        "ttflux.tracking.temporal.tracklets"
    )


def test_temporal_tracklet_behavior_is_preserved() -> None:
    candidates = [
        make_point(
            frame,
            100.0 + 3.0 * frame,
            200.0 + frame,
        )
        for frame in range(20)
    ]

    config = CanonicalConfig(
        max_track_span_frames=12,
        max_output_tracks=4,
        min_points=5,
        seed_stride_frames=3,
    )

    canonical_tracks, canonical_counters = (
        CanonicalBuild(
            candidates,
            config,
        )
    )

    legacy_tracks, legacy_counters = (
        LegacyBuild(
            candidates,
            config,
        )
    )

    assert legacy_counters == canonical_counters
    assert legacy_tracks == canonical_tracks
    assert canonical_tracks
    assert max(
        track.span_frames
        for track in canonical_tracks
    ) <= 12
