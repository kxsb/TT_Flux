from __future__ import annotations

import numpy as np

import ttflux.tracking as tracking
from ttflux.analysis.candidates import (
    CandidateConfig as LegacyConfig,
)
from ttflux.analysis.candidates import (
    analyze_candidates as LegacyAnalyze,
)
from ttflux.analysis.candidates import (
    detect_frame_candidates as LegacyDetect,
)
from ttflux.analysis.candidates import (
    summarize_candidate_counts as LegacySummarize,
)
from ttflux.tracking.candidates import (
    CandidateConfig as PackageConfig,
)
from ttflux.tracking.candidates.generator import (
    CandidateConfig as CanonicalConfig,
)
from ttflux.tracking.candidates.generator import (
    analyze_candidates as CanonicalAnalyze,
)
from ttflux.tracking.candidates.generator import (
    detect_frame_candidates as CanonicalDetect,
)
from ttflux.tracking.candidates.generator import (
    summarize_candidate_counts as CanonicalSummarize,
)


def test_legacy_generator_imports_are_identity_shims() -> None:
    assert LegacyConfig is CanonicalConfig
    assert LegacyAnalyze is CanonicalAnalyze
    assert LegacyDetect is CanonicalDetect
    assert LegacySummarize is CanonicalSummarize


def test_candidate_package_and_public_api_use_canonical_config() -> None:
    assert PackageConfig is CanonicalConfig
    assert tracking.CandidateConfig is CanonicalConfig

    assert CanonicalConfig.__module__ == (
        "ttflux.tracking.candidates.generator"
    )
    assert CanonicalAnalyze.__module__ == (
        "ttflux.tracking.candidates.generator"
    )
    assert CanonicalDetect.__module__ == (
        "ttflux.tracking.candidates.generator"
    )


def test_candidate_detection_behavior_is_preserved() -> None:
    previous = np.zeros(
        (80, 100),
        dtype=np.uint8,
    )
    current = np.zeros(
        (80, 100),
        dtype=np.uint8,
    )
    following = np.zeros(
        (80, 100),
        dtype=np.uint8,
    )

    current[38:43, 48:53] = 255

    config = CanonicalConfig(
        motion_threshold=10,
        min_area=1,
        max_area=100,
        max_dimension=20,
        min_fill_ratio=0.05,
        max_candidates_per_frame=10,
        blur_kernel=1,
    )

    canonical = CanonicalDetect(
        previous,
        current,
        following,
        config,
    )
    legacy = LegacyDetect(
        previous,
        current,
        following,
        config,
    )

    assert legacy == canonical
