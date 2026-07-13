from __future__ import annotations

import numpy as np

import ttflux.tracking as tracking
from ttflux.tracking.candidates import (
    CandidateConfig as PackageConfig,
)
from ttflux.tracking.candidates.generator import (
    CandidateConfig,
    analyze_candidates,
    detect_frame_candidates,
    summarize_candidate_counts,
)


def test_candidate_package_and_public_api_use_canonical_config() -> None:
    assert PackageConfig is CandidateConfig
    assert tracking.CandidateConfig is CandidateConfig


def test_generator_exports_are_canonical() -> None:
    expected_module = "ttflux.tracking.candidates.generator"

    assert CandidateConfig.__module__ == expected_module
    assert analyze_candidates.__module__ == expected_module
    assert detect_frame_candidates.__module__ == expected_module
    assert summarize_candidate_counts.__module__ == expected_module


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

    config = CandidateConfig(
        motion_threshold=10,
        min_area=1,
        max_area=100,
        max_dimension=20,
        min_fill_ratio=0.05,
        max_candidates_per_frame=10,
        blur_kernel=1,
    )

    candidates = detect_frame_candidates(
        previous,
        current,
        following,
        config,
    )

    assert candidates
    assert abs(float(candidates[0]["x"]) - 50.0) <= 1.0
    assert abs(float(candidates[0]["y"]) - 40.0) <= 1.0
