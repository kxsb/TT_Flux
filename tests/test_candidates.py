from __future__ import annotations

import cv2
import numpy as np

from ttflux.analysis.candidates import (
    CandidateConfig,
    detect_frame_candidates,
    summarize_candidate_counts,
)


def test_detect_frame_candidates_finds_middle_object() -> None:
    previous = np.zeros((100, 120), dtype=np.uint8)
    current = np.zeros((100, 120), dtype=np.uint8)
    following = np.zeros((100, 120), dtype=np.uint8)

    cv2.circle(previous, (20, 50), 3, 255, thickness=-1)
    cv2.circle(current, (60, 50), 3, 255, thickness=-1)
    cv2.circle(following, (100, 50), 3, 255, thickness=-1)

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
    best = candidates[0]
    assert abs(float(best["x"]) - 60.0) <= 1.0
    assert abs(float(best["y"]) - 50.0) <= 1.0
    assert int(best["area"]) >= 20


def test_summarize_candidate_counts() -> None:
    summary = summarize_candidate_counts(
        [0, 2, 1, 0],
        [0.5, 0.7, 0.9],
    )

    assert summary["analyzed_frames"] == 4
    assert summary["frames_with_candidates"] == 2
    assert summary["coverage_ratio"] == 0.5
    assert summary["total_candidates"] == 3
    assert summary["mean_candidates_per_frame"] == 0.75
    assert summary["median_candidates_per_frame"] == 0.5
    assert summary["max_candidates_per_frame"] == 2
    assert summary["max_candidate_score"] == 0.9
