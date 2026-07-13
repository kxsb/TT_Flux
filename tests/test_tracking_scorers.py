from __future__ import annotations

import numpy as np

import ttflux.tracking as tracking
from ttflux.tracking.candidates.scorers import (
    BallCandidateScorer,
    BallCandidateScoringInput,
    HeuristicV1BallCandidateScorer,
)


def test_public_api_uses_canonical_scorers() -> None:
    assert tracking.BallCandidateScorer is BallCandidateScorer
    assert (
        tracking.HeuristicV1BallCandidateScorer
        is HeuristicV1BallCandidateScorer
    )

    assert BallCandidateScorer.__module__ == (
        "ttflux.tracking.candidates.scorers"
    )
    assert BallCandidateScoringInput.__module__ == (
        "ttflux.tracking.candidates.scorers"
    )
    assert HeuristicV1BallCandidateScorer.__module__ == (
        "ttflux.tracking.candidates.scorers"
    )


def test_heuristic_v1_score_is_preserved_exactly() -> None:
    frame = np.zeros(
        (8, 8),
        dtype=np.uint8,
    )

    candidate = BallCandidateScoringInput(
        previous_gray=frame,
        current_gray=frame,
        next_gray=frame,
        x=4.0,
        y=4.0,
        bbox_x=3,
        bbox_y=3,
        bbox_w=2,
        bbox_h=2,
        area=28,
        mean_brightness=255.0,
        motion_strength=100.0,
        fill_ratio=0.70,
        circularity=1.0,
    )

    scorer = HeuristicV1BallCandidateScorer()

    assert scorer.scorer_id == "heuristic_v1"
    assert scorer.score(candidate) == 1.0
