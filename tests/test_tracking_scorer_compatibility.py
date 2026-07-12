from __future__ import annotations

import numpy as np

import ttflux.tracking as tracking
from ttflux.analysis.candidate_scorers import (
    BallCandidateScorer as LegacyScorer,
)
from ttflux.analysis.candidate_scorers import (
    BallCandidateScoringInput as LegacyInput,
)
from ttflux.analysis.candidate_scorers import (
    HeuristicV1BallCandidateScorer as LegacyHeuristic,
)
from ttflux.tracking.candidates.scorers import (
    BallCandidateScorer as CanonicalScorer,
)
from ttflux.tracking.candidates.scorers import (
    BallCandidateScoringInput as CanonicalInput,
)
from ttflux.tracking.candidates.scorers import (
    HeuristicV1BallCandidateScorer as CanonicalHeuristic,
)


def test_legacy_scorer_imports_are_identity_shims() -> None:
    assert LegacyScorer is CanonicalScorer
    assert LegacyInput is CanonicalInput
    assert LegacyHeuristic is CanonicalHeuristic


def test_public_api_uses_canonical_scorers() -> None:
    assert (
        tracking.BallCandidateScorer
        is CanonicalScorer
    )
    assert (
        tracking.HeuristicV1BallCandidateScorer
        is CanonicalHeuristic
    )

    assert CanonicalScorer.__module__ == (
        "ttflux.tracking.candidates.scorers"
    )
    assert CanonicalInput.__module__ == (
        "ttflux.tracking.candidates.scorers"
    )
    assert CanonicalHeuristic.__module__ == (
        "ttflux.tracking.candidates.scorers"
    )


def test_heuristic_v1_score_is_preserved_exactly() -> None:
    frame = np.zeros(
        (8, 8),
        dtype=np.uint8,
    )

    candidate = CanonicalInput(
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

    scorer = CanonicalHeuristic()

    assert scorer.scorer_id == "heuristic_v1"
    assert scorer.score(candidate) == 1.0
