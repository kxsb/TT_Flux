from ttflux.tracking.candidates.scorers import (
    HeuristicV1BallCandidateScorer,
)
from ttflux.analysis.candidates import CandidateConfig
from ttflux.analysis.tracks import TrackConfig
from ttflux.tracking import BallTrackingEngine


def test_public_and_remaining_legacy_imports_resolve() -> None:
    candidate_config = CandidateConfig()
    track_config = TrackConfig()
    scorer = HeuristicV1BallCandidateScorer()

    assert BallTrackingEngine.__name__ == "BallTrackingEngine"
    assert candidate_config.max_candidates_per_frame == 24
    assert track_config.max_output_tracks == 24
    assert scorer.scorer_id == "heuristic_v1"
