from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ttflux.analysis.candidates import CandidateConfig
from ttflux.analysis.tracks import TrackConfig


@dataclass(frozen=True)
class BallTrackingConfig:
    """
    Complete public configuration of the ball tracking engine.

    Candidate and temporal parameters remain separated internally,
    while callers manipulate a single stable configuration object.
    """

    candidates: CandidateConfig = field(
        default_factory=CandidateConfig
    )
    tracks: TrackConfig = field(
        default_factory=TrackConfig
    )

    def validate(self) -> None:
        self.candidates.validate()
        self.tracks.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": asdict(
                self.candidates
            ),
            "tracks": asdict(
                self.tracks
            ),
        }
