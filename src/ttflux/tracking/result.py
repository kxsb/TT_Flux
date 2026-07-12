from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ttflux.tracking.candidates.generator import CandidateConfig
from ttflux.analysis.tracks import TrackConfig
from ttflux.tracking.artifacts import (
    BallTrackingArtifacts,
)
from ttflux.tracking.metadata import (
    ENGINE_NAME,
    ENGINE_VERSION,
)


@dataclass(frozen=True)
class BallTrackingResult:
    """Structured result returned by the tracking engine."""

    status: str
    clip_path: Path
    output_dir: Path
    scorer_id: str
    candidate_config: CandidateConfig
    track_config: TrackConfig
    candidate_metrics: dict[str, Any]
    track_metrics: dict[str, Any]
    artifacts: BallTrackingArtifacts

    @property
    def candidate_summary(self) -> dict[str, Any]:
        summary = self.candidate_metrics.get("summary")

        if not isinstance(summary, dict):
            raise TypeError(
                "Candidate metrics do not contain "
                "a summary object."
            )

        return dict(summary)

    @property
    def track_summary(self) -> dict[str, Any]:
        summary = self.track_metrics.get("summary")

        if not isinstance(summary, dict):
            raise TypeError(
                "Track metrics do not contain "
                "a summary object."
            )

        return dict(summary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "engine": {
                "name": ENGINE_NAME,
                "version": ENGINE_VERSION,
            },
            "scorer_id": self.scorer_id,
            "clip": {
                "path": str(self.clip_path),
                "filename": self.clip_path.name,
            },
            "output_dir": str(self.output_dir),
            "configuration": {
                "candidates": asdict(
                    self.candidate_config
                ),
                "tracks": asdict(
                    self.track_config
                ),
            },
            "metrics": {
                "candidates": self.candidate_summary,
                "tracks": self.track_summary,
            },
            "artifacts": self.artifacts.as_dict(),
        }
