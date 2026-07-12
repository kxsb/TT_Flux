from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BallTrackingArtifacts:
    """Canonical artifacts produced by the tracking engine."""

    candidates: Path
    candidate_metrics: Path
    candidate_overlay: Path
    tracks: Path
    track_metrics: Path
    track_overlay: Path

    @classmethod
    def in_directory(
        cls,
        output_dir: Path,
    ) -> "BallTrackingArtifacts":
        resolved_output_dir = Path(output_dir)

        return cls(
            candidates=(
                resolved_output_dir / "candidates.csv"
            ),
            candidate_metrics=(
                resolved_output_dir
                / "candidates_metrics.json"
            ),
            candidate_overlay=(
                resolved_output_dir
                / "overlay_candidates.mp4"
            ),
            tracks=(
                resolved_output_dir / "tracks_probe.csv"
            ),
            track_metrics=(
                resolved_output_dir / "tracks_metrics.json"
            ),
            track_overlay=(
                resolved_output_dir
                / "overlay_tracks_probe.mp4"
            ),
        )

    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.candidates,
            self.candidate_metrics,
            self.candidate_overlay,
            self.tracks,
            self.track_metrics,
            self.track_overlay,
        )

    def missing_paths(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in self.all_paths()
            if not path.is_file()
        )

    def validate_created(self) -> None:
        missing = self.missing_paths()

        if not missing:
            return

        missing_names = ", ".join(
            path.name
            for path in missing
        )

        raise RuntimeError(
            "Tracking engine did not produce all "
            f"canonical artifacts: {missing_names}"
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "candidates": self.candidates.name,
            "candidate_metrics": (
                self.candidate_metrics.name
            ),
            "candidate_overlay": (
                self.candidate_overlay.name
            ),
            "tracks": self.tracks.name,
            "track_metrics": self.track_metrics.name,
            "track_overlay": self.track_overlay.name,
        }
