from __future__ import annotations

from pathlib import Path

from ttflux.tracking.candidates.scorers import (
    BallCandidateScorer,
    HeuristicV1BallCandidateScorer,
)
from ttflux.tracking.candidates.generator import (
    CandidateConfig,
    analyze_candidates,
)
from ttflux.tracking.temporal.tracklets import (
    TrackConfig,
    analyze_tracks,
)
from ttflux.tracking.artifacts import (
    BallTrackingArtifacts,
)
from ttflux.tracking.config import BallTrackingConfig
from ttflux.tracking.metadata import (
    ENGINE_NAME,
    ENGINE_VERSION,
)
from ttflux.tracking.result import BallTrackingResult


class BallTrackingEngine:
    """
    Stable facade around the current deterministic ball tracking stack.

    The facade owns orchestration only. Candidate generation, scoring
    and temporal tracklet construction remain implemented by their
    existing canonical modules.
    """

    engine_name = ENGINE_NAME
    engine_version = ENGINE_VERSION

    def __init__(
        self,
        *,
        config: BallTrackingConfig | None = None,
        candidate_config: CandidateConfig | None = None,
        track_config: TrackConfig | None = None,
        scorer: BallCandidateScorer | None = None,
    ) -> None:
        if config is not None and (
            candidate_config is not None
            or track_config is not None
        ):
            raise ValueError(
                "Use either aggregate config or component "
                "configuration arguments, not both."
            )

        resolved_config = config or BallTrackingConfig(
            candidates=(
                candidate_config or CandidateConfig()
            ),
            tracks=(
                track_config or TrackConfig()
            ),
        )
        resolved_config.validate()

        resolved_candidate_config = (
            resolved_config.candidates
        )
        resolved_track_config = (
            resolved_config.tracks
        )
        resolved_scorer = (
            scorer or HeuristicV1BallCandidateScorer()
        )

        scorer_id = getattr(
            resolved_scorer,
            "scorer_id",
            None,
        )

        if (
            not isinstance(scorer_id, str)
            or not scorer_id.strip()
        ):
            raise TypeError(
                "A ball candidate scorer must expose "
                "a non-empty scorer_id."
            )

        self._config = resolved_config
        self._candidate_config = (
            resolved_candidate_config
        )
        self._track_config = resolved_track_config
        self._scorer = resolved_scorer
        self._scorer_id = scorer_id.strip()

    @property
    def config(self) -> BallTrackingConfig:
        return self._config

    @property
    def candidate_config(self) -> CandidateConfig:
        return self._candidate_config

    @property
    def track_config(self) -> TrackConfig:
        return self._track_config

    @property
    def scorer(self) -> BallCandidateScorer:
        return self._scorer

    @property
    def scorer_id(self) -> str:
        return self._scorer_id

    def run(
        self,
        clip_path: Path,
        output_dir: Path,
    ) -> BallTrackingResult:
        """
        Run candidate generation and temporal tracklet construction.

        The input clip is never modified. All managed artifacts are
        written inside output_dir.
        """

        resolved_clip_path = Path(
            clip_path
        ).resolve()
        resolved_output_dir = Path(
            output_dir
        ).resolve()

        if not resolved_clip_path.is_file():
            raise FileNotFoundError(
                f"Clip video absent: {resolved_clip_path}"
            )

        resolved_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        artifacts = BallTrackingArtifacts.in_directory(
            resolved_output_dir
        )

        candidate_metrics = analyze_candidates(
            video_path=resolved_clip_path,
            csv_path=artifacts.candidates,
            metrics_path=artifacts.candidate_metrics,
            overlay_path=artifacts.candidate_overlay,
            config=self._candidate_config,
            scorer=self._scorer,
        )

        track_metrics = analyze_tracks(
            candidates_path=artifacts.candidates,
            video_path=resolved_clip_path,
            tracks_path=artifacts.tracks,
            metrics_path=artifacts.track_metrics,
            overlay_path=artifacts.track_overlay,
            config=self._track_config,
        )

        if not isinstance(candidate_metrics, dict):
            raise TypeError(
                "Candidate analysis returned an invalid result."
            )

        if not isinstance(track_metrics, dict):
            raise TypeError(
                "Track analysis returned an invalid result."
            )

        artifacts.validate_created()

        result = BallTrackingResult(
            status="completed",
            clip_path=resolved_clip_path,
            output_dir=resolved_output_dir,
            scorer_id=self._scorer_id,
            candidate_config=self._candidate_config,
            track_config=self._track_config,
            candidate_metrics=candidate_metrics,
            track_metrics=track_metrics,
            artifacts=artifacts,
        )

        result.candidate_summary
        result.track_summary

        return result
