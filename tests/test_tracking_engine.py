from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import ttflux.tracking.engine as engine_module
from ttflux.tracking.candidates.generator import CandidateConfig
from ttflux.analysis.tracks import TrackConfig
from ttflux.tracking import BallTrackingEngine


def test_tracking_engine_runs_canonical_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_path = tmp_path / "source_clip.mp4"
    output_dir = tmp_path / "run"

    clip_path.write_bytes(b"test-video")

    calls: list[str] = []

    def fake_analyze_candidates(
        *,
        video_path: Path,
        csv_path: Path,
        metrics_path: Path,
        overlay_path: Path,
        config: CandidateConfig,
        scorer: Any,
    ) -> dict[str, Any]:
        calls.append("candidates")

        assert video_path == clip_path.resolve()
        assert config == CandidateConfig()
        assert scorer.scorer_id == "heuristic_v1"

        csv_path.write_text(
            "candidate_id\n",
            encoding="utf-8",
        )
        metrics_path.write_text(
            "{}\n",
            encoding="utf-8",
        )
        overlay_path.write_bytes(
            b"candidate-overlay"
        )

        return {
            "summary": {
                "total_candidates": 12,
            }
        }

    def fake_analyze_tracks(
        *,
        candidates_path: Path,
        video_path: Path,
        tracks_path: Path,
        metrics_path: Path,
        overlay_path: Path,
        config: TrackConfig,
    ) -> dict[str, Any]:
        calls.append("tracks")

        assert candidates_path.is_file()
        assert video_path == clip_path.resolve()
        assert config == TrackConfig()

        tracks_path.write_text(
            "track_id\n",
            encoding="utf-8",
        )
        metrics_path.write_text(
            "{}\n",
            encoding="utf-8",
        )
        overlay_path.write_bytes(
            b"track-overlay"
        )

        return {
            "summary": {
                "selected_tracks": 3,
            }
        }

    monkeypatch.setattr(
        engine_module,
        "analyze_candidates",
        fake_analyze_candidates,
    )
    monkeypatch.setattr(
        engine_module,
        "analyze_tracks",
        fake_analyze_tracks,
    )

    engine = BallTrackingEngine()
    result = engine.run(
        clip_path,
        output_dir,
    )

    assert calls == [
        "candidates",
        "tracks",
    ]
    assert result.status == "completed"
    assert result.scorer_id == "heuristic_v1"
    assert result.candidate_summary == {
        "total_candidates": 12,
    }
    assert result.track_summary == {
        "selected_tracks": 3,
    }

    assert result.artifacts.as_dict() == {
        "candidates": "candidates.csv",
        "candidate_metrics": (
            "candidates_metrics.json"
        ),
        "candidate_overlay": (
            "overlay_candidates.mp4"
        ),
        "tracks": "tracks_probe.csv",
        "track_metrics": "tracks_metrics.json",
        "track_overlay": (
            "overlay_tracks_probe.mp4"
        ),
    }

    assert all(
        path.is_file()
        for path in result.artifacts.all_paths()
    )

    payload = result.to_dict()

    assert payload["engine"] == {
        "name": "ball_tracking_engine",
        "version": 1,
    }
    assert payload["metrics"]["candidates"] == {
        "total_candidates": 12,
    }
    assert payload["metrics"]["tracks"] == {
        "selected_tracks": 3,
    }


def test_tracking_engine_preserves_custom_dependencies() -> None:
    class CustomScorer:
        scorer_id = "test_scorer"

        def score(self, candidate: Any) -> float:
            return 0.5

    candidate_config = CandidateConfig(
        max_candidates_per_frame=12,
    )
    track_config = TrackConfig(
        max_output_tracks=8,
    )
    scorer = CustomScorer()

    engine = BallTrackingEngine(
        candidate_config=candidate_config,
        track_config=track_config,
        scorer=scorer,
    )

    assert engine.candidate_config is candidate_config
    assert engine.track_config is track_config
    assert engine.scorer is scorer
    assert engine.scorer_id == "test_scorer"


def test_tracking_engine_rejects_missing_clip(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"

    engine = BallTrackingEngine()

    with pytest.raises(
        FileNotFoundError,
        match="Clip video absent",
    ):
        engine.run(
            tmp_path / "missing.mp4",
            output_dir,
        )

    assert not output_dir.exists()
