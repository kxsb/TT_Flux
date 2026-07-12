from __future__ import annotations

from pathlib import Path

import pytest

import ttflux.pipeline.runs as runs
from ttflux.pipeline.artifacts import (
    get_run_clip_path,
    get_run_overlay_path,
    get_run_tracks_overlay_path,
)
from ttflux.pipeline.errors import (
    InvalidRunStateError,
)
from ttflux.pipeline.maintenance import (
    delete_run as delete_run_directory,
)
from ttflux.pipeline.storage import write_json_atomic


def make_run(
    root: Path,
    *,
    run_id: str = "run-test",
    status: str = "completed",
    artifacts: dict[str, str] | None = None,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir()

    write_json_atomic(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "status": status,
            "video_id": "video-123",
            "video_filename": "source.mp4",
            "artifacts": artifacts or {},
        },
    )

    return run_dir


def test_artifact_resolvers_return_declared_files(
    tmp_path: Path,
) -> None:
    run_dir = make_run(
        tmp_path,
        artifacts={
            "source_clip": "source_clip.mp4",
            "candidate_overlay": (
                "overlay_candidates.mp4"
            ),
            "track_overlay": (
                "overlay_tracks_probe.mp4"
            ),
        },
    )

    clip = run_dir / "source_clip.mp4"
    candidate_overlay = (
        run_dir / "overlay_candidates.mp4"
    )
    track_overlay = (
        run_dir / "overlay_tracks_probe.mp4"
    )

    clip.write_bytes(b"clip")
    candidate_overlay.write_bytes(b"candidate")
    track_overlay.write_bytes(b"tracks")

    assert get_run_clip_path(
        "run-test",
        tmp_path,
    ) == clip
    assert get_run_overlay_path(
        "run-test",
        tmp_path,
    ) == candidate_overlay
    assert get_run_tracks_overlay_path(
        "run-test",
        tmp_path,
    ) == track_overlay


def test_artifact_resolvers_reject_missing_data(
    tmp_path: Path,
) -> None:
    run_dir = make_run(
        tmp_path,
        artifacts={
            "source_clip": "source_clip.mp4",
        },
    )

    with pytest.raises(
        FileNotFoundError,
        match="Artefact absent",
    ):
        get_run_clip_path(
            "run-test",
            tmp_path,
        )

    with pytest.raises(
        FileNotFoundError,
        match="overlay candidat",
    ):
        get_run_overlay_path(
            "run-test",
            tmp_path,
        )

    assert run_dir.is_dir()


def test_canonical_delete_operation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    run_dir = make_run(tmp_path)

    deleted = delete_run_directory(
        "run-test",
        tmp_path,
    )

    assert deleted["run_id"] == "run-test"
    assert deleted["status"] == "completed"
    assert not run_dir.exists()
    assert source.read_bytes() == b"source"


def test_runs_wrappers_preserve_runtime_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed_dir = make_run(
        tmp_path,
        run_id="completed-run",
        artifacts={
            "source_clip": "source_clip.mp4",
        },
    )
    completed_clip = (
        completed_dir / "source_clip.mp4"
    )
    completed_clip.write_bytes(b"clip")

    running_dir = make_run(
        tmp_path,
        run_id="running-run",
        status="running",
    )

    monkeypatch.setattr(
        runs,
        "RUNS_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        runs,
        "ensure_project_layout",
        lambda: None,
    )

    assert runs.get_run_clip_path(
        "completed-run"
    ) == completed_clip

    deleted = runs.delete_run(
        "completed-run"
    )
    assert deleted["run_id"] == "completed-run"

    with pytest.raises(
        InvalidRunStateError,
        match="encore en cours",
    ):
        runs.delete_run("running-run")

    assert running_dir.is_dir()
