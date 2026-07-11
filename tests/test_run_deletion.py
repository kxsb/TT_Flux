from __future__ import annotations

import json
from pathlib import Path

import pytest

import ttflux.analysis.runs as run_store


def make_run(
    tmp_path: Path,
    status: str,
    run_id: str = "run-test",
) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir()

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "video_id": "video-123",
                "video_filename": "source.mp4",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "source_clip.mp4").write_bytes(b"clip")
    (run_dir / "overlay_tracks_probe.mp4").write_bytes(
        b"overlay"
    )

    return run_dir


def test_delete_run_removes_only_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    run_dir = make_run(tmp_path, "completed")

    monkeypatch.setattr(run_store, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(
        run_store,
        "ensure_project_layout",
        lambda: None,
    )

    deleted = run_store.delete_run("run-test")

    assert deleted["run_id"] == "run-test"
    assert deleted["status"] == "completed"
    assert not run_dir.exists()
    assert source_path.read_bytes() == b"source"


def test_delete_run_refuses_running_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = make_run(tmp_path, "running")

    monkeypatch.setattr(run_store, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(
        run_store,
        "ensure_project_layout",
        lambda: None,
    )

    with pytest.raises(
        run_store.InvalidRunStateError,
        match="encore en cours",
    ):
        run_store.delete_run("run-test")

    assert run_dir.is_dir()


def test_delete_run_rejects_unknown_or_unsafe_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_store, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(
        run_store,
        "ensure_project_layout",
        lambda: None,
    )

    with pytest.raises(run_store.UnknownRunError):
        run_store.delete_run("missing")

    with pytest.raises(run_store.UnknownRunError):
        run_store.delete_run("../outside")
