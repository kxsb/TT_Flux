from __future__ import annotations

import json
from pathlib import Path

import pytest

import ttflux.analysis.runs as run_store


def sample_video() -> dict[str, object]:
    return {
        "id": "video-123",
        "filename": "Match Démo 01.mp4",
        "relative_path": "Match Démo 01.mp4",
        "absolute_path": "C:/private/path/Match Démo 01.mp4",
        "extension": ".mp4",
        "size_mb": 12.5,
        "width": 1280,
        "height": 720,
        "fps": 50.0,
        "duration_s": 10.0,
        "frame_count": 500,
        "codec": "h264",
        "probe_error": None,
    }


def test_slugify() -> None:
    assert run_store.slugify("Match Démo 01") == "match-demo-01"
    assert run_store.slugify("***") == "video"


def test_create_run_writes_json_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = sample_video()
    monkeypatch.setattr(run_store, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(run_store, "ensure_project_layout", lambda: None)
    monkeypatch.setattr(run_store, "find_video", lambda video_id: video)

    run = run_store.create_run("video-123")
    run_dir = tmp_path / str(run["run_id"])

    assert (run_dir / "run.json").is_file()
    assert (run_dir / "video.json").is_file()

    saved_run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    saved_video = json.loads((run_dir / "video.json").read_text(encoding="utf-8"))

    assert saved_run["status"] == "created"
    assert saved_run["pipeline"]["name"] == "metadata_only"
    assert saved_run["video_id"] == "video-123"
    assert saved_video["filename"] == "Match Démo 01.mp4"
    assert "absolute_path" not in saved_video


def test_create_run_rejects_unknown_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_store, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(run_store, "ensure_project_layout", lambda: None)
    monkeypatch.setattr(run_store, "find_video", lambda video_id: None)

    with pytest.raises(run_store.UnknownVideoError):
        run_store.create_run("missing")


def test_list_runs_is_newest_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_store, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(run_store, "ensure_project_layout", lambda: None)

    older_dir = tmp_path / "older"
    newer_dir = tmp_path / "newer"
    older_dir.mkdir()
    newer_dir.mkdir()

    (older_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "older",
                "created_at": "2026-07-10T20:00:00+02:00",
            }
        ),
        encoding="utf-8",
    )
    (newer_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "newer",
                "created_at": "2026-07-10T21:00:00+02:00",
            }
        ),
        encoding="utf-8",
    )

    runs = run_store.list_runs()

    assert [run["run_id"] for run in runs] == ["newer", "older"]
    assert runs[0]["created_label"] == "10/07/2026 21:00:00"
