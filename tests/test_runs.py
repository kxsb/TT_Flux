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


def configure_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    video = sample_video()
    monkeypatch.setattr(run_store, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(run_store, "ensure_project_layout", lambda: None)
    monkeypatch.setattr(run_store, "find_video", lambda video_id: video)
    return video


def test_slugify() -> None:
    assert run_store.slugify("Match Démo 01") == "match-demo-01"
    assert run_store.slugify("***") == "video"


def test_create_run_writes_json_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_store(tmp_path, monkeypatch)

    run = run_store.create_run("video-123")
    run_dir = tmp_path / str(run["run_id"])

    assert (run_dir / "run.json").is_file()
    assert (run_dir / "video.json").is_file()
    assert not (run_dir / "analysis.json").exists()

    saved_run = json.loads(
        (run_dir / "run.json").read_text(encoding="utf-8")
    )
    saved_video = json.loads(
        (run_dir / "video.json").read_text(encoding="utf-8")
    )

    assert saved_run["status"] == "created"
    assert saved_run["pipeline"] == {
        "name": "metadata_only",
        "version": 2,
    }
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


def test_execute_run_writes_analysis_and_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_store(tmp_path, monkeypatch)
    created = run_store.create_run("video-123")

    completed = run_store.execute_run(str(created["run_id"]))
    run_dir = tmp_path / str(created["run_id"])

    saved_run = json.loads(
        (run_dir / "run.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (run_dir / "analysis.json").read_text(encoding="utf-8")
    )

    assert completed["status"] == "completed"
    assert saved_run["status"] == "completed"
    assert saved_run["artifacts"]["analysis"] == "analysis.json"
    assert "started_at" in saved_run
    assert "completed_at" in saved_run
    assert analysis["frame_count"] == 500
    assert analysis["duration_s"] == 10.0
    assert analysis["fps"] == 50.0
    assert analysis["resolution"] == {
        "width": 1280,
        "height": 720,
    }
    assert analysis["estimated_frame_interval_ms"] == 20.0


def test_running_state_is_persisted_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_store(tmp_path, monkeypatch)
    created = run_store.create_run("video-123")
    run_id = str(created["run_id"])
    run_path = tmp_path / run_id / "run.json"
    observed_statuses: list[str] = []
    original_builder = run_store._build_analysis

    def observing_builder(
        run_payload: dict[str, object],
        video_payload: dict[str, object],
        generated_at,
    ) -> dict[str, object]:
        persisted = json.loads(run_path.read_text(encoding="utf-8"))
        observed_statuses.append(str(persisted["status"]))
        return original_builder(
            run_payload,
            video_payload,
            generated_at,
        )

    monkeypatch.setattr(
        run_store,
        "_build_analysis",
        observing_builder,
    )

    run_store.execute_run(run_id)

    assert observed_statuses == ["running"]


def test_failed_execution_is_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_store(tmp_path, monkeypatch)
    created = run_store.create_run("video-123")
    run_id = str(created["run_id"])
    run_dir = tmp_path / run_id

    def fail_builder(*args, **kwargs):
        raise RuntimeError("échec simulé")

    monkeypatch.setattr(run_store, "_build_analysis", fail_builder)

    with pytest.raises(RuntimeError, match="échec simulé"):
        run_store.execute_run(run_id)

    failed_run = json.loads(
        (run_dir / "run.json").read_text(encoding="utf-8")
    )

    assert failed_run["status"] == "failed"
    assert failed_run["error"] == {
        "type": "RuntimeError",
        "message": "échec simulé",
    }
    assert "failed_at" in failed_run
    assert not (run_dir / "analysis.json").exists()


def test_execute_run_rejects_unknown_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_store, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(run_store, "ensure_project_layout", lambda: None)

    with pytest.raises(run_store.UnknownRunError):
        run_store.execute_run("missing")


def test_completed_run_cannot_be_executed_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_store(tmp_path, monkeypatch)
    created = run_store.create_run("video-123")
    run_id = str(created["run_id"])
    run_store.execute_run(run_id)

    with pytest.raises(run_store.InvalidRunStateError):
        run_store.execute_run(run_id)


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
