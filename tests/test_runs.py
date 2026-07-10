from __future__ import annotations

import json
from pathlib import Path

import pytest

import ttflux.analysis.runs as run_store


@pytest.fixture(autouse=True)
def stub_candidate_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_analyze(
        video_path: Path,
        csv_path: Path,
        metrics_path: Path,
        overlay_path: Path,
        config=None,
    ) -> dict[str, object]:
        summary = {
            "analyzed_frames": 498,
            "frames_with_candidates": 400,
            "coverage_ratio": 0.803213,
            "total_candidates": 1200,
            "mean_candidates_per_frame": 2.41,
            "median_candidates_per_frame": 2.0,
            "max_candidates_per_frame": 8,
            "mean_candidate_score": 0.5,
            "max_candidate_score": 0.9,
        }
        csv_path.write_text("candidate_id,frame\n", encoding="utf-8")
        metrics_path.write_text(
            json.dumps({"summary": summary}),
            encoding="utf-8",
        )
        overlay_path.write_bytes(b"overlay")
        return {"summary": summary}

    monkeypatch.setattr(run_store, "analyze_candidates", fake_analyze)


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
        "duration_s": 120.0,
        "frame_count": 6000,
        "codec": "h264",
        "probe_error": None,
    }


def configure_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    video = sample_video()
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    video["absolute_path"] = str(source_path)

    monkeypatch.setattr(run_store, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(run_store, "ensure_project_layout", lambda: None)
    monkeypatch.setattr(run_store, "find_video", lambda video_id: video)
    return video


def test_slugify() -> None:
    assert run_store.slugify("Match Démo 01") == "match-demo-01"
    assert run_store.slugify("***") == "video"


def test_validate_clip_range() -> None:
    video = sample_video()

    assert run_store.validate_clip_range(
        video,
        12.3456,
        15.5555,
    ) == (12.346, 15.556)

    with pytest.raises(run_store.InvalidClipRangeError):
        run_store.validate_clip_range(video, -1, 10)

    with pytest.raises(run_store.InvalidClipRangeError):
        run_store.validate_clip_range(video, 0, 0.5)

    with pytest.raises(run_store.InvalidClipRangeError):
        run_store.validate_clip_range(video, 0, 61)

    with pytest.raises(run_store.InvalidClipRangeError):
        run_store.validate_clip_range(video, 115, 10)


def test_create_run_writes_clip_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_store(tmp_path, monkeypatch)

    run = run_store.create_run(
        "video-123",
        clip_start_s=12.5,
        clip_duration_s=15.0,
    )
    run_dir = tmp_path / str(run["run_id"])

    saved_run = json.loads(
        (run_dir / "run.json").read_text(encoding="utf-8")
    )
    saved_video = json.loads(
        (run_dir / "video.json").read_text(encoding="utf-8")
    )

    assert saved_run["status"] == "created"
    assert saved_run["pipeline"] == {
        "name": "motion_candidates",
        "version": 1,
    }
    assert saved_run["configuration"]["clip_start_s"] == 12.5
    assert saved_run["configuration"]["clip_duration_s"] == 15.0
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


def test_execute_run_extracts_clip_and_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_store(tmp_path, monkeypatch)
    created = run_store.create_run(
        "video-123",
        clip_start_s=20.0,
        clip_duration_s=10.0,
    )
    run_id = str(created["run_id"])
    run_dir = tmp_path / run_id
    extraction_calls: list[tuple[float, float]] = []

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        start_s: float,
        duration_s: float,
    ) -> None:
        extraction_calls.append((start_s, duration_s))
        destination_path.write_bytes(b"clip")

    monkeypatch.setattr(run_store, "_extract_clip", fake_extract)
    monkeypatch.setattr(
        run_store,
        "probe_video",
        lambda path: {
            "width": 1280,
            "height": 720,
            "fps": 50.0,
            "duration_s": 10.0,
            "frame_count": 500,
            "codec": "h264",
            "probe_error": None,
        },
    )

    completed = run_store.execute_run(run_id)

    saved_run = json.loads(
        (run_dir / "run.json").read_text(encoding="utf-8")
    )
    clip = json.loads(
        (run_dir / "clip.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (run_dir / "analysis.json").read_text(encoding="utf-8")
    )

    assert extraction_calls == [(20.0, 10.0)]
    assert completed["status"] == "completed"
    assert saved_run["status"] == "completed"
    assert saved_run["artifacts"]["source_clip"] == "source_clip.mp4"
    assert (run_dir / "source_clip.mp4").is_file()
    assert clip["requested_start_s"] == 20.0
    assert clip["requested_duration_s"] == 10.0
    assert clip["frame_count"] == 500
    assert analysis["schema_version"] == 2
    assert analysis["clip"]["actual_duration_s"] == 10.0
    assert analysis["candidates"]["total_candidates"] == 1200
    assert saved_run["metrics"]["coverage_ratio"] == 0.803213
    assert saved_run["artifacts"]["candidates"] == "candidates.csv"
    assert saved_run["artifacts"]["candidate_overlay"] == (
        "overlay_candidates.mp4"
    )
    assert (run_dir / "candidates_metrics.json").is_file()
    assert (run_dir / "overlay_candidates.mp4").is_file()


def test_running_state_is_persisted_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_store(tmp_path, monkeypatch)
    created = run_store.create_run("video-123")
    run_id = str(created["run_id"])
    run_path = tmp_path / run_id / "run.json"
    observed_statuses: list[str] = []

    def observing_extract(
        source_path: Path,
        destination_path: Path,
        start_s: float,
        duration_s: float,
    ) -> None:
        persisted = json.loads(run_path.read_text(encoding="utf-8"))
        observed_statuses.append(str(persisted["status"]))
        destination_path.write_bytes(b"clip")

    monkeypatch.setattr(
        run_store,
        "_extract_clip",
        observing_extract,
    )
    monkeypatch.setattr(
        run_store,
        "probe_video",
        lambda path: {
            "width": 1280,
            "height": 720,
            "fps": 50.0,
            "duration_s": 15.0,
            "frame_count": 750,
            "codec": "h264",
            "probe_error": None,
        },
    )

    run_store.execute_run(run_id)

    assert observed_statuses == ["running"]


def test_failed_extraction_is_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_store(tmp_path, monkeypatch)
    created = run_store.create_run("video-123")
    run_id = str(created["run_id"])
    run_dir = tmp_path / run_id

    def fail_extract(*args, **kwargs):
        raise RuntimeError("échec simulé")

    monkeypatch.setattr(run_store, "_extract_clip", fail_extract)

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
    assert not (run_dir / "clip.json").exists()


def test_get_run_clip_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_store(tmp_path, monkeypatch)
    created = run_store.create_run("video-123")
    run_id = str(created["run_id"])
    run_dir = tmp_path / run_id
    clip_path = run_dir / "source_clip.mp4"
    clip_path.write_bytes(b"clip")

    run_payload = json.loads(
        (run_dir / "run.json").read_text(encoding="utf-8")
    )
    run_payload["artifacts"]["source_clip"] = "source_clip.mp4"
    (run_dir / "run.json").write_text(
        json.dumps(run_payload),
        encoding="utf-8",
    )

    assert run_store.get_run_clip_path(run_id) == clip_path


def test_completed_run_cannot_be_executed_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_store(tmp_path, monkeypatch)
    created = run_store.create_run("video-123")
    run_id = str(created["run_id"])

    monkeypatch.setattr(
        run_store,
        "_extract_clip",
        lambda source, destination, start, duration:
            destination.write_bytes(b"clip"),
    )
    monkeypatch.setattr(
        run_store,
        "probe_video",
        lambda path: {
            "width": 1280,
            "height": 720,
            "fps": 50.0,
            "duration_s": 15.0,
            "frame_count": 750,
            "codec": "h264",
            "probe_error": None,
        },
    )

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
