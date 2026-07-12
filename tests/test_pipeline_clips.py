from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

import ttflux.pipeline.clips as clips
import ttflux.pipeline.runs as runs
from ttflux.pipeline.contracts import (
    RunPayload,
)


def make_run_payload() -> RunPayload:
    return {
        "schema_version": 1,
        "run_id": "run-test",
        "video_id": "video-test",
        "video_filename": "source.mp4",
        "video_relative_path": "source.mp4",
        "created_at": (
            "2026-07-12T20:00:00+02:00"
        ),
        "status": "running",
        "pipeline": {
            "name": "motion_tracks_probe",
            "version": 1,
        },
        "configuration": {
            "clip_start_s": 1.25,
            "clip_duration_s": 2.5,
            "candidate_detection_enabled": True,
            "track_probe_enabled": True,
            "ball_tracking_enabled": False,
            "table_context_enabled": False,
            "pose_enabled": False,
        },
        "artifacts": {
            "run": "run.json",
            "video": "video.json",
        },
    }


def test_extract_clip_uses_expected_ffmpeg_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "clip.mp4"

    source.write_bytes(b"source")

    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        clips.shutil,
        "which",
        lambda executable: (
            "C:/tools/ffmpeg.exe"
        ),
    )

    def fake_run(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        observed["command"] = list(command)
        observed["kwargs"] = dict(kwargs)

        Path(command[-1]).write_bytes(
            b"encoded-clip"
        )

        return subprocess.CompletedProcess(
            command,
            0,
        )

    monkeypatch.setattr(
        clips.subprocess,
        "run",
        fake_run,
    )

    clips.extract_clip(
        source,
        destination,
        1.25,
        2.5,
    )

    command = observed["command"]
    kwargs = observed["kwargs"]

    assert command[0] == (
        "C:/tools/ffmpeg.exe"
    )
    assert command[
        command.index("-ss") + 1
    ] == "1.250"
    assert command[
        command.index("-t") + 1
    ] == "2.500"
    assert command[
        command.index("-c:v") + 1
    ] == "libx264"
    assert command[
        command.index("-pix_fmt") + 1
    ] == "yuv420p"

    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["encoding"] == "utf-8"

    assert destination.read_bytes() == (
        b"encoded-clip"
    )
    assert not (
        tmp_path / "clip.partial.mp4"
    ).exists()


def test_extract_clip_removes_partial_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "clip.mp4"
    partial = tmp_path / "clip.partial.mp4"

    source.write_bytes(b"source")

    monkeypatch.setattr(
        clips.shutil,
        "which",
        lambda executable: "ffmpeg",
    )

    def fail_run(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        Path(command[-1]).write_bytes(
            b"partial"
        )

        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=command,
            stderr="simulated failure",
        )

    monkeypatch.setattr(
        clips.subprocess,
        "run",
        fail_run,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated failure",
    ):
        clips.extract_clip(
            source,
            destination,
            0.0,
            2.0,
        )

    assert not partial.exists()
    assert not destination.exists()


def test_build_clip_payload_preserves_schema(
    tmp_path: Path,
) -> None:
    clip_path = tmp_path / "source_clip.mp4"
    clip_path.write_bytes(
        b"x" * 2048
    )

    probe_calls: list[Path] = []

    def fake_probe(
        path: Path,
    ) -> dict[str, Any]:
        probe_calls.append(path)

        return {
            "width": 1280,
            "height": 720,
            "fps": 50.0,
            "duration_s": 2.5,
            "frame_count": 125,
            "codec": "h264",
            "probe_error": None,
        }

    payload = clips.build_clip_payload(
        make_run_payload(),
        clip_path,
        probe_video_fn=fake_probe,
    )

    assert probe_calls == [clip_path]
    assert payload["schema_version"] == 1
    assert payload["run_id"] == "run-test"
    assert payload["video_id"] == "video-test"
    assert payload["artifact"] == (
        "source_clip.mp4"
    )
    assert payload["requested_start_s"] == 1.25
    assert payload[
        "requested_duration_s"
    ] == 2.5
    assert payload["size_mb"] == 0.002
    assert payload["frame_count"] == 125


def test_runs_wrappers_preserve_patch_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_path = tmp_path / "source_clip.mp4"
    clip_path.write_bytes(b"clip")

    assert runs._extract_clip_impl is (
        clips.extract_clip
    )
    assert runs._build_clip_payload_impl is (
        clips.build_clip_payload
    )

    monkeypatch.setattr(
        runs,
        "probe_video",
        lambda path: {
            "width": 640,
            "height": 360,
            "fps": 25.0,
            "duration_s": 2.5,
            "frame_count": 62,
            "codec": "h264",
            "probe_error": None,
        },
    )

    payload = runs._build_clip_payload(
        make_run_payload(),
        clip_path,
    )

    assert payload["width"] == 640
    assert payload["frame_count"] == 62
