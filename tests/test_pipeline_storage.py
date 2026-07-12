from __future__ import annotations

from pathlib import Path

import pytest

import ttflux.pipeline.runs as runs
from ttflux.pipeline.errors import UnknownRunError
from ttflux.pipeline.storage import (
    read_json_object,
    resolve_run_dir,
    try_read_json_object,
    write_json_atomic,
)


def test_json_storage_round_trip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.json"
    payload = {
        "run_id": "run-1",
        "status": "created",
    }

    write_json_atomic(path, payload)

    assert read_json_object(path) == payload
    assert path.read_text(
        encoding="utf-8"
    ).endswith("\n")
    assert not path.with_suffix(
        ".json.tmp"
    ).exists()


def test_tolerant_json_reader_skips_invalid_data(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.json"
    array = tmp_path / "array.json"

    malformed.write_text(
        "{invalid",
        encoding="utf-8",
    )
    array.write_text(
        "[]",
        encoding="utf-8",
    )

    assert try_read_json_object(malformed) is None
    assert try_read_json_object(array) is None


def test_run_directory_resolution_is_safe(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-safe"
    run_dir.mkdir()

    assert (
        resolve_run_dir(
            "run-safe",
            tmp_path,
        )
        == run_dir
    )

    for run_id in (
        "",
        ".",
        "..",
        "../outside",
        "folder/run",
        "missing",
    ):
        with pytest.raises(UnknownRunError):
            resolve_run_dir(
                run_id,
                tmp_path,
            )


def test_runs_module_uses_canonical_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run-test"
    run_dir.mkdir()

    monkeypatch.setattr(
        runs,
        "RUNS_DIR",
        tmp_path,
    )

    assert (
        runs._write_json_atomic
        is write_json_atomic
    )
    assert (
        runs._read_json_object
        is read_json_object
    )
    assert (
        runs._try_read_json_object
        is try_read_json_object
    )
    assert runs._resolve_run_dir(
        "run-test"
    ) == run_dir
