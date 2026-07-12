from __future__ import annotations

from pathlib import Path

import ttflux.pipeline.catalog as catalog
import ttflux.pipeline.runs as runs
from ttflux.pipeline.storage import (
    write_json_atomic,
)


def test_created_label_formats_iso_dates() -> None:
    assert catalog.created_label(
        "2026-07-12T21:30:15+02:00"
    ) == "12/07/2026 21:30:15"

    assert catalog.created_label(
        "invalid-date"
    ) == "invalid-date"

    assert catalog.created_label(
        None
    ) == "Date inconnue"


def test_catalog_skips_invalid_runs_and_sorts(
    tmp_path: Path,
) -> None:
    older_dir = tmp_path / "older-directory"
    newer_dir = tmp_path / "newer-directory"
    malformed_dir = tmp_path / "malformed"

    older_dir.mkdir()
    newer_dir.mkdir()
    malformed_dir.mkdir()

    write_json_atomic(
        older_dir / "run.json",
        {
            "run_id": "older",
            "created_at": (
                "2026-07-12T20:00:00+02:00"
            ),
            "status": "completed",
        },
    )

    write_json_atomic(
        newer_dir / "run.json",
        {
            "run_id": "newer",
            "created_at": (
                "2026-07-12T21:00:00+02:00"
            ),
            "status": "created",
        },
    )

    (
        malformed_dir / "run.json"
    ).write_text(
        "{invalid",
        encoding="utf-8",
    )

    found = catalog.list_runs(
        tmp_path
    )

    assert [
        item["run_id"]
        for item in found
    ] == [
        "newer",
        "older",
    ]

    assert found[0]["created_label"] == (
        "12/07/2026 21:00:00"
    )
    assert found[0]["directory"] == (
        "newer-directory"
    )


def test_runs_wrappers_preserve_runtime_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run-directory"
    run_dir.mkdir()

    write_json_atomic(
        run_dir / "run.json",
        {
            "run_id": "run-test",
            "created_at": (
                "2026-07-12T22:00:00+02:00"
            ),
        },
    )

    layout_calls: list[bool] = []

    monkeypatch.setattr(
        runs,
        "RUNS_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        runs,
        "ensure_project_layout",
        lambda: layout_calls.append(True),
    )

    assert runs._list_runs_impl is (
        catalog.list_runs
    )
    assert runs._created_label_impl is (
        catalog.created_label
    )

    found = runs.list_runs()

    assert layout_calls == [True]
    assert len(found) == 1
    assert found[0]["run_id"] == "run-test"

    assert runs._created_label(
        "2026-07-12T22:00:00+02:00"
    ) == "12/07/2026 22:00:00"
