from __future__ import annotations

from importlib import import_module
from pathlib import Path

import ttflux.pipeline as pipeline


canonical = import_module(
    "ttflux.pipeline.runs"
)
legacy = import_module(
    "ttflux.analysis.runs"
)


def test_legacy_run_module_is_canonical_alias() -> None:
    assert legacy is canonical


def test_pipeline_package_exposes_canonical_run_api() -> None:
    assert pipeline.create_run is canonical.create_run
    assert pipeline.execute_run is canonical.execute_run
    assert (
        pipeline.create_and_execute_clip_run
        is canonical.create_and_execute_clip_run
    )
    assert pipeline.list_runs is canonical.list_runs

    assert canonical.create_run.__module__ == (
        "ttflux.pipeline.runs"
    )
    assert canonical.execute_run.__module__ == (
        "ttflux.pipeline.runs"
    )


def test_legacy_monkeypatch_state_is_shared(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        legacy,
        "RUNS_DIR",
        tmp_path,
    )

    assert canonical.RUNS_DIR is tmp_path
