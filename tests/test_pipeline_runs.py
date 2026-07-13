from __future__ import annotations

from importlib import import_module

import ttflux.pipeline as pipeline


runs = import_module(
    "ttflux.pipeline.runs"
)


def test_pipeline_package_exposes_canonical_run_api() -> None:
    assert pipeline.create_run is runs.create_run
    assert pipeline.execute_run is runs.execute_run
    assert (
        pipeline.create_and_execute_clip_run
        is runs.create_and_execute_clip_run
    )
    assert pipeline.list_runs is runs.list_runs

    assert runs.create_run.__module__ == (
        "ttflux.pipeline.runs"
    )
    assert runs.execute_run.__module__ == (
        "ttflux.pipeline.runs"
    )
