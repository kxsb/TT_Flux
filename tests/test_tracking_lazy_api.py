from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


def run_clean_python(code: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()

    existing_pythonpath = environment.get(
        "PYTHONPATH",
        "",
    )

    pythonpath_parts = [
        str(SOURCE_ROOT),
    ]

    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    environment["PYTHONPATH"] = os.pathsep.join(
        pythonpath_parts
    )

    return subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_tracking_package_does_not_eagerly_load_api() -> None:
    completed = run_clean_python(
        """
import sys
import ttflux.tracking as tracking

assert tracking.__all__
assert "ttflux.tracking.api" not in sys.modules
assert "ttflux.tracking.engine" not in sys.modules

print("lazy_package_ok")
"""
    )

    assert completed.stdout.strip() == (
        "lazy_package_ok"
    )


def test_public_object_is_loaded_on_demand() -> None:
    completed = run_clean_python(
        """
import sys
import ttflux.tracking as tracking

assert "ttflux.tracking.api" not in sys.modules

engine_class = tracking.BallTrackingEngine

assert "ttflux.tracking.api" in sys.modules
assert "ttflux.tracking.engine" in sys.modules
assert engine_class.__name__ == "BallTrackingEngine"

print("lazy_resolution_ok")
"""
    )

    assert completed.stdout.strip() == (
        "lazy_resolution_ok"
    )


def test_legacy_analysis_imports_remain_standalone() -> None:
    completed = run_clean_python(
        """
from ttflux.analysis.candidate_scorers import (
    HeuristicV1BallCandidateScorer,
)
from ttflux.analysis.candidates import CandidateConfig
from ttflux.analysis.tracks import TrackConfig

candidate_config = CandidateConfig()
track_config = TrackConfig()
scorer = HeuristicV1BallCandidateScorer()

assert candidate_config.max_candidates_per_frame == 24
assert track_config.max_output_tracks == 24
assert scorer.scorer_id == "heuristic_v1"

print("legacy_imports_ok")
"""
    )

    assert completed.stdout.strip() == (
        "legacy_imports_ok"
    )
