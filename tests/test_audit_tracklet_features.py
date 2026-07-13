from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ttflux.review.audit_tracklet_features import (
    audit,
    validate_label_rows,
)


def write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_audit(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    write(
        run / "candidates.csv",
        [
            "candidate_id", "bbox_w", "bbox_h", "area",
            "mean_brightness", "motion_strength", "fill_ratio",
            "circularity", "score",
        ],
        [
            {
                "candidate_id": "a", "bbox_w": 4, "bbox_h": 2,
                "area": 8, "mean_brightness": 200,
                "motion_strength": 40, "fill_ratio": 1,
                "circularity": .5, "score": .7,
            },
            {
                "candidate_id": "b", "bbox_w": 6, "bbox_h": 2,
                "area": 10, "mean_brightness": 210,
                "motion_strength": 42, "fill_ratio": .8,
                "circularity": .4, "score": .8,
            },
        ],
    )
    write(
        run / "tracks_probe.csv",
        [
            "track_id", "track_rank", "point_index", "candidate_id",
            "frame", "x", "y", "prediction_error_px",
        ],
        [
            {
                "track_id": "T001", "track_rank": 1, "point_index": 1,
                "candidate_id": "a", "frame": 1, "x": 0, "y": 0,
                "prediction_error_px": 0,
            },
            {
                "track_id": "T001", "track_rank": 1, "point_index": 2,
                "candidate_id": "b", "frame": 2, "x": 10, "y": 0,
                "prediction_error_px": 1,
            },
        ],
    )
    labels = tmp_path / "labels.csv"
    write(
        labels,
        ["run_id", "track_id", "label", "notes"],
        [{
            "run_id": "run",
            "track_id": "T001",
            "label": "ball",
            "notes": "",
        }],
    )
    payload = audit(run, labels)
    assert payload["summary"]["track_count"] == 1
    assert payload["tracks"][0]["median_speed_px_per_frame"] == 10
    assert payload["tracks"][0]["median_area"] == 9
    assert (run / "tracklet_feature_audit.html").is_file()



def test_label_run_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="autre run",
    ):
        validate_label_rows(
            "run_b",
            [{
                "run_id": "run_a",
                "track_id": "T001",
                "label": "ball",
            }],
        )


def test_legacy_labels_without_run_id_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="run_id",
    ):
        validate_label_rows(
            "run_a",
            [{
                "track_id": "T001",
                "label": "ball",
            }],
        )
