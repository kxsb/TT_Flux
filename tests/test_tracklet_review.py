from __future__ import annotations

import csv
import json
from pathlib import Path

from ttflux.analysis.review_tracklets import (
    build_rows,
    load_metrics,
    load_tracks,
    render_html,
    write_manifest,
)


def test_review_manifest_and_html(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks_probe.csv"
    tracks_path.write_text(
        "track_id,track_rank,frame,x,y,prediction_error_px\n"
        "T001,1,10,100,200,0\n"
        "T001,1,11,105,203,2.5\n",
        encoding="utf-8",
    )
    metrics_path = tmp_path / "tracks_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"tracks": [{
                "track_id": "T001",
                "rank": 1,
                "point_count": 2,
                "first_frame": 10,
                "last_frame": 11,
                "span_frames": 2,
                "score": 2.4,
                "mean_prediction_error_px": 1.25,
            }]}
        ),
        encoding="utf-8",
    )

    rows = build_rows(
        load_tracks(tracks_path),
        load_metrics(metrics_path),
    )
    assert rows[0]["clip_path"] == "clips/T001.mp4"

    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    with manifest.open("r", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))

    assert saved[0]["track_id"] == "T001"
    page = render_html(rows, "run_demo")
    assert "clips/T001.mp4" in page
    assert 'data-label="ball"' in page
    assert "tracklet_labels.csv" in page
