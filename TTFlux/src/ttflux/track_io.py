from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrackPoint:
    frame: int
    x: float
    y: float


def read_track_points(
    track_csv: str | Path,
    clip_id: str,
) -> list[TrackPoint]:
    track_csv = Path(track_csv)

    if not track_csv.exists():
        raise FileNotFoundError(
            f"Track CSV introuvable: {track_csv}"
        )

    points: list[TrackPoint] = []

    with track_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        required = {"clip_id", "frame", "x", "y"}
        missing = required - set(fieldnames)

        if missing:
            raise ValueError(
                "Colonnes manquantes: "
                f"{missing}. Colonnes={fieldnames}"
            )

        for row in reader:
            row_clip = str(row.get("clip_id", "")).strip()

            if row_clip != clip_id:
                continue

            try:
                frame = int(
                    float(str(row["frame"]).replace(",", "."))
                )
                x = float(str(row["x"]).replace(",", "."))
                y = float(str(row["y"]).replace(",", "."))
            except Exception:
                continue

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            points.append(
                TrackPoint(frame=frame, x=x, y=y)
            )

    return sorted(points, key=lambda p: p.frame)


def write_segment_csv(
    path: str | Path,
    points: list[TrackPoint],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "frame", "x", "y"],
        )
        writer.writeheader()

        for idx, point in enumerate(points):
            writer.writerow(
                {
                    "idx": idx,
                    "frame": point.frame,
                    "x": point.x,
                    "y": point.y,
                }
            )
