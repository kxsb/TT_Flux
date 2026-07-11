from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


FEATURES = (
    "point_count",
    "span_frames",
    "coverage_ratio",
    "net_displacement_px",
    "total_distance_px",
    "straightness_ratio",
    "median_speed_px_per_frame",
    "max_speed_px_per_frame",
    "speed_mad_px_per_frame",
    "median_acceleration_px_per_frame2",
    "max_acceleration_px_per_frame2",
    "turns_over_45_deg",
    "turns_over_90_deg",
    "median_prediction_error_px",
    "median_area",
    "area_relative_mad",
    "median_elongation",
    "elongation_relative_mad",
    "median_brightness",
    "median_motion_strength",
    "median_fill_ratio",
    "median_circularity",
    "median_candidate_score",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def mad(values: list[float]) -> float:
    if not values:
        return 0.0
    center = median(values)
    return median([abs(value - center) for value in values])


def relative_mad(values: list[float]) -> float:
    center = median(values)
    return mad(values) / abs(center) if abs(center) > 1e-9 else 0.0


def percentile(values: list[float], proportion: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def angle(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_norm = math.hypot(*first)
    second_norm = math.hypot(*second)
    if first_norm < 1e-9 or second_norm < 1e-9:
        return 0.0
    cosine = (
        first[0] * second[0] + first[1] * second[1]
    ) / (first_norm * second_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def track_features(
    track_id: str,
    points: list[dict[str, str]],
    candidates: dict[str, dict[str, str]],
    label: dict[str, str],
) -> dict[str, Any]:
    points.sort(
        key=lambda row: (
            int(number(row.get("frame"))),
            int(number(row.get("point_index"))),
        )
    )
    frames = [int(number(row.get("frame"))) for row in points]
    positions = [
        (number(row.get("x")), number(row.get("y")))
        for row in points
    ]
    velocities: list[tuple[float, float]] = []
    distances: list[float] = []
    speeds: list[float] = []

    for index in range(1, len(points)):
        frame_delta = max(1, frames[index] - frames[index - 1])
        dx = positions[index][0] - positions[index - 1][0]
        dy = positions[index][1] - positions[index - 1][1]
        distance = math.hypot(dx, dy)
        velocities.append((dx / frame_delta, dy / frame_delta))
        distances.append(distance)
        speeds.append(distance / frame_delta)

    accelerations = [
        math.hypot(
            velocities[index][0] - velocities[index - 1][0],
            velocities[index][1] - velocities[index - 1][1],
        )
        for index in range(1, len(velocities))
    ]
    turns = [
        angle(velocities[index - 1], velocities[index])
        for index in range(1, len(velocities))
    ]

    candidate_rows = [
        candidates[row["candidate_id"]]
        for row in points
        if row.get("candidate_id") in candidates
    ]

    def values(column: str) -> list[float]:
        return [number(row.get(column)) for row in candidate_rows]

    widths = values("bbox_w")
    heights = values("bbox_h")
    areas = values("area")
    elongations = [
        max(width / height, height / width)
        for width, height in zip(widths, heights)
        if width > 0 and height > 0
    ]

    span = frames[-1] - frames[0] + 1 if frames else 0
    total_distance = sum(distances)
    net = (
        math.dist(positions[0], positions[-1])
        if len(positions) > 1
        else 0.0
    )

    return {
        "track_id": track_id,
        "label": label.get("label", ""),
        "notes": label.get("notes", ""),
        "track_rank": int(number(points[0].get("track_rank"))) if points else 0,
        "point_count": len(points),
        "first_frame": frames[0] if frames else 0,
        "last_frame": frames[-1] if frames else 0,
        "span_frames": span,
        "coverage_ratio": len(points) / span if span else 0.0,
        "net_displacement_px": net,
        "total_distance_px": total_distance,
        "straightness_ratio": net / total_distance if total_distance else 0.0,
        "median_speed_px_per_frame": median(speeds),
        "max_speed_px_per_frame": max(speeds, default=0.0),
        "speed_mad_px_per_frame": mad(speeds),
        "median_acceleration_px_per_frame2": median(accelerations),
        "max_acceleration_px_per_frame2": max(accelerations, default=0.0),
        "turns_over_45_deg": sum(value >= 45 for value in turns),
        "turns_over_90_deg": sum(value >= 90 for value in turns),
        "median_prediction_error_px": median(
            [number(row.get("prediction_error_px")) for row in points]
        ),
        "median_area": median(areas),
        "area_relative_mad": relative_mad(areas),
        "median_elongation": median(elongations),
        "elongation_relative_mad": relative_mad(elongations),
        "median_brightness": median(values("mean_brightness")),
        "median_motion_strength": median(values("motion_strength")),
        "median_fill_ratio": median(values("fill_ratio")),
        "median_circularity": median(values("circularity")),
        "median_candidate_score": median(values("score")),
    }


def comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for feature in FEATURES:
        ball = [
            float(row[feature]) for row in rows
            if row["label"] == "ball"
        ]
        negative = [
            float(row[feature]) for row in rows
            if row["label"] == "not_ball"
        ]
        combined = ball + negative
        scale = max(
            percentile(combined, 0.75) - percentile(combined, 0.25),
            1e-9,
        )
        separation = (median(ball) - median(negative)) / scale
        result.append(
            {
                "feature": feature,
                "ball_median": median(ball),
                "not_ball_median": median(negative),
                "robust_separation": separation,
            }
        )
    return sorted(
        result,
        key=lambda row: abs(row["robust_separation"]),
        reverse=True,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_html(
    run_id: str,
    rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> str:
    summary = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['feature'])}</td>"
        f"<td>{row['ball_median']:.3f}</td>"
        f"<td>{row['not_ball_median']:.3f}</td>"
        f"<td>{row['robust_separation']:.3f}</td>"
        "</tr>"
        for row in comparison_rows
    )
    columns = ["track_id", "label", "track_rank", *FEATURES]
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    tracks = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row[column]))}</td>"
            for column in columns
        )
        + "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<meta charset="utf-8">
<title>TTFlux — audit tracklets</title>
<style>
body{{font:13px system-ui;color:#e8edf5;background:#0a0d12;padding:24px}}
section{{margin:20px 0;padding:16px;background:#11161d;border:1px solid #29313d}}
div{{overflow:auto;max-height:65vh}}table{{border-collapse:collapse;min-width:100%}}
th,td{{padding:7px;border-bottom:1px solid #29313d;white-space:nowrap}}
th{{position:sticky;top:0;background:#1a222d;text-align:left}}
</style>
<h1>Audit descriptif des tracklets</h1>
<p>{html.escape(run_id)} — aucun seuil automatique appliqué.</p>
<section><h2>Comparaison ball / not_ball</h2><div><table>
<tr><th>Caractéristique</th><th>Médiane ball</th><th>Médiane not_ball</th><th>Séparation robuste</th></tr>
{summary}</table></div></section>
<section><h2>Tracklets</h2><div><table><tr>{headers}</tr>{tracks}</table></div></section>
"""


def audit(run_dir: Path, labels_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    candidate_rows = read_csv(run_dir / "candidates.csv")
    track_rows = read_csv(run_dir / "tracks_probe.csv")
    label_rows = read_csv(labels_path)

    candidates = {
        row["candidate_id"]: row
        for row in candidate_rows
    }
    labels = {
        row["track_id"]: row
        for row in label_rows
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in track_rows:
        grouped[row["track_id"]].append(row)

    rows = [
        track_features(track_id, points, candidates, labels.get(track_id, {}))
        for track_id, points in sorted(grouped.items())
    ]
    comparison_rows = comparisons(rows)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["label"])] += 1

    csv_path = run_dir / "tracklet_feature_audit.csv"
    json_path = run_dir / "tracklet_feature_audit.json"
    html_path = run_dir / "tracklet_feature_audit.html"
    write_csv(csv_path, rows)
    payload = {
        "schema_version": 1,
        "summary": {
            "track_count": len(rows),
            "label_counts": dict(counts),
        },
        "comparisons": comparison_rows,
        "tracks": rows,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(
        build_html(run_dir.name, rows, comparison_rows),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    args = parser.parse_args()
    payload = audit(args.run_dir, args.labels)
    print(f"Audit terminé : {payload['summary']['track_count']} tracklets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
