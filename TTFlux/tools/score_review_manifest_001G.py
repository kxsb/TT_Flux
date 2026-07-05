# PATCH 001G - TTFlux trajectory scoring
#
# Reads review_manifest_001F2_repaired.csv and each segment CSV.
# Adds trajectory-level metrics:
# - frame gaps
# - speed median / p95 / max
# - acceleration p95 / max
# - heading turns
# - teleport-like jumps
# - risk score and review guess
#
# Usage:
#   python tools\score_review_manifest_001G.py ^
#     --run-dir runs\batch_001E ^
#     --manifest runs\batch_001E\review_manifest_001F2_repaired.csv ^
#     --out-csv runs\batch_001E\review_manifest_001G_trajectory.csv ^
#     --out-html runs\batch_001E\review_manifest_001G_trajectory.html

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from pathlib import Path
from typing import Any


FRAME_COLS = ["frame", "frame_idx", "frame_index", "f"]
X_COLS = ["x", "ball_x", "cx", "center_x"]
Y_COLS = ["y", "ball_y", "cy", "center_y"]


def norm_col(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def first_existing_col(fieldnames: list[str], candidates: list[str]) -> str | None:
    normalized = {norm_col(c): c for c in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip().replace(",", ".")
    if not text or text.lower() == "nan":
        return None

    try:
        val = float(text)
    except ValueError:
        return None

    if not math.isfinite(val):
        return None

    return val


def safe_int(value: Any, default: int = 0) -> int:
    val = safe_float(value)
    if val is None:
        return default
    return int(round(val))


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None

    values = sorted(values)
    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return values[lo]

    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(8192)
            f.seek(0)

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel

            reader = csv.DictReader(f, dialect=dialect)
            if not reader.fieldnames:
                return [], []

            rows = []
            for row in reader:
                rows.append({str(k): "" if v is None else str(v) for k, v in row.items()})

            return list(reader.fieldnames), rows

    except Exception as exc:
        print(f"[WARN] could not read CSV {path}: {exc}")
        return [], []


def read_manifest(path: Path) -> list[dict[str, str]]:
    fieldnames, rows = read_csv_rows(path)
    if not fieldnames:
        raise RuntimeError(f"Could not read manifest: {path}")
    return rows


def clean_first_path(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    return text.split("|")[0].strip()


def resolve_segment_csv(run_dir: Path, row: dict[str, str]) -> Path | None:
    path_text = clean_first_path(row.get("csv", ""))
    if not path_text:
        return None

    path_text = path_text.replace("\\", "/")
    p = Path(path_text)

    if p.is_absolute():
        return p

    return run_dir / p


def collapse_points_by_frame(points: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    buckets: dict[int, list[tuple[float, float]]] = {}

    for frame, x, y in points:
        buckets.setdefault(frame, []).append((x, y))

    collapsed = []
    for frame in sorted(buckets):
        vals = buckets[frame]
        x = sum(v[0] for v in vals) / len(vals)
        y = sum(v[1] for v in vals) / len(vals)
        collapsed.append((frame, x, y))

    return collapsed


def load_points(path: Path) -> list[tuple[int, float, float]]:
    fieldnames, rows = read_csv_rows(path)
    if not fieldnames or not rows:
        return []

    frame_col = first_existing_col(fieldnames, FRAME_COLS)
    x_col = first_existing_col(fieldnames, X_COLS)
    y_col = first_existing_col(fieldnames, Y_COLS)

    if not frame_col or not x_col or not y_col:
        return []

    points = []
    for row in rows:
        frame = safe_float(row.get(frame_col))
        x = safe_float(row.get(x_col))
        y = safe_float(row.get(y_col))

        if frame is None or x is None or y is None:
            continue

        points.append((int(round(frame)), x, y))

    return collapse_points_by_frame(points)


def angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float | None:
    x1, y1 = v1
    x2, y2 = v2

    n1 = math.hypot(x1, y1)
    n2 = math.hypot(x2, y2)

    if n1 <= 1e-9 or n2 <= 1e-9:
        return None

    c = (x1 * x2 + y1 * y2) / (n1 * n2)
    c = max(-1.0, min(1.0, c))

    return math.degrees(math.acos(c))


def trajectory_stats(points: list[tuple[int, float, float]]) -> dict[str, Any]:
    if not points:
        return {
            "trajectory_found": "0",
            "risk_score_001G": "100",
            "smoothness_score_001G": "0",
            "review_guess_001G": "missing",
            "risk_flags_001G": "missing_points",
        }

    points = sorted(points, key=lambda p: p[0])

    frames = [p[0] for p in points]
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]

    n = len(points)
    first = min(frames)
    last = max(frames)
    span = max(1, last - first + 1)
    density = n / span

    gaps = []
    speeds = []
    velocities = []

    for a, b in zip(points, points[1:]):
        f0, x0, y0 = a
        f1, x1, y1 = b

        dt = max(1, f1 - f0)
        dx = x1 - x0
        dy = y1 - y0

        gaps.append(dt)
        speeds.append(math.hypot(dx, dy) / dt)
        velocities.append((dx / dt, dy / dt))

    accels = []
    for v0, v1 in zip(velocities, velocities[1:]):
        ax = v1[0] - v0[0]
        ay = v1[1] - v0[1]
        accels.append(math.hypot(ax, ay))

    heading_turns = []
    for v0, v1 in zip(velocities, velocities[1:]):
        angle = angle_between(v0, v1)
        if angle is not None:
            heading_turns.append(angle)

    speed_median = statistics.median(speeds) if speeds else 0.0
    speed_p95 = percentile(speeds, 0.95) or 0.0
    speed_max = max(speeds) if speeds else 0.0

    accel_p95 = percentile(accels, 0.95) or 0.0
    accel_max = max(accels) if accels else 0.0

    heading_max = max(heading_turns) if heading_turns else 0.0
    heading_p95 = percentile(heading_turns, 0.95) or 0.0

    gap_count = sum(1 for g in gaps if g > 1)
    big_gap_count = sum(1 for g in gaps if g >= 6)
    max_gap = max(gaps) if gaps else 0

    teleport_count = sum(1 for s in speeds if s >= 90.0)
    high_speed_count = sum(1 for s in speeds if s >= 65.0)

    hard_turn_count = sum(1 for a in heading_turns if a >= 120.0)
    medium_turn_count = sum(1 for a in heading_turns if a >= 80.0)

    x_range = max(xs) - min(xs) if xs else 0.0
    y_range = max(ys) - min(ys) if ys else 0.0

    risk = 0
    flags: list[str] = []

    if n < 12:
        risk += 18
        flags.append("very_few_points")
    elif n < 16:
        risk += 10
        flags.append("few_points")

    if density < 0.22:
        risk += 25
        flags.append("very_low_density")
    elif density < 0.32:
        risk += 16
        flags.append("low_density")
    elif density < 0.42:
        risk += 7
        flags.append("medium_density")

    if max_gap >= 12:
        risk += 22
        flags.append("huge_frame_gap")
    elif max_gap >= 7:
        risk += 12
        flags.append("big_frame_gap")
    elif max_gap >= 4:
        risk += 5
        flags.append("frame_gap")

    if big_gap_count >= 4:
        risk += 10
        flags.append("many_big_gaps")
    elif big_gap_count >= 2:
        risk += 5
        flags.append("some_big_gaps")

    if speed_max >= 140:
        risk += 25
        flags.append("extreme_speed_spike")
    elif speed_max >= 100:
        risk += 16
        flags.append("speed_spike")
    elif speed_max >= 75:
        risk += 8
        flags.append("high_speed")

    ratio = speed_p95 / max(1e-6, speed_median)
    if speed_median > 0:
        if ratio >= 4.5:
            risk += 16
            flags.append("unstable_speed_ratio")
        elif ratio >= 3.0:
            risk += 8
            flags.append("speed_ratio")

    if teleport_count >= 2:
        risk += 16
        flags.append("teleport_steps")
    elif teleport_count == 1:
        risk += 8
        flags.append("teleport_step")

    if accel_p95 >= 95:
        risk += 14
        flags.append("high_accel_p95")
    elif accel_p95 >= 60:
        risk += 7
        flags.append("accel_p95")

    if hard_turn_count >= 3:
        risk += 12
        flags.append("many_hard_turns")
    elif hard_turn_count >= 1:
        risk += 6
        flags.append("hard_turn")

    if x_range >= 700:
        risk += 12
        flags.append("huge_x_range")
    elif x_range >= 480:
        risk += 6
        flags.append("wide_x_range")

    if y_range >= 420:
        risk += 12
        flags.append("huge_y_range")
    elif y_range >= 260:
        risk += 6
        flags.append("wide_y_range")

    if n >= 18 and density >= 0.45 and speed_max < 75 and accel_p95 < 60:
        risk -= 12
        flags.append("stable_track_bonus")

    if x_range <= 90 and y_range <= 90 and n >= 16 and density >= 0.45:
        risk -= 8
        flags.append("compact_track_bonus")

    risk = max(0, min(100, int(round(risk))))
    smoothness = max(0, min(100, 100 - risk))

    if risk >= 70:
        guess = "high_risk"
    elif risk >= 45:
        guess = "medium_risk"
    elif risk >= 25:
        guess = "check"
    else:
        guess = "plausible"

    if not flags:
        flags.append("none")

    return {
        "trajectory_found": "1",
        "traj_n_points": str(n),
        "traj_first_frame": str(first),
        "traj_last_frame": str(last),
        "traj_density": fmt(density, 4),
        "gap_count": str(gap_count),
        "big_gap_count": str(big_gap_count),
        "max_frame_gap": str(max_gap),
        "speed_median": fmt(speed_median, 3),
        "speed_p95": fmt(speed_p95, 3),
        "speed_max": fmt(speed_max, 3),
        "speed_p95_over_median": fmt(ratio if speed_median > 0 else 0.0, 3),
        "high_speed_count": str(high_speed_count),
        "teleport_count": str(teleport_count),
        "accel_p95": fmt(accel_p95, 3),
        "accel_max": fmt(accel_max, 3),
        "heading_p95": fmt(heading_p95, 2),
        "heading_max": fmt(heading_max, 2),
        "medium_turn_count": str(medium_turn_count),
        "hard_turn_count": str(hard_turn_count),
        "traj_x_range": fmt(x_range, 2),
        "traj_y_range": fmt(y_range, 2),
        "risk_score_001G": str(risk),
        "smoothness_score_001G": str(smoothness),
        "review_guess_001G": guess,
        "risk_flags_001G": ",".join(flags),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "triage_rank",
        "review_id",
        "clip_id",
        "segment_idx",
        "segment_name",
        "first_frame",
        "last_frame",
        "n_points",
        "density",
        "travel",
        "x_range",
        "y_range",
        "review_guess",
        "risk_flags",
        "risk_score_001G",
        "smoothness_score_001G",
        "review_guess_001G",
        "risk_flags_001G",
        "trajectory_found",
        "traj_n_points",
        "traj_density",
        "gap_count",
        "big_gap_count",
        "max_frame_gap",
        "speed_median",
        "speed_p95",
        "speed_max",
        "speed_p95_over_median",
        "teleport_count",
        "accel_p95",
        "accel_max",
        "heading_p95",
        "heading_max",
        "hard_turn_count",
        "traj_x_range",
        "traj_y_range",
        "mp4",
        "csv",
        "review_status",
        "notes",
    ]

    seen = set()
    columns = []

    for col in preferred:
        if col not in seen:
            columns.append(col)
            seen.add(col)

    for row in rows:
        for col in row.keys():
            if col not in seen:
                columns.append(col)
                seen.add(col)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def html_link(path_text: Any, label: str) -> str:
    text = clean_first_path(path_text)
    if not text:
        return ""

    href = html.escape(text.replace("\\", "/"))
    return f'<a href="{href}">{html.escape(label)}</a>'


def write_html(path: Path, rows: list[dict[str, str]], csv_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for row in rows:
        guess = row.get("review_guess_001G", "unknown")
        counts[guess] = counts.get(guess, 0) + 1

    counts_text = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items()))

    trs = []
    for row in rows:
        guess = row.get("review_guess_001G", "")
        cls = guess.replace("_", "-")

        trs.append(
            f"""
            <tr class="{html.escape(cls)}">
              <td>{html.escape(row.get("triage_rank", ""))}</td>
              <td>{html.escape(row.get("review_id", ""))}</td>
              <td><code>{html.escape(row.get("clip_id", ""))}</code></td>
              <td>{html.escape(row.get("segment_idx", ""))}</td>
              <td><code>{html.escape(row.get("segment_name", ""))}</code></td>
              <td>{html.escape(row.get("first_frame", ""))} → {html.escape(row.get("last_frame", ""))}</td>
              <td>{html.escape(row.get("risk_score_001G", ""))}</td>
              <td>{html.escape(row.get("smoothness_score_001G", ""))}</td>
              <td class="guess">{html.escape(row.get("review_guess_001G", ""))}</td>
              <td class="flags">{html.escape(row.get("risk_flags_001G", ""))}</td>
              <td>{html.escape(row.get("traj_density", ""))}</td>
              <td>{html.escape(row.get("max_frame_gap", ""))}</td>
              <td>{html.escape(row.get("speed_median", ""))} / {html.escape(row.get("speed_p95", ""))} / {html.escape(row.get("speed_max", ""))}</td>
              <td>{html.escape(row.get("accel_p95", ""))} / {html.escape(row.get("accel_max", ""))}</td>
              <td>{html.escape(row.get("hard_turn_count", ""))}</td>
              <td>{html_link(row.get("mp4", ""), "mp4")}</td>
              <td>{html_link(row.get("csv", ""), "csv")}</td>
              <td class="manual"></td>
              <td></td>
            </tr>
            """
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux review manifest 001G trajectory</title>
<style>
  :root {{
    --bg: #101218;
    --panel: #181b22;
    --line: #2b303b;
    --text: #edf0f7;
    --muted: #9ea7b8;
    --accent: #83d4ff;
  }}
  body {{
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  h1 {{
    margin: 0 0 8px;
    font-size: 24px;
  }}
  .meta {{
    margin-bottom: 20px;
    color: var(--muted);
  }}
  a {{
    color: var(--accent);
    text-decoration: none;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    border: 1px solid var(--line);
  }}
  th, td {{
    border-bottom: 1px solid var(--line);
    padding: 8px 10px;
    vertical-align: top;
    font-size: 13px;
  }}
  th {{
    position: sticky;
    top: 0;
    background: #20242e;
    z-index: 2;
    text-align: left;
  }}
  code {{
    color: #d7e4ff;
    font-size: 12px;
  }}
  tr.high-risk td {{
    background: rgba(255, 80, 80, 0.10);
  }}
  tr.medium-risk td {{
    background: rgba(255, 200, 80, 0.08);
  }}
  tr.check td {{
    background: rgba(100, 160, 255, 0.06);
  }}
  tr.plausible td {{
    background: rgba(100, 255, 150, 0.035);
  }}
  tr.missing td {{
    background: rgba(255, 80, 80, 0.16);
  }}
  .guess {{
    font-weight: 700;
    white-space: nowrap;
  }}
  .flags {{
    color: var(--muted);
    max-width: 320px;
  }}
  .manual {{
    min-width: 90px;
  }}
</style>
</head>
<body>
  <h1>TTFlux review manifest 001G trajectory</h1>
  <div class="meta">
    Segments : {len(rows)} · {html.escape(counts_text)} · trié par risque décroissant · CSV :
    <a href="{html.escape(csv_name)}">{html.escape(csv_name)}</a>
  </div>

  <table>
    <thead>
      <tr>
        <th>rank</th>
        <th>ID</th>
        <th>clip_id</th>
        <th>seg</th>
        <th>segment_name</th>
        <th>frames</th>
        <th>risk</th>
        <th>smooth</th>
        <th>guess</th>
        <th>flags</th>
        <th>density</th>
        <th>max gap</th>
        <th>speed med/p95/max</th>
        <th>accel p95/max</th>
        <th>hard turns</th>
        <th>mp4</th>
        <th>csv</th>
        <th>manual</th>
        <th>notes</th>
      </tr>
    </thead>
    <tbody>
      {''.join(trs)}
    </tbody>
  </table>
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def score_manifest(run_dir: Path, manifest: Path, out_csv: Path, out_html: Path | None) -> None:
    run_dir = run_dir.resolve()
    rows = read_manifest(manifest)

    enriched = []

    for row in rows:
        row = {str(k): "" if v is None else str(v) for k, v in row.items()}

        segment_csv = resolve_segment_csv(run_dir, row)

        if segment_csv is None or not segment_csv.exists():
            stats = {
                "trajectory_found": "0",
                "risk_score_001G": "100",
                "smoothness_score_001G": "0",
                "review_guess_001G": "missing",
                "risk_flags_001G": "missing_csv",
            }
        else:
            points = load_points(segment_csv)
            stats = trajectory_stats(points)

        row.update(stats)
        enriched.append(row)

    def sort_key(row: dict[str, str]) -> tuple[int, str, int, int]:
        risk = safe_int(row.get("risk_score_001G"), 100)
        clip_id = row.get("clip_id", "")
        seg_idx = safe_int(row.get("segment_idx"), 0)
        first = safe_int(row.get("first_frame"), 0)
        return -risk, clip_id, seg_idx, first

    enriched.sort(key=sort_key)

    for i, row in enumerate(enriched, start=1):
        row["triage_rank"] = str(i)

    write_csv(out_csv, enriched)

    if out_html:
        write_html(out_html, enriched, out_csv.name)

    counts: dict[str, int] = {}
    found = 0
    for row in enriched:
        if row.get("trajectory_found") == "1":
            found += 1
        guess = row.get("review_guess_001G", "unknown")
        counts[guess] = counts.get(guess, 0) + 1

    print(f"[001G] run_dir          : {run_dir}")
    print(f"[001G] manifest         : {manifest}")
    print(f"[001G] segments         : {len(enriched)}")
    print(f"[001G] trajectory_found : {found}/{len(enriched)}")
    print(f"[001G] guesses          : {counts}")
    print(f"[001G] wrote CSV        : {out_csv}")

    if out_html:
        print(f"[001G] wrote HTML       : {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--manifest", default="runs/batch_001E/review_manifest_001F2_repaired.csv")
    parser.add_argument("--out-csv", default="runs/batch_001E/review_manifest_001G_trajectory.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/review_manifest_001G_trajectory.html")
    args = parser.parse_args()

    score_manifest(
        run_dir=Path(args.run_dir),
        manifest=Path(args.manifest),
        out_csv=Path(args.out_csv),
        out_html=Path(args.out_html) if args.out_html else None,
    )


if __name__ == "__main__":
    main()
    