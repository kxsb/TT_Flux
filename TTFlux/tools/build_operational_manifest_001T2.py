from __future__ import annotations

import argparse
import csv
import html
import math
import re
from pathlib import Path
from typing import Any


VALIDATION_RE = re.compile(
    r"^validation_(?P<idx>\d+)_f(?P<first>\d+)_to_f(?P<last>\d+)_n(?P<n>\d+)$"
)


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def read_points(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        rows: list[dict[str, float]] = []

        for row in reader:
            frame = row.get("frame") or row.get("frame_idx") or row.get("f")
            x = row.get("x") or row.get("cx") or row.get("ball_x")
            y = row.get("y") or row.get("cy") or row.get("ball_y")

            if frame is None or x is None or y is None:
                continue

            rows.append(
                {
                    "frame": fnum(frame),
                    "x": fnum(x),
                    "y": fnum(y),
                }
            )

        return sorted(rows, key=lambda r: r["frame"])


def rel_to(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def compute_metrics(points: list[dict[str, float]]) -> dict[str, str]:
    if not points:
        return {
            "first_frame": "",
            "last_frame": "",
            "n_points": "0",
            "density": "0",
            "travel": "0",
            "x_range": "0",
            "y_range": "0",
            "guess": "missing",
            "flags": "no_points",
        }

    frames = [p["frame"] for p in points]
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]

    first = int(min(frames))
    last = int(max(frames))
    n = len(points)
    span = max(1, last - first + 1)
    density = n / span

    travel = 0.0
    for a, b in zip(points, points[1:]):
        travel += math.hypot(b["x"] - a["x"], b["y"] - a["y"])

    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)

    flags = []

    if n >= 24:
        flags.append("many_points")
    if density < 0.30:
        flags.append("low_density")
    elif density < 0.42:
        flags.append("medium_density")
    if travel > 900:
        flags.append("high_travel")
    if x_range > 500:
        flags.append("huge_x_range")
    elif x_range > 250:
        flags.append("wide_x_range")
    if y_range > 350:
        flags.append("huge_y_range")
    elif y_range > 180:
        flags.append("wide_y_range")

    if "low_density" in flags or "huge_x_range" in flags or "huge_y_range" in flags:
        guess = "high_risk"
    elif flags:
        guess = "medium_risk"
    else:
        guess = "plausible"

    return {
        "first_frame": str(first),
        "last_frame": str(last),
        "n_points": str(n),
        "density": f"{density:.4f}",
        "travel": f"{travel:.2f}",
        "x_range": f"{x_range:.2f}",
        "y_range": f"{y_range:.2f}",
        "guess": guess,
        "flags": ",".join(flags),
    }


def build_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    clip_dirs = [
        p for p in run_dir.iterdir()
        if p.is_dir() and re.match(r"^\d{2}_", p.name)
    ]

    for clip_dir in sorted(clip_dirs, key=lambda p: p.name):
        csv_files = sorted(clip_dir.glob("validation_*.csv"))

        for csv_path in csv_files:
            stem = csv_path.stem
            m = VALIDATION_RE.match(stem)
            if not m:
                continue

            points = read_points(csv_path)
            metrics = compute_metrics(points)

            segment_idx = int(m.group("idx"))
            mp4_path = csv_path.with_suffix(".mp4")
            png_path = csv_path.with_suffix(".png")

            row = {
                "review_id": "",
                "clip_id": clip_dir.name,
                "segment_idx": str(segment_idx),
                "segment_name": stem,
                "first_frame": metrics["first_frame"],
                "last_frame": metrics["last_frame"],
                "n_points": metrics["n_points"],
                "density": metrics["density"],
                "travel": metrics["travel"],
                "x_range": metrics["x_range"],
                "y_range": metrics["y_range"],
                "guess": metrics["guess"],
                "flags": metrics["flags"],
                "mp4": rel_to(mp4_path, run_dir) if mp4_path.exists() else "",
                "overlay_png": rel_to(png_path, run_dir) if png_path.exists() else "",
                "csv": rel_to(csv_path, run_dir),
                "review": "",
                "notes": "",
            }

            rows.append(row)

    rows = sorted(
        rows,
        key=lambda r: (
            r["clip_id"],
            int(r["segment_idx"]),
            int(r["first_frame"] or 0),
        ),
    )

    for idx, row in enumerate(rows, start=1):
        row["review_id"] = f"R{idx:04d}"

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
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
        "guess",
        "flags",
        "mp4",
        "overlay_png",
        "csv",
        "review",
        "notes",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def write_html(path: Path, rows: list[dict[str, str]], out_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    body = []

    for row in rows:
        cls = html.escape(row.get("guess", ""))
        mp4 = row.get("mp4", "")
        csv_path = row.get("csv", "")

        mp4_link = f'<a href="{html.escape(mp4)}">mp4</a>' if mp4 else ""
        csv_link = f'<a href="{html.escape(csv_path)}">csv</a>' if csv_path else ""

        body.append(
            f"""
<tr class="{cls}">
<td>{html.escape(row["review_id"])}</td>
<td><code>{html.escape(row["clip_id"])}</code></td>
<td>{html.escape(row["segment_idx"])}</td>
<td><code>{html.escape(row["segment_name"])}</code></td>
<td>{html.escape(row["first_frame"])} → {html.escape(row["last_frame"])}</td>
<td>{html.escape(row["n_points"])}</td>
<td>{html.escape(row["density"])}</td>
<td>{html.escape(row["travel"])}</td>
<td>{html.escape(row["x_range"])} / {html.escape(row["y_range"])}</td>
<td>{html.escape(row["guess"])}</td>
<td>{html.escape(row["flags"])}</td>
<td>{mp4_link}</td>
<td>{csv_link}</td>
</tr>
"""
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux operational manifest 001T2</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
th,td{{border-bottom:1px solid #2b303b;padding:7px 8px;vertical-align:top;font-size:13px}}
th{{background:#20242e;text-align:left;position:sticky;top:0}}
.high_risk td,.high-risk td{{background:rgba(255,80,80,.08)}}
.medium_risk td,.medium-risk td{{background:rgba(255,200,80,.06)}}
.plausible td{{background:rgba(100,255,150,.045)}}
.muted{{color:#9ea7b8}}
</style>
</head>
<body>
<h1>TTFlux operational manifest 001T2</h1>
<section>
<p class="muted">
Manifeste propre : uniquement les CSV <code>validation_*.csv</code> dans les dossiers clips directs.
</p>
<p>Segments : <b>{len(rows)}</b> · CSV : <a href="{html.escape(out_csv.name)}">{html.escape(out_csv.name)}</a></p>
<table>
<thead>
<tr>
<th>ID</th><th>clip_id</th><th>segment</th><th>segment_name</th><th>frames</th><th>points</th>
<th>density</th><th>travel</th><th>x/y range</th><th>guess</th><th>flags</th><th>mp4</th><th>csv</th>
</tr>
</thead>
<tbody>
{''.join(body)}
</tbody>
</table>
</section>
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--out-csv", default="runs/batch_001E/operational_manifest_001T2.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/operational_manifest_001T2.html")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)

    rows = build_rows(run_dir)

    write_csv(out_csv, rows)
    write_html(out_html, rows, out_csv)

    print(f"[001T2] run_dir  : {run_dir}")
    print(f"[001T2] segments : {len(rows)}")
    print(f"[001T2] wrote CSV : {out_csv}")
    print(f"[001T2] wrote HTML: {out_html}")


if __name__ == "__main__":
    main()
