from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


VERSION = "004G1"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def parse_validation_name(name: str):
    # validation_01_f300_to_f379_n77.csv/mp4
    m = re.search(r"validation_(\d+)_f(\d+)_to_f(\d+)_n(\d+)", name)
    if not m:
        return None
    return {
        "segment_idx": int(m.group(1)),
        "first_frame_name": int(m.group(2)),
        "last_frame_name": int(m.group(3)),
        "n_name": int(m.group(4)),
    }


def read_points(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    cols = {c.lower(): c for c in df.columns}

    frame_col = cols.get("frame")
    x_col = cols.get("x")
    y_col = cols.get("y")

    if frame_col is None or x_col is None or y_col is None:
        return None

    return df


def metrics_from_csv(csv_path: Path, fallback: dict):
    df = read_points(csv_path)

    if df is None or df.empty:
        return {
            "first_frame": fallback.get("first_frame_name", ""),
            "last_frame": fallback.get("last_frame_name", ""),
            "n_points": fallback.get("n_name", 0),
            "density": 0.0,
            "travel": 0.0,
            "x_range": 0.0,
            "y_range": 0.0,
        }

    frame_col = next(c for c in df.columns if c.lower() == "frame")
    x_col = next(c for c in df.columns if c.lower() == "x")
    y_col = next(c for c in df.columns if c.lower() == "y")

    frames = pd.to_numeric(df[frame_col], errors="coerce").dropna().to_numpy()
    xs = pd.to_numeric(df[x_col], errors="coerce").dropna().to_numpy()
    ys = pd.to_numeric(df[y_col], errors="coerce").dropna().to_numpy()

    n = min(len(frames), len(xs), len(ys))
    if n <= 0:
        return {
            "first_frame": fallback.get("first_frame_name", ""),
            "last_frame": fallback.get("last_frame_name", ""),
            "n_points": fallback.get("n_name", 0),
            "density": 0.0,
            "travel": 0.0,
            "x_range": 0.0,
            "y_range": 0.0,
        }

    frames = frames[:n]
    xs = xs[:n]
    ys = ys[:n]

    first = int(np.min(frames))
    last = int(np.max(frames))
    span = max(1, last - first + 1)

    if n >= 2:
        travel = float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)))
    else:
        travel = 0.0

    return {
        "first_frame": first,
        "last_frame": last,
        "n_points": int(n),
        "density": round(float(n / span), 6),
        "travel": round(travel, 3),
        "x_range": round(float(np.max(xs) - np.min(xs)), 3),
        "y_range": round(float(np.max(ys) - np.min(ys)), 3),
    }


def guess_segment(m):
    n = int(m.get("n_points") or 0)
    density = float(m.get("density") or 0)
    travel = float(m.get("travel") or 0)
    xr = float(m.get("x_range") or 0)
    yr = float(m.get("y_range") or 0)

    flags = []

    if n < 8:
        flags.append("few_points")
    if density < 0.20:
        flags.append("sparse")
    if travel < 80:
        flags.append("low_travel")
    if travel > 3500:
        flags.append("high_travel")
    if xr < 20 and yr < 20:
        flags.append("static_like")
    if xr > 900 or yr > 550:
        flags.append("large_jump_range")

    if n >= 20 and 0.25 <= density <= 1.05 and 120 <= travel <= 2600:
        guess = "plausible"
    elif n >= 12 and density >= 0.15:
        guess = "check"
    else:
        guess = "high_risk"

    return guess, ",".join(flags)


def write_html(path: Path, rows: list[dict], summary: dict):
    trs = []
    for r in rows:
        cls = r.get("guess", "check")
        mp4 = r.get("mp4", "")
        video = ""
        if mp4:
            video = f"<video controls preload='metadata' width='260' src='{esc(mp4)}'></video>"

        trs.append(
            f"<tr class='{esc(cls)}'>"
            f"<td>{esc(r.get('review_id'))}</td>"
            f"<td>{esc(r.get('clip_id'))}</td>"
            f"<td>{esc(r.get('segment_idx'))}</td>"
            f"<td>{esc(r.get('first_frame'))}</td>"
            f"<td>{esc(r.get('last_frame'))}</td>"
            f"<td>{esc(r.get('n_points'))}</td>"
            f"<td>{esc(r.get('density'))}</td>"
            f"<td>{esc(r.get('travel'))}</td>"
            f"<td>{esc(r.get('x_range'))}</td>"
            f"<td>{esc(r.get('y_range'))}</td>"
            f"<td>{esc(r.get('guess'))}</td>"
            f"<td>{esc(r.get('flags'))}</td>"
            f"<td>{video}</td>"
            f"<td>{esc(r.get('csv'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004G1 operational manifest rebuilt</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:78vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
tr.plausible td{{background:rgba(116,217,159,.07)}}
tr.check td{{background:rgba(255,200,80,.08)}}
tr.high_risk td{{background:rgba(255,80,80,.10)}}
video{{border-radius:8px;background:#000}}
</style>
</head>
<body>
<h1>TTFlux · 004G1 operational manifest rebuilt</h1>
<section><h2>Résumé</h2><pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre></section>
<section>
<h2>Segments</h2>
<div class="wrap">
<table>
<thead>
<tr>
<th>review</th><th>clip</th><th>seg</th><th>first</th><th>last</th><th>n</th><th>density</th><th>travel</th><th>x_range</th><th>y_range</th><th>guess</th><th>flags</th><th>preview</th><th>csv</th>
</tr>
</thead>
<tbody>
{''.join(trs)}
</tbody>
</table>
</div>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/batch_004F_full240")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--out-html", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    root = Path.cwd()
    run = Path(args.run_dir)
    if not run.is_absolute():
        run = root / run

    if not run.is_dir():
        raise SystemExit(f"Run dir absent: {run}")

    out_csv = Path(args.out_csv) if args.out_csv else run / "operational_manifest_001T2.csv"
    out_html = Path(args.out_html) if args.out_html else run / "operational_manifest_001T2.html"
    out_json = Path(args.out_json) if args.out_json else run / "operational_manifest_004G1_summary.json"

    if not out_csv.is_absolute():
        out_csv = root / out_csv
    if not out_html.is_absolute():
        out_html = root / out_html
    if not out_json.is_absolute():
        out_json = root / out_json

    rows = []

    item_dirs = sorted([p for p in run.iterdir() if p.is_dir() and p.name[:3].isdigit()])

    review_i = 1

    for d in item_dirs:
        clip_id = d.name[4:] if len(d.name) > 4 and d.name[3] == "_" else d.name

        for csv_path in sorted(d.glob("validation_*.csv")):
            parsed = parse_validation_name(csv_path.name)
            if not parsed:
                continue

            mp4_path = csv_path.with_suffix(".mp4")
            if not mp4_path.is_file():
                continue

            m = metrics_from_csv(csv_path, parsed)
            guess, flags = guess_segment(m)

            review_id = f"R{review_i:04d}"
            review_i += 1

            row = {
                "review_id": review_id,
                "clip_id": clip_id,
                "segment_idx": parsed["segment_idx"],
                "segment_name": csv_path.stem,
                "first_frame": m["first_frame"],
                "last_frame": m["last_frame"],
                "n_points": m["n_points"],
                "density": m["density"],
                "travel": m["travel"],
                "x_range": m["x_range"],
                "y_range": m["y_range"],
                "guess": guess,
                "flags": flags,
                "mp4": rel(mp4_path, run),
                "overlay_png": "",
                "csv": rel(csv_path, run),
                "review": "",
                "notes": "",
            }
            rows.append(row)

    by_guess = {}
    for r in rows:
        by_guess[r["guess"]] = by_guess.get(r["guess"], 0) + 1

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "rebuild_operational_manifest_from_validation_outputs",
        "run_dir": str(run),
        "item_dirs": len(item_dirs),
        "segments": len(rows),
        "by_guess": by_guess,
        "out_csv": str(out_csv),
        "out_html": str(out_html),
    }

    fieldnames = [
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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, rows, summary)

    print("004G1 status=OK")
    print("item_dirs=", len(item_dirs))
    print("segments=", len(rows))
    print("by_guess=", json.dumps(by_guess, ensure_ascii=False))
    print("wrote", out_csv)
    print("wrote", out_html)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
