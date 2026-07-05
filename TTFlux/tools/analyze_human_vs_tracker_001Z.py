from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cols: list[str] = []
    seen = set()

    for row in rows:
        for k in row:
            if k not in seen:
                cols.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def parse_dt(value: str) -> float:
    value = str(value or "").strip()

    if not value:
        return 0.0

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def merge_human_exports(input_dir: Path, pattern: str) -> tuple[list[dict[str, str]], list[Path]]:
    files = sorted(input_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

    merged: dict[tuple[str, int], dict[str, str]] = {}

    for file_index, path in enumerate(files):
        rows = read_csv(path)

        for row in rows:
            rid = row.get("review_id", "").strip()
            frame = fnum(row.get("frame"), None)

            if not rid or frame is None:
                continue

            key = (rid, int(round(frame)))

            item = dict(row)
            item["_source_file"] = str(path)
            item["_source_file_index"] = str(file_index)
            item["_source_mtime"] = str(path.stat().st_mtime)

            previous = merged.get(key)

            if previous is None:
                merged[key] = item
                continue

            t_new = parse_dt(item.get("updated_at", "")) or float(item["_source_mtime"])
            t_old = parse_dt(previous.get("updated_at", "")) or float(previous.get("_source_mtime", 0))

            if t_new >= t_old:
                merged[key] = item

    out = sorted(
        merged.values(),
        key=lambda r: (
            r.get("review_id", ""),
            int(round(fnum(r.get("frame"), 0) or 0)),
        ),
    )

    return out, files


def resolve_path(value: str, root: Path) -> Path | None:
    value = str(value or "").strip()
    if not value:
        return None

    p = Path(value)

    if p.is_absolute():
        return p if p.exists() else None

    p1 = root / p
    if p1.exists():
        return p1

    p2 = Path.cwd() / p
    if p2.exists():
        return p2

    return None


def load_auto_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    return {r.get("review_id", ""): r for r in rows if r.get("review_id", "")}


def load_track(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.exists():
        return []

    rows = read_csv(path)
    points = []

    for row in rows:
        frame = row.get("frame") or row.get("frame_idx") or row.get("f")
        x = row.get("x") or row.get("cx") or row.get("ball_x")
        y = row.get("y") or row.get("cy") or row.get("ball_y")

        frame_f = fnum(frame, None)
        x_f = fnum(x, None)
        y_f = fnum(y, None)

        if frame_f is None or x_f is None or y_f is None:
            continue

        points.append({
            "frame": float(frame_f),
            "x": float(x_f),
            "y": float(y_f),
        })

    return sorted(points, key=lambda p: p["frame"])


def interp_track(points: list[dict[str, float]], frame: float) -> tuple[float, float] | None:
    if not points:
        return None

    if frame < points[0]["frame"] or frame > points[-1]["frame"]:
        return None

    for p in points:
        if abs(p["frame"] - frame) < 1e-6:
            return p["x"], p["y"]

    lo = None
    hi = None

    for p in points:
        if p["frame"] <= frame:
            lo = p
        if p["frame"] >= frame:
            hi = p
            break

    if lo is None or hi is None:
        return None

    if abs(hi["frame"] - lo["frame"]) < 1e-6:
        return lo["x"], lo["y"]

    t = (frame - lo["frame"]) / (hi["frame"] - lo["frame"])

    x = lo["x"] + t * (hi["x"] - lo["x"])
    y = lo["y"] + t * (hi["y"] - lo["y"])

    return x, y


def median(values: list[float]) -> float | None:
    if not values:
        return None

    xs = sorted(values)
    n = len(xs)

    if n % 2:
        return xs[n // 2]

    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None

    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, round((len(xs) - 1) * q)))
    return xs[idx]


def fmt(value: Any, ndigits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def analyze_segment(
    rid: str,
    human_rows: list[dict[str, str]],
    manifest_row: dict[str, str] | None,
    run_dir: Path,
    offsets: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = len(human_rows)

    visible_rows = [
        r for r in human_rows
        if str(r.get("quality", "")).strip() == "visible"
        and str(r.get("visible", "")).strip() in ("1", "1.0", "true", "True")
        and fnum(r.get("x"), None) is not None
        and fnum(r.get("y"), None) is not None
    ]

    invisible_rows = [r for r in human_rows if str(r.get("quality", "")).strip() == "invisible"]
    uncertain_rows = [r for r in human_rows if str(r.get("quality", "")).strip() == "uncertain"]
    inferred_rows = [
        r for r in human_rows
        if str(r.get("quality", "")).strip() in ("occluded_inferred", "inferred")
    ]

    track_points: list[dict[str, float]] = []

    if manifest_row is not None:
        csv_path = resolve_path(manifest_row.get("csv", ""), run_dir)
        track_points = load_track(csv_path)

    offset_stats = []

    for off in offsets:
        distances = []

        for row in visible_rows:
            hf = fnum(row.get("frame"), None)
            hx = fnum(row.get("x"), None)
            hy = fnum(row.get("y"), None)

            if hf is None or hx is None or hy is None:
                continue

            auto = interp_track(track_points, hf + off)
            if auto is None:
                continue

            ax, ay = auto
            distances.append(math.hypot(ax - hx, ay - hy))

        offset_stats.append({
            "offset": off,
            "n": len(distances),
            "median": median(distances),
            "p80": percentile(distances, 0.80),
            "mean": sum(distances) / len(distances) if distances else None,
            "hit25": sum(1 for d in distances if d <= 25) / len(distances) if distances else None,
            "hit50": sum(1 for d in distances if d <= 50) / len(distances) if distances else None,
        })

    usable_stats = [s for s in offset_stats if s["n"] >= max(2, min(4, len(visible_rows))) and s["median"] is not None]
    best = min(usable_stats, key=lambda s: s["median"]) if usable_stats else None
    zero = next((s for s in offset_stats if s["offset"] == 0), None)

    if len(visible_rows) < 3:
        classification = "insufficient_visible_points"
    elif best is None:
        classification = "no_auto_comparison"
    elif best["median"] is not None and best["median"] <= 25 and (best["hit50"] or 0) >= 0.75:
        classification = "auto_track_good"
    elif best["median"] is not None and best["median"] <= 50:
        classification = "auto_track_usable_or_lagged"
    elif best["median"] is not None and best["median"] <= 90:
        classification = "auto_track_partial"
    else:
        classification = "auto_track_bad_or_wrong_object"

    frame_rows: list[dict[str, Any]] = []

    best_offset = int(best["offset"]) if best else 0

    for row in visible_rows:
        hf = fnum(row.get("frame"), None)
        hx = fnum(row.get("x"), None)
        hy = fnum(row.get("y"), None)

        if hf is None or hx is None or hy is None:
            continue

        auto0 = interp_track(track_points, hf)
        autob = interp_track(track_points, hf + best_offset)

        ax0 = ay0 = d0 = ""
        axb = ayb = db = ""

        if auto0 is not None:
            ax0, ay0 = auto0
            d0 = math.hypot(ax0 - hx, ay0 - hy)

        if autob is not None:
            axb, ayb = autob
            db = math.hypot(axb - hx, ayb - hy)

        frame_rows.append({
            "review_id": rid,
            "clip_id": row.get("clip_id", ""),
            "segment_name": row.get("segment_name", ""),
            "frame": int(round(hf)),
            "human_x": fmt(hx),
            "human_y": fmt(hy),
            "quality": row.get("quality", ""),
            "auto_x_offset0": fmt(ax0) if ax0 != "" else "",
            "auto_y_offset0": fmt(ay0) if ay0 != "" else "",
            "dist_offset0": fmt(d0) if d0 != "" else "",
            "best_offset_frames": best_offset,
            "auto_x_best": fmt(axb) if axb != "" else "",
            "auto_y_best": fmt(ayb) if ayb != "" else "",
            "dist_best": fmt(db) if db != "" else "",
        })

    summary = {
        "review_id": rid,
        "clip_id": human_rows[0].get("clip_id", "") if human_rows else "",
        "segment_name": human_rows[0].get("segment_name", "") if human_rows else "",
        "human_points_total": total,
        "human_visible": len(visible_rows),
        "human_invisible": len(invisible_rows),
        "human_uncertain": len(uncertain_rows),
        "human_inferred": len(inferred_rows),
        "auto_points": len(track_points),
        "best_offset_frames": best["offset"] if best else "",
        "best_n": best["n"] if best else "",
        "best_median_dist": fmt(best["median"]) if best else "",
        "best_p80_dist": fmt(best["p80"]) if best else "",
        "best_hit25": fmt(best["hit25"], 3) if best and best["hit25"] is not None else "",
        "best_hit50": fmt(best["hit50"], 3) if best and best["hit50"] is not None else "",
        "zero_median_dist": fmt(zero["median"]) if zero else "",
        "zero_p80_dist": fmt(zero["p80"]) if zero else "",
        "classification_001Z": classification,
        "decision_001U": manifest_row.get("decision_001U", "") if manifest_row else "",
        "decision_reason_001U": manifest_row.get("decision_reason_001U", "") if manifest_row else "",
        "risk_score_001G": manifest_row.get("risk_score_001G", "") if manifest_row else "",
        "center_blob_false_score_001N": manifest_row.get("center_blob_false_score_001N", "") if manifest_row else "",
        "micro_keep_score_001O": manifest_row.get("micro_keep_score_001O", "") if manifest_row else "",
        "micro_false_score_001O": manifest_row.get("micro_false_score_001O", "") if manifest_row else "",
    }

    return summary, frame_rows


def write_html(path: Path, summaries: list[dict[str, Any]], files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows_html = []

    cls_map = {
        "auto_track_good": "good",
        "auto_track_usable_or_lagged": "usable",
        "auto_track_partial": "partial",
        "auto_track_bad_or_wrong_object": "bad",
        "insufficient_visible_points": "unclear",
        "no_auto_comparison": "unclear",
    }

    for row in summaries:
        cls = cls_map.get(row.get("classification_001Z", ""), "")

        rows_html.append(f"""
<tr class="{cls}">
<td>{html.escape(str(row.get("review_id", "")))}</td>
<td><code>{html.escape(str(row.get("segment_name", "")))}</code></td>
<td>{html.escape(str(row.get("human_visible", "")))}/{html.escape(str(row.get("human_points_total", "")))}</td>
<td>{html.escape(str(row.get("best_offset_frames", "")))}</td>
<td>{html.escape(str(row.get("best_median_dist", "")))}</td>
<td>{html.escape(str(row.get("best_p80_dist", "")))}</td>
<td>{html.escape(str(row.get("best_hit50", "")))}</td>
<td>{html.escape(str(row.get("zero_median_dist", "")))}</td>
<td>{html.escape(str(row.get("classification_001Z", "")))}</td>
<td>{html.escape(str(row.get("decision_reason_001U", "")))}</td>
</tr>
""")

    file_items = "".join(f"<li><code>{html.escape(str(p))}</code></li>" for p in files)

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux human tracker analysis 001Z</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
th,td{{border-bottom:1px solid #2b303b;padding:7px 8px;vertical-align:top;font-size:13px}}
th{{background:#20242e;text-align:left;position:sticky;top:0}}
.good td{{background:rgba(100,255,150,.055)}}
.usable td{{background:rgba(150,210,255,.050)}}
.partial td{{background:rgba(255,200,80,.060)}}
.bad td{{background:rgba(255,80,80,.080)}}
.unclear td{{background:rgba(150,170,255,.050)}}
code{{color:#dce4ff}}
.muted{{color:#9ea7b8}}
</style>
</head>
<body>
<h1>TTFlux human tracker analysis 001Z</h1>

<section>
<h2>Sources fusionnées</h2>
<ul>{file_items}</ul>
</section>

<section>
<h2>Résumé segment par segment</h2>
<table>
<thead>
<tr>
<th>ID</th>
<th>segment</th>
<th>visible/total humain</th>
<th>best offset</th>
<th>median dist best</th>
<th>p80 dist best</th>
<th>hit50 best</th>
<th>median dist offset0</th>
<th>classification 001Z</th>
<th>raison 001U</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</section>

<section>
<h2>Lecture rapide</h2>
<p class="muted">
Distance en pixels entre ta trajectoire humaine et la trajectoire automatique.
<code>best offset</code> teste les décalages temporels entre -8 et +8 frames.
Si <code>median dist best</code> est nettement meilleur que <code>median dist offset0</code>, le tracker est probablement décalé temporellement.
</p>
</section>
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-dir", default="runs/humain_review")
    parser.add_argument("--pattern", default="ball_trace_annotations_001Y*.csv")
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--arbiter-csv", default="runs/batch_001E/arbiter_001V/arbiter_001U_all.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/human_analysis_001Z")
    args = parser.parse_args()

    human_dir = Path(args.human_dir)
    run_dir = Path(args.run_dir).resolve()
    arbiter_csv = Path(args.arbiter_csv)
    out_dir = Path(args.out_dir)

    human_rows, files = merge_human_exports(human_dir, args.pattern)

    if not files:
        raise SystemExit(f"[001Z] Aucun CSV trouvé dans {human_dir} avec pattern {args.pattern}")

    if not human_rows:
        raise SystemExit("[001Z] CSV trouvés, mais aucune annotation exploitable.")

    manifest = load_auto_manifest(arbiter_csv)

    grouped: dict[str, list[dict[str, str]]] = {}

    for row in human_rows:
        rid = row.get("review_id", "")
        grouped.setdefault(rid, []).append(row)

    offsets = list(range(-8, 9))

    summaries = []
    all_frame_rows = []

    for rid, rows in sorted(grouped.items()):
        summary, frame_rows = analyze_segment(
            rid=rid,
            human_rows=rows,
            manifest_row=manifest.get(rid),
            run_dir=run_dir,
            offsets=offsets,
        )

        summaries.append(summary)
        all_frame_rows.extend(frame_rows)

    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "human_annotations_001Z_merged.csv", human_rows)
    write_csv(out_dir / "human_tracker_summary_001Z.csv", summaries)
    write_csv(out_dir / "human_tracker_frame_errors_001Z.csv", all_frame_rows)

    summary_json = {
        "input_files": [str(p) for p in files],
        "human_rows_merged": len(human_rows),
        "segments": len(summaries),
        "classifications": {},
    }

    for row in summaries:
        c = row.get("classification_001Z", "")
        summary_json["classifications"][c] = summary_json["classifications"].get(c, 0) + 1

    (out_dir / "human_tracker_summary_001Z.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_html(out_dir / "human_tracker_analysis_001Z.html", summaries, files)

    print(f"[001Z] input files       : {len(files)}")
    print(f"[001Z] human rows merged : {len(human_rows)}")
    print(f"[001Z] segments          : {len(summaries)}")
    print(f"[001Z] classifications   : {summary_json['classifications']}")
    print(f"[001Z] out dir           : {out_dir}")
    print(f"[001Z] html              : {out_dir / 'human_tracker_analysis_001Z.html'}")


if __name__ == "__main__":
    main()
