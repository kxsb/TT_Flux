from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FEATURES = [
    "has_table_model",
    "table_keyframe_distance_frames",
    "point_inside_table_quad_ratio",
    "point_distance_to_table_px_median",
    "point_distance_to_table_px_p90",
    "point_distance_to_net_px_median",
    "u_min",
    "u_max",
    "v_min",
    "v_max",
    "u_range",
    "v_range",
    "trajectory_crosses_net_line",
    "trajectory_crosses_table_bounds",
    "trajectory_side_flips",
    "trajectory_side_consistency",
    "below_table_risk",
    "near_table_edge_ratio",
    "far_from_table_ratio",
    "table_context_score_003B2",
]


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value or "").strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in csv.DictReader(f, dialect=dialect)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "review_id",
        "target_class",
        "clip_id_003B2",
        "segment_name",
        "final_decision_002B",
        "human_decision_002A",
        "track_csv_resolved_003B2",
        *FEATURES,
        "table_model_source_003B2",
        "table_model_reason_003B2",
    ]

    cols = []
    seen = set()

    for c in preferred:
        cols.append(c)
        seen.add(c)

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


def median(values: list[float]) -> float:
    if not values:
        return 0.0

    xs = sorted(values)
    n = len(xs)
    mid = n // 2

    if n % 2:
        return xs[mid]

    return (xs[mid - 1] + xs[mid]) / 2.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0

    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return xs[lo]

    t = pos - lo
    return xs[lo] * (1 - t) + xs[hi] * t


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def resolve_existing_path(value: str, roots: list[Path]) -> Path | None:
    value = str(value or "").strip().strip('"')

    if not value:
        return None

    p = Path(value)

    if p.is_absolute() and p.exists():
        return p

    for root in roots:
        q = root / p
        if q.exists():
            return q

    return None


def find_track_csv(row: dict[str, str], roots: list[Path], run_dir: Path) -> Path | None:
    for col in [
        "csv",
        "csv_path",
        "segment_csv",
        "validation_csv",
        "track_csv",
        "track_csv_resolved",
        "track_csv_resolved_001T2",
        "track_csv_resolved_002G2",
        "csv_resolved",
    ]:
        p = resolve_existing_path(row.get(col, ""), roots)

        if p is not None:
            return p

    segment_name = row.get("segment_name", "").strip()

    if not segment_name:
        return None

    hits = [p for p in run_dir.rglob(f"{segment_name}.csv") if p.is_file()]

    if hits:
        return sorted(hits, key=lambda p: len(str(p)))[0]

    hits = [
        p for p in run_dir.rglob("*.csv")
        if segment_name in p.name
    ]

    if hits:
        return sorted(hits, key=lambda p: len(str(p)))[0]

    return None


def load_track_points(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.exists():
        return []

    rows = read_csv(path)
    points = []

    for r in rows:
        frame = fnum(r.get("frame") or r.get("frame_idx") or r.get("f"), None)
        x = fnum(r.get("x") or r.get("cx") or r.get("ball_x"), None)
        y = fnum(r.get("y") or r.get("cy") or r.get("ball_y"), None)

        if frame is None or x is None or y is None:
            continue

        points.append({
            "frame": float(frame),
            "x": float(x),
            "y": float(y),
        })

    dedup: dict[int, dict[str, float]] = {}

    for p in points:
        dedup[int(round(p["frame"]))] = p

    return [dedup[k] for k in sorted(dedup)]


def infer_frame_range(row: dict[str, str]) -> tuple[int | None, int | None]:
    for a, b in [
        ("first_frame", "last_frame"),
        ("frame_first", "frame_last"),
        ("start_frame", "end_frame"),
    ]:
        if row.get(a) and row.get(b):
            try:
                return int(float(row[a])), int(float(row[b]))
            except Exception:
                pass

    blob = " ".join(str(v) for v in row.values())
    m = re.search(r"f(\d+)_to_f(\d+)", blob)

    if m:
        return int(m.group(1)), int(m.group(2))

    return None, None


def target_class(row: dict[str, str]) -> str:
    human = row.get("human_decision_002A", "")
    final = row.get("final_decision_002B", "")
    source = row.get("final_source_002B", "")

    if human == "reject_human_verified":
        return "human_reject"

    if human == "review_partial_human_verified":
        return "human_partial"

    if final == "keep":
        return "keep"

    if final == "reject" and source != "human_002A":
        return "auto_reject"

    if final == "review":
        return "auto_review"

    return "other"


def valid_point(p: Any) -> bool:
    return isinstance(p, dict) and p.get("x") is not None and p.get("y") is not None


def quad_complete(q: Any) -> bool:
    return isinstance(q, list) and len(q) >= 4 and all(valid_point(p) for p in q[:4])


def quad_np(q: list[dict[str, Any]]) -> np.ndarray:
    return np.array([[float(p["x"]), float(p["y"])] for p in q[:4]], dtype=np.float32)


def line_distance(pt: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    px, py = pt
    ax, ay = a
    bx, by = b

    vx = bx - ax
    vy = by - ay

    den = math.hypot(vx, vy)

    if den <= 1e-9:
        return math.hypot(px - ax, py - ay)

    return abs(vy * px - vx * py + bx * ay - by * ax) / den


def transform_uv_to_img(h_uv_to_img: np.ndarray, uv: tuple[float, float]) -> tuple[float, float]:
    arr = np.array([[[uv[0], uv[1]]]], dtype=np.float32)
    out = cv2.perspectiveTransform(arr, h_uv_to_img)[0][0]
    return float(out[0]), float(out[1])


def detect_clip_id(row: dict[str, str], track_csv: Path | None, table_models: dict[str, Any]) -> str:
    blob = " ".join(str(v) for v in row.values())

    if track_csv is not None:
        blob += " " + str(track_csv)

    for clip_id, model in table_models.items():
        filename = str(model.get("filename", ""))
        stem = Path(filename).stem

        if clip_id in blob or stem in blob:
            return clip_id

    return ""


def compute_table_features(points: list[dict[str, float]], row: dict[str, str], model: dict[str, Any] | None) -> dict[str, Any]:
    if not model or not model.get("has_table_model") or not quad_complete(model.get("table_quad")):
        return {
            "has_table_model": 0,
            "table_keyframe_distance_frames": "",
            "point_inside_table_quad_ratio": "",
            "point_distance_to_table_px_median": "",
            "point_distance_to_table_px_p90": "",
            "point_distance_to_net_px_median": "",
            "u_min": "",
            "u_max": "",
            "v_min": "",
            "v_max": "",
            "u_range": "",
            "v_range": "",
            "trajectory_crosses_net_line": "",
            "trajectory_crosses_table_bounds": "",
            "trajectory_side_flips": "",
            "trajectory_side_consistency": "",
            "below_table_risk": "",
            "near_table_edge_ratio": "",
            "far_from_table_ratio": "",
            "table_context_score_003B2": "",
        }

    if not points:
        return {
            "has_table_model": 1,
            "table_keyframe_distance_frames": "",
            "point_inside_table_quad_ratio": 0,
            "point_distance_to_table_px_median": "",
            "point_distance_to_table_px_p90": "",
            "point_distance_to_net_px_median": "",
            "u_min": "",
            "u_max": "",
            "v_min": "",
            "v_max": "",
            "u_range": "",
            "v_range": "",
            "trajectory_crosses_net_line": 0,
            "trajectory_crosses_table_bounds": "",
            "trajectory_side_flips": "",
            "trajectory_side_consistency": "",
            "below_table_risk": "",
            "near_table_edge_ratio": "",
            "far_from_table_ratio": "",
            "table_context_score_003B2": 0,
        }

    quad = model["table_quad"]
    src = quad_np(quad)
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    h_img_to_uv = cv2.getPerspectiveTransform(src, dst)
    h_uv_to_img = cv2.getPerspectiveTransform(dst, src)

    pts_arr = np.array([[[p["x"], p["y"]]] for p in points], dtype=np.float32)
    uv_arr = cv2.perspectiveTransform(pts_arr, h_img_to_uv).reshape(-1, 2)

    us = [float(x) for x in uv_arr[:, 0]]
    vs = [float(x) for x in uv_arr[:, 1]]

    contour = src.astype(np.float32)

    signed_distances = []
    outside_distances = []
    near_edge = []
    far_from_table = []
    inside = []

    for p in points:
        d = float(cv2.pointPolygonTest(contour, (float(p["x"]), float(p["y"])), True))
        signed_distances.append(d)
        outside_distances.append(max(0.0, -d))
        near_edge.append(1 if abs(d) <= 25.0 else 0)
        far_from_table.append(1 if max(0.0, -d) >= 90.0 else 0)
        inside.append(1 if d >= 0 else 0)

    net_a = transform_uv_to_img(h_uv_to_img, (0.0, 0.5))
    net_b = transform_uv_to_img(h_uv_to_img, (1.0, 0.5))

    net_distances = [
        line_distance((float(p["x"]), float(p["y"])), net_a, net_b)
        for p in points
    ]

    side_values = []
    for v in vs:
        if v < 0.5:
            side_values.append(-1)
        else:
            side_values.append(1)

    side_flips = 0
    for a, b in zip(side_values[:-1], side_values[1:]):
        if a != b:
            side_flips += 1

    if side_values:
        dominant = max(side_values.count(-1), side_values.count(1))
        side_consistency = dominant / len(side_values)
    else:
        side_consistency = 0.0

    net_cross = 0
    for a, b in zip(vs[:-1], vs[1:]):
        if (a - 0.5) * (b - 0.5) <= 0 and a != b:
            net_cross = 1
            break

    u_min = min(us)
    u_max = max(us)
    v_min = min(vs)
    v_max = max(vs)

    crosses_bounds = 1 if (u_min < 0 or u_max > 1 or v_min < 0 or v_max > 1) else 0

    below_table_risk = sum(1 for v in vs if v < -0.10) / max(1, len(vs))

    f1, f2 = infer_frame_range(row)
    mid = None
    if f1 is not None and f2 is not None:
        mid = int(round((f1 + f2) / 2))

    frame_ref = int(float(model.get("frame_ref") or 0))
    keyframe_distance = abs(mid - frame_ref) if mid is not None else ""

    inside_ratio = mean([float(x) for x in inside])
    near_edge_ratio = mean([float(x) for x in near_edge])
    far_ratio = mean([float(x) for x in far_from_table])

    # Score souple : pas une vérité physique, juste une feature synthétique lisible.
    score = (
        0.45 * inside_ratio
        + 0.20 * (1.0 - min(1.0, far_ratio))
        + 0.15 * (1.0 - min(1.0, below_table_risk))
        + 0.10 * min(1.0, side_consistency)
        + 0.10 * (1.0 if net_cross else 0.5)
    )

    return {
        "has_table_model": 1,
        "table_keyframe_distance_frames": keyframe_distance,
        "point_inside_table_quad_ratio": round(inside_ratio, 6),
        "point_distance_to_table_px_median": round(median(outside_distances), 3),
        "point_distance_to_table_px_p90": round(percentile(outside_distances, 0.90), 3),
        "point_distance_to_net_px_median": round(median(net_distances), 3),
        "u_min": round(u_min, 6),
        "u_max": round(u_max, 6),
        "v_min": round(v_min, 6),
        "v_max": round(v_max, 6),
        "u_range": round(u_max - u_min, 6),
        "v_range": round(v_max - v_min, 6),
        "trajectory_crosses_net_line": int(net_cross),
        "trajectory_crosses_table_bounds": int(crosses_bounds),
        "trajectory_side_flips": int(side_flips),
        "trajectory_side_consistency": round(side_consistency, 6),
        "below_table_risk": round(below_table_risk, 6),
        "near_table_edge_ratio": round(near_edge_ratio, 6),
        "far_from_table_ratio": round(far_ratio, 6),
        "table_context_score_003B2": round(score, 6),
    }


def load_table_models(path: Path) -> dict[str, Any]:
    data = read_json(path)
    models = data.get("table_models", {})
    if not isinstance(models, dict):
        raise SystemExit("[003B2] table_models invalide.")
    return models


def build_rows(final_csv: Path, table_models: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(final_csv)
    roots = [Path.cwd(), run_dir, run_dir.resolve()]
    out = []

    for row in rows:
        track_csv = find_track_csv(row, roots, run_dir)
        points = load_track_points(track_csv)
        clip_id = detect_clip_id(row, track_csv, table_models)
        model = table_models.get(clip_id) if clip_id else None

        feats = compute_table_features(points, row, model)

        item: dict[str, Any] = dict(row)
        item.update(feats)

        item["target_class"] = target_class(row)
        item["clip_id_003B2"] = clip_id
        item["track_csv_resolved_003B2"] = str(track_csv) if track_csv else ""
        item["table_model_source_003B2"] = model.get("source", "") if isinstance(model, dict) else ""
        item["table_model_reason_003B2"] = model.get("reason_003B1B", "") if isinstance(model, dict) else ""

        out.append(item)

    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, int] = {}
    with_model = 0
    without_model = 0

    for r in rows:
        t = str(r.get("target_class", ""))
        by_target[t] = by_target.get(t, 0) + 1

        if int(fnum(r.get("has_table_model"), 0) or 0) == 1:
            with_model += 1
        else:
            without_model += 1

    means_by_target: dict[str, dict[str, float]] = {}

    for target in sorted(by_target):
        subset = [
            r for r in rows
            if r.get("target_class") == target
            and int(fnum(r.get("has_table_model"), 0) or 0) == 1
        ]

        means_by_target[target] = {}

        for f in FEATURES:
            vals = []
            for r in subset:
                x = fnum(r.get(f), None)
                if x is not None:
                    vals.append(float(x))

            means_by_target[target][f] = round(mean(vals), 6) if vals else 0.0

    missing_rows = [
        {
            "review_id": r.get("review_id", ""),
            "segment_name": r.get("segment_name", ""),
            "clip_id": r.get("clip_id_003B2", ""),
            "target_class": r.get("target_class", ""),
        }
        for r in rows
        if int(fnum(r.get("has_table_model"), 0) or 0) == 0
    ]

    return {
        "rows": len(rows),
        "segments_with_table_model": with_model,
        "segments_without_table_model": without_model,
        "by_target_class": dict(sorted(by_target.items())),
        "means_by_target_with_table_model": means_by_target,
        "missing_table_segments": missing_rows,
        "warning": "003B2 table_context uses curated table models only. Missing table models are kept as has_table_model=0.",
    }


def write_html(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["by_target_class"].items()
    )

    segment_rows = []

    for r in rows:
        model = int(fnum(r.get("has_table_model"), 0) or 0)
        cls = str(r.get("target_class", "")).replace("_", "-")
        if not model:
            cls += " missing-table"

        segment_rows.append(f"""
<tr class="{html.escape(cls)}">
<td>{html.escape(str(r.get("review_id", "")))}</td>
<td>{html.escape(str(r.get("target_class", "")))}</td>
<td>{html.escape(str(r.get("clip_id_003B2", "")))}</td>
<td><code>{html.escape(str(r.get("segment_name", "")))}</code></td>
<td>{html.escape(str(r.get("has_table_model", "")))}</td>
<td>{html.escape(str(r.get("point_inside_table_quad_ratio", "")))}</td>
<td>{html.escape(str(r.get("point_distance_to_table_px_median", "")))}</td>
<td>{html.escape(str(r.get("point_distance_to_net_px_median", "")))}</td>
<td>{html.escape(str(r.get("u_min", "")))}</td>
<td>{html.escape(str(r.get("u_max", "")))}</td>
<td>{html.escape(str(r.get("v_min", "")))}</td>
<td>{html.escape(str(r.get("v_max", "")))}</td>
<td>{html.escape(str(r.get("trajectory_crosses_net_line", "")))}</td>
<td>{html.escape(str(r.get("below_table_risk", "")))}</td>
<td>{html.escape(str(r.get("far_from_table_ratio", "")))}</td>
<td>{html.escape(str(r.get("table_context_score_003B2", "")))}</td>
</tr>
""")

    mean_sections = []

    for target, feats in summary["means_by_target_with_table_model"].items():
        body = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in feats.items()
        )

        mean_sections.append(f"""
<section>
<h2>Moyennes table_context · {html.escape(target)}</h2>
<table>
<thead><tr><th>feature</th><th>mean</th></tr></thead>
<tbody>{body}</tbody>
</table>
</section>
""")

    missing_body = "".join(
        f"<tr><td>{html.escape(str(r['review_id']))}</td><td>{html.escape(str(r['target_class']))}</td><td>{html.escape(str(r['clip_id']))}</td><td><code>{html.escape(str(r['segment_name']))}</code></td></tr>"
        for r in summary["missing_table_segments"]
    )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux table context 003B2</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}}
th{{background:#20242e;position:sticky;top:0}}
code,pre{{color:#dce4ff}}
pre{{white-space:pre-wrap;word-break:break-word;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.human-reject td{{background:rgba(255,80,80,.09)}}
.human-partial td{{background:rgba(255,200,80,.08)}}
.keep td{{background:rgba(100,255,150,.055)}}
.auto-reject td{{background:rgba(255,80,80,.045)}}
.auto-review td{{background:rgba(150,170,255,.045)}}
.missing-table td{{opacity:.55;background:rgba(255,200,80,.08)}}
.warn{{color:#ffd37a}}
</style>
</head>
<body>
<h1>TTFlux · table_context 003B2</h1>

<section>
<h2>Résumé</h2>
<table>
<tr><th>rows</th><td>{summary["rows"]}</td></tr>
<tr><th>segments_with_table_model</th><td>{summary["segments_with_table_model"]}</td></tr>
<tr><th>segments_without_table_model</th><td>{summary["segments_without_table_model"]}</td></tr>
<tr><th>warning</th><td class="warn">{html.escape(summary["warning"])}</td></tr>
</table>

<table>
<thead><tr><th>target</th><th>count</th></tr></thead>
<tbody>{summary_rows}</tbody>
</table>
</section>

<section>
<h2>Segments sans modèle table</h2>
<table>
<thead><tr><th>review</th><th>target</th><th>clip</th><th>segment</th></tr></thead>
<tbody>{missing_body}</tbody>
</table>
</section>

<section>
<h2>Segments · table_context</h2>
<table>
<thead>
<tr>
<th>ID</th><th>target</th><th>clip</th><th>segment</th><th>model</th>
<th>inside</th><th>dist table med</th><th>dist net med</th>
<th>u min</th><th>u max</th><th>v min</th><th>v max</th>
<th>cross net</th><th>below risk</th><th>far ratio</th><th>score</th>
</tr>
</thead>
<tbody>{''.join(segment_rows)}</tbody>
</table>
</section>

{''.join(mean_sections)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-csv", default="runs/batch_001E/final_decisions_002B/final_decisions_002B_all.csv")
    parser.add_argument("--table-models", default="runs/batch_001E/table_bootstrap_003B1B/table_models_curated_003B1B.json")
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--out-dir", default="runs/batch_001E/table_context_003B2")
    args = parser.parse_args()

    final_csv = Path(args.final_csv)
    table_models_path = Path(args.table_models)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_models = load_table_models(table_models_path)
    rows = build_rows(final_csv, table_models, run_dir)
    summary = summarize(rows)

    csv_path = out_dir / "table_context_features_003B2.csv"
    json_path = out_dir / "table_context_summary_003B2.json"
    html_path = out_dir / "table_context_003B2.html"

    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html(html_path, rows, summary)

    print(f"[003B2] rows                    : {summary['rows']}")
    print(f"[003B2] with table model        : {summary['segments_with_table_model']}")
    print(f"[003B2] without table model     : {summary['segments_without_table_model']}")
    print(f"[003B2] by target               : {summary['by_target_class']}")
    print(f"[003B2] out dir                 : {out_dir}")
    print(f"[003B2] html                    : {html_path}")


if __name__ == "__main__":
    main()
