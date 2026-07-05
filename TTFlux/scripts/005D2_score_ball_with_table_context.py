from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005D2"

TABLE_LENGTH_M = 2.740
TABLE_WIDTH_M = 1.525
TABLE_HALF_L = TABLE_LENGTH_M / 2.0
TABLE_HALF_W = TABLE_WIDTH_M / 2.0

CANON_W = 520
CANON_H = 936


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def imwrite_unicode(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def find_col(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise SystemExit(f"missing column among {names}")


def xy_m_to_px(x, y, w=CANON_W, h=CANON_H):
    px = int(round((float(x) + TABLE_HALF_W) / TABLE_WIDTH_M * (w - 1)))
    py = int(round((float(y) + TABLE_HALF_L) / TABLE_LENGTH_M * (h - 1)))
    return px, py


def draw_topdown_base():
    img = np.zeros((CANON_H, CANON_W, 3), dtype=np.uint8)
    img[:, :] = (24, 74, 58)

    cv2.rectangle(img, (0, 0), (CANON_W - 1, CANON_H - 1), (235, 235, 235), 3, cv2.LINE_AA)

    net_y = xy_m_to_px(0, 0)[1]
    cv2.line(img, (0, net_y), (CANON_W - 1, net_y), (40, 210, 255), 3, cv2.LINE_AA)

    center_x = xy_m_to_px(0, 0)[0]
    cv2.line(img, (center_x, 0), (center_x, CANON_H - 1), (255, 255, 80), 2, cv2.LINE_AA)

    for x in np.linspace(-TABLE_HALF_W, TABLE_HALF_W, 5):
        px, _ = xy_m_to_px(x, 0)
        cv2.line(img, (px, 0), (px, CANON_H - 1), (72, 118, 100), 1, cv2.LINE_AA)

    for y in np.linspace(-TABLE_HALF_L, TABLE_HALF_L, 7):
        _, py = xy_m_to_px(0, y)
        cv2.line(img, (0, py), (CANON_W - 1, py), (72, 118, 100), 1, cv2.LINE_AA)

    cv2.putText(img, "FAR SIDE", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(img, "NEAR SIDE", (14, CANON_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 2, cv2.LINE_AA)

    return img


def table_distance_outside(x: float, y: float):
    dx = max(0.0, abs(float(x)) - TABLE_HALF_W)
    dy = max(0.0, abs(float(y)) - TABLE_HALF_L)
    return math.hypot(dx, dy), dx, dy


def score_group(g: pd.DataFrame, frame_col: str):
    g = g.copy()
    g[frame_col] = to_num(g[frame_col]).fillna(-1).astype(int)

    g["table_project_ok_005C9B"] = to_num(g["table_project_ok_005C9B"]).fillna(0).astype(int)
    g["ball_inside_table_005C9B"] = to_num(g["ball_inside_table_005C9B"]).fillna(0).astype(int)
    g["table_gate_keep_005D1"] = to_num(g["table_gate_keep_005D1"]).fillna(1).astype(int)

    g["ball_table_x_m_005C9B"] = to_num(g["ball_table_x_m_005C9B"])
    g["ball_table_y_m_005C9B"] = to_num(g["ball_table_y_m_005C9B"])

    g = g.sort_values(frame_col).reset_index(drop=True)

    frames = g[frame_col].to_numpy(dtype=int)

    coords = []
    for _, r in g.iterrows():
        if int(r["table_project_ok_005C9B"]) == 1 and not pd.isna(r["ball_table_x_m_005C9B"]) and not pd.isna(r["ball_table_y_m_005C9B"]):
            coords.append((float(r["ball_table_x_m_005C9B"]), float(r["ball_table_y_m_005C9B"])))
        else:
            coords.append(None)

    scores = []
    zones = []
    dist_outs = []
    speed_prev = []
    speed_next = []
    split_flags = []
    tracklet_ids = []

    tracklet_id = 0

    for i, r in g.iterrows():
        c = coords[i]
        project_ok = int(r["table_project_ok_005C9B"]) == 1
        gate_keep = int(r["table_gate_keep_005D1"]) == 1

        if not project_ok or c is None:
            score = 0.42
            zone = "NO_METRIC_PROJECTION"
            dist_out = np.nan
        else:
            x, y = c
            d, dx, dy = table_distance_outside(x, y)
            dist_out = d

            inside = abs(x) <= TABLE_HALF_W and abs(y) <= TABLE_HALF_L
            near = abs(x) <= TABLE_HALF_W + 0.85 and abs(y) <= TABLE_HALF_L + 1.25
            far = abs(x) > TABLE_HALF_W + 1.80 or abs(y) > TABLE_HALF_L + 2.70

            if inside:
                zone = "INSIDE_TABLE"
                score = 0.92
            elif near:
                zone = "NEAR_TABLE_SCENE"
                score = 0.67
            elif far:
                zone = "FAR_OUTSIDE_TABLE_SCENE"
                score = 0.05
            else:
                zone = "OUTSIDE_SOFT_SCENE"
                score = 0.43

            if not gate_keep:
                score = min(score, 0.08)

            score -= min(0.25, d * 0.035)
            score = max(0.0, min(1.0, score))

        ps = np.nan
        ns = np.nan

        if i > 0 and coords[i - 1] is not None and c is not None:
            dt = max(1, frames[i] - frames[i - 1])
            if dt <= 12:
                ps = math.hypot(c[0] - coords[i - 1][0], c[1] - coords[i - 1][1]) / dt

        if i + 1 < len(g) and coords[i + 1] is not None and c is not None:
            dt = max(1, frames[i + 1] - frames[i])
            if dt <= 12:
                ns = math.hypot(coords[i + 1][0] - c[0], coords[i + 1][1] - c[1]) / dt

        # pénalité vitesse scène
        speed_penalty = 0.0
        for s in [ps, ns]:
            if not pd.isna(s):
                if s > 1.25:
                    speed_penalty = max(speed_penalty, min(0.38, (s - 1.25) * 0.22))

        score = max(0.0, score - speed_penalty)

        split = 0

        if i > 0:
            gap = frames[i] - frames[i - 1]
            if gap > 18:
                split = 1

            if not pd.isna(ps) and ps > 1.85:
                split = 1

            if score < 0.18:
                split = 1

        if split:
            tracklet_id += 1

        scores.append(round(float(score), 5))
        zones.append(zone)
        dist_outs.append("" if pd.isna(dist_out) else round(float(dist_out), 5))
        speed_prev.append("" if pd.isna(ps) else round(float(ps), 5))
        speed_next.append("" if pd.isna(ns) else round(float(ns), 5))
        split_flags.append(split)
        tracklet_ids.append(tracklet_id)

    g["ball_table_context_zone_005D2"] = zones
    g["ball_table_dist_outside_m_005D2"] = dist_outs
    g["ball_table_speed_prev_mpf_005D2"] = speed_prev
    g["ball_table_speed_next_mpf_005D2"] = speed_next
    g["ball_table_score_005D2"] = scores
    g["ball_table_tracklet_split_005D2"] = split_flags
    g["ball_table_tracklet_id_005D2"] = tracklet_ids

    g["ball_table_safe_005D2"] = (
        to_num(g["ball_table_score_005D2"]).fillna(0).ge(0.55)
        | g["ball_table_context_zone_005D2"].astype(str).eq("NO_METRIC_PROJECTION")
    ).astype(int)

    g["ball_table_metric_safe_005D2"] = (
        to_num(g["ball_table_score_005D2"]).fillna(0).ge(0.55)
        & to_num(g["table_project_ok_005C9B"]).fillna(0).astype(int).eq(1)
    ).astype(int)

    return g


def draw_segment(g: pd.DataFrame, frame_col: str, title: str):
    img = draw_topdown_base()

    g = g.copy()
    g[frame_col] = to_num(g[frame_col]).fillna(-1).astype(int)
    g["table_project_ok_005C9B"] = to_num(g["table_project_ok_005C9B"]).fillna(0).astype(int)
    g["ball_table_x_m_005C9B"] = to_num(g["ball_table_x_m_005C9B"])
    g["ball_table_y_m_005C9B"] = to_num(g["ball_table_y_m_005C9B"])
    g["ball_table_score_005D2"] = to_num(g["ball_table_score_005D2"]).fillna(0)
    g["ball_table_metric_safe_005D2"] = to_num(g["ball_table_metric_safe_005D2"]).fillna(0).astype(int)

    g = g[g["table_project_ok_005C9B"].eq(1)].sort_values(frame_col).copy()

    last = None

    for _, r in g.iterrows():
        x = float(r["ball_table_x_m_005C9B"])
        y = float(r["ball_table_y_m_005C9B"])
        px, py = xy_m_to_px(x, y)

        px = max(-90, min(CANON_W + 90, px))
        py = max(-130, min(CANON_H + 130, py))

        score = float(r["ball_table_score_005D2"])

        if score >= 0.70:
            color = (0, 255, 0)
        elif score >= 0.55:
            color = (0, 200, 255)
        elif score >= 0.25:
            color = (0, 120, 255)
        else:
            color = (0, 0, 255)

        if score >= 0.55:
            if last is not None:
                cv2.line(img, last, (px, py), (180, 160, 110), 1, cv2.LINE_AA)
            last = (px, py)
        else:
            last = None

        cv2.circle(img, (px, py), 4, color, -1, cv2.LINE_AA)

    cv2.putText(img, title, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def segment_summary(scored: pd.DataFrame, frame_col: str):
    rows = []

    for (review_id, seg), g in scored.groupby(["review_id", "camera_segment_id_005B2"], dropna=False):
        metric = to_num(g["table_project_ok_005C9B"]).fillna(0).astype(int)
        score = to_num(g["ball_table_score_005D2"]).fillna(0)
        safe = to_num(g["ball_table_metric_safe_005D2"]).fillna(0).astype(int)
        inside = to_num(g["ball_inside_table_005C9B"]).fillna(0).astype(int)

        mg = g[metric.eq(1)].copy()

        rows.append({
            "review_id": str(review_id),
            "camera_segment_id_005B2": int(seg),
            "points_total": int(len(g)),
            "metric_points": int(metric.sum()),
            "metric_safe": int((metric & safe).sum()),
            "metric_low_score": int((metric & safe.eq(0)).sum()),
            "inside_points": int((metric & inside).sum()),
            "inside_ratio_metric": round(float((metric & inside).sum() / max(1, metric.sum())), 4),
            "score_mean_metric": round(float(score[metric.eq(1)].mean()), 5) if metric.sum() else 0.0,
            "score_med_metric": round(float(score[metric.eq(1)].median()), 5) if metric.sum() else 0.0,
            "tracklets_metric": int(mg["ball_table_tracklet_id_005D2"].nunique()) if len(mg) else 0,
            "zone_counts": json.dumps(g["ball_table_context_zone_005D2"].astype(str).value_counts().to_dict(), ensure_ascii=False),
        })

    return pd.DataFrame(rows)


def build_html(cards):
    sections = []

    for c in cards:
        sections.append(f"""
<section class="card">
  <h2>{html.escape(c["review_id"])} · seg {c["seg"]}</h2>
  <p>
    metric={c["metric_points"]}
    · safe={c["metric_safe"]}
    · low={c["metric_low"]}
    · score_med={c["score_med"]}
    · tracklets={c["tracklets"]}
  </p>
  <img src="topdown/{html.escape(Path(c["image"]).name)}">
</section>
""")

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>005D2 table-aware ball score</title>
<style>
body{{margin:0;background:#101219;color:#eef1f8;font-family:system-ui,Segoe UI,sans-serif}}
header{{padding:16px 22px;background:#171b25;border-bottom:1px solid #303746}}
.card{{margin:18px;padding:16px;background:#181d27;border:1px solid #303746;border-radius:12px}}
img{{max-width:100%;border-radius:8px;border:1px solid #303746}}
h1{{margin:0;font-size:20px}}
h2{{font-size:17px;margin:0 0 8px}}
p{{color:#c3cada}}
</style>
</head>
<body>
<header>
<h1>TTFlux 005D2 · score balle table/caméra</h1>
<p>Vert = bon score. Jaune/orange = douteux. Rouge = rejet fort. Ce n’est pas encore le nouveau tracker, c’est la couche de scoring.</p>
</header>
{''.join(sections)}
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated-points", default="runs/rally_ball_table_gate_005D1/ball_points_table_gated_005D1.csv")
    ap.add_argument("--out-dir", default="runs/rally_ball_table_score_005D2")
    ap.add_argument("--make-topdown", action="store_true")
    ap.add_argument("--max-topdown", type=int, default=80)
    args = ap.parse_args()

    root = Path.cwd()

    points_path = Path(args.gated_points)
    out_dir = Path(args.out_dir)

    if not points_path.is_absolute():
        points_path = root / points_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    pts = pd.read_csv(points_path).fillna("")

    frame_col = find_col(pts, ["frame_num", "frame", "frame_idx"])
    seg_col = find_col(pts, ["camera_segment_id_005B2", "camera_segment_id"])

    pts[seg_col] = to_num(pts[seg_col]).fillna(1).astype(int)

    groups = []

    for (_, _), g in pts.groupby(["review_id", seg_col], dropna=False):
        groups.append(score_group(g, frame_col=frame_col))

    scored = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()

    out_points = out_dir / "ball_points_table_scored_005D2.csv"
    out_safe = out_dir / "ball_points_table_safe_005D2.csv"
    out_segments = out_dir / "ball_table_score_segment_summary_005D2.csv"
    out_json = out_dir / "ball_table_score_summary_005D2.json"
    out_html = out_dir / "ball_table_score_topdown_report_005D2.html"

    scored.to_csv(out_points, index=False, encoding="utf-8")

    safe = scored[to_num(scored["ball_table_safe_005D2"]).fillna(0).astype(int).eq(1)].copy()
    safe.to_csv(out_safe, index=False, encoding="utf-8")

    seg_summary = segment_summary(scored, frame_col=frame_col)
    seg_summary.to_csv(out_segments, index=False, encoding="utf-8")

    html_path = ""

    if args.make_topdown:
        img_dir = out_dir / "topdown"
        img_dir.mkdir(parents=True, exist_ok=True)

        cards = []

        view = seg_summary[to_num(seg_summary["metric_points"]).fillna(0).astype(int).gt(0)].copy()
        view = view.sort_values(["metric_low_score", "metric_points"], ascending=[False, False])

        for i, (_, s) in enumerate(view.iterrows(), start=1):
            if i > args.max_topdown:
                break

            review_id = str(s["review_id"])
            seg = int(s["camera_segment_id_005B2"])

            g = scored[
                scored["review_id"].astype(str).eq(review_id)
                & to_num(scored[seg_col]).fillna(1).astype(int).eq(seg)
            ].copy()

            title = f"{review_id} seg={seg} safe={int(s['metric_safe'])}/{int(s['metric_points'])} score_med={float(s['score_med_metric']):.2f}"
            img = draw_segment(g, frame_col=frame_col, title=title)

            fname = f"{i:03d}_{review_id}_seg{seg}_005D2_score.jpg"
            path = img_dir / fname
            imwrite_unicode(path, img)

            cards.append({
                "review_id": review_id,
                "seg": seg,
                "metric_points": int(s["metric_points"]),
                "metric_safe": int(s["metric_safe"]),
                "metric_low": int(s["metric_low_score"]),
                "score_med": float(s["score_med_metric"]),
                "tracklets": int(s["tracklets_metric"]),
                "image": str(path),
            })

        out_html.write_text(build_html(cards), encoding="utf-8")
        html_path = str(out_html)

    total = int(len(scored))
    metric_total = int(to_num(scored["table_project_ok_005C9B"]).fillna(0).astype(int).sum())
    safe_total = int(to_num(scored["ball_table_safe_005D2"]).fillna(0).astype(int).sum())
    metric_safe = int(to_num(scored["ball_table_metric_safe_005D2"]).fillna(0).astype(int).sum())

    zone_counts = scored["ball_table_context_zone_005D2"].astype(str).value_counts().to_dict()
    score = to_num(scored["ball_table_score_005D2"]).fillna(0)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "score_existing_ball_points_with_table_camera_context_for_future_candidate_reranking",
        "gated_points": str(points_path),
        "points_total": total,
        "metric_points_total": metric_total,
        "safe_total": safe_total,
        "metric_safe": metric_safe,
        "metric_low_score": int(metric_total - metric_safe),
        "score_mean_all": round(float(score.mean()), 5) if total else 0.0,
        "score_median_all": round(float(score.median()), 5) if total else 0.0,
        "zone_counts": {str(k): int(v) for k, v in zone_counts.items()},
        "outputs": {
            "scored_points": str(out_points),
            "safe_points": str(out_safe),
            "segment_summary": str(out_segments),
            "topdown_html": html_path,
        },
        "next": "Use score_005D2 as a feature in candidate-level Viterbi/reranking. Current step is scoring, not final tracking."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005D2 status=OK")
    print("points_total=", summary["points_total"])
    print("metric_points_total=", summary["metric_points_total"])
    print("safe_total=", summary["safe_total"])
    print("metric_safe=", summary["metric_safe"])
    print("metric_low_score=", summary["metric_low_score"])
    print("score_median_all=", summary["score_median_all"])
    print("zone_counts=", json.dumps(summary["zone_counts"], ensure_ascii=False))
    print("wrote", out_points)
    print("wrote", out_safe)
    print("wrote", out_segments)
    if html_path:
        print("wrote", html_path)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
