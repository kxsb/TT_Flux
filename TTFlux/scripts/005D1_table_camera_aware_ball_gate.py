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


VERSION = "005D1"

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


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"Missing required column among {candidates}")


def xy_m_to_px(x, y, w=CANON_W, h=CANON_H):
    px = int(round((float(x) + TABLE_HALF_W) / TABLE_WIDTH_M * (w - 1)))
    py = int(round((float(y) + TABLE_HALF_L) / TABLE_LENGTH_M * (h - 1)))
    return px, py


def draw_topdown_base(w=CANON_W, h=CANON_H):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (24, 74, 58)

    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (235, 235, 235), 3, cv2.LINE_AA)

    net_y = xy_m_to_px(0, 0, w, h)[1]
    cv2.line(img, (0, net_y), (w - 1, net_y), (40, 210, 255), 3, cv2.LINE_AA)

    center_x = xy_m_to_px(0, 0, w, h)[0]
    cv2.line(img, (center_x, 0), (center_x, h - 1), (255, 255, 80), 2, cv2.LINE_AA)

    for x in np.linspace(-TABLE_HALF_W, TABLE_HALF_W, 5):
        px, _ = xy_m_to_px(x, 0, w, h)
        cv2.line(img, (px, 0), (px, h - 1), (72, 118, 100), 1, cv2.LINE_AA)

    for y in np.linspace(-TABLE_HALF_L, TABLE_HALF_L, 7):
        _, py = xy_m_to_px(0, y, w, h)
        cv2.line(img, (0, py), (w - 1, py), (72, 118, 100), 1, cv2.LINE_AA)

    cv2.putText(img, "FAR SIDE", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(img, "NEAR SIDE", (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 2, cv2.LINE_AA)

    return img


def dist_xy(a, b):
    if a is None or b is None:
        return np.nan
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def classify_group(
    g: pd.DataFrame,
    frame_col: str,
    x_col: str,
    y_col: str,
    soft_x_margin: float,
    soft_y_margin: float,
    hard_x_margin: float,
    hard_y_margin: float,
    spike_speed_mpf: float,
    max_neighbor_dt: int,
):
    g = g.copy()
    g[frame_col] = to_num(g[frame_col]).fillna(-1).astype(int)
    g[x_col] = to_num(g[x_col])
    g[y_col] = to_num(g[y_col])
    g["table_project_ok_005C9B"] = to_num(g["table_project_ok_005C9B"]).fillna(0).astype(int)

    g = g.sort_values(frame_col).reset_index(drop=True)

    n = len(g)

    keep = []
    reason = []
    soft_ok_arr = []
    hard_outside_arr = []
    prev_speed_arr = []
    next_speed_arr = []
    spike_arr = []

    pts = []

    for _, r in g.iterrows():
        if int(r["table_project_ok_005C9B"]) != 1 or pd.isna(r[x_col]) or pd.isna(r[y_col]):
            pts.append(None)
        else:
            pts.append((float(r[x_col]), float(r[y_col])))

    frames = g[frame_col].to_numpy(dtype=int)

    for i, r in g.iterrows():
        p = pts[i]

        if p is None:
            keep.append(1)
            reason.append("keep_no_metric_projection")
            soft_ok_arr.append(0)
            hard_outside_arr.append(0)
            prev_speed_arr.append(np.nan)
            next_speed_arr.append(np.nan)
            spike_arr.append(0)
            continue

        x, y = p

        inside = int(abs(x) <= TABLE_HALF_W and abs(y) <= TABLE_HALF_L)

        soft_ok = int(
            abs(x) <= TABLE_HALF_W + soft_x_margin
            and abs(y) <= TABLE_HALF_L + soft_y_margin
        )

        hard_outside = int(
            abs(x) > TABLE_HALF_W + hard_x_margin
            or abs(y) > TABLE_HALF_L + hard_y_margin
        )

        prev_speed = np.nan
        next_speed = np.nan
        prev_next_speed = np.nan

        if i > 0 and pts[i - 1] is not None:
            dt = max(1, int(frames[i] - frames[i - 1]))
            if dt <= max_neighbor_dt:
                prev_speed = dist_xy(pts[i - 1], p) / dt

        if i + 1 < n and pts[i + 1] is not None:
            dt = max(1, int(frames[i + 1] - frames[i]))
            if dt <= max_neighbor_dt:
                next_speed = dist_xy(p, pts[i + 1]) / dt

        if i > 0 and i + 1 < n and pts[i - 1] is not None and pts[i + 1] is not None:
            dt = max(1, int(frames[i + 1] - frames[i - 1]))
            if dt <= max_neighbor_dt * 2:
                prev_next_speed = dist_xy(pts[i - 1], pts[i + 1]) / dt

        spike = 0

        if not pd.isna(prev_speed) and not pd.isna(next_speed):
            if prev_speed >= spike_speed_mpf and next_speed >= spike_speed_mpf:
                if pd.isna(prev_next_speed) or prev_next_speed < max(prev_speed, next_speed) * 0.72:
                    spike = 1

        # Politique conservatrice :
        # - on drop les points très loin de la scène table ;
        # - on drop les spikes isolés, surtout hors table/zone proche ;
        # - on garde les points douteux mais non absurdes pour analyse suivante.
        if hard_outside:
            keep_i = 0
            reason_i = "drop_hard_outside_table_scene"
        elif spike and not inside:
            keep_i = 0
            reason_i = "drop_temporal_spike_off_table"
        elif spike and not soft_ok:
            keep_i = 0
            reason_i = "drop_temporal_spike_outside_soft_scene"
        elif inside:
            keep_i = 1
            reason_i = "keep_inside_table"
        elif soft_ok:
            keep_i = 1
            reason_i = "keep_near_table_scene"
        else:
            keep_i = 1
            reason_i = "keep_outside_soft_but_not_absurd"

        keep.append(keep_i)
        reason.append(reason_i)
        soft_ok_arr.append(soft_ok)
        hard_outside_arr.append(hard_outside)
        prev_speed_arr.append(prev_speed)
        next_speed_arr.append(next_speed)
        spike_arr.append(spike)

    g["table_scene_soft_ok_005D1"] = soft_ok_arr
    g["table_scene_hard_outside_005D1"] = hard_outside_arr
    g["table_scene_prev_speed_mpf_005D1"] = ["" if pd.isna(x) else round(float(x), 5) for x in prev_speed_arr]
    g["table_scene_next_speed_mpf_005D1"] = ["" if pd.isna(x) else round(float(x), 5) for x in next_speed_arr]
    g["table_scene_temporal_spike_005D1"] = spike_arr
    g["table_gate_keep_005D1"] = keep
    g["table_gate_reason_005D1"] = reason

    return g


def draw_segment_topdown(g: pd.DataFrame, frame_col: str, title: str):
    img = draw_topdown_base()

    g = g.copy()
    g[frame_col] = to_num(g[frame_col]).fillna(-1).astype(int)
    g["ball_table_x_m_005C9B"] = to_num(g["ball_table_x_m_005C9B"])
    g["ball_table_y_m_005C9B"] = to_num(g["ball_table_y_m_005C9B"])
    g["table_project_ok_005C9B"] = to_num(g["table_project_ok_005C9B"]).fillna(0).astype(int)
    g["table_gate_keep_005D1"] = to_num(g["table_gate_keep_005D1"]).fillna(1).astype(int)

    g = g[g["table_project_ok_005C9B"].eq(1)].sort_values(frame_col).copy()

    last_kept = None

    for _, r in g.iterrows():
        x = float(r["ball_table_x_m_005C9B"])
        y = float(r["ball_table_y_m_005C9B"])
        px, py = xy_m_to_px(x, y)

        kept = int(r["table_gate_keep_005D1"]) == 1

        # Clamp visuel pour voir les points très hors-table sans exploser.
        pxv = max(-80, min(CANON_W + 80, px))
        pyv = max(-120, min(CANON_H + 120, py))

        if kept:
            if last_kept is not None:
                cv2.line(img, last_kept, (pxv, pyv), (180, 160, 110), 1, cv2.LINE_AA)
            cv2.circle(img, (pxv, pyv), 3, (0, 255, 0), -1, cv2.LINE_AA)
            last_kept = (pxv, pyv)
        else:
            cv2.circle(img, (pxv, pyv), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (pxv, pyv), 8, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, title, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def build_html(cards):
    sections = []

    for c in cards:
        sections.append(f"""
<section class="card">
  <h2>{html.escape(c["review_id"])} · seg {c["seg"]}</h2>
  <p>
    points={c["points"]}
    · kept={c["kept"]}
    · dropped={c["dropped"]}
    · inside_before={c["inside_before"]}
    · inside_after={c["inside_after"]}
  </p>
  <img src="topdown/{html.escape(Path(c["image"]).name)}">
</section>
""")

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>005D1 table-aware ball gate</title>
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
<h1>TTFlux 005D1 · gate tracking balle avec table/caméra</h1>
<p>Vert = point gardé. Rouge = point rejeté par contrainte table/caméra. La table est fixe en top-down.</p>
</header>
{''.join(sections)}
</body>
</html>
"""


def make_segment_summary(df: pd.DataFrame):
    rows = []

    if df.empty:
        return pd.DataFrame()

    for (review_id, seg), g in df.groupby(["review_id", "camera_segment_id_005B2"], dropna=False):
        ok = to_num(g["table_project_ok_005C9B"]).fillna(0).astype(int)
        keep = to_num(g["table_gate_keep_005D1"]).fillna(1).astype(int)
        inside = to_num(g["ball_inside_table_005C9B"]).fillna(0).astype(int)

        metric_g = g[ok.eq(1)].copy()
        keep_g = g[ok.eq(1) & keep.eq(1)].copy()

        inside_before = int(to_num(metric_g["ball_inside_table_005C9B"]).fillna(0).sum()) if len(metric_g) else 0
        inside_after = int(to_num(keep_g["ball_inside_table_005C9B"]).fillna(0).sum()) if len(keep_g) else 0

        rows.append({
            "review_id": str(review_id),
            "camera_segment_id_005B2": int(seg),
            "points_total": int(len(g)),
            "metric_points": int(ok.sum()),
            "kept_total": int(keep.sum()),
            "dropped_total": int((1 - keep).sum()),
            "metric_kept": int((ok & keep).sum()),
            "metric_dropped": int((ok & (1 - keep)).sum()),
            "inside_before": inside_before,
            "inside_after": inside_after,
            "inside_ratio_before": round(float(inside_before / max(1, len(metric_g))), 4),
            "inside_ratio_after": round(float(inside_after / max(1, len(keep_g))), 4),
            "hard_outside_drop": int((to_num(g["table_scene_hard_outside_005D1"]).fillna(0).astype(int).eq(1) & keep.eq(0)).sum()),
            "temporal_spike_drop": int((to_num(g["table_scene_temporal_spike_005D1"]).fillna(0).astype(int).eq(1) & keep.eq(0)).sum()),
            "camera_signature_005C9B": str(g["camera_signature_005C9B"].dropna().astype(str).iloc[0]) if "camera_signature_005C9B" in g.columns and len(g) else "",
            "scene_table_status_005C9B": str(g["scene_table_status_005C9B"].dropna().astype(str).iloc[0]) if "scene_table_status_005C9B" in g.columns and len(g) else "",
        })

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-tables", default="runs/rally_scene_table_objects_005C9B/scene_table_objects_005C9B.csv")
    ap.add_argument("--projected-points", default="runs/rally_scene_table_objects_005C9B/ball_points_scene_table_projected_005C9B.csv")
    ap.add_argument("--out-dir", default="runs/rally_ball_table_gate_005D1")

    ap.add_argument("--soft-x-margin", type=float, default=0.85)
    ap.add_argument("--soft-y-margin", type=float, default=1.25)
    ap.add_argument("--hard-x-margin", type=float, default=1.80)
    ap.add_argument("--hard-y-margin", type=float, default=2.70)
    ap.add_argument("--spike-speed-mpf", type=float, default=1.10)
    ap.add_argument("--max-neighbor-dt", type=int, default=8)

    ap.add_argument("--make-topdown", action="store_true")
    ap.add_argument("--max-topdown", type=int, default=60)

    args = ap.parse_args()

    root = Path.cwd()

    scene_path = Path(args.scene_tables)
    points_path = Path(args.projected_points)
    out_dir = Path(args.out_dir)

    if not scene_path.is_absolute():
        scene_path = root / scene_path
    if not points_path.is_absolute():
        points_path = root / points_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    scene = pd.read_csv(scene_path).fillna("")
    pts = pd.read_csv(points_path).fillna("")

    frame_col = find_col(pts, ["frame_num", "frame", "frame_idx"])
    seg_col = find_col(pts, ["camera_segment_id_005B2", "camera_segment_id"])
    x_col = find_col(pts, ["ball_table_x_m_005C9B"])
    y_col = find_col(pts, ["ball_table_y_m_005C9B"])

    pts[seg_col] = to_num(pts[seg_col]).fillna(1).astype(int)

    gated_groups = []

    for (review_id, seg), g in pts.groupby(["review_id", seg_col], dropna=False):
        gg = classify_group(
            g=g,
            frame_col=frame_col,
            x_col=x_col,
            y_col=y_col,
            soft_x_margin=args.soft_x_margin,
            soft_y_margin=args.soft_y_margin,
            hard_x_margin=args.hard_x_margin,
            hard_y_margin=args.hard_y_margin,
            spike_speed_mpf=args.spike_speed_mpf,
            max_neighbor_dt=args.max_neighbor_dt,
        )
        gated_groups.append(gg)

    gated = pd.concat(gated_groups, ignore_index=True) if gated_groups else pd.DataFrame()

    out_points = out_dir / "ball_points_table_gated_005D1.csv"
    out_segments = out_dir / "ball_table_gate_segment_summary_005D1.csv"
    out_json = out_dir / "ball_table_gate_summary_005D1.json"
    out_html = out_dir / "ball_table_gate_topdown_report_005D1.html"

    gated.to_csv(out_points, index=False, encoding="utf-8")

    seg_summary = make_segment_summary(gated)
    seg_summary.to_csv(out_segments, index=False, encoding="utf-8")

    html_path = ""

    if args.make_topdown:
        img_dir = out_dir / "topdown"
        img_dir.mkdir(parents=True, exist_ok=True)

        cards = []

        metric_segments = seg_summary[
            to_num(seg_summary["metric_points"]).fillna(0).astype(int).gt(0)
        ].sort_values(
            ["metric_dropped", "metric_points"],
            ascending=[False, False],
        ).copy()

        for i, (_, s) in enumerate(metric_segments.iterrows(), start=1):
            if i > args.max_topdown:
                break

            review_id = str(s["review_id"])
            seg = int(s["camera_segment_id_005B2"])

            g = gated[
                gated["review_id"].astype(str).eq(review_id)
                & to_num(gated[seg_col]).fillna(1).astype(int).eq(seg)
            ].copy()

            title = f"{review_id} seg={seg} kept={int(s['metric_kept'])}/{int(s['metric_points'])} drop={int(s['metric_dropped'])}"
            img = draw_segment_topdown(g, frame_col=frame_col, title=title)

            fname = f"{i:03d}_{review_id}_seg{seg}_005D1_gate.jpg"
            path = img_dir / fname
            imwrite_unicode(path, img)

            cards.append({
                "review_id": review_id,
                "seg": seg,
                "points": int(s["metric_points"]),
                "kept": int(s["metric_kept"]),
                "dropped": int(s["metric_dropped"]),
                "inside_before": float(s["inside_ratio_before"]),
                "inside_after": float(s["inside_ratio_after"]),
                "image": str(path),
            })

        out_html.write_text(build_html(cards), encoding="utf-8")
        html_path = str(out_html)

    total = int(len(gated))
    metric_total = int(to_num(gated["table_project_ok_005C9B"]).fillna(0).astype(int).sum()) if len(gated) else 0
    kept_total = int(to_num(gated["table_gate_keep_005D1"]).fillna(1).astype(int).sum()) if len(gated) else 0
    dropped_total = total - kept_total

    metric_keep_mask = (
        to_num(gated["table_project_ok_005C9B"]).fillna(0).astype(int).eq(1)
        & to_num(gated["table_gate_keep_005D1"]).fillna(1).astype(int).eq(1)
    ) if len(gated) else pd.Series([], dtype=bool)

    metric_drop_mask = (
        to_num(gated["table_project_ok_005C9B"]).fillna(0).astype(int).eq(1)
        & to_num(gated["table_gate_keep_005D1"]).fillna(1).astype(int).eq(0)
    ) if len(gated) else pd.Series([], dtype=bool)

    inside_before = int(to_num(gated["ball_inside_table_005C9B"]).fillna(0).astype(int)[
        to_num(gated["table_project_ok_005C9B"]).fillna(0).astype(int).eq(1)
    ].sum()) if len(gated) else 0

    inside_after = int(to_num(gated["ball_inside_table_005C9B"]).fillna(0).astype(int)[metric_keep_mask].sum()) if len(gated) else 0

    reason_counts = gated["table_gate_reason_005D1"].astype(str).value_counts().to_dict() if len(gated) else {}

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "conservative_table_camera_aware_gate_for_existing_ball_tracking_points",
        "scene_tables": str(scene_path),
        "projected_points": str(points_path),
        "params": {
            "soft_x_margin": args.soft_x_margin,
            "soft_y_margin": args.soft_y_margin,
            "hard_x_margin": args.hard_x_margin,
            "hard_y_margin": args.hard_y_margin,
            "spike_speed_mpf": args.spike_speed_mpf,
            "max_neighbor_dt": args.max_neighbor_dt,
        },
        "points_total": total,
        "metric_points_total": metric_total,
        "kept_total": kept_total,
        "dropped_total": dropped_total,
        "metric_kept": int(metric_keep_mask.sum()) if len(gated) else 0,
        "metric_dropped": int(metric_drop_mask.sum()) if len(gated) else 0,
        "inside_before": inside_before,
        "inside_after": inside_after,
        "inside_ratio_before_metric": round(float(inside_before / max(1, metric_total)), 4),
        "inside_ratio_after_metric_kept": round(float(inside_after / max(1, int(metric_keep_mask.sum()) if len(gated) else 0)), 4),
        "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "outputs": {
            "gated_points": str(out_points),
            "segment_summary": str(out_segments),
            "topdown_html": html_path,
        },
        "next": "Review topdown report. Then use dropped points to tune table-aware Viterbi/candidate ranking in 005D2."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005D1 status=OK")
    print("points_total=", summary["points_total"])
    print("metric_points_total=", summary["metric_points_total"])
    print("kept_total=", summary["kept_total"])
    print("dropped_total=", summary["dropped_total"])
    print("metric_kept=", summary["metric_kept"])
    print("metric_dropped=", summary["metric_dropped"])
    print("inside_ratio_before_metric=", summary["inside_ratio_before_metric"])
    print("inside_ratio_after_metric_kept=", summary["inside_ratio_after_metric_kept"])
    print("reason_counts=", json.dumps(summary["reason_counts"], ensure_ascii=False))
    print("wrote", out_points)
    print("wrote", out_segments)
    if html_path:
        print("wrote", html_path)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
