from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005D4"

TABLE_LENGTH_M = 2.740
TABLE_WIDTH_M = 1.525
TABLE_HALF_L = TABLE_LENGTH_M / 2.0
TABLE_HALF_W = TABLE_WIDTH_M / 2.0


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def find_col(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise SystemExit(f"missing column among {names}")


def open_writer(path: Path, fps: float, w: int, h: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not wr.isOpened():
        raise RuntimeError(f"cannot open writer: {path}")
    return wr


def build_clip_map(scene: pd.DataFrame):
    out = {}

    if "clip_path" not in scene.columns:
        return out

    for review_id, g in scene.groupby("review_id", dropna=False):
        vals = [str(x) for x in g["clip_path"].tolist() if str(x).strip()]
        if vals:
            out[str(review_id)] = vals[0]

    return out


def metric_xy(row):
    ok = int(to_num(pd.Series([row.get("table_project_ok_005C9B", 0)])).fillna(0).iloc[0]) == 1
    if not ok:
        return None

    x = to_num(pd.Series([row.get("ball_table_x_m_005C9B", "")])).iloc[0]
    y = to_num(pd.Series([row.get("ball_table_y_m_005C9B", "")])).iloc[0]

    if pd.isna(x) or pd.isna(y):
        return None

    return float(x), float(y)


def pixel_xy(row, x_col, y_col):
    x = to_num(pd.Series([row.get(x_col, "")])).iloc[0]
    y = to_num(pd.Series([row.get(y_col, "")])).iloc[0]

    if pd.isna(x) or pd.isna(y):
        return None

    return float(x), float(y)


def dist(a, b):
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def inside_table(x, y):
    return int(abs(float(x)) <= TABLE_HALF_W and abs(float(y)) <= TABLE_HALF_L)


def is_safe_endpoint(row, min_score):
    safe = int(to_num(pd.Series([row.get("ball_table_safe_005D2", 1)])).fillna(1).iloc[0]) == 1
    score = float(to_num(pd.Series([row.get("ball_table_score_005D2", 0.42)])).fillna(0.42).iloc[0])
    project_ok = int(to_num(pd.Series([row.get("table_project_ok_005C9B", 0)])).fillna(0).iloc[0]) == 1

    if project_ok:
        return safe and score >= min_score

    # Hors métrique : on garde, mais on ne l'utilise pas comme preuve table.
    return safe


def can_bridge(a, b, frame_col, x_col, y_col, max_gap, max_metric_speed, max_pixel_speed):
    fa = int(a[frame_col])
    fb = int(b[frame_col])
    gap = fb - fa

    if gap <= 1 or gap > max_gap:
        return False, "gap_outside_range"

    am = metric_xy(a)
    bm = metric_xy(b)

    if am is not None and bm is not None:
        speed = dist(am, bm) / max(1, gap)
        if speed <= max_metric_speed:
            return True, f"metric_bridge_speed_{speed:.3f}"
        return False, f"metric_speed_too_high_{speed:.3f}"

    ap = pixel_xy(a, x_col, y_col)
    bp = pixel_xy(b, x_col, y_col)

    if ap is not None and bp is not None:
        speed = dist(ap, bp) / max(1, gap)
        if speed <= max_pixel_speed:
            return True, f"pixel_bridge_speed_{speed:.3f}"
        return False, f"pixel_speed_too_high_{speed:.3f}"

    return False, "missing_coordinates"


def interpolate_row(a, b, f, frame_col, x_col, y_col, bridge_reason, tracklet_id):
    fa = int(a[frame_col])
    fb = int(b[frame_col])
    alpha = (f - fa) / max(1, fb - fa)

    out = dict(a)

    out[frame_col] = int(f)

    ax = float(a[x_col])
    ay = float(a[y_col])
    bx = float(b[x_col])
    by = float(b[y_col])

    out[x_col] = round(float(ax + alpha * (bx - ax)), 4)
    out[y_col] = round(float(ay + alpha * (by - ay)), 4)

    am = metric_xy(a)
    bm = metric_xy(b)

    if am is not None and bm is not None:
        tx = am[0] + alpha * (bm[0] - am[0])
        ty = am[1] + alpha * (bm[1] - am[1])

        out["table_project_ok_005C9B"] = 1
        out["ball_table_x_m_005C9B"] = round(float(tx), 5)
        out["ball_table_y_m_005C9B"] = round(float(ty), 5)
        out["ball_inside_table_005C9B"] = inside_table(tx, ty)
        out["ball_table_side_005C9B"] = "near" if ty >= 0 else "far"

    score_a = float(to_num(pd.Series([a.get("ball_table_score_005D2", 0.55)])).fillna(0.55).iloc[0])
    score_b = float(to_num(pd.Series([b.get("ball_table_score_005D2", 0.55)])).fillna(0.55).iloc[0])
    score = max(0.0, min(score_a, score_b) - 0.04)

    out["ball_table_score_005D2"] = round(float(score), 5)
    out["ball_table_safe_005D2"] = 1
    out["ball_table_metric_safe_005D2"] = int(am is not None and bm is not None and score >= 0.55)

    out["point_source_005D4"] = "interp_bridge"
    out["is_interpolated_005D4"] = 1
    out["bridge_reason_005D4"] = bridge_reason
    out["bridge_gap_005D4"] = int(fb - fa - 1)
    out["tracklet_id_005D4"] = int(tracklet_id)

    return out


def relink_group(g, frame_col, x_col, y_col, min_score, max_gap, max_metric_speed, max_pixel_speed, split_metric_speed):
    g = g.copy()
    g[frame_col] = to_num(g[frame_col]).fillna(-1).astype(int)
    g[x_col] = to_num(g[x_col])
    g[y_col] = to_num(g[y_col])
    g = g.sort_values(frame_col).reset_index(drop=True)

    safe_rows = []

    for _, r in g.iterrows():
        row = r.to_dict()
        if is_safe_endpoint(row, min_score=min_score):
            row["point_source_005D4"] = "original_safe"
            row["is_interpolated_005D4"] = 0
            row["bridge_reason_005D4"] = ""
            row["bridge_gap_005D4"] = 0
            safe_rows.append(row)

    if not safe_rows:
        return pd.DataFrame(), {
            "original_safe": 0,
            "interpolated": 0,
            "bridges": 0,
            "tracklets": 0,
        }

    out_rows = []
    tracklet_id = 0
    bridges = 0
    interpolated = 0

    first = dict(safe_rows[0])
    first["tracklet_id_005D4"] = tracklet_id
    out_rows.append(first)

    for prev, cur in zip(safe_rows[:-1], safe_rows[1:]):
        fa = int(prev[frame_col])
        fb = int(cur[frame_col])
        gap = fb - fa

        bridge, reason = can_bridge(
            prev,
            cur,
            frame_col=frame_col,
            x_col=x_col,
            y_col=y_col,
            max_gap=max_gap,
            max_metric_speed=max_metric_speed,
            max_pixel_speed=max_pixel_speed,
        )

        split = False

        pm = metric_xy(prev)
        cm = metric_xy(cur)

        if pm is not None and cm is not None:
            speed = dist(pm, cm) / max(1, gap)
            if speed > split_metric_speed:
                split = True

        if gap > max_gap:
            split = True

        if not bridge and gap > 1:
            split = True

        if split:
            tracklet_id += 1

        if bridge:
            bridges += 1
            for f in range(fa + 1, fb):
                out_rows.append(
                    interpolate_row(
                        prev,
                        cur,
                        f=f,
                        frame_col=frame_col,
                        x_col=x_col,
                        y_col=y_col,
                        bridge_reason=reason,
                        tracklet_id=tracklet_id,
                    )
                )
                interpolated += 1

        cur2 = dict(cur)
        cur2["tracklet_id_005D4"] = tracklet_id
        out_rows.append(cur2)

    return pd.DataFrame(out_rows), {
        "original_safe": len(safe_rows),
        "interpolated": interpolated,
        "bridges": bridges,
        "tracklets": int(tracklet_id + 1),
    }


def draw_label(img, text, y):
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)


def draw_point(img, x, y, source, score):
    x = int(round(float(x)))
    y = int(round(float(y)))

    if source == "interp_bridge":
        color = (255, 0, 255)       # magenta interpolation
        radius = 5
    elif score >= 0.70:
        color = (0, 255, 0)
        radius = 6
    elif score >= 0.55:
        color = (0, 210, 255)
        radius = 6
    else:
        color = (0, 0, 255)
        radius = 6

    cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), radius + 2, (0, 0, 0), 1, cv2.LINE_AA)


def draw_tail(frame, clean, frame_col, x_col, y_col, current_frame, tail_age):
    gg = clean.copy()
    gg[frame_col] = to_num(gg[frame_col]).fillna(-1).astype(int)
    gg = gg[gg[frame_col].between(current_frame - tail_age, current_frame)].sort_values(frame_col)

    pts = []

    for _, r in gg.iterrows():
        try:
            pts.append((int(round(float(r[x_col]))), int(round(float(r[y_col])))))
        except Exception:
            pass

    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(frame, a, b, (90, 220, 90), 2, cv2.LINE_AA)


def render_review(review_id, clean_g, orig_g, clip_path, out_dir, frame_col, x_col, y_col, max_frames, tail_age):
    clip_path = Path(str(clip_path))

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print("skip cannot_open_video", review_id, clip_path)
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if w <= 0 or h <= 0:
        cap.release()
        return None

    clean_g = clean_g.copy()
    clean_g[frame_col] = to_num(clean_g[frame_col]).fillna(-1).astype(int)

    orig_g = orig_g.copy()
    orig_g[frame_col] = to_num(orig_g[frame_col]).fillna(-1).astype(int)

    first = max(0, int(min(clean_g[frame_col].min(), orig_g[frame_col].min())) - 20)
    last = min(n_frames - 1, int(max(clean_g[frame_col].max(), orig_g[frame_col].max())) + 20)

    if max_frames > 0:
        last = min(last, first + max_frames - 1)

    clean_by_frame = {int(k): v.copy() for k, v in clean_g.groupby(frame_col)}

    out_path = out_dir / "overlays" / f"{review_id}_005D4_relinked_overlay.mp4"
    wr = open_writer(out_path, fps=fps, w=w, h=h)

    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    frame_idx = first
    written = 0

    interp_count = int(to_num(clean_g["is_interpolated_005D4"]).fillna(0).astype(int).sum())
    tracklets = int(clean_g["tracklet_id_005D4"].nunique()) if len(clean_g) else 0

    while frame_idx <= last:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        draw_tail(frame, clean_g, frame_col, x_col, y_col, frame_idx, tail_age)

        rows = clean_by_frame.get(frame_idx)

        if rows is not None and len(rows):
            for _, r in rows.iterrows():
                try:
                    x = float(r[x_col])
                    y = float(r[y_col])
                except Exception:
                    continue

                source = str(r.get("point_source_005D4", "original_safe"))
                score = float(to_num(pd.Series([r.get("ball_table_score_005D2", 0.55)])).fillna(0.55).iloc[0])
                draw_point(frame, x, y, source, score)

        draw_label(frame, f"{review_id} 005D4 relink/interp  interp={interp_count} tracklets={tracklets}", 30)
        draw_label(frame, "green=safe original  magenta=interpolated bridge", 58)

        wr.write(frame)
        written += 1
        frame_idx += 1

    cap.release()
    wr.release()

    return {
        "review_id": str(review_id),
        "overlay": str(out_path),
        "clip_path": str(clip_path),
        "first_frame": first,
        "last_frame": frame_idx - 1,
        "written_frames": written,
        "interpolated": interp_count,
        "tracklets": tracklets,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-points", default="runs/rally_ball_table_score_005D2/ball_points_table_scored_005D2.csv")
    ap.add_argument("--scene-tables", default="runs/rally_scene_table_objects_005C9B/scene_table_objects_005C9B.csv")
    ap.add_argument("--out-dir", default="runs/rally_ball_table_relink_005D4")

    ap.add_argument("--min-score", type=float, default=0.55)
    ap.add_argument("--max-gap", type=int, default=12)
    ap.add_argument("--max-metric-bridge-speed", type=float, default=1.25)
    ap.add_argument("--max-pixel-bridge-speed", type=float, default=140.0)
    ap.add_argument("--split-metric-speed", type=float, default=1.85)

    ap.add_argument("--make-videos", action="store_true")
    ap.add_argument("--max-videos", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--tail-age", type=int, default=18)
    args = ap.parse_args()

    root = Path.cwd()

    points_path = Path(args.scored_points)
    scene_path = Path(args.scene_tables)
    out_dir = Path(args.out_dir)

    if not points_path.is_absolute():
        points_path = root / points_path
    if not scene_path.is_absolute():
        scene_path = root / scene_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    pts = pd.read_csv(points_path).fillna("")
    scene = pd.read_csv(scene_path).fillna("") if scene_path.is_file() else pd.DataFrame()
    clip_map = build_clip_map(scene)

    frame_col = find_col(pts, ["frame_num", "frame", "frame_idx"])
    seg_col = find_col(pts, ["camera_segment_id_005B2", "camera_segment_id"])
    x_col = find_col(pts, ["x_num", "x", "cx"])
    y_col = find_col(pts, ["y_num", "y", "cy"])

    pts[seg_col] = to_num(pts[seg_col]).fillna(1).astype(int)

    clean_groups = []
    summary_rows = []

    for (review_id, seg), g in pts.groupby(["review_id", seg_col], dropna=False):
        clean, stats = relink_group(
            g,
            frame_col=frame_col,
            x_col=x_col,
            y_col=y_col,
            min_score=args.min_score,
            max_gap=args.max_gap,
            max_metric_speed=args.max_metric_bridge_speed,
            max_pixel_speed=args.max_pixel_bridge_speed,
            split_metric_speed=args.split_metric_speed,
        )

        if len(clean):
            clean["review_id"] = str(review_id)
            clean[seg_col] = int(seg)
            clean_groups.append(clean)

        summary_rows.append({
            "review_id": str(review_id),
            "camera_segment_id_005B2": int(seg),
            "input_points": int(len(g)),
            "original_safe": int(stats["original_safe"]),
            "interpolated": int(stats["interpolated"]),
            "bridges": int(stats["bridges"]),
            "relinked_points": int(stats["original_safe"] + stats["interpolated"]),
            "tracklets": int(stats["tracklets"]),
        })

    clean_all = pd.concat(clean_groups, ignore_index=True) if clean_groups else pd.DataFrame()
    seg_summary = pd.DataFrame(summary_rows)

    out_points = out_dir / "ball_points_table_relinked_005D4.csv"
    out_segments = out_dir / "ball_table_relink_segment_summary_005D4.csv"
    out_json = out_dir / "ball_table_relink_summary_005D4.json"

    clean_all.to_csv(out_points, index=False, encoding="utf-8")
    seg_summary.to_csv(out_segments, index=False, encoding="utf-8")

    overlays = []

    if args.make_videos and len(clean_all):
        view = seg_summary.sort_values(["interpolated", "bridges", "input_points"], ascending=[False, False, False]).copy()

        chosen = []

        for r in view["review_id"].astype(str).tolist():
            if r not in chosen:
                chosen.append(r)
            if len(chosen) >= args.max_videos:
                break

        for review_id in chosen:
            clean_g = clean_all[clean_all["review_id"].astype(str).eq(str(review_id))].copy()
            orig_g = pts[pts["review_id"].astype(str).eq(str(review_id))].copy()

            clip_path = clip_map.get(str(review_id))

            if not clip_path and "clip_path" in orig_g.columns:
                vals = [str(x) for x in orig_g["clip_path"].tolist() if str(x).strip()]
                if vals:
                    clip_path = vals[0]

            if not clip_path:
                print("skip no_clip_path", review_id)
                continue

            info = render_review(
                review_id=review_id,
                clean_g=clean_g,
                orig_g=orig_g,
                clip_path=clip_path,
                out_dir=out_dir,
                frame_col=frame_col,
                x_col=x_col,
                y_col=y_col,
                max_frames=args.max_frames,
                tail_age=args.tail_age,
            )

            if info:
                overlays.append(info)
                print("overlay", info["overlay"])

    source_counts = clean_all["point_source_005D4"].astype(str).value_counts().to_dict() if len(clean_all) else {}

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "conservative_table_aware_temporal_relink_and_short_gap_interpolation_for_ball_tracking",
        "scored_points": str(points_path),
        "scene_tables": str(scene_path),
        "params": {
            "min_score": args.min_score,
            "max_gap": args.max_gap,
            "max_metric_bridge_speed": args.max_metric_bridge_speed,
            "max_pixel_bridge_speed": args.max_pixel_bridge_speed,
            "split_metric_speed": args.split_metric_speed,
        },
        "input_points_total": int(len(pts)),
        "relinked_points_total": int(len(clean_all)),
        "original_safe_total": int((clean_all["point_source_005D4"].astype(str).eq("original_safe")).sum()) if len(clean_all) else 0,
        "interpolated_total": int(to_num(clean_all["is_interpolated_005D4"]).fillna(0).astype(int).sum()) if len(clean_all) else 0,
        "bridges_total": int(seg_summary["bridges"].sum()) if len(seg_summary) else 0,
        "tracklets_total": int(seg_summary["tracklets"].sum()) if len(seg_summary) else 0,
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "outputs": {
            "relinked_points": str(out_points),
            "segment_summary": str(out_segments),
            "overlays": overlays,
        },
        "next": "Review relinked overlays. If short gaps are fixed, integrate this as display/track smoothing; for long fast losses, candidate-level detector/Viterbi must be improved."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005D4 status=OK")
    print("input_points_total=", summary["input_points_total"])
    print("relinked_points_total=", summary["relinked_points_total"])
    print("original_safe_total=", summary["original_safe_total"])
    print("interpolated_total=", summary["interpolated_total"])
    print("bridges_total=", summary["bridges_total"])
    print("tracklets_total=", summary["tracklets_total"])
    print("source_counts=", json.dumps(summary["source_counts"], ensure_ascii=False))
    print("overlays=", len(overlays))
    print("wrote", out_points)
    print("wrote", out_segments)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
