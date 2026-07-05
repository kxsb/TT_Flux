from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd


VERSION = "005D3B"


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


def draw_label(img, text, y):
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)


def draw_point(img, x, y, score, project_ok, safe):
    x = int(round(float(x)))
    y = int(round(float(y)))

    if not project_ok:
        color = (255, 220, 40)      # cyan : pas de table métrique
        radius = 6
    elif score >= 0.70:
        color = (0, 255, 0)         # vert : safe
        radius = 6
    elif score >= 0.55:
        color = (0, 210, 255)       # orange : gardé mais douteux
        radius = 6
    elif safe:
        color = (0, 150, 255)       # douteux gardé
        radius = 6
    else:
        color = (0, 0, 255)         # rouge : low/reject
        radius = 7

    cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), radius + 2, (0, 0, 0), 1, cv2.LINE_AA)

    if project_ok and not safe:
        cv2.line(img, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(img, (x - 10, y + 10), (x + 10, y - 10), (0, 0, 255), 2, cv2.LINE_AA)


def build_clip_map(scene: pd.DataFrame):
    out = {}

    if "clip_path" not in scene.columns:
        return out

    for review_id, g in scene.groupby("review_id", dropna=False):
        vals = [str(x) for x in g["clip_path"].tolist() if str(x).strip()]
        if vals:
            out[str(review_id)] = vals[0]

    return out


def choose_reviews(seg_summary: pd.DataFrame, pts: pd.DataFrame, max_videos: int):
    reviews = []

    if not seg_summary.empty and "review_id" in seg_summary.columns:
        df = seg_summary.copy()

        for c in ["metric_low_score", "metric_points"]:
            if c in df.columns:
                df[c] = to_num(df[c]).fillna(0)

        sort_cols = [c for c in ["metric_low_score", "metric_points"] if c in df.columns]

        if sort_cols:
            df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))

        for r in df["review_id"].astype(str).tolist():
            if r not in reviews:
                reviews.append(r)
            if len(reviews) >= max_videos:
                break

    if len(reviews) < max_videos:
        for r in pts["review_id"].astype(str).drop_duplicates().tolist():
            if r not in reviews:
                reviews.append(r)
            if len(reviews) >= max_videos:
                break

    return reviews


def draw_tail(frame, g, frame_col, x_col, y_col, current_frame, tail_age):
    gg = g.copy()
    gg[frame_col] = to_num(gg[frame_col]).fillna(-1).astype(int)
    gg = gg[gg[frame_col].between(current_frame - tail_age, current_frame)].sort_values(frame_col)

    pts = []

    for _, r in gg.iterrows():
        score = float(to_num(pd.Series([r.get("ball_table_score_005D2", 0.42)])).fillna(0.42).iloc[0])
        safe = int(to_num(pd.Series([r.get("ball_table_safe_005D2", 1)])).fillna(1).iloc[0]) == 1

        if not safe or score < 0.55:
            continue

        try:
            pts.append((int(round(float(r[x_col]))), int(round(float(r[y_col])))))
        except Exception:
            pass

    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(frame, a, b, (90, 220, 90), 2, cv2.LINE_AA)


def render_review(review_id, g, clip_path, out_dir, frame_col, x_col, y_col, max_frames, tail_age):
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

    g = g.copy()
    g[frame_col] = to_num(g[frame_col]).fillna(-1).astype(int)
    g[x_col] = to_num(g[x_col])
    g[y_col] = to_num(g[y_col])

    first = max(0, int(g[frame_col].min()) - 20)
    last = min(n_frames - 1, int(g[frame_col].max()) + 20)

    if max_frames > 0:
        last = min(last, first + max_frames - 1)

    by_frame = {int(k): v.copy() for k, v in g.groupby(frame_col)}

    out_path = out_dir / "overlays" / f"{review_id}_005D3B_table_score_overlay.mp4"
    wr = open_writer(out_path, fps=fps, w=w, h=h)

    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    written = 0
    frame_idx = first

    metric_total = int(to_num(g.get("table_project_ok_005C9B", pd.Series([]))).fillna(0).astype(int).sum())
    metric_safe = int(
        (
            to_num(g.get("table_project_ok_005C9B", pd.Series([]))).fillna(0).astype(int).eq(1)
            & to_num(g.get("ball_table_metric_safe_005D2", pd.Series([]))).fillna(0).astype(int).eq(1)
        ).sum()
    )

    while frame_idx <= last:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        draw_tail(frame, g, frame_col, x_col, y_col, frame_idx, tail_age)

        rows = by_frame.get(frame_idx)

        if rows is not None and len(rows):
            for _, r in rows.iterrows():
                try:
                    x = float(r[x_col])
                    y = float(r[y_col])
                except Exception:
                    continue

                score = float(to_num(pd.Series([r.get("ball_table_score_005D2", 0.42)])).fillna(0.42).iloc[0])
                project_ok = int(to_num(pd.Series([r.get("table_project_ok_005C9B", 0)])).fillna(0).iloc[0]) == 1
                safe = int(to_num(pd.Series([r.get("ball_table_safe_005D2", 1)])).fillna(1).iloc[0]) == 1

                draw_point(frame, x, y, score, project_ok, safe)

        draw_label(frame, f"{review_id} 005D3B table-score overlay metric_safe={metric_safe}/{metric_total}", 30)
        draw_label(frame, "green=safe  orange=doubt  red=low/reject  cyan=no metric table", 58)

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
        "metric_total": metric_total,
        "metric_safe": metric_safe,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-points", default="runs/rally_ball_table_score_005D2/ball_points_table_scored_005D2.csv")
    ap.add_argument("--segment-summary", default="runs/rally_ball_table_score_005D2/ball_table_score_segment_summary_005D2.csv")
    ap.add_argument("--scene-tables", default="runs/rally_scene_table_objects_005C9B/scene_table_objects_005C9B.csv")
    ap.add_argument("--out-dir", default="runs/rally_ball_table_score_overlay_005D3B")
    ap.add_argument("--max-videos", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--tail-age", type=int, default=16)
    args = ap.parse_args()

    root = Path.cwd()

    points_path = Path(args.scored_points)
    segment_path = Path(args.segment_summary)
    scene_path = Path(args.scene_tables)
    out_dir = Path(args.out_dir)

    if not points_path.is_absolute():
        points_path = root / points_path
    if not segment_path.is_absolute():
        segment_path = root / segment_path
    if not scene_path.is_absolute():
        scene_path = root / scene_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    pts = pd.read_csv(points_path).fillna("")
    segs = pd.read_csv(segment_path).fillna("") if segment_path.is_file() else pd.DataFrame()
    scene = pd.read_csv(scene_path).fillna("") if scene_path.is_file() else pd.DataFrame()

    clip_map = build_clip_map(scene)

    frame_col = find_col(pts, ["frame_num", "frame", "frame_idx"])
    x_col = find_col(pts, ["x_num", "x", "cx"])
    y_col = find_col(pts, ["y_num", "y", "cy"])

    reviews = choose_reviews(segs, pts, args.max_videos)

    overlays = []

    for review_id in reviews:
        g = pts[pts["review_id"].astype(str).eq(str(review_id))].copy()

        if g.empty:
            continue

        clip_path = None

        if "clip_path" in g.columns:
            vals = [str(x) for x in g["clip_path"].tolist() if str(x).strip()]
            if vals:
                clip_path = vals[0]

        if not clip_path:
            clip_path = clip_map.get(str(review_id))

        if not clip_path:
            print("skip no_clip_path", review_id)
            continue

        info = render_review(
            review_id=review_id,
            g=g,
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

    out_json = out_dir / "ball_table_score_overlay_summary_005D3B.json"

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "visual_video_overlay_for_table_aware_ball_score_validation_with_scene_table_clip_path_lookup",
        "scored_points": str(points_path),
        "segment_summary": str(segment_path),
        "scene_tables": str(scene_path),
        "clip_map_count": int(len(clip_map)),
        "reviews_requested": int(len(reviews)),
        "overlays": overlays,
        "next": "Review overlays. If red/orange points correspond to false positives, integrate table score into candidate reranking."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005D3B status=OK")
    print("clip_map_count=", summary["clip_map_count"])
    print("reviews_requested=", summary["reviews_requested"])
    print("overlays=", len(overlays))
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
