from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005D3"


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


def open_writer(path: Path, fps: float, w: int, h: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer: {path}")
    return writer


def draw_label(img, text, y=28):
    cv2.putText(
        img,
        text,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_point(img, x, y, score, zone, project_ok, safe):
    x = int(round(float(x)))
    y = int(round(float(y)))

    if not project_ok:
        color = (255, 220, 40)      # cyan/bleu clair : pas de métrique
        radius = 6
    elif score >= 0.70:
        color = (0, 255, 0)         # vert
        radius = 6
    elif score >= 0.55:
        color = (0, 210, 255)       # orange/jaune
        radius = 6
    elif safe:
        color = (0, 150, 255)       # douteux mais gardé
        radius = 6
    else:
        color = (0, 0, 255)         # rouge rejet/low
        radius = 7

    cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), radius + 2, (0, 0, 0), 1, cv2.LINE_AA)

    if project_ok and not safe:
        cv2.line(img, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(img, (x - 10, y + 10), (x + 10, y - 10), (0, 0, 255), 2, cv2.LINE_AA)


def draw_track_tail(img, rows, x_col, y_col, frame_col, current_frame, max_age=20):
    if rows.empty:
        return

    rows = rows.copy()
    rows[frame_col] = to_num(rows[frame_col]).fillna(-1).astype(int)
    rows = rows[
        rows[frame_col].between(current_frame - max_age, current_frame)
    ].sort_values(frame_col).copy()

    pts = []

    for _, r in rows.iterrows():
        safe = int(to_num(pd.Series([r.get("ball_table_safe_005D2", 1)])).fillna(1).iloc[0]) == 1
        score = float(to_num(pd.Series([r.get("ball_table_score_005D2", 0.42)])).fillna(0.42).iloc[0])
        project_ok = int(to_num(pd.Series([r.get("table_project_ok_005C9B", 0)])).fillna(0).iloc[0]) == 1

        if not safe or score < 0.55:
            continue

        try:
            pts.append((int(round(float(r[x_col]))), int(round(float(r[y_col])))))
        except Exception:
            continue

    if len(pts) >= 2:
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (90, 220, 90), 2, cv2.LINE_AA)


def choose_reviews(segment_summary: pd.DataFrame, max_videos: int):
    if segment_summary.empty:
        return []

    df = segment_summary.copy()

    for c in ["metric_low_score", "metric_points", "metric_safe", "score_med_metric"]:
        if c in df.columns:
            df[c] = to_num(df[c]).fillna(0)

    # Priorité : segments où le score a quelque chose à retirer / diagnostiquer.
    df = df.sort_values(
        ["metric_low_score", "metric_points"],
        ascending=[False, False],
    )

    reviews = []

    for r in df["review_id"].astype(str).tolist():
        if r not in reviews:
            reviews.append(r)
        if len(reviews) >= max_videos:
            break

    return reviews


def render_review(
    review_id: str,
    g: pd.DataFrame,
    out_dir: Path,
    frame_col: str,
    x_col: str,
    y_col: str,
    max_frames: int,
    tail_age: int,
):
    g = g.copy()
    g[frame_col] = to_num(g[frame_col]).fillna(-1).astype(int)
    g[x_col] = to_num(g[x_col])
    g[y_col] = to_num(g[y_col])

    clip_path = Path(str(g["clip_path"].dropna().astype(str).iloc[0]))

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if w <= 0 or h <= 0:
        cap.release()
        return None

    first = max(0, int(g[frame_col].min()) - 20)
    last = min(n_frames - 1, int(g[frame_col].max()) + 20)

    if max_frames > 0:
        last = min(last, first + max_frames - 1)

    out_path = out_dir / "overlays" / f"{review_id}_005D3_table_score_overlay.mp4"
    writer = open_writer(out_path, fps=fps, w=w, h=h)

    by_frame = {int(k): v.copy() for k, v in g.groupby(frame_col)}

    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    frame_idx = first
    written = 0

    while frame_idx <= last:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        rows = by_frame.get(frame_idx, pd.DataFrame())

        tail_rows = g[g[frame_col].between(frame_idx - tail_age, frame_idx)].copy()
        draw_track_tail(frame, tail_rows, x_col=x_col, y_col=y_col, frame_col=frame_col, current_frame=frame_idx, max_age=tail_age)

        if not rows.empty:
            for _, r in rows.iterrows():
                try:
                    x = float(r[x_col])
                    y = float(r[y_col])
                except Exception:
                    continue

                score = float(to_num(pd.Series([r.get("ball_table_score_005D2", 0.42)])).fillna(0.42).iloc[0])
                zone = str(r.get("ball_table_context_zone_005D2", ""))
                project_ok = int(to_num(pd.Series([r.get("table_project_ok_005C9B", 0)])).fillna(0).iloc[0]) == 1
                safe = int(to_num(pd.Series([r.get("ball_table_safe_005D2", 1)])).fillna(1).iloc[0]) == 1

                draw_point(frame, x, y, score=score, zone=zone, project_ok=project_ok, safe=safe)

        metric = int(to_num(g["table_project_ok_005C9B"]).fillna(0).sum())
        safe_metric = int(
            (
                to_num(g["table_project_ok_005C9B"]).fillna(0).astype(int).eq(1)
                & to_num(g["ball_table_metric_safe_005D2"]).fillna(0).astype(int).eq(1)
            ).sum()
        )

        draw_label(frame, f"{review_id} 005D3 table-score overlay  metric_safe={safe_metric}/{metric}", y=30)
        draw_label(frame, "green=safe  orange=doubt  red=low/reject  cyan=no metric table", y=58)

        writer.write(frame)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()

    return {
        "review_id": review_id,
        "overlay": str(out_path),
        "clip_path": str(clip_path),
        "first_frame": first,
        "last_frame": frame_idx - 1,
        "written_frames": written,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-points", default="runs/rally_ball_table_score_005D2/ball_points_table_scored_005D2.csv")
    ap.add_argument("--segment-summary", default="runs/rally_ball_table_score_005D2/ball_table_score_segment_summary_005D2.csv")
    ap.add_argument("--out-dir", default="runs/rally_ball_table_score_overlay_005D3")
    ap.add_argument("--max-videos", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--tail-age", type=int, default=16)
    args = ap.parse_args()

    root = Path.cwd()

    points_path = Path(args.scored_points)
    segment_path = Path(args.segment_summary)
    out_dir = Path(args.out_dir)

    if not points_path.is_absolute():
        points_path = root / points_path
    if not segment_path.is_absolute():
        segment_path = root / segment_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    pts = pd.read_csv(points_path).fillna("")
    segs = pd.read_csv(segment_path).fillna("") if segment_path.is_file() else pd.DataFrame()

    frame_col = find_col(pts, ["frame_num", "frame", "frame_idx"])
    x_col = find_col(pts, ["x_num", "x", "cx"])
    y_col = find_col(pts, ["y_num", "y", "cy"])

    reviews = choose_reviews(segs, max_videos=args.max_videos)

    if not reviews:
        reviews = pts["review_id"].astype(str).drop_duplicates().head(args.max_videos).tolist()

    overlays = []

    for review_id in reviews:
        g = pts[pts["review_id"].astype(str).eq(str(review_id))].copy()
        if g.empty:
            continue

        info = render_review(
            review_id=review_id,
            g=g,
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

    out_json = out_dir / "ball_table_score_overlay_summary_005D3.json"

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "visual_video_overlay_for_table_aware_ball_score_validation",
        "scored_points": str(points_path),
        "segment_summary": str(segment_path),
        "max_videos": args.max_videos,
        "max_frames": args.max_frames,
        "tail_age": args.tail_age,
        "reviews": reviews,
        "overlays": overlays,
        "next": "Human review overlays. If red/orange points correspond to false positives, integrate score_005D2 into candidate Viterbi/reranking."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005D3 status=OK")
    print("reviews_requested=", len(reviews))
    print("overlays=", len(overlays))
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
