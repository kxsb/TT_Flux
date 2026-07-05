from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "004Y"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def imwrite_unicode(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def draw_cross(img, x, y, color, label):
    x = int(round(float(x)))
    y = int(round(float(y)))

    cv2.circle(img, (x, y), 7, color, 2, cv2.LINE_AA)
    cv2.line(img, (x - 12, y), (x + 12, y), color, 2, cv2.LINE_AA)
    cv2.line(img, (x, y - 12), (x, y + 12), color, 2, cv2.LINE_AA)
    cv2.putText(
        img,
        label,
        (x + 9, max(20, y - 9)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
        cv2.LINE_AA,
    )


def group_segments(g: pd.DataFrame, max_gap_frames: int) -> pd.DataFrame:
    if g.empty:
        g["tracklet_id_004Y"] = []
        return g

    g = g.sort_values("frame_num").copy()

    seg = 0
    prev_frame = None
    ids = []

    for _, r in g.iterrows():
        f = int(r["frame_num"])

        if prev_frame is None or f - prev_frame > max_gap_frames:
            seg += 1

        ids.append(seg)
        prev_frame = f

    g["tracklet_id_004Y"] = ids
    return g


def nearest_emitted(path_df: pd.DataFrame, frame: int, window: int):
    if path_df.empty:
        return None

    g = path_df[
        (path_df["frame_num"] >= frame - window) &
        (path_df["frame_num"] <= frame + window)
    ].copy()

    if g.empty:
        return None

    g["frame_delta"] = (g["frame_num"] - frame).abs()
    g = g.sort_values(["frame_delta", "score_num"], ascending=[True, False])
    return g.iloc[0].to_dict()


def score_against_clicks(gated: pd.DataFrame, all_reviews: list[str], clicks_path: Path, frame_window: int):
    if not clicks_path.is_file():
        return {}, pd.DataFrame(), pd.DataFrame()

    clicks = pd.read_csv(clicks_path).fillna("")
    clicks = clicks[
        clicks["visibility"].astype(str).eq("ball")
        & clicks["review_id"].astype(str).isin(all_reviews)
    ].copy()

    for c in ["local_frame", "x", "y"]:
        clicks[c + "_num"] = to_num(clicks[c])

    clicks = clicks.dropna(subset=["local_frame_num", "x_num", "y_num"]).copy()
    clicks["local_frame_num"] = clicks["local_frame_num"].astype(int)

    by_review = {
        str(k): g.copy()
        for k, g in gated.groupby("review_id", dropna=False)
    }

    rows = []

    for _, c in clicks.iterrows():
        review_id = str(c["review_id"])
        frame = int(c["local_frame_num"])
        hx = float(c["x_num"])
        hy = float(c["y_num"])

        near = nearest_emitted(by_review.get(review_id, pd.DataFrame()), frame, frame_window)

        if near is None:
            rows.append({
                "review_id": review_id,
                "click_frame": frame,
                "human_x": round(hx, 3),
                "human_y": round(hy, 3),
                "path_found": 0,
                "path_frame": "",
                "path_x": "",
                "path_y": "",
                "path_score": "",
                "dist": "",
                "hit10": 0,
                "hit20": 0,
                "hit50": 0,
            })
            continue

        px = float(near["x_num"])
        py = float(near["y_num"])
        d = math.hypot(px - hx, py - hy)

        rows.append({
            "review_id": review_id,
            "click_frame": frame,
            "human_x": round(hx, 3),
            "human_y": round(hy, 3),
            "path_found": 1,
            "path_frame": int(near["frame_num"]),
            "path_x": round(px, 3),
            "path_y": round(py, 3),
            "path_score": round(float(near["score_num"]), 6),
            "dist": round(d, 3),
            "hit10": int(d <= 10),
            "hit20": int(d <= 20),
            "hit50": int(d <= 50),
        })

    scored = pd.DataFrame(rows)

    review_rows = []
    if not scored.empty:
        for review_id, g in scored.groupby("review_id", dropna=False):
            dist = to_num(g["dist"]).dropna()
            score = to_num(g["path_score"]).dropna()

            review_rows.append({
                "review_id": review_id,
                "clicks": int(len(g)),
                "found_ratio": round(float(to_num(g["path_found"]).fillna(0).mean()), 4),
                "hit10_total": round(float(to_num(g["hit10"]).fillna(0).mean()), 4),
                "hit20_total": round(float(to_num(g["hit20"]).fillna(0).mean()), 4),
                "hit50_total": round(float(to_num(g["hit50"]).fillna(0).mean()), 4),
                "hit10_found": round(float(to_num(g[g["path_found"] == 1]["hit10"]).fillna(0).mean()), 4) if int(to_num(g["path_found"]).sum()) else 0,
                "hit20_found": round(float(to_num(g[g["path_found"] == 1]["hit20"]).fillna(0).mean()), 4) if int(to_num(g["path_found"]).sum()) else 0,
                "hit50_found": round(float(to_num(g[g["path_found"] == 1]["hit50"]).fillna(0).mean()), 4) if int(to_num(g["path_found"]).sum()) else 0,
                "dist_med_found": round(float(dist.median()), 3) if len(dist) else "",
                "score_med_found": round(float(score.median()), 6) if len(score) else "",
            })

    review_df = pd.DataFrame(review_rows)

    if not review_df.empty:
        review_df = review_df.sort_values(["hit50_found", "found_ratio", "hit20_found"], ascending=[False, False, False])

    def rate(col):
        if scored.empty:
            return 0.0
        return round(float(to_num(scored[col]).fillna(0).mean()), 4)

    found = scored[scored["path_found"] == 1].copy() if not scored.empty else pd.DataFrame()
    dist_found = to_num(found["dist"]).dropna() if not found.empty else pd.Series(dtype=float)

    score_summary = {
        "clicks": int(len(scored)),
        "reviews": int(scored["review_id"].nunique()) if not scored.empty else 0,
        "found_ratio": rate("path_found"),
        "hit10_total": rate("hit10"),
        "hit20_total": rate("hit20"),
        "hit50_total": rate("hit50"),
        "hit10_found": round(float(to_num(found["hit10"]).fillna(0).mean()), 4) if not found.empty else 0,
        "hit20_found": round(float(to_num(found["hit20"]).fillna(0).mean()), 4) if not found.empty else 0,
        "hit50_found": round(float(to_num(found["hit50"]).fillna(0).mean()), 4) if not found.empty else 0,
        "dist_med_found": round(float(dist_found.median()), 3) if len(dist_found) else None,
    }

    return score_summary, scored, review_df


def make_overlay_video(
    review_id: str,
    clip_path: Path,
    points: pd.DataFrame,
    out_path: Path,
    max_frames: int = 0,
):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if max_frames > 0:
        frame_limit = min(frame_count, max_frames)
    else:
        frame_limit = frame_count

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    by_frame = {
        int(r["frame_num"]): r
        for _, r in points.iterrows()
    }

    trail = []

    fidx = 0

    while fidx < frame_limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        p = by_frame.get(fidx)

        if p is not None:
            x = float(p["x_num"])
            y = float(p["y_num"])
            score = float(p["score_num"])

            trail.append((x, y))
            if len(trail) > 18:
                trail = trail[-18:]

            for a, b in zip(trail[:-1], trail[1:]):
                cv2.line(
                    frame,
                    (int(round(a[0])), int(round(a[1]))),
                    (int(round(b[0])), int(round(b[1]))),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            draw_cross(frame, x, y, (0, 255, 0), f"BALL_OK {score:.2f}")
            state = "BALL_OK"
        else:
            state = "NO_BALL"
            cv2.putText(
                frame,
                "NO_BALL",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"{review_id} f={fidx} state={state}",
            (24, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="runs/rally_apply_ranker_004V_annotated17/rally_ranker_paths_004V.csv")
    ap.add_argument("--review-summary", default="runs/rally_apply_ranker_004V_annotated17/rally_ranker_review_summary_004V.csv")
    ap.add_argument("--clicks", default="runs/rally_goldset_004U2_reused_clicks/rally_reused_clicks_004U2.csv")
    ap.add_argument("--out-dir", default="runs/rally_gated_paths_004Y_annotated17")
    ap.add_argument("--point-score-min", type=float, default=0.85)
    ap.add_argument("--review-score-min", type=float, default=0.50)
    ap.add_argument("--max-gap-frames", type=int, default=8)
    ap.add_argument("--frame-window", type=int, default=2)
    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--max-videos", type=int, default=12)
    ap.add_argument("--max-video-frames", type=int, default=0)
    args = ap.parse_args()

    root = Path.cwd()

    paths_path = Path(args.paths)
    review_path = Path(args.review_summary)
    clicks_path = Path(args.clicks)
    out_dir = Path(args.out_dir)

    if not paths_path.is_absolute():
        paths_path = root / paths_path
    if not review_path.is_absolute():
        review_path = root / review_path
    if not clicks_path.is_absolute():
        clicks_path = root / clicks_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    paths = pd.read_csv(paths_path).fillna("")
    reviews = pd.read_csv(review_path).fillna("")

    for c in ["frame", "x", "y", "score"]:
        paths[c + "_num"] = to_num(paths[c])

    paths = paths.dropna(subset=["frame_num", "x_num", "y_num", "score_num"]).copy()
    paths["frame_num"] = paths["frame_num"].astype(int)

    if "path_score_med" in reviews.columns:
        reviews["review_score_med_004Y"] = to_num(reviews["path_score_med"]).fillna(-1)
    else:
        reviews["review_score_med_004Y"] = reviews["review_id"].astype(str).map(
            paths.groupby(paths["review_id"].astype(str))["score_num"].median().to_dict()
        ).fillna(-1)

    review_score = dict(zip(reviews["review_id"].astype(str), reviews["review_score_med_004Y"]))

    paths["review_score_med_004Y"] = paths["review_id"].astype(str).map(review_score).fillna(-1)

    paths["emitted_004Y"] = (
        (paths["score_num"] >= args.point_score_min)
        & (paths["review_score_med_004Y"] >= args.review_score_min)
    ).astype(int)

    paths["state_004Y"] = np.where(paths["emitted_004Y"] == 1, "BALL_OK", "NO_BALL")

    gated = paths[paths["emitted_004Y"] == 1].copy()

    gated_parts = []
    for review_id, g in gated.groupby("review_id", dropna=False):
        gated_parts.append(group_segments(g, max_gap_frames=args.max_gap_frames))

    gated = pd.concat(gated_parts, ignore_index=True) if gated_parts else pd.DataFrame(columns=list(paths.columns) + ["tracklet_id_004Y"])

    review_rows = []

    for review_id, g_all in paths.groupby("review_id", dropna=False):
        review_id = str(review_id)
        g_emit = gated[gated["review_id"].astype(str).eq(review_id)].copy()

        clip_path = ""
        if "clip_path" in reviews.columns:
            m = reviews[reviews["review_id"].astype(str).eq(review_id)]
            if not m.empty:
                clip_path = str(m.iloc[0].get("clip_path", ""))

        segments = int(g_emit["tracklet_id_004Y"].nunique()) if not g_emit.empty and "tracklet_id_004Y" in g_emit.columns else 0

        if not g_emit.empty and "tracklet_id_004Y" in g_emit.columns:
            longest = int(g_emit.groupby("tracklet_id_004Y").size().max())
        else:
            longest = 0

        review_rows.append({
            "review_id": review_id,
            "video_id": str(g_all.iloc[0].get("video_id", "")),
            "rally_id": str(g_all.iloc[0].get("rally_id", "")),
            "path_points": int(len(g_all)),
            "emitted_points": int(len(g_emit)),
            "coverage_points": round(float(len(g_emit) / max(1, len(g_all))), 4),
            "tracklets": segments,
            "longest_tracklet_points": longest,
            "score_med_all": round(float(g_all["score_num"].median()), 6) if len(g_all) else "",
            "score_med_emitted": round(float(g_emit["score_num"].median()), 6) if len(g_emit) else "",
            "review_score_med_004Y": round(float(review_score.get(review_id, -1)), 6),
            "first_emit_frame": int(g_emit["frame_num"].min()) if len(g_emit) else "",
            "last_emit_frame": int(g_emit["frame_num"].max()) if len(g_emit) else "",
            "clip_path": clip_path,
            "overlay": "",
        })

    review_df = pd.DataFrame(review_rows)

    overlays = []

    if args.make_video and not review_df.empty:
        # Priorité : reviews avec beaucoup de points émis, puis quelques mauvaises.
        viz = review_df.sort_values(["emitted_points", "review_score_med_004Y"], ascending=[False, False]).head(args.max_videos).copy()

        for i, (_, r) in enumerate(viz.iterrows(), start=1):
            review_id = str(r["review_id"])
            clip_path = Path(str(r.get("clip_path", "")))

            if not clip_path.is_absolute():
                clip_path = root / clip_path

            if not clip_path.is_file():
                continue

            pts = gated[gated["review_id"].astype(str).eq(review_id)].copy()

            out_video = out_dir / "overlays" / f"{i:02d}_{review_id}_004Y_gated_overlay.mp4"

            ok = make_overlay_video(
                review_id=review_id,
                clip_path=clip_path,
                points=pts,
                out_path=out_video,
                max_frames=args.max_video_frames,
            )

            if ok:
                overlays.append({
                    "review_id": review_id,
                    "overlay": str(out_video),
                })
                review_df.loc[review_df["review_id"].astype(str).eq(review_id), "overlay"] = str(out_video)
                print(f"  overlay {i}/{len(viz)} {review_id} -> {out_video}")

    score_summary, scored_clicks, scored_reviews = score_against_clicks(
        gated=gated,
        all_reviews=sorted(paths["review_id"].astype(str).unique().tolist()),
        clicks_path=clicks_path,
        frame_window=args.frame_window,
    )

    out_flags = out_dir / "path_gate_flags_004Y.csv"
    out_gated = out_dir / "gated_points_004Y.csv"
    out_reviews = out_dir / "gated_review_summary_004Y.csv"
    out_clicks = out_dir / "gated_vs_clicks_004Y.csv"
    out_click_reviews = out_dir / "gated_vs_clicks_reviews_004Y.csv"
    out_json = out_dir / "gated_summary_004Y.json"

    paths.to_csv(out_flags, index=False, encoding="utf-8")
    gated.to_csv(out_gated, index=False, encoding="utf-8")
    review_df.to_csv(out_reviews, index=False, encoding="utf-8")
    scored_clicks.to_csv(out_clicks, index=False, encoding="utf-8")
    scored_reviews.to_csv(out_click_reviews, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "emit_intermittent_high_confidence_ball_points_only",
        "paths": str(paths_path),
        "review_summary": str(review_path),
        "clicks": str(clicks_path),
        "point_score_min": args.point_score_min,
        "review_score_min": args.review_score_min,
        "max_gap_frames": args.max_gap_frames,
        "input_path_points": int(len(paths)),
        "emitted_points": int(len(gated)),
        "coverage_points": round(float(len(gated) / max(1, len(paths))), 4),
        "reviews": int(paths["review_id"].nunique()),
        "reviews_with_emit": int(gated["review_id"].nunique()) if len(gated) else 0,
        "click_score": score_summary,
        "overlays": overlays,
        "outputs": {
            "flags": str(out_flags),
            "gated_points": str(out_gated),
            "review_summary": str(out_reviews),
            "gated_vs_clicks": str(out_clicks),
            "gated_vs_clicks_reviews": str(out_click_reviews),
        },
        "next": "Review gated overlays. If good, run 004V without videos on all 160 rallies, then apply this same 004Y gate."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004Y status=OK")
    print("input_path_points=", summary["input_path_points"])
    print("emitted_points=", summary["emitted_points"])
    print("coverage_points=", summary["coverage_points"])
    print("reviews=", summary["reviews"])
    print("reviews_with_emit=", summary["reviews_with_emit"])
    print("click_score=", json.dumps(score_summary, ensure_ascii=False))
    print("wrote", out_flags)
    print("wrote", out_gated)
    print("wrote", out_reviews)
    print("wrote", out_clicks)
    print("wrote", out_click_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
