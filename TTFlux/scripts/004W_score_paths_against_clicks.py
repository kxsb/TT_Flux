from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "004W"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def nearest_path(path_df: pd.DataFrame, frame: int, window: int):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="runs/rally_apply_ranker_004V_probe12/rally_ranker_paths_004V.csv")
    ap.add_argument("--clicks", default="runs/rally_goldset_004U2_reused_clicks/rally_reused_clicks_004U2.csv")
    ap.add_argument("--out-dir", default="runs/rally_path_score_004W_probe12")
    ap.add_argument("--frame-window", type=int, default=2)
    args = ap.parse_args()

    root = Path.cwd()

    paths_path = Path(args.paths)
    clicks_path = Path(args.clicks)
    out_dir = Path(args.out_dir)

    if not paths_path.is_absolute():
        paths_path = root / paths_path
    if not clicks_path.is_absolute():
        clicks_path = root / clicks_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    paths = pd.read_csv(paths_path).fillna("")
    clicks = pd.read_csv(clicks_path).fillna("")

    clicks = clicks[clicks["visibility"].astype(str).eq("ball")].copy()

    for c in ["local_frame", "source_frame", "x", "y"]:
        clicks[c + "_num"] = to_num(clicks[c])

    clicks = clicks.dropna(subset=["local_frame_num", "x_num", "y_num"]).copy()
    clicks["local_frame_num"] = clicks["local_frame_num"].astype(int)

    for c in ["frame", "x", "y", "score"]:
        paths[c + "_num"] = to_num(paths[c])

    paths = paths.dropna(subset=["frame_num", "x_num", "y_num"]).copy()
    paths["frame_num"] = paths["frame_num"].astype(int)
    paths["score_num"] = to_num(paths["score"]).fillna(0)

    paths_by_review = {
        str(k): g.copy()
        for k, g in paths.groupby("review_id", dropna=False)
    }

    rows = []

    for _, c in clicks.iterrows():
        review_id = str(c["review_id"])
        frame = int(c["local_frame_num"])
        hx = float(c["x_num"])
        hy = float(c["y_num"])

        pg = paths_by_review.get(review_id, pd.DataFrame())
        near = nearest_path(pg, frame, args.frame_window)

        if near is None:
            rows.append({
                "review_id": review_id,
                "video_id": str(c.get("video_id", "")),
                "clip_id": str(c.get("clip_id", "")),
                "click_frame": frame,
                "human_x": round(hx, 3),
                "human_y": round(hy, 3),
                "path_found": 0,
                "path_frame": "",
                "path_x": "",
                "path_y": "",
                "path_score": "",
                "frame_delta": "",
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
            "video_id": str(c.get("video_id", "")),
            "clip_id": str(c.get("clip_id", "")),
            "click_frame": frame,
            "human_x": round(hx, 3),
            "human_y": round(hy, 3),
            "path_found": 1,
            "path_frame": int(near["frame_num"]),
            "path_x": round(px, 3),
            "path_y": round(py, 3),
            "path_score": round(float(near["score_num"]), 6),
            "frame_delta": int(abs(int(near["frame_num"]) - frame)),
            "dist": round(d, 3),
            "hit10": int(d <= 10),
            "hit20": int(d <= 20),
            "hit50": int(d <= 50),
        })

    scored = pd.DataFrame(rows)

    review_rows = []
    if not scored.empty:
        for review_id, g in scored.groupby("review_id", dropna=False):
            found = pd.to_numeric(g["path_found"], errors="coerce").fillna(0)
            dist = pd.to_numeric(g["dist"], errors="coerce")
            score = pd.to_numeric(g["path_score"], errors="coerce")

            review_rows.append({
                "review_id": review_id,
                "clicks": int(len(g)),
                "path_found_ratio": round(float(found.mean()), 4),
                "dist_med": round(float(dist.dropna().median()), 3) if dist.dropna().size else "",
                "hit10": round(float(pd.to_numeric(g["hit10"], errors="coerce").fillna(0).mean()), 4),
                "hit20": round(float(pd.to_numeric(g["hit20"], errors="coerce").fillna(0).mean()), 4),
                "hit50": round(float(pd.to_numeric(g["hit50"], errors="coerce").fillna(0).mean()), 4),
                "path_score_med": round(float(score.dropna().median()), 6) if score.dropna().size else "",
                "path_score_p10": round(float(np.percentile(score.dropna(), 10)), 6) if score.dropna().size else "",
            })

    review_df = pd.DataFrame(review_rows)
    if not review_df.empty:
        review_df = review_df.sort_values(["hit50", "hit20", "dist_med"], ascending=[False, False, True])

    def rate(col):
        if scored.empty:
            return 0.0
        return round(float(pd.to_numeric(scored[col], errors="coerce").fillna(0).mean()), 4)

    dist_all = pd.to_numeric(scored["dist"], errors="coerce").dropna() if not scored.empty else pd.Series(dtype=float)
    score_all = pd.to_numeric(scored["path_score"], errors="coerce").dropna() if not scored.empty else pd.Series(dtype=float)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "score_004V_paths_against_reused_manual_ball_clicks",
        "paths": str(paths_path),
        "clicks": str(clicks_path),
        "frame_window": args.frame_window,
        "ball_clicks_scored": int(len(scored)),
        "reviews_scored": int(scored["review_id"].nunique()) if not scored.empty else 0,
        "global": {
            "path_found_ratio": rate("path_found"),
            "hit10": rate("hit10"),
            "hit20": rate("hit20"),
            "hit50": rate("hit50"),
            "dist_med": round(float(dist_all.median()), 3) if len(dist_all) else None,
            "path_score_med": round(float(score_all.median()), 6) if len(score_all) else None,
            "path_score_p10": round(float(np.percentile(score_all, 10)), 6) if len(score_all) else None,
        },
        "interpretation": {
            "hit50_high": "path sometimes follows the ball on annotated frames",
            "hit50_low": "current continuous Viterbi path is not usable as tracker",
            "next_if_low": "implement 004X intermittent/gated path with no-ball state"
        }
    }

    out_clicks = out_dir / "path_vs_clicks_004W.csv"
    out_reviews = out_dir / "path_vs_clicks_reviews_004W.csv"
    out_json = out_dir / "path_score_summary_004W.json"

    scored.to_csv(out_clicks, index=False, encoding="utf-8")
    review_df.to_csv(out_reviews, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004W status=OK")
    print("ball_clicks_scored=", summary["ball_clicks_scored"])
    print("reviews_scored=", summary["reviews_scored"])
    print("global=", json.dumps(summary["global"], ensure_ascii=False))
    print("wrote", out_clicks)
    print("wrote", out_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
