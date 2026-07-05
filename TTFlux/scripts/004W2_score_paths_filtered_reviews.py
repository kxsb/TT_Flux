from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "004W2"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def nearest_path(path_df: pd.DataFrame, frame: int, window: int):
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
    ap.add_argument("--out-dir", default="runs/rally_path_score_004W2_probe12_filtered")
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

    paths["frame_num"] = to_num(paths["frame"])
    paths["x_num"] = to_num(paths["x"])
    paths["y_num"] = to_num(paths["y"])
    paths["score_num"] = to_num(paths["score"]).fillna(0.0)
    paths = paths.dropna(subset=["frame_num", "x_num", "y_num"]).copy()
    paths["frame_num"] = paths["frame_num"].astype(int)

    path_reviews = sorted(paths["review_id"].astype(str).unique().tolist())

    clicks = clicks[
        clicks["visibility"].astype(str).eq("ball")
        & clicks["review_id"].astype(str).isin(path_reviews)
    ].copy()

    clicks["local_frame_num"] = to_num(clicks["local_frame"])
    clicks["x_num"] = to_num(clicks["x"])
    clicks["y_num"] = to_num(clicks["y"])
    clicks = clicks.dropna(subset=["local_frame_num", "x_num", "y_num"]).copy()
    clicks["local_frame_num"] = clicks["local_frame_num"].astype(int)

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

        near = nearest_path(paths_by_review.get(review_id, pd.DataFrame()), frame, args.frame_window)

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
    for review_id, g in scored.groupby("review_id", dropna=False):
        dist = to_num(g["dist"]).dropna()
        score = to_num(g["path_score"]).dropna()

        review_rows.append({
            "review_id": review_id,
            "clicks": int(len(g)),
            "path_found_ratio": round(float(to_num(g["path_found"]).fillna(0).mean()), 4),
            "hit10": round(float(to_num(g["hit10"]).fillna(0).mean()), 4),
            "hit20": round(float(to_num(g["hit20"]).fillna(0).mean()), 4),
            "hit50": round(float(to_num(g["hit50"]).fillna(0).mean()), 4),
            "dist_med": round(float(dist.median()), 3) if len(dist) else "",
            "path_score_med": round(float(score.median()), 6) if len(score) else "",
        })

    review_df = pd.DataFrame(review_rows)
    if not review_df.empty:
        review_df = review_df.sort_values(["hit50", "hit20", "dist_med"], ascending=[False, False, True])

    def rate(col):
        if scored.empty:
            return 0.0
        return round(float(to_num(scored[col]).fillna(0).mean()), 4)

    dist_all = to_num(scored["dist"]).dropna() if not scored.empty else pd.Series(dtype=float)
    score_all = to_num(scored["path_score"]).dropna() if not scored.empty else pd.Series(dtype=float)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "score_only_clicks_for_reviews_present_in_004V_paths",
        "paths": str(paths_path),
        "clicks": str(clicks_path),
        "path_reviews": path_reviews,
        "path_review_count": len(path_reviews),
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
        }
    }

    out_clicks = out_dir / "path_vs_clicks_004W2_filtered.csv"
    out_reviews = out_dir / "path_vs_clicks_reviews_004W2_filtered.csv"
    out_json = out_dir / "path_score_summary_004W2_filtered.json"

    scored.to_csv(out_clicks, index=False, encoding="utf-8")
    review_df.to_csv(out_reviews, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004W2 status=OK")
    print("path_review_count=", summary["path_review_count"])
    print("ball_clicks_scored=", summary["ball_clicks_scored"])
    print("reviews_scored=", summary["reviews_scored"])
    print("global=", json.dumps(summary["global"], ensure_ascii=False))
    print("wrote", out_clicks)
    print("wrote", out_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
