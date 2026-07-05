from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "004X"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def eval_subset(df: pd.DataFrame, emit_mask: pd.Series) -> dict:
    total = len(df)
    emitted = int(emit_mask.sum())

    if total == 0:
        return {}

    if emitted == 0:
        return {
            "clicks": total,
            "emitted": 0,
            "coverage": 0.0,
            "hit10_emitted": 0.0,
            "hit20_emitted": 0.0,
            "hit50_emitted": 0.0,
            "hit10_total": 0.0,
            "hit20_total": 0.0,
            "hit50_total": 0.0,
            "dist_med_emitted": None,
        }

    sub = df[emit_mask].copy()

    return {
        "clicks": total,
        "emitted": emitted,
        "coverage": round(emitted / total, 4),
        "hit10_emitted": round(float(to_num(sub["hit10"]).fillna(0).mean()), 4),
        "hit20_emitted": round(float(to_num(sub["hit20"]).fillna(0).mean()), 4),
        "hit50_emitted": round(float(to_num(sub["hit50"]).fillna(0).mean()), 4),
        "hit10_total": round(float((to_num(df["hit10"]).fillna(0) * emit_mask.astype(int)).mean()), 4),
        "hit20_total": round(float((to_num(df["hit20"]).fillna(0) * emit_mask.astype(int)).mean()), 4),
        "hit50_total": round(float((to_num(df["hit50"]).fillna(0) * emit_mask.astype(int)).mean()), 4),
        "dist_med_emitted": round(float(to_num(sub["dist"]).dropna().median()), 3) if to_num(sub["dist"]).dropna().size else None,
        "score_med_emitted": round(float(to_num(sub["path_score"]).dropna().median()), 6) if to_num(sub["path_score"]).dropna().size else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--click-scores", default="runs/rally_path_score_004W2_annotated17/path_vs_clicks_004W2_filtered.csv")
    ap.add_argument("--review-scores", default="runs/rally_path_score_004W2_annotated17/path_vs_clicks_reviews_004W2_filtered.csv")
    ap.add_argument("--out-dir", default="runs/rally_confidence_gate_004X")
    args = ap.parse_args()

    root = Path.cwd()

    click_scores = Path(args.click_scores)
    review_scores = Path(args.review_scores)
    out_dir = Path(args.out_dir)

    if not click_scores.is_absolute():
        click_scores = root / click_scores
    if not review_scores.is_absolute():
        review_scores = root / review_scores
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    clicks = pd.read_csv(click_scores).fillna("")
    reviews = pd.read_csv(review_scores).fillna("")

    clicks["path_score_num"] = to_num(clicks["path_score"]).fillna(-1)
    clicks["dist_num"] = to_num(clicks["dist"])
    clicks["hit10_num"] = to_num(clicks["hit10"]).fillna(0)
    clicks["hit20_num"] = to_num(clicks["hit20"]).fillna(0)
    clicks["hit50_num"] = to_num(clicks["hit50"]).fillna(0)

    # Normalise column aliases for eval_subset.
    clicks["hit10"] = clicks["hit10_num"]
    clicks["hit20"] = clicks["hit20_num"]
    clicks["hit50"] = clicks["hit50_num"]
    clicks["dist"] = clicks["dist_num"]
    clicks["path_score"] = clicks["path_score_num"]

    reviews["path_score_med_num"] = to_num(reviews["path_score_med"]).fillna(-1)
    reviews["hit50_num"] = to_num(reviews["hit50"]).fillna(0)
    reviews["hit20_num"] = to_num(reviews["hit20"]).fillna(0)
    reviews["dist_med_num"] = to_num(reviews["dist_med"])

    review_score_map = dict(zip(reviews["review_id"].astype(str), reviews["path_score_med_num"]))
    clicks["review_path_score_med"] = clicks["review_id"].astype(str).map(review_score_map).fillna(-1)

    point_rows = []
    for th in np.round(np.arange(0.05, 0.951, 0.025), 3):
        emit = clicks["path_score_num"] >= th
        row = {"gate_type": "point_score", "point_score_min": float(th), "review_score_min": ""}
        row.update(eval_subset(clicks, emit))
        point_rows.append(row)

    review_rows = []
    for th in np.round(np.arange(0.05, 0.951, 0.025), 3):
        emit = clicks["review_path_score_med"] >= th
        row = {"gate_type": "review_score", "point_score_min": "", "review_score_min": float(th)}
        row.update(eval_subset(clicks, emit))
        review_rows.append(row)

    combo_rows = []
    for rth in np.round(np.arange(0.20, 0.951, 0.05), 3):
        for pth in np.round(np.arange(0.20, 0.951, 0.05), 3):
            emit = (clicks["review_path_score_med"] >= rth) & (clicks["path_score_num"] >= pth)
            row = {"gate_type": "combo", "point_score_min": float(pth), "review_score_min": float(rth)}
            row.update(eval_subset(clicks, emit))
            combo_rows.append(row)

    all_rows = pd.DataFrame(point_rows + review_rows + combo_rows)

    # Score de choix : on veut un bon hit50 sur points émis, sans couverture ridicule.
    all_rows["coverage_num"] = to_num(all_rows["coverage"]).fillna(0)
    all_rows["hit50_emitted_num"] = to_num(all_rows["hit50_emitted"]).fillna(0)
    all_rows["hit20_emitted_num"] = to_num(all_rows["hit20_emitted"]).fillna(0)
    all_rows["hit50_total_num"] = to_num(all_rows["hit50_total"]).fillna(0)

    viable = all_rows[all_rows["coverage_num"] >= 0.25].copy()
    if viable.empty:
        viable = all_rows.copy()

    viable["choice_score"] = (
        viable["hit50_emitted_num"] * 100
        + viable["hit20_emitted_num"] * 35
        + viable["hit50_total_num"] * 25
        - (1.0 - viable["coverage_num"]) * 12
    )

    best = viable.sort_values("choice_score", ascending=False).head(20).copy()

    # Résumé par review pour classifier les contextes.
    reviews_out = reviews.copy()
    reviews_out["quality_bucket_004X"] = "bad"
    reviews_out.loc[reviews_out["hit50_num"] >= 0.75, "quality_bucket_004X"] = "good"
    reviews_out.loc[(reviews_out["hit50_num"] >= 0.45) & (reviews_out["hit50_num"] < 0.75), "quality_bucket_004X"] = "medium"

    bucket_counts = reviews_out["quality_bucket_004X"].value_counts().to_dict()

    out_sweep = out_dir / "confidence_gate_sweep_004X.csv"
    out_best = out_dir / "confidence_gate_best_004X.csv"
    out_reviews = out_dir / "confidence_gate_reviews_004X.csv"
    out_json = out_dir / "confidence_gate_summary_004X.json"

    all_rows.to_csv(out_sweep, index=False, encoding="utf-8")
    best.to_csv(out_best, index=False, encoding="utf-8")
    reviews_out.to_csv(out_reviews, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "calibrate_confidence_gate_for_intermittent_ball_tracking",
        "click_scores": str(click_scores),
        "review_scores": str(review_scores),
        "clicks": int(len(clicks)),
        "reviews": int(clicks["review_id"].nunique()),
        "baseline_no_gate": eval_subset(clicks, pd.Series([True] * len(clicks), index=clicks.index)),
        "bucket_counts": bucket_counts,
        "best_candidates": best.head(10).to_dict(orient="records"),
        "next": "Use chosen gate to emit intermittent path points only where confidence is high."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004X status=OK")
    print("clicks=", summary["clicks"])
    print("reviews=", summary["reviews"])
    print("baseline_no_gate=", json.dumps(summary["baseline_no_gate"], ensure_ascii=False))
    print("bucket_counts=", json.dumps(bucket_counts, ensure_ascii=False))
    print("best_top3=", json.dumps(summary["best_candidates"][:3], ensure_ascii=False))
    print("wrote", out_sweep)
    print("wrote", out_best)
    print("wrote", out_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
