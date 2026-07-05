from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "004Z3"

RATING_SCORE = {
    "bon": 3,
    "moyen": 2,
    "mauvais": 1,
    "inutilisable": 0,
}


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def describe_group(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {}

    for c in cols:
        if c not in df.columns:
            continue

        v = to_num(df[c]).dropna()

        if len(v) == 0:
            continue

        out[c] = {
            "n": int(len(v)),
            "min": round(float(v.min()), 4),
            "p25": round(float(v.quantile(0.25)), 4),
            "med": round(float(v.median()), 4),
            "p75": round(float(v.quantile(0.75)), 4),
            "max": round(float(v.max()), 4),
        }

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes-merged", default="runs/rally_video_vote_004Z2_summary_p080_r030/video_votes_merged_004Z2.csv")
    ap.add_argument("--gated-review-summary", default="runs/rally_gated_paths_004Y_annotated17_p080_r030_vote/gated_review_summary_004Y.csv")
    ap.add_argument("--gated-vs-clicks-reviews", default="runs/rally_gated_paths_004Y_annotated17_p080_r030/gated_vs_clicks_reviews_004Y.csv")
    ap.add_argument("--out-dir", default="runs/rally_video_vote_004Z3_diagnosis_p080_r030")
    args = ap.parse_args()

    root = Path.cwd()

    votes_path = Path(args.votes_merged)
    review_path = Path(args.gated_review_summary)
    clicks_review_path = Path(args.gated_vs_clicks_reviews)
    out_dir = Path(args.out_dir)

    if not votes_path.is_absolute():
        votes_path = root / votes_path
    if not review_path.is_absolute():
        review_path = root / review_path
    if not clicks_review_path.is_absolute():
        clicks_review_path = root / clicks_review_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    votes = pd.read_csv(votes_path).fillna("")
    review = pd.read_csv(review_path).fillna("")

    if clicks_review_path.is_file():
        clicks_review = pd.read_csv(clicks_review_path).fillna("")
    else:
        clicks_review = pd.DataFrame()

    votes["rating_norm"] = votes["rating"].astype(str).str.strip().str.lower()
    votes["rating_score_004Z3"] = votes["rating_norm"].map(RATING_SCORE).fillna(-1).astype(int)
    votes["accepted_004Z3"] = votes["rating_norm"].isin(["bon", "moyen"]).astype(int)
    votes["rejected_004Z3"] = votes["rating_norm"].isin(["mauvais", "inutilisable"]).astype(int)

    merged = votes.merge(
        review,
        on="review_id",
        how="left",
        suffixes=("", "_review"),
    )

    if not clicks_review.empty and "review_id" in clicks_review.columns:
        keep_cols = [
            c for c in [
                "review_id",
                "clicks",
                "found_ratio",
                "hit10_total",
                "hit20_total",
                "hit50_total",
                "hit10_found",
                "hit20_found",
                "hit50_found",
                "dist_med_found",
                "score_med_found",
            ]
            if c in clicks_review.columns
        ]

        merged = merged.merge(
            clicks_review[keep_cols],
            on="review_id",
            how="left",
            suffixes=("", "_click_eval"),
        )

    metric_cols = [
        "duration_vote_sec",
        "duration_src_sec",
        "path_points",
        "emitted_points",
        "coverage_points",
        "tracklets",
        "longest_tracklet_points",
        "score_med_emitted",
        "review_score_med_004Y",
        "found_ratio",
        "hit10_total",
        "hit20_total",
        "hit50_total",
        "hit10_found",
        "hit20_found",
        "hit50_found",
        "dist_med_found",
        "score_med_found",
    ]

    accepted = merged[merged["accepted_004Z3"] == 1].copy()
    rejected = merged[merged["rejected_004Z3"] == 1].copy()

    # Heuristique : profils observés dans les votes acceptés.
    accepted_profile = describe_group(accepted, metric_cols)
    rejected_profile = describe_group(rejected, metric_cols)

    # Classement lisible.
    sort_cols = ["rating_score_004Z3", "review_score_med_004Y", "emitted_points"]
    sort_cols = [c for c in sort_cols if c in merged.columns]

    for c in sort_cols:
        merged[c + "_numsort"] = to_num(merged[c]).fillna(-999)

    merged = merged.sort_values(
        [c + "_numsort" for c in sort_cols],
        ascending=[False] * len(sort_cols),
    )

    # Propose un sous-ensemble strict : votes acceptés uniquement.
    safe = accepted.copy()

    if "rating_score_004Z3" in safe.columns:
        safe = safe.sort_values(["rating_score_004Z3"], ascending=False)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "diagnose_human_video_votes_against_gated_tracking_metrics",
        "votes": int(len(merged)),
        "accepted": int(len(accepted)),
        "rejected": int(len(rejected)),
        "accepted_review_ids": accepted["review_id"].astype(str).tolist(),
        "rejected_review_ids": rejected["review_id"].astype(str).tolist(),
        "accepted_profile": accepted_profile,
        "rejected_profile": rejected_profile,
        "decision": {
            "run_all160": False,
            "reason": "Human vote acceptance too low for global rollout.",
            "next": "Use accepted reviews as positive context, rejected as negative context; improve review-level/context gate before all160."
        }
    }

    out_merged = out_dir / "vote_metric_diagnosis_004Z3.csv"
    out_safe = out_dir / "accepted_reviews_004Z3.csv"
    out_rejected = out_dir / "rejected_reviews_004Z3.csv"
    out_json = out_dir / "vote_metric_diagnosis_summary_004Z3.json"

    merged.to_csv(out_merged, index=False, encoding="utf-8")
    safe.to_csv(out_safe, index=False, encoding="utf-8")
    rejected.to_csv(out_rejected, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004Z3 status=OK")
    print("votes=", summary["votes"])
    print("accepted=", summary["accepted"])
    print("rejected=", summary["rejected"])
    print("accepted_review_ids=", json.dumps(summary["accepted_review_ids"], ensure_ascii=False))
    print("rejected_review_ids=", json.dumps(summary["rejected_review_ids"], ensure_ascii=False))
    print("wrote", out_merged)
    print("wrote", out_safe)
    print("wrote", out_rejected)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
