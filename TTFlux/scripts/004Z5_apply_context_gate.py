from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "004Z5"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-summary", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--emitted-min", type=int, default=141)
    ap.add_argument("--coverage-min", type=float, default=0.0)
    args = ap.parse_args()

    root = Path.cwd()

    review_path = Path(args.review_summary)
    out_dir = Path(args.out_dir)

    if not review_path.is_absolute():
        review_path = root / review_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(review_path).fillna("")

    df["emitted_points_num"] = to_num(df.get("emitted_points", 0)).fillna(0).astype(int)
    df["coverage_points_num"] = to_num(df.get("coverage_points", 0)).fillna(0.0)

    df["context_gate_004Z5"] = (
        (df["emitted_points_num"] >= args.emitted_min)
        & (df["coverage_points_num"] >= args.coverage_min)
    ).astype(int)

    df["context_gate_reason_004Z5"] = df.apply(
        lambda r: "pass"
        if int(r["context_gate_004Z5"]) == 1
        else f"reject: emitted_points={int(r['emitted_points_num'])} < {args.emitted_min}",
        axis=1,
    )

    passed = df[df["context_gate_004Z5"] == 1].copy()
    rejected = df[df["context_gate_004Z5"] == 0].copy()

    passed = passed.sort_values(
        ["emitted_points_num", "coverage_points_num"],
        ascending=[False, False],
    )

    rejected = rejected.sort_values(
        ["emitted_points_num", "coverage_points_num"],
        ascending=[False, False],
    )

    by_video_pass = passed["video_id"].astype(str).value_counts().to_dict() if "video_id" in passed.columns else {}
    by_video_all = df["video_id"].astype(str).value_counts().to_dict() if "video_id" in df.columns else {}

    out_all = out_dir / "context_gate_all_reviews_004Z5.csv"
    out_pass = out_dir / "context_gate_pass_reviews_004Z5.csv"
    out_reject = out_dir / "context_gate_rejected_reviews_004Z5.csv"
    out_json = out_dir / "context_gate_summary_004Z5.json"

    df.to_csv(out_all, index=False, encoding="utf-8")
    passed.to_csv(out_pass, index=False, encoding="utf-8")
    rejected.to_csv(out_reject, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "apply_human_vote_context_gate_to_gated_rally_review_summary",
        "review_summary": str(review_path),
        "emitted_min": args.emitted_min,
        "coverage_min": args.coverage_min,
        "reviews_total": int(len(df)),
        "reviews_pass": int(len(passed)),
        "reviews_rejected": int(len(rejected)),
        "pass_ratio": round(float(len(passed) / max(1, len(df))), 4),
        "by_video_all": by_video_all,
        "by_video_pass": by_video_pass,
        "top_pass_review_ids": passed["review_id"].astype(str).head(40).tolist() if "review_id" in passed.columns else [],
        "outputs": {
            "all": str(out_all),
            "pass": str(out_pass),
            "rejected": str(out_reject),
        },
        "next": "Generate video vote packet only for context_gate pass reviews.",
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004Z5 status=OK")
    print("reviews_total=", summary["reviews_total"])
    print("reviews_pass=", summary["reviews_pass"])
    print("reviews_rejected=", summary["reviews_rejected"])
    print("pass_ratio=", summary["pass_ratio"])
    print("by_video_pass=", json.dumps(by_video_pass, ensure_ascii=False))
    print("top_pass_review_ids=", json.dumps(summary["top_pass_review_ids"][:20], ensure_ascii=False))
    print("wrote", out_all)
    print("wrote", out_pass)
    print("wrote", out_reject)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
