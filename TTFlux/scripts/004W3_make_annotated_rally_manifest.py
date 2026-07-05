from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "004W3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clicks", default="runs/rally_goldset_004U2_reused_clicks/rally_reused_clicks_004U2.csv")
    ap.add_argument("--manifest", default="runs/rally_dataset_004T/rally_manifest_004T.csv")
    ap.add_argument("--out-dir", default="runs/rally_annotated_manifest_004W3")
    args = ap.parse_args()

    root = Path.cwd()

    clicks_path = Path(args.clicks)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    if not clicks_path.is_absolute():
        clicks_path = root / clicks_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    clicks = pd.read_csv(clicks_path).fillna("")
    manifest = pd.read_csv(manifest_path).fillna("")

    ball = clicks[clicks["visibility"].astype(str).eq("ball")].copy()
    annotated_reviews = sorted(ball["review_id"].astype(str).unique().tolist())

    sub = manifest[manifest["review_id"].astype(str).isin(annotated_reviews)].copy()

    missing = sorted(set(annotated_reviews) - set(sub["review_id"].astype(str).unique()))

    by_video = sub["video_id"].astype(str).value_counts().to_dict()
    click_counts = ball["review_id"].astype(str).value_counts().to_dict()

    sub["ball_clicks_004W3"] = sub["review_id"].astype(str).map(click_counts).fillna(0).astype(int)
    sub = sub.sort_values(["ball_clicks_004W3", "activity_score"], ascending=[False, False])

    out_csv = out_dir / "rally_annotated_manifest_004W3.csv"
    out_json = out_dir / "rally_annotated_manifest_summary_004W3.json"

    sub.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "subset_rally_manifest_to_reviews_with_reused_manual_ball_clicks",
        "clicks": str(clicks_path),
        "manifest": str(manifest_path),
        "annotated_review_count": len(annotated_reviews),
        "manifest_rows": int(len(sub)),
        "missing_reviews": missing,
        "ball_clicks": int(len(ball)),
        "by_video": by_video,
        "top_click_counts": dict(list(click_counts.items())[:20]),
        "out_csv": str(out_csv),
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004W3 status=OK")
    print("annotated_review_count=", summary["annotated_review_count"])
    print("manifest_rows=", summary["manifest_rows"])
    print("ball_clicks=", summary["ball_clicks"])
    print("missing_reviews=", json.dumps(missing, ensure_ascii=False))
    print("by_video=", json.dumps(by_video, ensure_ascii=False))
    print("wrote", out_csv)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
