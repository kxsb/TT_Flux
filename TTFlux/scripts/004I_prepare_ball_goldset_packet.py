from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "004I"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/batch_004F_full240/operational_manifest_001T2.csv")
    ap.add_argument("--out-dir", default="runs/ball_goldset_004I")
    ap.add_argument("--count", type=int, default=30)
    args = ap.parse_args()

    root = Path.cwd()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = root / manifest

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)

    for c in ["travel", "density", "n_points", "x_range", "y_range"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "video_id" not in df.columns:
        df["video_id"] = df["clip_id"].astype(str).str.split("_").str[-1]

    picks = []

    # 1) quelques trajectoires très longues = souvent faux tracking corps
    picks.append(df.sort_values("travel", ascending=False).head(args.count // 3))

    # 2) quelques plausibles moyens
    mid = df[
        (df["travel"] >= df["travel"].quantile(0.35)) &
        (df["travel"] <= df["travel"].quantile(0.70))
    ]
    if len(mid):
        picks.append(mid.sample(min(args.count // 3, len(mid)), random_state=2026))

    # 3) diversité par vidéo
    per_video = []
    for vid, g in df.groupby("video_id"):
        per_video.append(g.sample(min(3, len(g)), random_state=2027))
    if per_video:
        picks.append(pd.concat(per_video, ignore_index=False))

    out = pd.concat(picks, ignore_index=False).drop_duplicates("review_id")
    out = out.head(args.count).copy()
    out.insert(0, "gold_rank_004I", range(1, len(out) + 1))

    # colonnes utiles pour l'interface suivante
    out["annotation_status_004I"] = ""
    out["annotation_notes_004I"] = ""

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(manifest),
        "source_rows": int(len(df)),
        "goldset_rows": int(len(out)),
        "instruction": {
            "goal": "Manually click real ball positions, not body motion.",
            "labels": ["ball", "not_visible", "unsure"],
            "target": "5-15 ball points per segment is enough to start."
        }
    }

    out_csv = out_dir / "ball_goldset_packet_004I.csv"
    out_json = out_dir / "ball_goldset_packet_summary_004I.json"

    out.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004I status=OK")
    print("source_rows=", len(df))
    print("goldset_rows=", len(out))
    print("wrote", out_csv)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
