from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


RULE_ID = "003Y_center_blob_fill_med_ge_082379"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_002A")
    ap.add_argument("--candidate", default="appearance_reject_candidate_003Y.csv")
    ap.add_argument("--threshold", type=float, default=0.82379)
    args = ap.parse_args()

    root = Path.cwd()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    candidate = Path(args.candidate)
    if not candidate.is_absolute():
        candidate = run_dir / candidate

    if not candidate.is_file():
        raise SystemExit(f"Candidate CSV introuvable: {candidate}")

    df = pd.read_csv(candidate)

    required = [
        "review_id",
        "003Y_candidate_hit",
        "003Y_center_fill_value",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Colonnes manquantes dans 003Y: {missing}")

    hits = df[df["003Y_candidate_hit"].astype(bool)].copy()

    hit_ids = hits["review_id"].astype(str).tolist()

    shadow_dir = run_dir / "shadow_rules"
    shadow_dir.mkdir(parents=True, exist_ok=True)

    out_json = shadow_dir / f"{RULE_ID}.json"

    rule = {
        "version": "003Y2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rule_id": RULE_ID,
        "rule_family": "appearance_logo_shoes_reject",
        "policy": "frozen_shadow_rule_only_no_live_filter_no_delete",
        "source_run": str(run_dir),
        "source_candidate_csv": str(candidate),
        "feature_source": "center_blob_features_001T2.csv",
        "feature_col": "center_blob_fill_med",
        "operator": ">=",
        "threshold": args.threshold,
        "expression": f"center_blob_fill_med >= {args.threshold}",
        "validated_on_source": {
            "rows": int(len(df)),
            "hit_total": int(len(hits)),
            "hit_ids": hit_ids,
            "dangerous_hit_total": int(
                len(
                    hits[
                        hits.get("human_label_003S", pd.Series([""] * len(hits)))
                        .astype(str)
                        .str.lower()
                        .isin(["keep", "partial"])
                    ]
                )
            )
            if "human_label_003S" in hits.columns
            else None,
        },
        "warning": "Shadow only. Needs cross-batch validation before any live integration.",
    }

    out_json.write_text(json.dumps(rule, ensure_ascii=False, indent=2), encoding="utf-8")

    print("003Y2 status=OK_RULE_FROZEN")
    print("rule_json=", out_json)
    print("rule_id=", RULE_ID)
    print("hit_ids=" + (",".join(hit_ids) or "-"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
