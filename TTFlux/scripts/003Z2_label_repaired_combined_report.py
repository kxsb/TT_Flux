from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003Z2"

# Confirmé par les audits 003K/003L/003M sur batch_001E.
HISTORICAL_CONFIRMED_REJECTS = {
    "batch_001E/R0008": "confirmed_reject_003K_003L",
    "batch_001E/R0009": "confirmed_reject_003K_003L",
    "batch_001E/R0010": "confirmed_reject_003K_003L",
    "batch_001E/R0014": "confirmed_reject_003K_003L",
    "batch_001E/R0020": "confirmed_reject_003K_003L",
    "batch_001E/R0021": "confirmed_reject_003K_003L",
}


def clean_label(x) -> str:
    s = str(x or "").strip().lower()
    return s if s in {"reject", "keep", "partial", "unsure"} else ""


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes"}


def summarize(df: pd.DataFrame, hit_col: str) -> dict:
    hit = df[hit_col].map(boolish)
    sub = df[hit].copy()

    labels = sub["label_final_003Z2"].map(clean_label)

    dangerous = labels.isin(["keep", "partial"])

    return {
        "hit_total": int(len(sub)),
        "hit_ids": sub["sample_id"].astype(str).tolist(),
        "label_counts": labels.value_counts(dropna=False).to_dict(),
        "reject_ids": sub.loc[labels.eq("reject"), "sample_id"].astype(str).tolist(),
        "keep_ids": sub.loc[labels.eq("keep"), "sample_id"].astype(str).tolist(),
        "partial_ids": sub.loc[labels.eq("partial"), "sample_id"].astype(str).tolist(),
        "unsure_ids": sub.loc[labels.eq("unsure"), "sample_id"].astype(str).tolist(),
        "unknown_ids": sub.loc[labels.eq(""), "sample_id"].astype(str).tolist(),
        "dangerous_hit_total": int(dangerous.sum()),
        "dangerous_hit_ids": sub.loc[dangerous, "sample_id"].astype(str).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="runs/combined_shadow_003Z/combined_shadow_rules_003Z.csv")
    ap.add_argument("--out-dir", default="runs/combined_shadow_003Z2")
    args = ap.parse_args()

    root = Path.cwd()

    input_csv = Path(args.input)
    if not input_csv.is_absolute():
        input_csv = root / input_csv

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    if not input_csv.is_file():
        raise SystemExit(f"Input introuvable: {input_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df["sample_id"] = df["sample_id"].astype(str)

    if "human_label" not in df.columns:
        df["human_label"] = ""

    if "human_comment" not in df.columns:
        df["human_comment"] = ""

    df["label_final_003Z2"] = df["human_label"].map(clean_label)
    df["label_source_003Z2"] = ""

    # Source labels directs.
    direct_mask = df["label_final_003Z2"] != ""
    df.loc[direct_mask, "label_source_003Z2"] = "direct_human_label"

    # Injection historique pour 001E.
    for sample_id, source in HISTORICAL_CONFIRMED_REJECTS.items():
        mask = df["sample_id"].eq(sample_id)
        df.loc[mask, "label_final_003Z2"] = "reject"
        df.loc[mask, "label_source_003Z2"] = source

        old_comment = df.loc[mask, "human_comment"].fillna("").astype(str)
        if not old_comment.empty:
            df.loc[mask, "human_comment"] = old_comment + " | historical confirmed reject"

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "combined_shadow_report_with_historical_labels_only_no_live_filter_no_delete",
        "input": str(input_csv),
        "rows": int(len(df)),
        "label_counts_final": df["label_final_003Z2"].map(clean_label).value_counts(dropna=False).to_dict(),
        "historical_labels_injected": HISTORICAL_CONFIRMED_REJECTS,
        "combined_strict_reject_shadow": summarize(df, "combined_strict_reject_shadow"),
        "combined_review_shadow": summarize(df, "combined_review_shadow"),
        "decision": {
            "strict": "Promising shadow reject candidate: no known keep/partial/unsure false positive after historical label repair.",
            "review": "Keep as human review queue only.",
            "live_filter": "NO. Needs one more fresh batch validation before promotion.",
        },
    }

    out_csv = out_dir / "combined_shadow_rules_labeled_003Z2.csv"
    out_json = out_dir / "combined_shadow_rules_labeled_summary_003Z2.json"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("003Z2 status=OK")
    print("rows=", len(df))
    print("label_counts_final=", summary["label_counts_final"])
    print("combined_strict=", json.dumps(summary["combined_strict_reject_shadow"], ensure_ascii=False))
    print("combined_review=", json.dumps(summary["combined_review_shadow"], ensure_ascii=False))
    print("wrote", out_csv)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
