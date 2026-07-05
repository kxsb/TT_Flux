from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003Y6"


def clean_label(x) -> str:
    s = str(x or "").strip().lower()
    return s if s in {"reject", "keep", "partial", "unsure"} else ""


def thresholds(vals: pd.Series) -> list[float]:
    v = pd.to_numeric(vals, errors="coerce").dropna().astype(float)
    uniq = sorted(set(v.tolist()))

    if len(uniq) <= 1:
        return uniq

    mids = [(a + b) / 2 for a, b in zip(uniq[:-1], uniq[1:])]
    return sorted(set(uniq + mids))


def eval_rule(df: pd.DataFrame, cth: float, mth: float) -> dict:
    c = pd.to_numeric(df["center_fill_med_003Y5"], errors="coerce")
    m = pd.to_numeric(df["micro_distance_med_003Y5"], errors="coerce")
    lab = df["human_label"].map(clean_label)

    hit = (c >= cth) & (m >= mth)

    reject = hit & lab.eq("reject")
    keep = hit & lab.eq("keep")
    partial = hit & lab.eq("partial")
    unsure = hit & lab.eq("unsure")
    unknown = hit & lab.eq("")

    dangerous = keep | partial

    return {
        "expression": f"center_fill_med >= {round(float(cth), 6)} AND micro_distance_med >= {round(float(mth), 6)}",
        "center_threshold": float(cth),
        "micro_threshold": float(mth),
        "hit_total": int(hit.sum()),
        "hit_ids": df.loc[hit, "sample_id"].astype(str).tolist(),
        "reject_hit_total": int(reject.sum()),
        "reject_hit_ids": df.loc[reject, "sample_id"].astype(str).tolist(),
        "dangerous_hit_total": int(dangerous.sum()),
        "dangerous_hit_ids": df.loc[dangerous, "sample_id"].astype(str).tolist(),
        "unsure_hit_total": int(unsure.sum()),
        "unsure_hit_ids": df.loc[unsure, "sample_id"].astype(str).tolist(),
        "unknown_hit_total": int(unknown.sum()),
        "unknown_hit_ids": df.loc[unknown, "sample_id"].astype(str).tolist(),
        "score": int(reject.sum()) * 100 - int(dangerous.sum()) * 1000 - int(unsure.sum()) * 200 - int(unknown.sum()) * 100 - int(hit.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/cross_batch_003Y5/cross_batch_label_features_003Y5.csv")
    ap.add_argument("--out-dir", default="runs/cross_batch_003Y6")
    args = ap.parse_args()

    root = Path.cwd()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    if not data_path.is_file():
        raise SystemExit(f"Data introuvable: {data_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    required = ["sample_id", "human_label", "center_fill_med_003Y5", "micro_distance_med_003Y5"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Colonnes manquantes: {missing}")

    cths = thresholds(df["center_fill_med_003Y5"])
    mths = thresholds(df["micro_distance_med_003Y5"])

    results = []

    for cth in cths:
        for mth in mths:
            results.append(eval_rule(df, cth, mth))

    all_sorted = sorted(
        results,
        key=lambda r: (
            r["dangerous_hit_total"] == 0,
            r["unknown_hit_total"] == 0,
            r["unsure_hit_total"] == 0,
            r["reject_hit_total"],
            r["score"],
            -r["hit_total"],
        ),
        reverse=True,
    )

    strict = [
        r for r in all_sorted
        if r["dangerous_hit_total"] == 0
        and r["unknown_hit_total"] == 0
        and r["unsure_hit_total"] == 0
        and r["reject_hit_total"] > 0
    ]

    no_danger = [
        r for r in all_sorted
        if r["dangerous_hit_total"] == 0
        and r["unknown_hit_total"] == 0
        and r["reject_hit_total"] > 0
    ]

    best_strict = strict[0] if strict else None
    best_no_danger = no_danger[0] if no_danger else None

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "strict_sweep_only_no_live_filter_no_delete",
        "data": str(data_path),
        "rows": int(len(df)),
        "label_counts": df["human_label"].map(clean_label).value_counts(dropna=False).to_dict(),
        "best_strict_no_danger_no_unsure_no_unknown": best_strict,
        "best_no_danger_allow_unsure": best_no_danger,
        "top_strict": strict[:20],
        "top_no_danger": no_danger[:20],
    }

    out_json = out_dir / "strict_sweep_summary_003Y6.json"
    out_csv = out_dir / "strict_sweep_all_003Y6.csv"

    pd.DataFrame(all_sorted).to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("003Y6 status=OK")
    print("rows=", len(df))
    print("label_counts=", summary["label_counts"])

    if best_strict:
        print("best_strict=", json.dumps(best_strict, ensure_ascii=False))
    else:
        print("best_strict=-")

    if best_no_danger:
        print("best_no_danger_allow_unsure=", json.dumps(best_no_danger, ensure_ascii=False))
    else:
        print("best_no_danger_allow_unsure=-")

    print("wrote", out_json)
    print("wrote", out_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
