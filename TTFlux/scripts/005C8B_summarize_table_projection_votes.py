from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "005C8B"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_counts(series):
    if series is None or len(series) == 0:
        return {}
    return {str(k): int(v) for k, v in series.astype(str).value_counts().to_dict().items()}


def prf(tp, fp, fn, tn):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1), 4),
    }


def eval_pred(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    return prf(tp, fp, fn, tn)


def mine_single_rules(df: pd.DataFrame, target_col: str):
    numeric_cols = []

    skip = {
        target_col,
        "rating_1_10",
        "usable_metric",
        "review_metric",
        "reject_metric",
        "camera_segment_id",
        "camera_segment_id_005B2",
        "frame",
    }

    for c in df.columns:
        if c in skip:
            continue

        vals = to_num(df[c])
        if vals.notna().sum() >= max(5, int(len(df) * 0.35)):
            numeric_cols.append(c)

    y = to_num(df[target_col]).fillna(0).astype(int).to_numpy()

    rules = []

    for c in numeric_cols:
        vals = to_num(df[c])

        clean = vals.dropna().to_numpy(dtype=float)
        if len(clean) < 5:
            continue

        qs = sorted(set(np.quantile(clean, q) for q in np.linspace(0.05, 0.95, 19)))

        for thr in qs:
            for op in [">=", "<="]:
                if op == ">=":
                    pred = (vals.fillna(-1e18) >= thr).astype(int).to_numpy()
                else:
                    pred = (vals.fillna(1e18) <= thr).astype(int).to_numpy()

                m = eval_pred(y, pred)

                # On favorise précision + rappel, mais précision d'abord.
                score = (
                    100.0 * m["precision"]
                    + 75.0 * m["recall"]
                    + 45.0 * m["f1"]
                    - 15.0 * (m["fp"] / max(1, len(df)))
                )

                rules.append({
                    "rule_type": "single",
                    "feature": c,
                    "op": op,
                    "threshold": round(float(thr), 6),
                    **m,
                    "predicted_accept": int(pred.sum()),
                    "choice_score": round(float(score), 4),
                })

    rules = sorted(
        rules,
        key=lambda r: (
            r["choice_score"],
            r["precision"],
            r["recall"],
            -r["fp"],
        ),
        reverse=True,
    )

    return rules


def mine_combo_rules(df: pd.DataFrame, target_col: str, top_single: list[dict], max_rules: int = 60):
    y = to_num(df[target_col]).fillna(0).astype(int).to_numpy()

    base = top_single[:max_rules]
    rules = []

    for i, r1 in enumerate(base):
        for r2 in base[i + 1:]:
            c1, c2 = r1["feature"], r2["feature"]

            if c1 == c2:
                continue

            v1 = to_num(df[c1])
            v2 = to_num(df[c2])

            if r1["op"] == ">=":
                p1 = v1.fillna(-1e18) >= float(r1["threshold"])
            else:
                p1 = v1.fillna(1e18) <= float(r1["threshold"])

            if r2["op"] == ">=":
                p2 = v2.fillna(-1e18) >= float(r2["threshold"])
            else:
                p2 = v2.fillna(1e18) <= float(r2["threshold"])

            pred = (p1 & p2).astype(int).to_numpy()

            m = eval_pred(y, pred)

            score = (
                110.0 * m["precision"]
                + 80.0 * m["recall"]
                + 50.0 * m["f1"]
                - 18.0 * (m["fp"] / max(1, len(df)))
            )

            rules.append({
                "rule_type": "combo_and",
                "feature_1": c1,
                "op_1": r1["op"],
                "threshold_1": r1["threshold"],
                "feature_2": c2,
                "op_2": r2["op"],
                "threshold_2": r2["threshold"],
                **m,
                "predicted_accept": int(pred.sum()),
                "choice_score": round(float(score), 4),
            })

    rules = sorted(
        rules,
        key=lambda r: (
            r["choice_score"],
            r["precision"],
            r["recall"],
            -r["fp"],
        ),
        reverse=True,
    )

    return rules


def apply_rule(df: pd.DataFrame, rule: dict):
    if not rule:
        return np.zeros((len(df),), dtype=int)

    if rule["rule_type"] == "single":
        v = to_num(df[rule["feature"]])
        if rule["op"] == ">=":
            return (v.fillna(-1e18) >= float(rule["threshold"])).astype(int).to_numpy()
        return (v.fillna(1e18) <= float(rule["threshold"])).astype(int).to_numpy()

    if rule["rule_type"] == "combo_and":
        v1 = to_num(df[rule["feature_1"]])
        v2 = to_num(df[rule["feature_2"]])

        if rule["op_1"] == ">=":
            p1 = v1.fillna(-1e18) >= float(rule["threshold_1"])
        else:
            p1 = v1.fillna(1e18) <= float(rule["threshold_1"])

        if rule["op_2"] == ">=":
            p2 = v2.fillna(-1e18) >= float(rule["threshold_2"])
        else:
            p2 = v2.fillna(1e18) <= float(rule["threshold_2"])

        return (p1 & p2).astype(int).to_numpy()

    return np.zeros((len(df),), dtype=int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", default="runs/rally_table_projection_vote_005C8A/table_projection_votes_005C8A.csv")
    ap.add_argument("--metric-csv", default="runs/rally_table_metric_qa_005C7B/table_metric_quality_005C7B.csv")
    ap.add_argument("--out-dir", default="runs/rally_table_projection_vote_summary_005C8B")
    ap.add_argument("--usable-min", type=int, default=8)
    ap.add_argument("--review-min", type=int, default=5)
    args = ap.parse_args()

    root = Path.cwd()

    votes_path = Path(args.votes)
    metric_path = Path(args.metric_csv)
    out_dir = Path(args.out_dir)

    if not votes_path.is_absolute():
        votes_path = root / votes_path
    if not metric_path.is_absolute():
        metric_path = root / metric_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    votes = pd.read_csv(votes_path).fillna("")
    metric = pd.read_csv(metric_path).fillna("")

    votes["camera_segment_id"] = to_num(votes["camera_segment_id"]).fillna(-1).astype(int)
    metric["camera_segment_id_005B2"] = to_num(metric["camera_segment_id_005B2"]).fillna(-1).astype(int)

    # Recalcule proprement les classes depuis la note, même si colonnes déjà présentes.
    votes["rating_1_10"] = to_num(votes["rating_1_10"]).fillna(0).astype(int)
    votes["usable_metric_human_005C8B"] = (votes["rating_1_10"] >= args.usable_min).astype(int)
    votes["review_metric_human_005C8B"] = (
        (votes["rating_1_10"] >= args.review_min)
        & (votes["rating_1_10"] < args.usable_min)
    ).astype(int)
    votes["reject_metric_human_005C8B"] = (votes["rating_1_10"] < args.review_min).astype(int)

    merged = votes.merge(
        metric,
        left_on=["review_id", "camera_segment_id"],
        right_on=["review_id", "camera_segment_id_005B2"],
        how="left",
        suffixes=("_vote", ""),
    )

    single_rules = mine_single_rules(merged, target_col="usable_metric_human_005C8B")
    combo_rules = mine_combo_rules(merged, target_col="usable_metric_human_005C8B", top_single=single_rules)

    best_rules = sorted(
        single_rules[:20] + combo_rules[:20],
        key=lambda r: (
            r["choice_score"],
            r["precision"],
            r["recall"],
            -r["fp"],
        ),
        reverse=True,
    )

    best_rule = best_rules[0] if best_rules else {}

    pred = apply_rule(merged, best_rule) if best_rule else np.zeros((len(merged),), dtype=int)
    merged["predicted_usable_by_best_rule_005C8B"] = pred

    accepted = merged[merged["usable_metric_human_005C8B"].eq(1)].copy()
    review = merged[merged["review_metric_human_005C8B"].eq(1)].copy()
    rejected = merged[merged["reject_metric_human_005C8B"].eq(1)].copy()
    predicted = merged[merged["predicted_usable_by_best_rule_005C8B"].eq(1)].copy()

    out_merged = out_dir / "table_projection_votes_merged_005C8B.csv"
    out_accepted = out_dir / "table_projection_votes_accepted_005C8B.csv"
    out_review = out_dir / "table_projection_votes_review_005C8B.csv"
    out_rejected = out_dir / "table_projection_votes_rejected_005C8B.csv"
    out_predicted = out_dir / "table_projection_best_rule_predicted_005C8B.csv"
    out_rules = out_dir / "table_projection_rules_005C8B.csv"
    out_json = out_dir / "table_projection_vote_summary_005C8B.json"

    merged.to_csv(out_merged, index=False, encoding="utf-8")
    accepted.to_csv(out_accepted, index=False, encoding="utf-8")
    review.to_csv(out_review, index=False, encoding="utf-8")
    rejected.to_csv(out_rejected, index=False, encoding="utf-8")
    predicted.to_csv(out_predicted, index=False, encoding="utf-8")

    pd.DataFrame(best_rules[:80]).to_csv(out_rules, index=False, encoding="utf-8")

    rating_counts = {str(k): int(v) for k, v in votes["rating_1_10"].value_counts().sort_index().to_dict().items()}
    tag_counts = safe_counts(votes["tag"]) if "tag" in votes.columns else {}

    by_quality = {}
    if "metric_quality_005C7B" in merged.columns:
        for q, g in merged.groupby("metric_quality_005C7B", dropna=False):
            by_quality[str(q)] = {
                "count": int(len(g)),
                "rating_mean": round(float(to_num(g["rating_1_10"]).mean()), 4),
                "accepted": int(g["usable_metric_human_005C8B"].sum()),
                "accepted_ratio": round(float(g["usable_metric_human_005C8B"].mean()), 4),
            }

    by_snap = {}
    if "snap_status_005C7A" in merged.columns:
        for q, g in merged.groupby("snap_status_005C7A", dropna=False):
            by_snap[str(q)] = {
                "count": int(len(g)),
                "rating_mean": round(float(to_num(g["rating_1_10"]).mean()), 4),
                "accepted": int(g["usable_metric_human_005C8B"].sum()),
                "accepted_ratio": round(float(g["usable_metric_human_005C8B"].mean()), 4),
            }

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "summarize_human_rating_for_reference_table_projection_and_mine_quality_rules",
        "votes": str(votes_path),
        "metric_csv": str(metric_path),
        "usable_min": args.usable_min,
        "review_min": args.review_min,
        "votes_count": int(len(votes)),
        "rating_counts": rating_counts,
        "rating_mean": round(float(votes["rating_1_10"].mean()), 4) if len(votes) else 0,
        "accepted": int(votes["usable_metric_human_005C8B"].sum()),
        "review": int(votes["review_metric_human_005C8B"].sum()),
        "rejected": int(votes["reject_metric_human_005C8B"].sum()),
        "accepted_ratio": round(float(votes["usable_metric_human_005C8B"].mean()), 4) if len(votes) else 0,
        "review_ratio": round(float(votes["review_metric_human_005C8B"].mean()), 4) if len(votes) else 0,
        "rejected_ratio": round(float(votes["reject_metric_human_005C8B"].mean()), 4) if len(votes) else 0,
        "tag_counts": tag_counts,
        "by_metric_quality_005C7B": by_quality,
        "by_snap_status_005C7A": by_snap,
        "best_rule": best_rule,
        "top_rules": best_rules[:10],
        "outputs": {
            "merged": str(out_merged),
            "accepted": str(out_accepted),
            "review": str(out_review),
            "rejected": str(out_rejected),
            "predicted_by_rule": str(out_predicted),
            "rules": str(out_rules),
        },
        "next": "Use accepted projections as reliable metric table references. Use rejected cases to train/improve reference-table fitting."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005C8B status=OK")
    print("votes_count=", summary["votes_count"])
    print("rating_counts=", json.dumps(rating_counts, ensure_ascii=False))
    print("accepted=", summary["accepted"])
    print("review=", summary["review"])
    print("rejected=", summary["rejected"])
    print("accepted_ratio=", summary["accepted_ratio"])
    print("best_rule=", json.dumps(best_rule, ensure_ascii=False))
    print("wrote", out_merged)
    print("wrote", out_accepted)
    print("wrote", out_review)
    print("wrote", out_rejected)
    print("wrote", out_rules)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
