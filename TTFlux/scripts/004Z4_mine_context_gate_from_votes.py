from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "004Z4"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def metrics_for_mask(y: pd.Series, pred: pd.Series) -> dict:
    y = y.astype(int)
    pred = pred.astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "accepted_pred": int(pred.sum()),
    }


def describe_rule(feature: str, op: str, threshold: float) -> str:
    return f"{feature} {op} {threshold:.6g}"


def eval_single_rules(df: pd.DataFrame, y: pd.Series, features: list[str]) -> list[dict]:
    rows = []

    for f in features:
        vals = to_num(df[f]).dropna()
        if len(vals) < 4:
            continue

        qs = sorted(set(float(vals.quantile(q)) for q in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]))

        for th in qs:
            for op in [">=", "<="]:
                if op == ">=":
                    pred = to_num(df[f]).fillna(-1e18) >= th
                else:
                    pred = to_num(df[f]).fillna(1e18) <= th

                m = metrics_for_mask(y, pred)
                if m["accepted_pred"] == 0:
                    continue

                rows.append({
                    "rule_type": "single",
                    "rule": describe_rule(f, op, th),
                    "feature_1": f,
                    "op_1": op,
                    "threshold_1": round(th, 6),
                    "feature_2": "",
                    "op_2": "",
                    "threshold_2": "",
                    **m,
                })

    return rows


def eval_pair_rules(df: pd.DataFrame, y: pd.Series, features: list[str], max_features: int = 16) -> list[dict]:
    rows = []

    # Limite volontaire : petit dataset, règles interprétables seulement.
    usable = []
    for f in features:
        vals = to_num(df[f]).dropna()
        if len(vals) >= 4 and vals.nunique() > 2:
            usable.append(f)

    usable = usable[:max_features]

    rule_atoms = []

    for f in usable:
        vals = to_num(df[f]).dropna()
        qs = sorted(set(float(vals.quantile(q)) for q in [0.25, 0.40, 0.50, 0.60, 0.75]))

        for th in qs:
            rule_atoms.append((f, ">=", th))
            rule_atoms.append((f, "<=", th))

    for (f1, op1, th1), (f2, op2, th2) in itertools.combinations(rule_atoms, 2):
        if f1 == f2:
            continue

        s1 = to_num(df[f1])
        s2 = to_num(df[f2])

        p1 = (s1.fillna(-1e18) >= th1) if op1 == ">=" else (s1.fillna(1e18) <= th1)
        p2 = (s2.fillna(-1e18) >= th2) if op2 == ">=" else (s2.fillna(1e18) <= th2)

        pred = p1 & p2
        m = metrics_for_mask(y, pred)

        if m["accepted_pred"] == 0:
            continue

        rows.append({
            "rule_type": "pair_and",
            "rule": f"{describe_rule(f1, op1, th1)} AND {describe_rule(f2, op2, th2)}",
            "feature_1": f1,
            "op_1": op1,
            "threshold_1": round(th1, 6),
            "feature_2": f2,
            "op_2": op2,
            "threshold_2": round(th2, 6),
            **m,
        })

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vote-metrics", default="runs/rally_video_vote_004Z3_diagnosis_p080_r030/vote_metric_diagnosis_004Z3.csv")
    ap.add_argument("--out-dir", default="runs/rally_context_gate_004Z4_p080_r030")
    args = ap.parse_args()

    root = Path.cwd()

    vote_metrics = Path(args.vote_metrics)
    if not vote_metrics.is_absolute():
        vote_metrics = root / vote_metrics

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(vote_metrics).fillna("")

    if "accepted_004Z3" not in df.columns:
        raise SystemExit("Colonne accepted_004Z3 absente. Relance 004Z3.")

    y = to_num(df["accepted_004Z3"]).fillna(0).astype(int)

    # Features candidates. On garde seulement celles présentes.
    candidate_features = [
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

    features = [f for f in candidate_features if f in df.columns]

    # Ajoute ratios utiles.
    work = df.copy()

    if "emitted_points" in work.columns and "tracklets" in work.columns:
        work["emitted_per_tracklet_004Z4"] = (
            to_num(work["emitted_points"]) / to_num(work["tracklets"]).replace(0, np.nan)
        )
        features.append("emitted_per_tracklet_004Z4")

    if "longest_tracklet_points" in work.columns and "emitted_points" in work.columns:
        work["longest_tracklet_ratio_004Z4"] = (
            to_num(work["longest_tracklet_points"]) / to_num(work["emitted_points"]).replace(0, np.nan)
        )
        features.append("longest_tracklet_ratio_004Z4")

    if "emitted_points" in work.columns and "path_points" in work.columns:
        work["emit_ratio_recalc_004Z4"] = (
            to_num(work["emitted_points"]) / to_num(work["path_points"]).replace(0, np.nan)
        )
        features.append("emit_ratio_recalc_004Z4")

    # Numeric clean.
    for f in features:
        work[f] = to_num(work[f])

    single = eval_single_rules(work, y, features)
    pair = eval_pair_rules(work, y, features)

    rules = pd.DataFrame(single + pair)

    if not rules.empty:
        # Priorité : éviter les faux positifs, puis garder du rappel.
        rules["choice_score"] = (
            to_num(rules["precision"]).fillna(0) * 100
            + to_num(rules["recall"]).fillna(0) * 55
            + to_num(rules["specificity"]).fillna(0) * 35
            + to_num(rules["f1"]).fillna(0) * 40
            - to_num(rules["fp"]).fillna(0) * 20
        )

        rules = rules.sort_values(
            ["choice_score", "precision", "recall", "specificity", "f1"],
            ascending=[False, False, False, False, False],
        )

    # Applique meilleure règle pour audit.
    best_rule = rules.iloc[0].to_dict() if not rules.empty else {}

    work["context_gate_pred_004Z4"] = 0

    if best_rule:
        f1 = best_rule["feature_1"]
        op1 = best_rule["op_1"]
        th1 = float(best_rule["threshold_1"])

        p = (work[f1] >= th1) if op1 == ">=" else (work[f1] <= th1)

        f2 = str(best_rule.get("feature_2", ""))
        if f2:
            op2 = best_rule["op_2"]
            th2 = float(best_rule["threshold_2"])
            p2 = (work[f2] >= th2) if op2 == ">=" else (work[f2] <= th2)
            p = p & p2

        work["context_gate_pred_004Z4"] = p.astype(int)

    work["vote_label_004Z4"] = np.where(y == 1, "accepted", "rejected")

    out_features = out_dir / "vote_context_features_004Z4.csv"
    out_rules = out_dir / "context_gate_rules_004Z4.csv"
    out_json = out_dir / "context_gate_summary_004Z4.json"

    work.to_csv(out_features, index=False, encoding="utf-8")
    rules.to_csv(out_rules, index=False, encoding="utf-8")

    accepted_ids = work[work["accepted_004Z3"].astype(int) == 1]["review_id"].astype(str).tolist()
    rejected_ids = work[work["accepted_004Z3"].astype(int) == 0]["review_id"].astype(str).tolist()
    predicted_ids = work[work["context_gate_pred_004Z4"].astype(int) == 1]["review_id"].astype(str).tolist()

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "mine_simple_context_gate_from_human_video_votes_tiny_dataset",
        "warning": "Dataset is tiny: 17 votes. Use rules as diagnostic, not final model.",
        "votes": int(len(work)),
        "accepted": int(y.sum()),
        "rejected": int((y == 0).sum()),
        "features": features,
        "accepted_review_ids": accepted_ids,
        "rejected_review_ids": rejected_ids,
        "best_rule": best_rule,
        "predicted_accept_review_ids": predicted_ids,
        "outputs": {
            "features": str(out_features),
            "rules": str(out_rules),
        },
        "next": "If best rule separates accepted/rejected reasonably, use it as review-context prefilter for next vote packet."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004Z4 status=OK")
    print("votes=", summary["votes"])
    print("accepted=", summary["accepted"])
    print("rejected=", summary["rejected"])
    print("best_rule=", json.dumps(best_rule, ensure_ascii=False))
    print("predicted_accept_review_ids=", json.dumps(predicted_ids, ensure_ascii=False))
    print("wrote", out_features)
    print("wrote", out_rules)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
