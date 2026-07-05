from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "003M"

RULE = {
    "id": "003L_hardened_v1_player060_dist210_accel20",
    "version": VERSION,
    "status": "preferred_shadow_candidate",
    "policy": "shadow_only_no_live_filter_no_delete",
    "expression": (
        "inside_player_motion_mask_ratio_003C >= 0.60 AND "
        "point_distance_to_table_px_median <= 210.0 AND "
        "max_accel >= 20.0"
    ),
    "clauses": [
        {
            "name": "player_motion_inside",
            "column": "inside_player_motion_mask_ratio_003C",
            "op": ">=",
            "threshold": 0.60,
        },
        {
            "name": "old_table_distance",
            "column": "point_distance_to_table_px_median",
            "op": "<=",
            "threshold": 210.0,
        },
        {
            "name": "max_accel",
            "column": "max_accel",
            "op": ">=",
            "threshold": 20.0,
        },
    ],
    "rationale": [
        "003L catches all 6 confirmed rejects from batch_001E.",
        "003L catches no protected sentinel from batch_001E.",
        "Manual table outside alone is unsafe; R0015,R0019,R0022 are protected but outside-table.",
        "Keep as shadow candidate until tested on larger batches.",
    ],
}

EXPECTED_001E = {
    "caught_ids": ["R0010", "R0009", "R0020", "R0014", "R0008", "R0021"],
    "caught_confirmed_reject_ids": ["R0008", "R0009", "R0010", "R0014", "R0020", "R0021"],
    "caught_protected_sentinel_ids": [],
}

CONFIRMED_REJECT_IDS = {"R0008", "R0009", "R0010", "R0014", "R0020", "R0021"}
PROTECTED_SENTINEL_IDS = {"R0003", "R0006", "R0015", "R0018", "R0019", "R0022"}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def num(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def normalize_review_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["review_id", "review_id_003G", "review_id_003F", "review_id_003E"]:
        if c in df.columns:
            df["review_id"] = df[c].astype(str)
            return df

    raise SystemExit("Aucune colonne review_id trouvée.")


def find_source_csv(run_dir: Path) -> Path:
    candidates = [
        run_dir / "hardened_shadow_packet_003L.csv",
        run_dir / "shadow_filter_simulation_003G.csv",
        run_dir / "multi_object_arbiter_shadow_003E.csv",
        run_dir / "multi_object_audit_003D" / "multi_object_features_003D.csv",
    ]

    for p in candidates:
        if p.is_file():
            return p

    raise SystemExit(
        "Aucun CSV source trouvé. Attendu un de :\n"
        + "\n".join(str(p) for p in candidates)
    )


def eval_rule(row: pd.Series) -> dict:
    vals = {}
    passes = []
    failed = []
    margins = {}

    for clause in RULE["clauses"]:
        col = clause["column"]
        op = clause["op"]
        th = float(clause["threshold"])
        value = num(row.get(col))

        vals[col] = value

        if not np.isfinite(value):
            ok = False
            margin = np.nan
        elif op == ">=":
            ok = value >= th
            margin = value - th
        elif op == "<=":
            ok = value <= th
            margin = th - value
        else:
            raise ValueError(f"Unsupported op: {op}")

        passes.append(ok)
        margins[clause["name"]] = margin

        if not ok:
            failed.append(clause["name"])

    hit = bool(all(passes))

    return {
        "003M_shadow_hit": hit,
        "003M_failed_clauses": ",".join(failed),
        "003M_player_value": vals.get("inside_player_motion_mask_ratio_003C", np.nan),
        "003M_player_margin": round(float(margins.get("player_motion_inside", np.nan)), 6)
            if np.isfinite(margins.get("player_motion_inside", np.nan)) else np.nan,
        "003M_table_distance_value": vals.get("point_distance_to_table_px_median", np.nan),
        "003M_table_distance_margin": round(float(margins.get("old_table_distance", np.nan)), 3)
            if np.isfinite(margins.get("old_table_distance", np.nan)) else np.nan,
        "003M_accel_value": vals.get("max_accel", np.nan),
        "003M_accel_margin": round(float(margins.get("max_accel", np.nan)), 3)
            if np.isfinite(margins.get("max_accel", np.nan)) else np.nan,
    }


def classify(row: pd.Series) -> tuple[str, str]:
    rid = str(row["review_id"])
    hit = boolish(row.get("003M_shadow_hit"))

    if rid in CONFIRMED_REJECT_IDS:
        if hit:
            return "CONFIRMED_REJECT_CAUGHT", "expected_confirmed_reject_hit"
        return "CONFIRMED_REJECT_MISSED", "confirmed_reject_not_hit"

    if rid in PROTECTED_SENTINEL_IDS:
        if hit:
            return "PROTECTED_SENTINEL_CAUGHT_DANGER", "protected_sentinel_hit"
        return "PROTECTED_SENTINEL_SAFE", "protected_sentinel_not_hit"

    if hit:
        return "OTHER_CAUGHT_REVIEW", "unlabeled_or_other_hit"

    return "OTHER_NOT_CAUGHT", "not_hit"


def write_html(path: Path, df: pd.DataFrame, summary: dict) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section,.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card.good{border-color:rgba(116,217,159,.75);box-shadow:inset 4px 0 0 rgba(116,217,159,.85)}
.card.warn{border-color:rgba(255,200,80,.75);box-shadow:inset 4px 0 0 rgba(255,200,80,.85)}
.card.bad{border-color:rgba(255,80,80,.75);box-shadow:inset 4px 0 0 rgba(255,80,80,.85)}
.card.other{border-color:#2b303b}
h1,h2,h3{margin-top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{width:280px;background:#20242e}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.good{border-color:rgba(116,217,159,.8);background:rgba(116,217,159,.08)}
.badge.warn{border-color:rgba(255,200,80,.8);background:rgba(255,200,80,.08)}
.badge.bad{border-color:rgba(255,80,80,.8);background:rgba(255,80,80,.08)}
"""

    priority = {
        "CONFIRMED_REJECT_CAUGHT": 0,
        "CONFIRMED_REJECT_MISSED": 1,
        "PROTECTED_SENTINEL_CAUGHT_DANGER": 2,
        "PROTECTED_SENTINEL_SAFE": 3,
        "OTHER_CAUGHT_REVIEW": 4,
        "OTHER_NOT_CAUGHT": 5,
    }

    work = df.copy()
    work["_prio"] = work["003M_audit_class"].map(lambda x: priority.get(str(x), 99))
    work = work.sort_values(["_prio", "review_id"]).drop(columns=["_prio"])

    cols = [
        "review_id",
        "target_class_003G",
        "003M_audit_class",
        "003M_audit_reason",
        "003M_shadow_hit",
        "003M_failed_clauses",
        "003M_player_value",
        "003M_player_margin",
        "003M_table_distance_value",
        "003M_table_distance_margin",
        "003M_accel_value",
        "003M_accel_margin",
        "would_reject_shadow_003G",
        "multi_object_shadow_hit_003E",
    ]

    cards = []

    for _, r in work.iterrows():
        cls_name = str(r.get("003M_audit_class", ""))

        if cls_name == "CONFIRMED_REJECT_CAUGHT":
            card = "card good"
            badge = "badge good"
        elif cls_name in {"CONFIRMED_REJECT_MISSED", "PROTECTED_SENTINEL_CAUGHT_DANGER"}:
            card = "card bad"
            badge = "badge bad"
        elif cls_name == "PROTECTED_SENTINEL_SAFE":
            card = "card warn"
            badge = "badge warn"
        else:
            card = "card other"
            badge = "badge"

        trs = []
        for c in cols:
            if c in r.index:
                trs.append(f"<tr><th>{esc(c)}</th><td>{esc(r.get(c, ''))}</td></tr>")

        cards.append(f"""
<div class="{card}">
<h2>{esc(r.get('review_id', ''))}</h2>
<div><span class="{badge}">{esc(cls_name)}</span></div>
<table><tbody>{''.join(trs)}</tbody></table>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003M shadow rule freeze regression</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003M shadow rule freeze regression</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

{''.join(cards)}

</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_001E")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run)

    source_csv = find_source_csv(run_dir)
    df = normalize_review_id(pd.read_csv(source_csv))

    missing_cols = [c["column"] for c in RULE["clauses"] if c["column"] not in df.columns]
    if missing_cols:
        raise SystemExit("Colonnes manquantes pour la règle: " + ",".join(missing_cols))

    eval_rows = [eval_rule(row) for _, row in df.iterrows()]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(eval_rows).reset_index(drop=True)], axis=1)

    audit_classes = []
    audit_reasons = []
    for _, row in df.iterrows():
        c, r = classify(row)
        audit_classes.append(c)
        audit_reasons.append(r)

    df["003M_audit_class"] = audit_classes
    df["003M_audit_reason"] = audit_reasons

    caught_ids = df[df["003M_shadow_hit"].map(boolish)]["review_id"].astype(str).tolist()
    caught_confirmed = sorted([x for x in caught_ids if x in CONFIRMED_REJECT_IDS])
    caught_protected = sorted([x for x in caught_ids if x in PROTECTED_SENTINEL_IDS])
    missed_confirmed = sorted(list(CONFIRMED_REJECT_IDS - set(caught_confirmed)))

    by_class = (
        df.groupby("003M_audit_class")["review_id"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    warnings = []
    if missed_confirmed:
        warnings.append("missed confirmed rejects: " + ",".join(missed_confirmed))
    if caught_protected:
        warnings.append("caught protected sentinels: " + ",".join(caught_protected))

    regression_ok = (
        caught_confirmed == EXPECTED_001E["caught_confirmed_reject_ids"]
        and caught_protected == EXPECTED_001E["caught_protected_sentinel_ids"]
        and missed_confirmed == []
    )

    status = "OK_RULE_FROZEN" if not warnings and regression_ok else "WARN_REGRESSION_REVIEW"

    rules_dir = run_dir / "shadow_rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    rule_json = rules_dir / "003L_hardened_v1_player060_dist210_accel20.json"
    rule_payload = {
        **RULE,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_from_run": str(run_dir),
        "expected_001E": EXPECTED_001E,
    }
    rule_json.write_text(json.dumps(rule_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "policy": "freeze_shadow_rule_and_regression_only_no_live_filter_no_delete",
        "source_csv": str(source_csv),
        "rule_json": str(rule_json),
        "rule": RULE,
        "rows": int(len(df)),
        "caught_total": int(len(caught_ids)),
        "caught_ids": caught_ids,
        "caught_confirmed_reject_ids": caught_confirmed,
        "missed_confirmed_reject_ids": missed_confirmed,
        "caught_protected_sentinel_ids": caught_protected,
        "audit_class_by_id": by_class,
        "regression_expected_001E": EXPECTED_001E,
        "regression_ok": regression_ok,
        "warnings": warnings,
        "next_step": (
            "003N: run this frozen shadow rule on the next batch / larger sample. "
            "Do not promote live before cross-batch validation."
        ),
    }

    out_csv = run_dir / "shadow_rule_freeze_regression_003M.csv"
    out_json = run_dir / "shadow_rule_freeze_regression_summary_003M.json"
    out_html = run_dir / "shadow_rule_freeze_regression_003M.html"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, df, summary)

    print(f"003M status={status}")
    print("source_csv=", source_csv)
    print("rule_json=", rule_json)
    print("caught_ids=" + (",".join(caught_ids) or "-"))
    print("caught_confirmed=" + (",".join(caught_confirmed) or "-"))
    print("missed_confirmed=" + (",".join(missed_confirmed) or "-"))
    print("caught_protected=" + (",".join(caught_protected) or "-"))
    print("regression_ok=" + str(regression_ok))
    print("warnings=" + (" | ".join(warnings) if warnings else "-"))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    if args.strict and status != "OK_RULE_FROZEN":
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
