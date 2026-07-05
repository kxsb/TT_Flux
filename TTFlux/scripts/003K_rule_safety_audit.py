from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "003K"

# Règle 003E actuelle.
CURRENT_RULE = {
    "name": "003E_current_multi_object_table_player_safe_v1",
    "player_col": "inside_player_motion_mask_ratio_003C",
    "player_min": 0.516666,
    "table_distance_col": "point_distance_to_table_px_median",
    "table_distance_max": 216.719,
    "accel_col": "max_accel",
    "accel_min": 17.672,
}

# Variante durcie candidate, à auditer seulement.
# But : bloquer les near-miss sentinelles R0019/R0022 sans perdre les 6 rejets confirmés.
HARDENED_RULES = [
    {
        "name": "003K_hardened_v1_player060_dist210_accel20",
        "player_col": "inside_player_motion_mask_ratio_003C",
        "player_min": 0.60,
        "table_distance_col": "point_distance_to_table_px_median",
        "table_distance_max": 210.0,
        "accel_col": "max_accel",
        "accel_min": 20.0,
    },
    {
        "name": "003K_hardened_v2_player062_dist205_accel20",
        "player_col": "inside_player_motion_mask_ratio_003C",
        "player_min": 0.62,
        "table_distance_col": "point_distance_to_table_px_median",
        "table_distance_max": 205.0,
        "accel_col": "max_accel",
        "accel_min": 20.0,
    },
    {
        "name": "003K_hardened_v3_player060_dist200_accel21",
        "player_col": "inside_player_motion_mask_ratio_003C",
        "player_min": 0.60,
        "table_distance_col": "point_distance_to_table_px_median",
        "table_distance_max": 200.0,
        "accel_col": "max_accel",
        "accel_min": 21.0,
    },
]

CONFIRMED_REJECT_IDS = {"R0008", "R0009", "R0014", "R0020", "R0021", "R0010"}
PROTECTED_SENTINEL_IDS = {"R0019", "R0022", "R0015", "R0003", "R0006", "R0018"}

# Seuils de near-miss : un segment protégé qui rate une seule clause de très peu est fragile.
NEAR_MISS_TOLERANCE = {
    "player": 0.05,
    "table_distance": 25.0,
    "accel": 2.0,
}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def num(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Fichier introuvable: {path}")
    return pd.read_csv(path)


def normalize_review_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    candidates = [
        "review_id",
        "review_id_003G",
        "review_id_003F",
        "review_id_003E",
        "review_id_003D",
    ]

    rid_col = None
    for c in candidates:
        if c in df.columns:
            rid_col = c
            break

    if rid_col is None:
        raise SystemExit("Impossible de trouver une colonne review_id dans un CSV.")

    df["review_id"] = df[rid_col].astype(str)
    return df


def merge_on_review_id(base: pd.DataFrame, other: pd.DataFrame, suffix: str) -> pd.DataFrame:
    other = normalize_review_id_cols(other)

    # Évite les collisions illisibles.
    rename = {}
    for c in other.columns:
        if c == "review_id":
            continue
        if c in base.columns:
            rename[c] = f"{c}_{suffix}"

    other = other.rename(columns=rename)
    return base.merge(other, on="review_id", how="left")


def eval_rule_on_row(row: pd.Series, rule: dict, prefix: str) -> dict:
    player = num(row.get(rule["player_col"]))
    dist = num(row.get(rule["table_distance_col"]))
    accel = num(row.get(rule["accel_col"]))

    player_pass = np.isfinite(player) and player >= rule["player_min"]
    dist_pass = np.isfinite(dist) and dist <= rule["table_distance_max"]
    accel_pass = np.isfinite(accel) and accel >= rule["accel_min"]

    hit = bool(player_pass and dist_pass and accel_pass)

    player_margin = player - rule["player_min"] if np.isfinite(player) else np.nan
    dist_margin = rule["table_distance_max"] - dist if np.isfinite(dist) else np.nan
    accel_margin = accel - rule["accel_min"] if np.isfinite(accel) else np.nan

    failed = []
    if not player_pass:
        failed.append("player")
    if not dist_pass:
        failed.append("table_distance")
    if not accel_pass:
        failed.append("accel")

    near_failed = []
    if "player" in failed and np.isfinite(player_margin) and player_margin >= -NEAR_MISS_TOLERANCE["player"]:
        near_failed.append("player")
    if "table_distance" in failed and np.isfinite(dist_margin) and dist_margin >= -NEAR_MISS_TOLERANCE["table_distance"]:
        near_failed.append("table_distance")
    if "accel" in failed and np.isfinite(accel_margin) and accel_margin >= -NEAR_MISS_TOLERANCE["accel"]:
        near_failed.append("accel")

    near_miss = len(failed) == 1 and len(near_failed) == 1

    return {
        f"{prefix}_hit": hit,
        f"{prefix}_player_value": player,
        f"{prefix}_table_distance_value": dist,
        f"{prefix}_accel_value": accel,
        f"{prefix}_player_pass": bool(player_pass),
        f"{prefix}_table_distance_pass": bool(dist_pass),
        f"{prefix}_accel_pass": bool(accel_pass),
        f"{prefix}_player_margin": round(float(player_margin), 6) if np.isfinite(player_margin) else np.nan,
        f"{prefix}_table_distance_margin": round(float(dist_margin), 3) if np.isfinite(dist_margin) else np.nan,
        f"{prefix}_accel_margin": round(float(accel_margin), 3) if np.isfinite(accel_margin) else np.nan,
        f"{prefix}_failed_clauses": ",".join(failed),
        f"{prefix}_near_failed_clauses": ",".join(near_failed),
        f"{prefix}_near_miss": bool(near_miss),
    }


def classify_row(row: pd.Series) -> tuple[str, list[str]]:
    rid = str(row["review_id"])
    target = str(row.get("target_class_003G", row.get("target_class", "")))

    current_hit = boolish(row.get("current_rule_hit"))
    manual_outside = boolish(row.get("manual_outside_table_003K"))
    manual_overlap = boolish(row.get("manual_overlap_table_003K"))

    reasons = []

    if rid in CONFIRMED_REJECT_IDS:
        if current_hit and manual_outside:
            return "CONFIRMED_REJECT_CAUGHT", ["current_rule_hit", "manual_track_outside"]
        if not current_hit and manual_outside:
            return "CONFIRMED_REJECT_MISSED", ["current_rule_miss", "manual_track_outside"]
        return "CONFIRMED_REJECT_UNCLEAR", ["unexpected_manual_overlap_or_missing"]

    if rid in PROTECTED_SENTINEL_IDS:
        if current_hit:
            return "PROTECTED_SENTINEL_DANGER_CAUGHT", ["protected_segment_would_be_rejected"]
        if manual_outside:
            reasons.append("protected_but_manual_track_outside")
        if manual_overlap:
            reasons.append("protected_has_table_overlap")
        if boolish(row.get("current_rule_near_miss")):
            reasons.append("near_miss_current_rule")
        if not reasons:
            reasons.append("protected_not_caught")
        return "PROTECTED_SENTINEL_SAFE_NOT_CAUGHT", reasons

    if current_hit:
        return "UNLABELED_OR_OTHER_CAUGHT", ["current_rule_hit"]

    return "UNLABELED_OR_OTHER_NOT_CAUGHT", ["current_rule_miss"]


def summarize_rule(df: pd.DataFrame, hit_col: str, rule_name: str) -> dict:
    caught = df[df[hit_col].map(boolish)].copy()

    caught_ids = caught["review_id"].astype(str).tolist()
    caught_confirmed = sorted([x for x in caught_ids if x in CONFIRMED_REJECT_IDS])
    caught_protected = sorted([x for x in caught_ids if x in PROTECTED_SENTINEL_IDS])

    missed_confirmed = sorted(list(CONFIRMED_REJECT_IDS - set(caught_confirmed)))

    return {
        "rule_name": rule_name,
        "caught_total": int(len(caught_ids)),
        "caught_ids": caught_ids,
        "caught_confirmed_reject_ids": caught_confirmed,
        "missed_confirmed_reject_ids": missed_confirmed,
        "caught_protected_sentinel_ids": caught_protected,
        "safe_on_protected_sentinels": len(caught_protected) == 0,
        "catches_all_confirmed_rejects": len(missed_confirmed) == 0,
        "promotable_on_current_audit": len(caught_protected) == 0 and len(missed_confirmed) == 0,
    }


def write_html(path: Path, df: pd.DataFrame, summary: dict) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section,.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card.confirmed{border-color:rgba(116,217,159,.75);box-shadow:inset 4px 0 0 rgba(116,217,159,.85)}
.card.protected{border-color:rgba(255,200,80,.75);box-shadow:inset 4px 0 0 rgba(255,200,80,.85)}
.card.danger{border-color:rgba(255,80,80,.75);box-shadow:inset 4px 0 0 rgba(255,80,80,.85)}
.card.other{border-color:#2b303b}
h1,h2,h3{margin-top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{width:280px;background:#20242e}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:950px){.grid{grid-template-columns:1fr}}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.confirmed{border-color:rgba(116,217,159,.8);background:rgba(116,217,159,.08)}
.badge.protected{border-color:rgba(255,200,80,.8);background:rgba(255,200,80,.08)}
.badge.danger{border-color:rgba(255,80,80,.8);background:rgba(255,80,80,.08)}
.muted{color:#aab2c5}
"""

    cards = []

    cols_left = [
        "review_id",
        "target_class_003G",
        "audit_class_003K",
        "audit_reasons_003K",
        "would_reject_shadow_003G",
        "multi_object_shadow_hit_003E",
        "manual_outside_table_003K",
        "manual_overlap_table_003K",
        "inside_ratio_003J3",
        "distance_median_003J3",
        "signed_median_003J3",
    ]

    cols_current = [
        "current_rule_hit",
        "current_rule_near_miss",
        "current_rule_player_value",
        "current_rule_player_pass",
        "current_rule_player_margin",
        "current_rule_table_distance_value",
        "current_rule_table_distance_pass",
        "current_rule_table_distance_margin",
        "current_rule_accel_value",
        "current_rule_accel_pass",
        "current_rule_accel_margin",
        "current_rule_failed_clauses",
        "current_rule_near_failed_clauses",
    ]

    hardened_cols = []
    for rule in HARDENED_RULES:
        prefix = rule["name"]
        hardened_cols.extend([
            f"{prefix}_hit",
            f"{prefix}_failed_clauses",
            f"{prefix}_near_miss",
        ])

    def render_table(row, cols):
        trs = []
        for c in cols:
            if c in row.index:
                trs.append(f"<tr><th>{esc(c)}</th><td>{esc(row.get(c, ''))}</td></tr>")
        return "<table><tbody>" + "".join(trs) + "</tbody></table>"

    priority = {
        "CONFIRMED_REJECT_CAUGHT": 0,
        "CONFIRMED_REJECT_MISSED": 1,
        "PROTECTED_SENTINEL_DANGER_CAUGHT": 2,
        "PROTECTED_SENTINEL_SAFE_NOT_CAUGHT": 3,
        "UNLABELED_OR_OTHER_CAUGHT": 4,
        "UNLABELED_OR_OTHER_NOT_CAUGHT": 5,
    }

    work = df.copy()
    work["_p"] = work["audit_class_003K"].map(lambda x: priority.get(str(x), 99))
    work = work.sort_values(["_p", "review_id"]).drop(columns=["_p"])

    for _, row in work.iterrows():
        cls_name = str(row.get("audit_class_003K", ""))

        if cls_name.startswith("CONFIRMED"):
            cls = "card confirmed"
            badge = "badge confirmed"
        elif "DANGER" in cls_name:
            cls = "card danger"
            badge = "badge danger"
        elif cls_name.startswith("PROTECTED"):
            cls = "card protected"
            badge = "badge protected"
        else:
            cls = "card other"
            badge = "badge"

        rid = row.get("review_id", "")

        cards.append(f"""
<div class="{cls}">
<h2>{esc(rid)}</h2>
<div><span class="{badge}">{esc(cls_name)}</span></div>
<div class="grid">
  <div>
    <h3>Audit</h3>
    {render_table(row, cols_left)}
  </div>
  <div>
    <h3>Règle actuelle 003E</h3>
    {render_table(row, cols_current)}
    <h3>Variantes durcies testées</h3>
    {render_table(row, hardened_cols)}
  </div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003K rule safety audit</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003K rule safety audit</h1>

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

    g_csv = run_dir / "shadow_filter_simulation_003G.csv"
    e_csv = run_dir / "multi_object_arbiter_shadow_003E.csv"
    f_csv = run_dir / "consolidated_shadow_arbiter_003F.csv"
    j2_csv = run_dir / "manual_table_metric_recompute_003J2B.csv"
    j3_csv = run_dir / "manual_table_track_sanity_003J3.csv"

    base = normalize_review_id_cols(read_csv_required(g_csv))

    optional_sources = [
        (e_csv, "003E"),
        (f_csv, "003F"),
        (j2_csv, "003J2B"),
        (j3_csv, "003J3"),
    ]

    loaded_sources = {"003G": str(g_csv)}

    for path, suffix in optional_sources:
        if path.is_file():
            df = read_csv_required(path)
            base = merge_on_review_id(base, df, suffix)
            loaded_sources[suffix] = str(path)

    df = base.copy()

    # Colonnes manuelles J3, avec fallback selon collisions éventuelles.
    inside_col = "inside_ratio_003J3"
    if inside_col not in df.columns:
        candidates = [c for c in df.columns if "inside_ratio_003J3" in c]
        if candidates:
            inside_col = candidates[0]

    df["manual_outside_table_003K"] = pd.to_numeric(df.get(inside_col, np.nan), errors="coerce").fillna(np.nan).map(
        lambda x: bool(np.isfinite(x) and x < 0.20)
    )
    df["manual_overlap_table_003K"] = pd.to_numeric(df.get(inside_col, np.nan), errors="coerce").fillna(np.nan).map(
        lambda x: bool(np.isfinite(x) and x >= 0.20)
    )

    # Règle actuelle.
    current_eval = []
    for _, row in df.iterrows():
        current_eval.append(eval_rule_on_row(row, CURRENT_RULE, "current_rule"))
    current_eval_df = pd.DataFrame(current_eval)
    df = pd.concat([df.reset_index(drop=True), current_eval_df.reset_index(drop=True)], axis=1)

    # Variantes durcies.
    for rule in HARDENED_RULES:
        prefix = rule["name"]
        rows = []
        for _, row in df.iterrows():
            rows.append(eval_rule_on_row(row, rule, prefix))
        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows).reset_index(drop=True)], axis=1)

    # Classification audit.
    audit_classes = []
    audit_reasons = []
    for _, row in df.iterrows():
        cls, reasons = classify_row(row)
        audit_classes.append(cls)
        audit_reasons.append(",".join(reasons))

    df["audit_class_003K"] = audit_classes
    df["audit_reasons_003K"] = audit_reasons

    # Résumés.
    current_summary = summarize_rule(df, "current_rule_hit", CURRENT_RULE["name"])
    hardened_summaries = [
        summarize_rule(df, f"{rule['name']}_hit", rule["name"])
        for rule in HARDENED_RULES
    ]

    by_audit_class = (
        df.groupby("audit_class_003K")["review_id"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    protected_df = df[df["review_id"].isin(PROTECTED_SENTINEL_IDS)].copy()
    protected_near_miss_ids = protected_df[
        protected_df["current_rule_near_miss"].map(boolish)
    ]["review_id"].astype(str).tolist()

    protected_outside_ids = protected_df[
        protected_df["manual_outside_table_003K"].map(boolish)
    ]["review_id"].astype(str).tolist()

    protected_overlap_ids = protected_df[
        protected_df["manual_overlap_table_003K"].map(boolish)
    ]["review_id"].astype(str).tolist()

    confirmed_df = df[df["review_id"].isin(CONFIRMED_REJECT_IDS)].copy()
    confirmed_current_caught = confirmed_df[
        confirmed_df["current_rule_hit"].map(boolish)
    ]["review_id"].astype(str).tolist()

    confirmed_current_missed = sorted(list(CONFIRMED_REJECT_IDS - set(confirmed_current_caught)))

    candidate_promotable = [
        s for s in hardened_summaries
        if s["promotable_on_current_audit"]
    ]

    warnings = []

    if current_summary["caught_protected_sentinel_ids"]:
        warnings.append(
            "current rule catches protected sentinels: "
            + ",".join(current_summary["caught_protected_sentinel_ids"])
        )

    if protected_near_miss_ids:
        warnings.append(
            "protected sentinels are near-miss under current rule: "
            + ",".join(protected_near_miss_ids)
        )

    if protected_outside_ids:
        warnings.append(
            "manual outside-table is not sufficient; protected sentinels outside table: "
            + ",".join(protected_outside_ids)
        )

    if confirmed_current_missed:
        warnings.append(
            "current rule misses confirmed rejects: "
            + ",".join(confirmed_current_missed)
        )

    status = "OK_WITH_WARNINGS" if warnings else "OK"

    recommendation = {
        "do_not_use_manual_outside_alone": True,
        "current_003E_live_recommendation": (
            "shadow_only" if protected_near_miss_ids else "candidate_with_more_data"
        ),
        "best_hardened_candidates_from_this_audit": candidate_promotable,
        "recommended_next_step": (
            "003L: run hardened candidate rules as shadow on all 22 rows and generate visual packet. "
            "Do not delete or filter live yet."
        ),
    }

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "policy": "safety_audit_only_no_arbiter_change",
        "loaded_sources": loaded_sources,
        "rows": int(len(df)),
        "confirmed_reject_ids": sorted(CONFIRMED_REJECT_IDS),
        "protected_sentinel_ids": sorted(PROTECTED_SENTINEL_IDS),
        "audit_class_by_id": by_audit_class,
        "current_rule": CURRENT_RULE,
        "current_rule_summary": current_summary,
        "hardened_rule_summaries": hardened_summaries,
        "protected_sentinel_near_miss_current_rule_ids": protected_near_miss_ids,
        "protected_sentinel_manual_outside_ids": protected_outside_ids,
        "protected_sentinel_manual_overlap_ids": protected_overlap_ids,
        "confirmed_current_caught_ids": confirmed_current_caught,
        "confirmed_current_missed_ids": confirmed_current_missed,
        "warnings": warnings,
        "recommendation": recommendation,
    }

    out_csv = run_dir / "rule_safety_audit_003K.csv"
    out_json = run_dir / "rule_safety_audit_summary_003K.json"
    out_html = run_dir / "rule_safety_audit_003K.html"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, df, summary)

    print(f"003K status={status}")
    print("current_caught_confirmed=" + ",".join(current_summary["caught_confirmed_reject_ids"]))
    print("current_missed_confirmed=" + (",".join(current_summary["missed_confirmed_reject_ids"]) or "-"))
    print("current_caught_protected=" + (",".join(current_summary["caught_protected_sentinel_ids"]) or "-"))
    print("protected_near_miss_current=" + (",".join(protected_near_miss_ids) or "-"))
    print("protected_manual_outside=" + (",".join(protected_outside_ids) or "-"))

    for hs in hardened_summaries:
        print(
            "hardened",
            hs["rule_name"],
            "promotable_on_current_audit=" + str(hs["promotable_on_current_audit"]),
            "caught_confirmed=" + ",".join(hs["caught_confirmed_reject_ids"]),
            "missed_confirmed=" + (",".join(hs["missed_confirmed_reject_ids"]) or "-"),
            "caught_protected=" + (",".join(hs["caught_protected_sentinel_ids"]) or "-"),
        )

    for w in warnings:
        print("warning:", w)

    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    if args.strict and status != "OK":
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
