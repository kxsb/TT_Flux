from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003V"

RULE = {
    "player_col": "inside_player_motion_mask_ratio_003C",
    "player_min": 0.60,
    "table_col": "point_distance_to_table_px_median",
    "table_max": 210.0,
    "accel_col": "max_accel",
    "accel_min": 20.0,
}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def num(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def clean_label(x) -> str:
    s = str(x or "").strip().lower()
    if s in {"reject", "keep", "partial", "unsure"}:
        return s
    return ""


def passfail(v, op: str, th: float) -> bool:
    if v is None:
        return False
    if op == ">=":
        return v >= th
    if op == "<=":
        return v <= th
    return False


def row_eval(r: pd.Series) -> dict:
    player = num(r.get(RULE["player_col"]))
    table = num(r.get(RULE["table_col"]))
    accel = num(r.get(RULE["accel_col"]))

    player_pass = passfail(player, ">=", RULE["player_min"])
    table_pass = passfail(table, "<=", RULE["table_max"])
    accel_pass = passfail(accel, ">=", RULE["accel_min"])

    fails = []
    if not player_pass:
        fails.append("player")
    if not table_pass:
        fails.append("table")
    if not accel_pass:
        fails.append("accel")

    player_margin = None if player is None else player - RULE["player_min"]
    table_margin = None if table is None else RULE["table_max"] - table
    accel_margin = None if accel is None else accel - RULE["accel_min"]

    return {
        "003V_player_value": player,
        "003V_table_value": table,
        "003V_accel_value": accel,
        "003V_player_pass": player_pass,
        "003V_table_pass": table_pass,
        "003V_accel_pass": accel_pass,
        "003V_failed_clauses": ",".join(fails),
        "003V_rule_hit_recomputed": bool(player_pass and table_pass and accel_pass),
        "003V_player_margin": player_margin,
        "003V_table_margin": table_margin,
        "003V_accel_margin": accel_margin,
    }


def write_html(path: Path, df: pd.DataFrame, summary: dict) -> None:
    cards = []

    for _, r in df.iterrows():
        label = clean_label(r.get("human_label_003S", ""))
        hit = bool(r.get("003V_rule_hit_recomputed", False))

        if label == "reject" and not hit:
            cls = "missed-reject"
        elif label in {"keep", "partial"} and hit:
            cls = "false-positive"
        elif hit:
            cls = "hit"
        elif label:
            cls = "labeled-safe"
        else:
            cls = "unlabeled"

        trs = []
        for c in [
            "review_id",
            "human_label_003S",
            "comment_003S",
            "003V_rule_hit_recomputed",
            "003V_failed_clauses",
            "003V_player_value",
            "003V_player_margin",
            "003V_table_value",
            "003V_table_margin",
            "003V_accel_value",
            "003V_accel_margin",
            "target_class_003G",
            "multi_object_shadow_hit_003E",
            "would_reject_shadow_003G",
            "mp4",
            "csv",
        ]:
            if c in r.index:
                trs.append(f"<tr><th>{esc(c)}</th><td>{esc(r.get(c, ''))}</td></tr>")

        cards.append(f"""
<div class="card {cls}">
<h2>{esc(r.get('review_id', ''))} · {esc(label or 'unlabeled')}</h2>
<table><tbody>{''.join(trs)}</tbody></table>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003V true feature diagnosis</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section,.card{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
.card.missed-reject{{border-color:rgba(255,80,80,.85);box-shadow:inset 4px 0 0 rgba(255,80,80,.9)}}
.card.false-positive{{border-color:rgba(255,200,80,.85);box-shadow:inset 4px 0 0 rgba(255,200,80,.9)}}
.card.hit{{border-color:rgba(116,217,159,.75);box-shadow:inset 4px 0 0 rgba(116,217,159,.85)}}
.card.labeled-safe{{border-color:#3a4050}}
.card.unlabeled{{opacity:.82}}
h1,h2{{margin-top:0}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{width:280px;background:#20242e}}
</style>
</head>
<body>
<h1>TTFlux · 003V true feature diagnosis</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
{''.join(cards)}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def sweep_thresholds(df: pd.DataFrame) -> list[dict]:
    labeled = df[df["human_label_003S"].map(clean_label) != ""].copy()
    if labeled.empty:
        return []

    labeled["is_reject"] = labeled["human_label_003S"].map(clean_label).eq("reject")
    labeled["is_danger"] = labeled["human_label_003S"].map(clean_label).isin(["keep", "partial"])

    player_vals = [0.20, 0.30, 0.40, 0.50, 0.60]
    table_vals = [120, 160, 210, 260, 320, 420]
    accel_vals = [8, 12, 16, 20, 25, 30]

    results = []

    for pth in player_vals:
        for tth in table_vals:
            for ath in accel_vals:
                p = pd.to_numeric(labeled[RULE["player_col"]], errors="coerce")
                t = pd.to_numeric(labeled[RULE["table_col"]], errors="coerce")
                a = pd.to_numeric(labeled[RULE["accel_col"]], errors="coerce")

                hit = (p >= pth) & (t <= tth) & (a >= ath)

                reject_hit = int((hit & labeled["is_reject"]).sum())
                danger_hit = int((hit & labeled["is_danger"]).sum())
                unsure_hit = int((hit & labeled["human_label_003S"].eq("unsure")).sum())
                total_hit = int(hit.sum())

                results.append({
                    "player_min": pth,
                    "table_max": tth,
                    "accel_min": ath,
                    "total_hit": total_hit,
                    "reject_hit": reject_hit,
                    "danger_hit_keep_or_partial": danger_hit,
                    "unsure_hit": unsure_hit,
                    "score": reject_hit * 10 - danger_hit * 20 - unsure_hit * 3 - max(0, total_hit - reject_hit) * 2,
                    "hit_ids": labeled.loc[hit, "review_id"].astype(str).tolist(),
                    "reject_hit_ids": labeled.loc[hit & labeled["is_reject"], "review_id"].astype(str).tolist(),
                    "danger_hit_ids": labeled.loc[hit & labeled["is_danger"], "review_id"].astype(str).tolist(),
                })

    return sorted(results, key=lambda x: (x["score"], x["reject_hit"], -x["danger_hit_keep_or_partial"]), reverse=True)[:20]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_002A")
    ap.add_argument("--features", default="multi_object_arbiter_shadow_003E.csv")
    ap.add_argument("--labels", default="human_review_003S_labels.csv")
    args = ap.parse_args()

    root = Path.cwd()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    features = Path(args.features)
    if not features.is_absolute():
        features = run_dir / features

    labels = Path(args.labels)
    if not labels.is_absolute():
        labels = run_dir / labels

    if not features.is_file():
        raise SystemExit(f"Features introuvables: {features}")

    df = pd.read_csv(features)

    if labels.is_file():
        lab = pd.read_csv(labels)
        lab["review_id"] = lab["review_id"].astype(str)
        lab["human_label_003S"] = lab["human_label_003S"].map(clean_label)
        df["review_id"] = df["review_id"].astype(str)
        df = df.merge(
            lab[["review_id", "human_label_003S", "comment_003S", "updated_at"]],
            on="review_id",
            how="left",
        )
    else:
        df["human_label_003S"] = ""
        df["comment_003S"] = ""

    eval_rows = []
    for _, r in df.iterrows():
        eval_rows.append(row_eval(r))

    eval_df = pd.DataFrame(eval_rows)
    out = pd.concat([df.reset_index(drop=True), eval_df], axis=1)

    label_counts = out["human_label_003S"].map(clean_label).value_counts().to_dict()

    hits = out[out["003V_rule_hit_recomputed"].astype(bool)]
    missed_rejects = out[
        (out["human_label_003S"].map(clean_label) == "reject")
        & (~out["003V_rule_hit_recomputed"].astype(bool))
    ]

    dangerous_hits = out[
        (out["human_label_003S"].map(clean_label).isin(["keep", "partial"]))
        & (out["003V_rule_hit_recomputed"].astype(bool))
    ]

    sweep = sweep_thresholds(out)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "diagnostic_only_no_live_filter_no_delete",
        "run_dir": str(run_dir),
        "features": str(features),
        "labels": str(labels) if labels.is_file() else "",
        "rows": int(len(out)),
        "human_label_counts": label_counts,
        "rule": RULE,
        "recomputed_hit_total": int(len(hits)),
        "recomputed_hit_ids": hits["review_id"].astype(str).tolist(),
        "missed_reject_total": int(len(missed_rejects)),
        "missed_reject_ids": missed_rejects["review_id"].astype(str).tolist(),
        "dangerous_hit_total": int(len(dangerous_hits)),
        "dangerous_hit_ids": dangerous_hits["review_id"].astype(str).tolist(),
        "top_threshold_sweep_labeled_only": sweep[:10],
        "warning": "Uses human labels from proxy review when present. Unlabeled rows remain unknown.",
    }

    out_csv = run_dir / "true_feature_diagnosis_003V.csv"
    out_json = run_dir / "true_feature_diagnosis_summary_003V.json"
    out_html = run_dir / "true_feature_diagnosis_003V.html"

    out.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, out, summary)

    print("003V status=OK")
    print("rows=", len(out))
    print("human_label_counts=", label_counts)
    print("recomputed_hit_total=", len(hits))
    print("recomputed_hit_ids=" + (",".join(summary["recomputed_hit_ids"]) or "-"))
    print("missed_reject_total=", len(missed_rejects))
    print("missed_reject_ids=" + (",".join(summary["missed_reject_ids"]) or "-"))
    print("dangerous_hit_total=", len(dangerous_hits))
    print("dangerous_hit_ids=" + (",".join(summary["dangerous_hit_ids"]) or "-"))

    if sweep:
        best = sweep[0]
        print("best_sweep=", json.dumps(best, ensure_ascii=False))

    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
