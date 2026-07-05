from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003F"

EXPECTED_REJECT_SUGGESTED = {"R0010", "R0009", "R0020", "R0014", "R0008", "R0021"}
EXPECTED_003B4 = {"R0010", "R0009", "R0020", "R0008", "R0021"}
EXPECTED_003E_NEW = {"R0014"}

SAFE_TARGETS_NOT_TO_CATCH = {"keep", "human_partial"}


def norm_target(x) -> str:
    s = str(x or "").strip().lower().replace("-", "_").replace(" ", "_")
    if s in {"humanreject", "human_rejected"}:
        return "human_reject"
    if s in {"humanpartial", "partial"}:
        return "human_partial"
    if s in {"autoreject", "auto_rejected"}:
        return "auto_reject"
    if s in {"autoreview", "review"}:
        return "auto_review"
    if s in {"keep", "ok", "valid"}:
        return "keep"
    return s


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def reason_for_row(row: pd.Series) -> str:
    reasons = []
    if bool(row["contextual_hit_003B4_resolved"]):
        reasons.append("003B4_contextual_table_kinematic")
    if bool(row["multi_object_shadow_hit_003E"]):
        reasons.append("003E_multi_object_table_player_kinematic")
    return " + ".join(reasons)


def write_html(path: Path, df: pd.DataFrame, summary: dict) -> None:
    cols = [
        "review_id_003F",
        "target_class_003F",
        "contextual_hit_003B4_resolved",
        "multi_object_shadow_hit_003E",
        "reject_suggested_shadow_003F",
        "new_reject_suggested_vs_003B4_003F",
        "arbiter_reason_003F",
        "final_decision_policy_003F",
    ]

    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{background:#20242e;position:sticky;top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
.hit td{box-shadow:inset 3px 0 0 rgba(116,217,159,.95)}
.new td{background:rgba(116,217,159,.09)}
.unsafe td{background:rgba(255,80,80,.13)}
.keep td{background:rgba(100,255,150,.045)}
.partial td{background:rgba(255,200,80,.07)}
"""

    trs = []
    for _, r in df.iterrows():
        cls = []
        if bool(r["reject_suggested_shadow_003F"]):
            cls.append("hit")
        if bool(r["new_reject_suggested_vs_003B4_003F"]):
            cls.append("new")
        if r["target_class_003F"] == "keep":
            cls.append("keep")
        if r["target_class_003F"] == "human_partial":
            cls.append("partial")
        if bool(r["reject_suggested_shadow_003F"]) and r["target_class_003F"] in SAFE_TARGETS_NOT_TO_CATCH:
            cls.append("unsafe")

        tds = "".join(f"<td>{str(r.get(c, ''))}</td>" for c in cols)
        trs.append(f"<tr class='{' '.join(cls)}'>{tds}</tr>")

    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003F consolidated shadow arbiter</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003F consolidated shadow arbiter</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

<section>
<h2>Lignes</h2>
<table>
<thead>
<tr>{''.join(f'<th>{c}</th>' for c in cols)}</tr>
</thead>
<tbody>
{''.join(trs)}
</tbody>
</table>
</section>

</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_001E")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run)

    input_csv = run_dir / "multi_object_arbiter_shadow_003E.csv"
    input_json = run_dir / "multi_object_arbiter_shadow_summary_003E.json"

    if not input_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {input_csv}")
    if not input_json.is_file():
        raise SystemExit(f"Fichier introuvable: {input_json}")

    summary_003e = json.loads(input_json.read_text(encoding="utf-8"))
    if summary_003e.get("status") != "OK":
        raise SystemExit(f"003E non OK: {summary_003e.get('status')}")
    if summary_003e.get("mode") != "metrics":
        raise SystemExit(f"003E pas en mode metrics: {summary_003e.get('mode')}")

    df = pd.read_csv(input_csv)

    required = [
        "review_id_003E",
        "target_class_003E",
        "contextual_hit_003B4_resolved",
        "multi_object_shadow_hit_003E",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Colonnes manquantes dans 003E CSV: " + ", ".join(missing))

    out = df.copy()

    out["review_id_003F"] = out["review_id_003E"].astype(str)
    out["target_class_003F"] = out["target_class_003E"].map(norm_target)

    out["contextual_hit_003B4_resolved"] = out["contextual_hit_003B4_resolved"].map(boolish)
    out["multi_object_shadow_hit_003E"] = out["multi_object_shadow_hit_003E"].map(boolish)

    out["reject_suggested_shadow_003F"] = (
        out["contextual_hit_003B4_resolved"] | out["multi_object_shadow_hit_003E"]
    )

    out["new_reject_suggested_vs_003B4_003F"] = (
        out["reject_suggested_shadow_003F"] & ~out["contextual_hit_003B4_resolved"]
    )

    out["arbiter_reason_003F"] = out.apply(reason_for_row, axis=1)

    # Important : 003F reste shadow. On ne modifie pas encore la décision finale.
    out["final_decision_policy_003F"] = "shadow_only_no_final_change"

    caught = out[out["reject_suggested_shadow_003F"]]["review_id_003F"].astype(str).tolist()
    caught_set = set(caught)

    new_ids = out[out["new_reject_suggested_vs_003B4_003F"]]["review_id_003F"].astype(str).tolist()
    unsafe_df = out[
        out["reject_suggested_shadow_003F"]
        & out["target_class_003F"].isin(SAFE_TARGETS_NOT_TO_CATCH)
    ]

    by_target = (
        out[out["reject_suggested_shadow_003F"]]
        .groupby("target_class_003F")["review_id_003F"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    expected_missing = sorted(EXPECTED_REJECT_SUGGESTED - caught_set)
    expected_extra = sorted(caught_set - EXPECTED_REJECT_SUGGESTED)
    new_missing = sorted(EXPECTED_003E_NEW - set(new_ids))
    new_extra = sorted(set(new_ids) - EXPECTED_003E_NEW)

    warnings = []
    if expected_missing:
        warnings.append("missing expected reject suggestions: " + ",".join(expected_missing))
    if expected_extra:
        warnings.append("unexpected reject suggestions: " + ",".join(expected_extra))
    if new_missing:
        warnings.append("missing expected new-vs-003B4 ids: " + ",".join(new_missing))
    if new_extra:
        warnings.append("unexpected new-vs-003B4 ids: " + ",".join(new_extra))
    if len(unsafe_df):
        warnings.append(
            "UNSAFE caught keep/human_partial: "
            + ",".join(unsafe_df["review_id_003F"].astype(str).tolist())
        )

    status = "OK" if not warnings else "WARN"

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "source_csv": str(input_csv),
        "source_json": str(input_json),
        "policy": "shadow_only_no_final_change",
        "rows": int(len(out)),
        "reject_suggested_total": int(len(caught)),
        "reject_suggested_ids": caught,
        "new_reject_suggested_vs_003B4_ids": new_ids,
        "by_target_caught": by_target,
        "unsafe_keep_or_human_partial_caught": int(len(unsafe_df)),
        "unsafe_keep_or_human_partial_ids": unsafe_df["review_id_003F"].astype(str).tolist(),
        "expected_missing": expected_missing,
        "expected_extra": expected_extra,
        "new_missing": new_missing,
        "new_extra": new_extra,
        "warnings": warnings,
    }

    out_csv = run_dir / "consolidated_shadow_arbiter_003F.csv"
    out_json = run_dir / "consolidated_shadow_arbiter_summary_003F.json"
    out_html = run_dir / "consolidated_shadow_arbiter_003F.html"

    out.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, out, summary)

    print(f"003F status={status}")
    print(f"reject_suggested={','.join(caught)}")
    print(f"new_vs_003B4={','.join(new_ids) or '-'}")
    print(f"unsafe_keep_or_human_partial={len(unsafe_df)}")
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
