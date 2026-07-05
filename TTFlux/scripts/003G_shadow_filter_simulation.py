from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003G"

EXPECTED_REJECT_IDS = {"R0010", "R0009", "R0020", "R0014", "R0008", "R0021"}
SAFE_TARGETS_NOT_TO_REJECT = {"keep", "human_partial"}


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


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


def bucket(row: pd.Series) -> str:
    if bool(row["would_reject_shadow_003G"]):
        return "REJECT_SUGGESTED"

    t = row["target_class_003G"]
    if t == "keep":
        return "KEEP_CONFIRMED"
    if t == "human_partial":
        return "HUMAN_PARTIAL_REVIEW"
    if t == "auto_review":
        return "AUTO_REVIEW_REMAINING"
    if t == "auto_reject":
        return "AUTO_REJECT_NOT_CAUGHT"
    if t == "human_reject":
        return "HUMAN_REJECT_NOT_CAUGHT"
    return "OTHER_REMAINING"


def write_html(path: Path, df: pd.DataFrame, summary: dict) -> None:
    cols = [
        "review_id_003G",
        "target_class_003G",
        "would_reject_shadow_003G",
        "would_keep_shadow_003G",
        "review_bucket_003G",
        "arbiter_reason_003F",
        "final_policy_003G",
    ]

    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{background:#20242e;position:sticky;top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
.reject td{box-shadow:inset 3px 0 0 rgba(255,80,80,.9);background:rgba(255,80,80,.07)}
.keep td{box-shadow:inset 3px 0 0 rgba(116,217,159,.75);background:rgba(116,217,159,.045)}
.partial td{background:rgba(255,200,80,.08)}
.review td{background:rgba(150,170,255,.045)}
.unsafe td{background:rgba(255,0,0,.18)}
"""

    trs = []
    for _, r in df.iterrows():
        cls = []
        if bool(r["would_reject_shadow_003G"]):
            cls.append("reject")
        elif r["target_class_003G"] == "keep":
            cls.append("keep")
        elif r["target_class_003G"] == "human_partial":
            cls.append("partial")
        elif r["target_class_003G"] == "auto_review":
            cls.append("review")

        if bool(r["would_reject_shadow_003G"]) and r["target_class_003G"] in SAFE_TARGETS_NOT_TO_REJECT:
            cls.append("unsafe")

        tds = "".join(f"<td>{str(r.get(c, ''))}</td>" for c in cols)
        trs.append(f"<tr class='{' '.join(cls)}'>{tds}</tr>")

    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003G shadow filter simulation</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003G shadow filter simulation</h1>

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

    input_csv = run_dir / "consolidated_shadow_arbiter_003F.csv"
    input_json = run_dir / "consolidated_shadow_arbiter_summary_003F.json"

    if not input_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {input_csv}")
    if not input_json.is_file():
        raise SystemExit(f"Fichier introuvable: {input_json}")

    summary_003f = json.loads(input_json.read_text(encoding="utf-8"))
    if summary_003f.get("status") != "OK":
        raise SystemExit(f"003F non OK: {summary_003f.get('status')}")

    df = pd.read_csv(input_csv)

    required = [
        "review_id_003F",
        "target_class_003F",
        "reject_suggested_shadow_003F",
        "arbiter_reason_003F",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Colonnes manquantes dans 003F CSV: " + ", ".join(missing))

    out = df.copy()

    out["review_id_003G"] = out["review_id_003F"].astype(str)
    out["target_class_003G"] = out["target_class_003F"].map(norm_target)
    out["would_reject_shadow_003G"] = out["reject_suggested_shadow_003F"].map(boolish)
    out["would_keep_shadow_003G"] = ~out["would_reject_shadow_003G"]
    out["review_bucket_003G"] = out.apply(bucket, axis=1)

    # Important : simulation uniquement. Aucun fichier vidéo/overlay n'est supprimé.
    out["final_policy_003G"] = "simulation_only_no_file_delete_no_final_change"

    reject_df = out[out["would_reject_shadow_003G"]]
    keep_df = out[out["would_keep_shadow_003G"]]

    reject_ids = reject_df["review_id_003G"].astype(str).tolist()
    reject_set = set(reject_ids)

    unsafe_df = reject_df[reject_df["target_class_003G"].isin(SAFE_TARGETS_NOT_TO_REJECT)]

    by_rejected_target = (
        reject_df.groupby("target_class_003G")["review_id_003G"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    by_remaining_target = (
        keep_df.groupby("target_class_003G")["review_id_003G"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    by_bucket = (
        out.groupby("review_bucket_003G")["review_id_003G"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    expected_missing = sorted(EXPECTED_REJECT_IDS - reject_set)
    expected_extra = sorted(reject_set - EXPECTED_REJECT_IDS)

    warnings = []
    if expected_missing:
        warnings.append("missing expected reject ids: " + ",".join(expected_missing))
    if expected_extra:
        warnings.append("unexpected reject ids: " + ",".join(expected_extra))
    if len(unsafe_df):
        warnings.append(
            "UNSAFE would reject keep/human_partial: "
            + ",".join(unsafe_df["review_id_003G"].astype(str).tolist())
        )

    status = "OK" if not warnings else "WARN"

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "source_csv": str(input_csv),
        "source_json": str(input_json),
        "policy": "simulation_only_no_file_delete_no_final_change",
        "input_rows": int(len(out)),
        "would_reject_total": int(len(reject_df)),
        "would_reject_ids": reject_ids,
        "would_keep_total": int(len(keep_df)),
        "would_keep_ids": keep_df["review_id_003G"].astype(str).tolist(),
        "by_rejected_target": by_rejected_target,
        "by_remaining_target": by_remaining_target,
        "by_bucket": by_bucket,
        "unsafe_keep_or_human_partial_rejected": int(len(unsafe_df)),
        "unsafe_keep_or_human_partial_ids": unsafe_df["review_id_003G"].astype(str).tolist(),
        "expected_missing": expected_missing,
        "expected_extra": expected_extra,
        "warnings": warnings,
        "promotion_note": (
            "003G is a non-destructive simulation. "
            "Next step should visually inspect the 6 would-reject segments, especially R0014, "
            "before any live arbiter integration."
        ),
    }

    out_csv = run_dir / "shadow_filter_simulation_003G.csv"
    out_json = run_dir / "shadow_filter_simulation_summary_003G.json"
    out_html = run_dir / "shadow_filter_simulation_003G.html"

    out.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, out, summary)

    print(f"003G status={status}")
    print(f"would_reject={','.join(reject_ids)}")
    print(f"would_keep_total={len(keep_df)}")
    print(f"unsafe_keep_or_human_partial_rejected={len(unsafe_df)}")
    print("remaining_by_target=" + json.dumps(by_remaining_target, ensure_ascii=False))
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
