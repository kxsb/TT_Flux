from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003W"

KEY_COLS = [
    "inside_player_motion_mask_ratio_003C",
    "candidate_on_body_risk_003C",
    "point_distance_to_table_px_median",
    "point_distance_to_table_px_p90",
    "track_inside_table_ratio",
    "inside_table_ratio",
    "max_accel",
    "v_max",
    "travel",
    "density",
    "score",
    "n_points",
    "micro_blob_score",
    "center_blob_touch_ratio",
]


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def clean_label(x) -> str:
    s = str(x or "").strip().lower()
    return s if s in {"reject", "keep", "partial", "unsure"} else ""


def num(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def summarize_col(df: pd.DataFrame, col: str) -> dict:
    vals = pd.to_numeric(df[col], errors="coerce")
    return {
        "col": col,
        "count": int(vals.notna().sum()),
        "nan": int(vals.isna().sum()),
        "min": None if vals.dropna().empty else float(vals.min()),
        "median": None if vals.dropna().empty else float(vals.median()),
        "max": None if vals.dropna().empty else float(vals.max()),
    }


def write_html(path: Path, df: pd.DataFrame, summary: dict, present_cols: list[str]) -> None:
    headers = ["review_id", "human_label_003S", "comment_003S"] + present_cols

    trs = []
    for _, r in df.iterrows():
        label = clean_label(r.get("human_label_003S", ""))

        if label == "reject":
            cls = "reject"
        elif label == "keep":
            cls = "keep"
        elif label == "partial":
            cls = "partial"
        elif label == "unsure":
            cls = "unsure"
        else:
            cls = "unlabeled"

        cells = []
        for c in headers:
            cells.append(f"<td>{esc(r.get(c, ''))}</td>")

        trs.append(f"<tr class='{cls}'>{''.join(cells)}</tr>")

    ths = "".join(f"<th>{esc(c)}</th>" for c in headers)

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003W true metrics audit</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
tr.reject td{{background:rgba(255,80,80,.12)}}
tr.keep td{{background:rgba(116,217,159,.08)}}
tr.partial td{{background:rgba(255,200,80,.08)}}
tr.unsure td{{background:rgba(157,180,255,.08)}}
.wrap{{overflow:auto;max-height:75vh;border:1px solid #2b303b;border-radius:10px}}
</style>
</head>
<body>
<h1>TTFlux · 003W true metrics audit</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

<section>
<h2>Table métriques</h2>
<div class="wrap">
<table>
<thead><tr>{ths}</tr></thead>
<tbody>{''.join(trs)}</tbody>
</table>
</div>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


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

    if not labels.is_file():
        raise SystemExit(f"Labels introuvables: {labels}")

    df = pd.read_csv(features)
    lab = pd.read_csv(labels)

    df["review_id"] = df["review_id"].astype(str)
    lab["review_id"] = lab["review_id"].astype(str)
    lab["human_label_003S"] = lab["human_label_003S"].map(clean_label)

    df = df.merge(
        lab[["review_id", "human_label_003S", "comment_003S", "updated_at"]],
        on="review_id",
        how="left",
    )

    df["human_label_003S"] = df["human_label_003S"].map(clean_label)

    present_cols = [c for c in KEY_COLS if c in df.columns]
    missing_cols = [c for c in KEY_COLS if c not in df.columns]

    numeric_summary = [summarize_col(df, c) for c in present_cols]

    reject_df = df[df["human_label_003S"] == "reject"].copy()
    danger_df = df[df["human_label_003S"].isin(["keep", "partial"])].copy()

    reject_rows = []
    for _, r in reject_df.iterrows():
        reject_rows.append({
            "review_id": str(r.get("review_id", "")),
            "comment": str(r.get("comment_003S", "")),
            "metrics": {c: r.get(c, None) for c in present_cols},
        })

    # Cherche colonnes numériques où reject est séparé du keep/partial.
    separations = []
    for c in present_cols:
        rv = pd.to_numeric(reject_df[c], errors="coerce").dropna()
        dv = pd.to_numeric(danger_df[c], errors="coerce").dropna()

        if rv.empty or dv.empty:
            continue

        r_min, r_max, r_med = float(rv.min()), float(rv.max()), float(rv.median())
        d_min, d_max, d_med = float(dv.min()), float(dv.max()), float(dv.median())

        sep = {
            "col": c,
            "reject_min": r_min,
            "reject_median": r_med,
            "reject_max": r_max,
            "danger_min": d_min,
            "danger_median": d_med,
            "danger_max": d_max,
            "possible_rule": "",
        }

        if r_min > d_max:
            sep["possible_rule"] = f"{c} > {round((r_min + d_max) / 2, 6)}"
            sep["gap"] = r_min - d_max
        elif r_max < d_min:
            sep["possible_rule"] = f"{c} < {round((r_max + d_min) / 2, 6)}"
            sep["gap"] = d_min - r_max
        else:
            sep["gap"] = 0.0

        separations.append(sep)

    separations = sorted(separations, key=lambda x: x["gap"], reverse=True)

    out_csv = run_dir / "true_metrics_audit_003W.csv"
    out_json = run_dir / "true_metrics_audit_summary_003W.json"
    out_html = run_dir / "true_metrics_audit_003W.html"

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "diagnostic_only_no_live_filter_no_delete",
        "run_dir": str(run_dir),
        "features": str(features),
        "labels": str(labels),
        "rows": int(len(df)),
        "label_counts": df["human_label_003S"].value_counts(dropna=False).to_dict(),
        "present_key_cols": present_cols,
        "missing_key_cols": missing_cols,
        "numeric_summary": numeric_summary,
        "reject_rows": reject_rows,
        "top_separations_reject_vs_keep_partial": separations[:20],
        "interpretation_hint": "If table/player columns are NaN or constant, the 003L rule cannot fire. Look at reject_rows for R0005/R0015.",
    }

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, df, summary, present_cols)

    print("003W status=OK")
    print("rows=", len(df))
    print("label_counts=", summary["label_counts"])
    print("present_key_cols=", ",".join(present_cols) or "-")
    print("missing_key_cols=", ",".join(missing_cols) or "-")
    print("reject_rows=", json.dumps(reject_rows, ensure_ascii=False))
    if separations:
        print("best_separation=", json.dumps(separations[0], ensure_ascii=False))
    else:
        print("best_separation=-")
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
