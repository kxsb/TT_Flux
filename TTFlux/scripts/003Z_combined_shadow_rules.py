from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003Z"

RULES = {
    "003L_context": {
        "type": "context",
        "expression": "inside_player_motion_mask_ratio_003C >= 0.60 AND point_distance_to_table_px_median <= 210.0 AND max_accel >= 20.0",
        "player_min": 0.60,
        "table_max": 210.0,
        "accel_min": 20.0,
    },
    "003Y_strict_appearance": {
        "type": "appearance",
        "expression": "center_fill_med >= 0.75924 AND micro_distance_med >= 506.0725",
        "center_min": 0.75924,
        "micro_min": 506.0725,
    },
    "003Y_review_appearance": {
        "type": "appearance",
        "expression": "center_fill_med >= 0.78588 AND micro_distance_med >= 9.0332",
        "center_min": 0.78588,
        "micro_min": 9.0332,
    },
}


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


def find_col(df: pd.DataFrame, exact: list[str], contains: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}

    for e in exact:
        if e.lower() in low:
            return low[e.lower()]

    for c in df.columns:
        cl = c.lower()
        if all(s.lower() in cl for s in contains):
            return c

    return None


def load_labels(run_dir: Path) -> pd.DataFrame:
    candidates = [
        ("human_review_003S_labels.csv", "human_label_003S", "comment_003S"),
        ("appearance_review_003Y4_labels.csv", "human_label_003Y4", "comment_003Y4"),
    ]

    out = []

    for fname, label_col, comment_col in candidates:
        p = run_dir / fname
        if not p.is_file():
            continue

        df = pd.read_csv(p)
        if "review_id" not in df.columns:
            continue

        df["review_id"] = df["review_id"].astype(str)
        df["human_label"] = df[label_col].map(clean_label) if label_col in df.columns else ""
        df["human_comment"] = df[comment_col].fillna("").astype(str) if comment_col in df.columns else ""
        out.append(df[["review_id", "human_label", "human_comment"]])

    if not out:
        return pd.DataFrame(columns=["review_id", "human_label", "human_comment"])

    lab = pd.concat(out, ignore_index=True)
    lab = lab.drop_duplicates("review_id", keep="last")
    return lab


def load_run(run_dir: Path) -> pd.DataFrame:
    center_path = run_dir / "center_blob_features_001T2.csv"
    micro_path = run_dir / "micro_blob_features_001T2.csv"
    context_path = run_dir / "multi_object_arbiter_shadow_003E.csv"

    if not center_path.is_file():
        raise RuntimeError(f"Center features absent: {center_path}")

    center = pd.read_csv(center_path)
    center["review_id"] = center["review_id"].astype(str)

    df = center.copy()

    if micro_path.is_file():
        micro = pd.read_csv(micro_path)
        micro["review_id"] = micro["review_id"].astype(str)
        micro = micro.rename(columns={c: f"micro__{c}" for c in micro.columns if c != "review_id"})
        df = df.merge(micro, on="review_id", how="left")

    if context_path.is_file():
        ctx = pd.read_csv(context_path)
        ctx["review_id"] = ctx["review_id"].astype(str)
        ctx = ctx.rename(columns={c: f"context__{c}" for c in ctx.columns if c != "review_id"})
        df = df.merge(ctx, on="review_id", how="left")

    labels = load_labels(run_dir)
    if not labels.empty:
        df = df.merge(labels, on="review_id", how="left", suffixes=("", "_labels"))

    # Robustesse : certains CSV peuvent d?j? contenir human_label/human_comment,
    # ou pandas peut les suffixer apr?s merge.
    if "human_label" not in df.columns:
        label_candidates = [c for c in df.columns if c.lower().startswith("human_label")]
        if label_candidates:
            df["human_label"] = df[label_candidates[-1]]
        else:
            df["human_label"] = ""

    if "human_comment" not in df.columns:
        comment_candidates = [c for c in df.columns if c.lower().startswith("human_comment")]
        if comment_candidates:
            df["human_comment"] = df[comment_candidates[-1]]
        else:
            df["human_comment"] = ""

    df["human_label"] = df["human_label"].map(clean_label)
    df["human_comment"] = df["human_comment"].fillna("").astype(str)

    center_col = find_col(df, ["center_blob_fill_med"], ["center", "fill", "med"])
    micro_col = find_col(df, ["micro__micro_distance_med_001O"], ["micro", "distance", "med"])

    if not center_col:
        raise RuntimeError(f"center_blob_fill_med absent dans {run_dir}")

    df["center_fill_med_003Z"] = pd.to_numeric(df[center_col], errors="coerce")
    df["micro_distance_med_003Z"] = pd.to_numeric(df[micro_col], errors="coerce") if micro_col else pd.NA

    # Context columns may already be prefixed.
    def col(name):
        if name in df.columns:
            return name
        pref = f"context__{name}"
        if pref in df.columns:
            return pref
        return None

    player_col = col("inside_player_motion_mask_ratio_003C")
    table_col = col("point_distance_to_table_px_median")
    accel_col = col("max_accel")

    df["context_player_003Z"] = pd.to_numeric(df[player_col], errors="coerce") if player_col else pd.NA
    df["context_table_dist_003Z"] = pd.to_numeric(df[table_col], errors="coerce") if table_col else pd.NA
    df["context_accel_003Z"] = pd.to_numeric(df[accel_col], errors="coerce") if accel_col else pd.NA

    df["run_id"] = run_dir.name
    df["sample_id"] = df["run_id"] + "/" + df["review_id"].astype(str)

    return df


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["hit_003L_context"] = (
        (out["context_player_003Z"] >= 0.60)
        & (out["context_table_dist_003Z"] <= 210.0)
        & (out["context_accel_003Z"] >= 20.0)
    )

    out["hit_003Y_strict"] = (
        (out["center_fill_med_003Z"] >= 0.75924)
        & (out["micro_distance_med_003Z"] >= 506.0725)
    )

    out["hit_003Y_review"] = (
        (out["center_fill_med_003Z"] >= 0.78588)
        & (out["micro_distance_med_003Z"] >= 9.0332)
    )

    out["combined_strict_reject_shadow"] = out["hit_003L_context"].fillna(False) | out["hit_003Y_strict"].fillna(False)
    out["combined_review_shadow"] = out["hit_003Y_review"].fillna(False) & ~out["combined_strict_reject_shadow"].fillna(False)

    return out


def summarize_hits(df: pd.DataFrame, col: str) -> dict:
    hit = df[col].fillna(False).astype(bool)
    lab = df["human_label"].map(clean_label)

    dangerous = hit & lab.isin(["keep", "partial"])

    return {
        "hit_total": int(hit.sum()),
        "hit_ids": df.loc[hit, "sample_id"].astype(str).tolist(),
        "label_counts": df.loc[hit, "human_label"].map(clean_label).value_counts(dropna=False).to_dict(),
        "reject_ids": df.loc[hit & lab.eq("reject"), "sample_id"].astype(str).tolist(),
        "keep_ids": df.loc[hit & lab.eq("keep"), "sample_id"].astype(str).tolist(),
        "partial_ids": df.loc[hit & lab.eq("partial"), "sample_id"].astype(str).tolist(),
        "unsure_ids": df.loc[hit & lab.eq("unsure"), "sample_id"].astype(str).tolist(),
        "unknown_ids": df.loc[hit & lab.eq(""), "sample_id"].astype(str).tolist(),
        "dangerous_hit_total": int(dangerous.sum()),
        "dangerous_hit_ids": df.loc[dangerous, "sample_id"].astype(str).tolist(),
    }


def write_html(path: Path, df: pd.DataFrame, summary: dict) -> None:
    cols = [
        "sample_id",
        "human_label",
        "human_comment",
        "hit_003L_context",
        "hit_003Y_strict",
        "hit_003Y_review",
        "combined_strict_reject_shadow",
        "combined_review_shadow",
        "center_fill_med_003Z",
        "micro_distance_med_003Z",
        "context_player_003Z",
        "context_table_dist_003Z",
        "context_accel_003Z",
    ]

    trs = []
    for _, r in df.iterrows():
        lab = clean_label(r.get("human_label", ""))
        cls = "hit" if bool(r.get("combined_strict_reject_shadow", False)) else "review" if bool(r.get("combined_review_shadow", False)) else lab or "none"

        cells = "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in cols)
        trs.append(f"<tr class='{cls}'>{cells}</tr>")

    ths = "".join(f"<th>{esc(c)}</th>" for c in cols)

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003Z combined shadow rules</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:75vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
tr.hit td{{background:rgba(116,217,159,.13)}}
tr.review td{{background:rgba(255,200,80,.10)}}
tr.reject td{{background:rgba(255,80,80,.10)}}
</style>
</head>
<body>
<h1>TTFlux · 003Z combined shadow rules</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
<section>
<h2>Rows</h2>
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
    ap.add_argument("--runs", nargs="+", default=["runs/batch_001E", "runs/batch_002A"])
    ap.add_argument("--out-dir", default="runs/combined_shadow_003Z")
    args = ap.parse_args()

    root = Path.cwd()

    frames = []
    for raw in args.runs:
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        frames.append(load_run(p))

    df = pd.concat(frames, ignore_index=True)
    df = apply_rules(df)

    # 003Z label repair:
    # 003Y5 is the current trusted cross-batch label table.
    # It contains sample_id, human_label, human_comment for batch_001E + batch_002A.
    trusted_labels_path = root / "runs" / "cross_batch_003Y5" / "cross_batch_label_features_003Y5.csv"
    if trusted_labels_path.is_file():
        trusted = pd.read_csv(trusted_labels_path)
        if {"sample_id", "human_label"}.issubset(set(trusted.columns)):
            keep_cols = ["sample_id", "human_label"]
            if "human_comment" in trusted.columns:
                keep_cols.append("human_comment")

            trusted = trusted[keep_cols].copy()
            trusted["sample_id"] = trusted["sample_id"].astype(str)
            trusted["trusted_human_label_003Z"] = trusted["human_label"].map(clean_label)

            if "human_comment" in trusted.columns:
                trusted["trusted_human_comment_003Z"] = trusted["human_comment"].fillna("").astype(str)
            else:
                trusted["trusted_human_comment_003Z"] = ""

            trusted = trusted[["sample_id", "trusted_human_label_003Z", "trusted_human_comment_003Z"]]
            df = df.merge(trusted, on="sample_id", how="left")

            df["human_label"] = df["trusted_human_label_003Z"].fillna("").map(clean_label)
            df["human_comment"] = df["trusted_human_comment_003Z"].fillna("").astype(str)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "combined_shadow_report_only_no_live_filter_no_delete",
        "runs": args.runs,
        "rows": int(len(df)),
        "label_counts": df["human_label"].map(clean_label).value_counts(dropna=False).to_dict(),
        "rules": RULES,
        "hit_003L_context": summarize_hits(df, "hit_003L_context"),
        "hit_003Y_strict": summarize_hits(df, "hit_003Y_strict"),
        "hit_003Y_review": summarize_hits(df, "hit_003Y_review"),
        "combined_strict_reject_shadow": summarize_hits(df, "combined_strict_reject_shadow"),
        "combined_review_shadow": summarize_hits(df, "combined_review_shadow"),
        "decision": "Do not live-filter. Strict hits are good candidates for future reject shadow; review hits need human review.",
    }

    out_csv = out_dir / "combined_shadow_rules_003Z.csv"
    out_json = out_dir / "combined_shadow_rules_summary_003Z.json"
    out_html = out_dir / "combined_shadow_rules_003Z.html"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, df, summary)

    print("003Z status=OK")
    print("rows=", len(df))
    print("label_counts=", summary["label_counts"])
    print("combined_strict=", json.dumps(summary["combined_strict_reject_shadow"], ensure_ascii=False))
    print("combined_review=", json.dumps(summary["combined_review_shadow"], ensure_ascii=False))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
