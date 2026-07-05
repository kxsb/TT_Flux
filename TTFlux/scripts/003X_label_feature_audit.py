from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003X"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def clean_label(x) -> str:
    s = str(x or "").strip().lower()
    return s if s in {"reject", "keep", "partial", "unsure"} else ""


def read_optional_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "review_id" not in df.columns:
        print(f"[003X] WARN {name}: review_id absent")
        return pd.DataFrame()

    df["review_id"] = df["review_id"].astype(str)

    # Préfixe les colonnes ambiguës, sauf review_id.
    rename = {}
    for c in df.columns:
        if c == "review_id":
            continue
        if c in {"target_class_003G", "human_label_003S", "comment_003S"}:
            continue
        rename[c] = f"{name}__{c}"

    return df.rename(columns=rename)


def numeric_cols(df: pd.DataFrame) -> list[str]:
    out = []

    skip = {"review_id"}

    for c in df.columns:
        if c in skip:
            continue

        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= 2:
            out.append(c)

    return out


def feature_summary(df: pd.DataFrame, col: str) -> dict:
    vals = pd.to_numeric(df[col], errors="coerce")

    def stats(mask):
        v = vals[mask].dropna()
        if v.empty:
            return {
                "count": 0,
                "min": None,
                "median": None,
                "max": None,
            }
        return {
            "count": int(len(v)),
            "min": float(v.min()),
            "median": float(v.median()),
            "max": float(v.max()),
        }

    lab = df["human_label_003S"].map(clean_label)

    return {
        "col": col,
        "reject": stats(lab.eq("reject")),
        "keep": stats(lab.eq("keep")),
        "partial": stats(lab.eq("partial")),
        "keep_or_partial": stats(lab.isin(["keep", "partial"])),
        "unsure": stats(lab.eq("unsure")),
        "all_labeled": stats(lab.ne("")),
        "nan_total": int(vals.isna().sum()),
    }


def separations(df: pd.DataFrame) -> list[dict]:
    lab = df["human_label_003S"].map(clean_label)
    reject_mask = lab.eq("reject")
    danger_mask = lab.isin(["keep", "partial"])

    rows = []

    for c in numeric_cols(df):
        vals = pd.to_numeric(df[c], errors="coerce")

        rv = vals[reject_mask].dropna()
        dv = vals[danger_mask].dropna()

        if rv.empty or dv.empty:
            continue

        r_min = float(rv.min())
        r_med = float(rv.median())
        r_max = float(rv.max())

        d_min = float(dv.min())
        d_med = float(dv.median())
        d_max = float(dv.max())

        possible = ""
        gap = 0.0

        if r_min > d_max:
            thr = round((r_min + d_max) / 2, 6)
            possible = f"{c} > {thr}"
            gap = r_min - d_max
        elif r_max < d_min:
            thr = round((r_max + d_min) / 2, 6)
            possible = f"{c} < {thr}"
            gap = d_min - r_max
        else:
            # séparation médiane seulement
            gap = abs(r_med - d_med)

        rows.append({
            "col": c,
            "reject_min": r_min,
            "reject_median": r_med,
            "reject_max": r_max,
            "keep_partial_min": d_min,
            "keep_partial_median": d_med,
            "keep_partial_max": d_max,
            "possible_clean_rule": possible,
            "gap": gap,
        })

    return sorted(rows, key=lambda x: x["gap"], reverse=True)


def sweep_single_features(df: pd.DataFrame) -> list[dict]:
    lab = df["human_label_003S"].map(clean_label)
    labeled = df[lab.ne("")].copy()
    lab = labeled["human_label_003S"].map(clean_label)

    if labeled.empty:
        return []

    results = []

    for c in numeric_cols(labeled):
        vals = pd.to_numeric(labeled[c], errors="coerce")

        uniq = sorted(set(float(x) for x in vals.dropna().tolist()))
        if len(uniq) < 2:
            continue

        thresholds = []
        for a, b in zip(uniq[:-1], uniq[1:]):
            thresholds.append((a + b) / 2)

        # limite simple pour éviter trop de bruit
        if len(thresholds) > 40:
            qs = vals.dropna().quantile([i / 20 for i in range(1, 20)]).tolist()
            thresholds = sorted(set(float(x) for x in qs))

        for th in thresholds:
            for op in ["<=", ">="]:
                if op == "<=":
                    hit = vals <= th
                else:
                    hit = vals >= th

                reject_hit = int((hit & lab.eq("reject")).sum())
                keep_hit = int((hit & lab.eq("keep")).sum())
                partial_hit = int((hit & lab.eq("partial")).sum())
                unsure_hit = int((hit & lab.eq("unsure")).sum())
                total_hit = int(hit.sum())

                false_danger = keep_hit + partial_hit
                score = reject_hit * 20 - false_danger * 25 - unsure_hit * 4 - max(0, total_hit - reject_hit) * 2

                if reject_hit == 0:
                    continue

                results.append({
                    "feature": c,
                    "op": op,
                    "threshold": round(float(th), 6),
                    "total_hit": total_hit,
                    "reject_hit": reject_hit,
                    "keep_hit": keep_hit,
                    "partial_hit": partial_hit,
                    "unsure_hit": unsure_hit,
                    "false_danger": false_danger,
                    "score": score,
                    "hit_ids": labeled.loc[hit, "review_id"].astype(str).tolist(),
                    "reject_hit_ids": labeled.loc[hit & lab.eq("reject"), "review_id"].astype(str).tolist(),
                    "false_danger_ids": labeled.loc[hit & lab.isin(["keep", "partial"]), "review_id"].astype(str).tolist(),
                })

    return sorted(
        results,
        key=lambda x: (x["score"], x["reject_hit"], -x["false_danger"], -x["total_hit"]),
        reverse=True,
    )


def write_html(path: Path, df: pd.DataFrame, summary: dict, sep: list[dict], sweep: list[dict]) -> None:
    cols = [
        "review_id",
        "human_label_003S",
        "comment_003S",
    ]

    preferred = [
        "micro__micro_guess",
        "micro__micro_blob_guess",
        "micro__micro_score",
        "micro__micro_blob_score",
        "micro__score",
        "micro__keep_score",
        "micro__false_score",
        "center__center_blob_guess",
        "center__center_guess",
        "center__touch_ratio",
        "center__center_blob_touch_ratio",
        "center__area_median",
        "center__area",
        "appearance__appearance_ok",
        "appearance__source",
        "true003E__inside_player_motion_mask_ratio_003C",
        "true003E__candidate_on_body_risk_003C",
        "true003E__point_distance_to_table_px_median",
        "true003E__max_accel",
        "true003E__travel",
        "true003E__density",
        "true003E__n_points",
    ]

    for c in preferred:
        if c in df.columns and c not in cols:
            cols.append(c)

    # ajoute quelques colonnes textuelles contenant guess/risk/score si elles existent
    for c in df.columns:
        cl = c.lower()
        if c not in cols and any(k in cl for k in ["guess", "risk", "score", "touch", "area"]):
            cols.append(c)

    cols = [c for c in cols if c in df.columns]

    trs = []
    for _, r in df.iterrows():
        lab = clean_label(r.get("human_label_003S", ""))
        cls = lab or "unlabeled"
        cells = "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in cols)
        trs.append(f"<tr class='{cls}'>{cells}</tr>")

    ths = "".join(f"<th>{esc(c)}</th>" for c in cols)

    sep_rows = []
    for r in sep[:30]:
        sep_rows.append(
            "<tr>"
            f"<td>{esc(r['col'])}</td>"
            f"<td>{esc(r['gap'])}</td>"
            f"<td>{esc(r['possible_clean_rule'])}</td>"
            f"<td>{esc(r['reject_median'])}</td>"
            f"<td>{esc(r['keep_partial_median'])}</td>"
            "</tr>"
        )

    sweep_rows = []
    for r in sweep[:30]:
        sweep_rows.append(
            "<tr>"
            f"<td>{esc(r['feature'])}</td>"
            f"<td>{esc(r['op'])}</td>"
            f"<td>{esc(r['threshold'])}</td>"
            f"<td>{esc(r['score'])}</td>"
            f"<td>{esc(r['reject_hit'])}</td>"
            f"<td>{esc(r['false_danger'])}</td>"
            f"<td>{esc(','.join(r['hit_ids']))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003X labels × appearance audit</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:75vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
tr.reject td{{background:rgba(255,80,80,.14)}}
tr.keep td{{background:rgba(116,217,159,.10)}}
tr.partial td{{background:rgba(255,200,80,.09)}}
tr.unsure td{{background:rgba(157,180,255,.10)}}
</style>
</head>
<body>
<h1>TTFlux · 003X labels × appearance/center/micro audit</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

<section>
<h2>Top séparations reject vs keep/partial</h2>
<table>
<thead><tr><th>feature</th><th>gap</th><th>possible clean rule</th><th>reject median</th><th>keep/partial median</th></tr></thead>
<tbody>{''.join(sep_rows)}</tbody>
</table>
</section>

<section>
<h2>Top single-feature rules</h2>
<table>
<thead><tr><th>feature</th><th>op</th><th>threshold</th><th>score</th><th>reject_hit</th><th>false_danger</th><th>hit_ids</th></tr></thead>
<tbody>{''.join(sweep_rows)}</tbody>
</table>
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
    ap.add_argument("--run", default="runs/batch_002A")
    args = ap.parse_args()

    root = Path.cwd()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    labels = run_dir / "human_review_003S_labels.csv"
    appearance = run_dir / "appearance_features_001T2.csv"
    center = run_dir / "center_blob_features_001T2.csv"
    micro = run_dir / "micro_blob_features_001T2.csv"
    true003e = run_dir / "multi_object_arbiter_shadow_003E.csv"

    if not labels.is_file():
        raise SystemExit(f"Labels introuvables: {labels}")

    base = pd.read_csv(labels)
    base["review_id"] = base["review_id"].astype(str)
    base["human_label_003S"] = base["human_label_003S"].map(clean_label)

    sources = {
        "appearance": appearance,
        "center": center,
        "micro": micro,
        "true003E": true003e,
    }

    loaded = {}

    df = base.copy()

    for name, path in sources.items():
        sub = read_optional_csv(path, name)
        loaded[name] = {
            "path": str(path),
            "loaded": not sub.empty,
            "rows": int(len(sub)) if not sub.empty else 0,
            "cols": list(sub.columns) if not sub.empty else [],
        }

        if not sub.empty:
            df = df.merge(sub, on="review_id", how="left")

    sep = separations(df)
    sweep = sweep_single_features(df)

    summaries = []
    for c in numeric_cols(df):
        summaries.append(feature_summary(df, c))

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "diagnostic_only_no_live_filter_no_delete",
        "run_dir": str(run_dir),
        "rows": int(len(df)),
        "label_counts": df["human_label_003S"].value_counts(dropna=False).to_dict(),
        "sources": loaded,
        "top_separations": sep[:20],
        "top_single_feature_rules": sweep[:20],
        "feature_summaries": summaries,
        "interpretation_hint": "Look for rules that catch R0005/R0015 while avoiding keep/partial. Beware: only 11 annotated rows, so this is exploratory.",
    }

    out_csv = run_dir / "label_feature_audit_003X.csv"
    out_json = run_dir / "label_feature_audit_summary_003X.json"
    out_html = run_dir / "label_feature_audit_003X.html"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, df, summary, sep, sweep)

    print("003X status=OK")
    print("rows=", len(df))
    print("label_counts=", summary["label_counts"])

    for name, info in loaded.items():
        print(f"source_{name}_loaded=", info["loaded"], "rows=", info["rows"])

    if sep:
        print("best_separation=", json.dumps(sep[0], ensure_ascii=False))
    else:
        print("best_separation=-")

    if sweep:
        print("best_single_feature_rule=", json.dumps(sweep[0], ensure_ascii=False))
    else:
        print("best_single_feature_rule=-")

    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
