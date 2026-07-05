from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003Y5"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def clean_label(x) -> str:
    s = str(x or "").strip().lower()
    return s if s in {"reject", "keep", "partial", "unsure"} else ""


def find_col(df: pd.DataFrame, exact: list[str], contains_all: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}

    for e in exact:
        if e.lower() in low:
            return low[e.lower()]

    for c in df.columns:
        cl = c.lower()
        if all(s.lower() in cl for s in contains_all):
            return c

    return None


def prefixed_csv(path: Path, prefix: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "review_id" not in df.columns:
        return pd.DataFrame()

    df["review_id"] = df["review_id"].astype(str)

    rename = {}
    for c in df.columns:
        if c != "review_id":
            rename[c] = f"{prefix}__{c}"

    return df.rename(columns=rename)


def load_labels(path: Path, label_col: str, comment_col: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["review_id", "human_label", "human_comment"])

    df = pd.read_csv(path)

    if "review_id" not in df.columns:
        return pd.DataFrame(columns=["review_id", "human_label", "human_comment"])

    df["review_id"] = df["review_id"].astype(str)

    if label_col not in df.columns:
        df["human_label"] = ""
    else:
        df["human_label"] = df[label_col].map(clean_label)

    if comment_col not in df.columns:
        df["human_comment"] = ""
    else:
        df["human_comment"] = df[comment_col].fillna("").astype(str)

    return df[["review_id", "human_label", "human_comment"]]


def load_run(run_dir: Path, label_file: str, label_col: str, comment_col: str) -> pd.DataFrame:
    center = prefixed_csv(run_dir / "center_blob_features_001T2.csv", "center")
    micro = prefixed_csv(run_dir / "micro_blob_features_001T2.csv", "micro")
    manifest = prefixed_csv(run_dir / "review_manifest_001T2_trajectory.csv", "manifest")
    labels = load_labels(run_dir / label_file, label_col, comment_col)

    if center.empty:
        raise RuntimeError(f"center_blob_features_001T2.csv absent/invalide dans {run_dir}")

    df = center.copy()

    for sub in [micro, manifest]:
        if not sub.empty:
            df = df.merge(sub, on="review_id", how="left")

    df = df.merge(labels, on="review_id", how="left")
    df["human_label"] = df["human_label"].map(clean_label)
    df["human_comment"] = df["human_comment"].fillna("").astype(str)

    df["run_id"] = run_dir.name
    df["sample_id"] = df["run_id"] + "/" + df["review_id"].astype(str)

    center_col = find_col(
        df,
        ["center__center_blob_fill_med"],
        ["center", "fill", "med"],
    )
    micro_col = find_col(
        df,
        ["micro__micro_distance_med_001O"],
        ["micro", "distance", "med"],
    )

    if not center_col:
        raise RuntimeError(f"center fill med introuvable dans {run_dir}")

    df["center_fill_med_003Y5"] = pd.to_numeric(df[center_col], errors="coerce")

    if micro_col:
        df["micro_distance_med_003Y5"] = pd.to_numeric(df[micro_col], errors="coerce")
    else:
        df["micro_distance_med_003Y5"] = pd.NA

    df["center_col_used_003Y5"] = center_col
    df["micro_col_used_003Y5"] = micro_col or ""

    return df


def eval_rule(df: pd.DataFrame, hit: pd.Series, name: str, expr: str) -> dict:
    lab = df["human_label"].map(clean_label)
    labeled = lab.ne("")

    hit = hit.fillna(False).astype(bool)

    hit_labeled = hit & labeled
    reject_hit = hit & lab.eq("reject")
    keep_hit = hit & lab.eq("keep")
    partial_hit = hit & lab.eq("partial")
    unsure_hit = hit & lab.eq("unsure")
    unknown_hit = hit & lab.eq("")

    dangerous = keep_hit | partial_hit

    return {
        "name": name,
        "expression": expr,
        "rows": int(len(df)),
        "hit_total": int(hit.sum()),
        "hit_ids": df.loc[hit, "sample_id"].astype(str).tolist(),
        "labeled_hit_total": int(hit_labeled.sum()),
        "reject_hit_total": int(reject_hit.sum()),
        "reject_hit_ids": df.loc[reject_hit, "sample_id"].astype(str).tolist(),
        "keep_hit_total": int(keep_hit.sum()),
        "keep_hit_ids": df.loc[keep_hit, "sample_id"].astype(str).tolist(),
        "partial_hit_total": int(partial_hit.sum()),
        "partial_hit_ids": df.loc[partial_hit, "sample_id"].astype(str).tolist(),
        "unsure_hit_total": int(unsure_hit.sum()),
        "unsure_hit_ids": df.loc[unsure_hit, "sample_id"].astype(str).tolist(),
        "unknown_hit_total": int(unknown_hit.sum()),
        "unknown_hit_ids": df.loc[unknown_hit, "sample_id"].astype(str).tolist(),
        "dangerous_hit_total": int(dangerous.sum()),
        "dangerous_hit_ids": df.loc[dangerous, "sample_id"].astype(str).tolist(),
        "score": int(reject_hit.sum()) * 30
                 - int(dangerous.sum()) * 45
                 - int(unsure_hit.sum()) * 8
                 - int(unknown_hit.sum()) * 5,
    }


def thresholds(vals: pd.Series, max_n: int = 40) -> list[float]:
    v = pd.to_numeric(vals, errors="coerce").dropna().astype(float)

    if v.empty:
        return []

    uniq = sorted(set(v.tolist()))

    if len(uniq) <= 1:
        return uniq

    mids = [(a + b) / 2 for a, b in zip(uniq[:-1], uniq[1:])]

    # Ajoute quelques valeurs exactes utiles.
    candidates = sorted(set(uniq + mids))

    if len(candidates) > max_n:
        qs = [i / 20 for i in range(1, 20)]
        candidates = sorted(set(float(x) for x in v.quantile(qs).tolist()))

    return candidates


def sweep(df: pd.DataFrame) -> list[dict]:
    results = []

    c = df["center_fill_med_003Y5"]
    m = df["micro_distance_med_003Y5"]

    cths = thresholds(c)
    mths = thresholds(m)

    for th in cths:
        hit = c >= th
        results.append(eval_rule(
            df,
            hit,
            "center_ge",
            f"center_fill_med >= {round(float(th), 6)}",
        ) | {"center_threshold": float(th), "micro_threshold": None, "mode": "center"})

    for th in mths:
        hit = m >= th
        results.append(eval_rule(
            df,
            hit,
            "micro_ge",
            f"micro_distance_med >= {round(float(th), 6)}",
        ) | {"center_threshold": None, "micro_threshold": float(th), "mode": "micro"})

    for cth in cths:
        for mth in mths:
            hit_both = (c >= cth) & (m >= mth)
            results.append(eval_rule(
                df,
                hit_both,
                "center_and_micro_ge",
                f"center_fill_med >= {round(float(cth), 6)} AND micro_distance_med >= {round(float(mth), 6)}",
            ) | {"center_threshold": float(cth), "micro_threshold": float(mth), "mode": "both"})

            hit_either = (c >= cth) | (m >= mth)
            results.append(eval_rule(
                df,
                hit_either,
                "center_or_micro_ge",
                f"center_fill_med >= {round(float(cth), 6)} OR micro_distance_med >= {round(float(mth), 6)}",
            ) | {"center_threshold": float(cth), "micro_threshold": float(mth), "mode": "either"})

    # Priorité : zéro keep/partial, moins d'unknown, puis plus de reject.
    return sorted(
        results,
        key=lambda r: (
            r["dangerous_hit_total"] == 0,
            r["reject_hit_total"],
            -r["unknown_hit_total"],
            -r["unsure_hit_total"],
            r["score"],
            -r["hit_total"],
        ),
        reverse=True,
    )


def write_html(path: Path, df: pd.DataFrame, summary: dict, top_rules: list[dict]) -> None:
    rule_rows = []

    for r in top_rules[:40]:
        rule_rows.append(
            "<tr>"
            f"<td>{esc(r['name'])}</td>"
            f"<td>{esc(r['expression'])}</td>"
            f"<td>{esc(r['hit_total'])}</td>"
            f"<td>{esc(r['reject_hit_total'])}</td>"
            f"<td>{esc(r['dangerous_hit_total'])}</td>"
            f"<td>{esc(r['unsure_hit_total'])}</td>"
            f"<td>{esc(r['unknown_hit_total'])}</td>"
            f"<td>{esc(','.join(r['hit_ids']))}</td>"
            "</tr>"
        )

    data_rows = []
    for _, r in df.iterrows():
        lab = clean_label(r.get("human_label", ""))
        cls = lab or "unlabeled"
        data_rows.append(
            f"<tr class='{cls}'>"
            f"<td>{esc(r.get('sample_id'))}</td>"
            f"<td>{esc(lab)}</td>"
            f"<td>{esc(r.get('human_comment'))}</td>"
            f"<td>{esc(r.get('center_fill_med_003Y5'))}</td>"
            f"<td>{esc(r.get('micro_distance_med_003Y5'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003Y5 cross-batch score</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
tr.reject td{{background:rgba(255,80,80,.14)}}
tr.keep td{{background:rgba(116,217,159,.10)}}
tr.partial td{{background:rgba(255,200,80,.10)}}
tr.unsure td{{background:rgba(157,180,255,.10)}}
.wrap{{overflow:auto;max-height:70vh;border:1px solid #2b303b;border-radius:10px}}
</style>
</head>
<body>
<h1>TTFlux · 003Y5 cross-batch appearance rule score</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

<section>
<h2>Top règles sweep</h2>
<div class="wrap">
<table>
<thead><tr><th>name</th><th>expression</th><th>hits</th><th>reject</th><th>danger</th><th>unsure</th><th>unknown</th><th>hit ids</th></tr></thead>
<tbody>{''.join(rule_rows)}</tbody>
</table>
</div>
</section>

<section>
<h2>Données annotées</h2>
<div class="wrap">
<table>
<thead><tr><th>sample</th><th>label</th><th>comment</th><th>center_fill</th><th>micro_distance</th></tr></thead>
<tbody>{''.join(data_rows)}</tbody>
</table>
</div>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/batch_002A", "runs/batch_001E"])
    ap.add_argument("--out-dir", default="runs/cross_batch_003Y5")
    args = ap.parse_args()

    root = Path.cwd()

    frames = []

    for raw in args.runs:
        run_dir = Path(raw)
        if not run_dir.is_absolute():
            run_dir = root / run_dir

        if run_dir.name == "batch_002A":
            df = load_run(
                run_dir,
                "human_review_003S_labels.csv",
                "human_label_003S",
                "comment_003S",
            )
        elif run_dir.name == "batch_001E":
            df = load_run(
                run_dir,
                "appearance_review_003Y4_labels.csv",
                "human_label_003Y4",
                "comment_003Y4",
            )
        else:
            # Convention par défaut pour prochains batchs.
            df = load_run(
                run_dir,
                "appearance_review_003Y4_labels.csv",
                "human_label_003Y4",
                "comment_003Y4",
            )

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    original = eval_rule(
        data,
        data["center_fill_med_003Y5"] >= 0.82379,
        "003Y_original_center_ge_082379",
        "center_fill_med >= 0.82379",
    )

    top_rules = sweep(data)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    label_counts = data["human_label"].map(clean_label).value_counts(dropna=False).to_dict()

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "cross_batch_diagnostic_only_no_live_filter_no_delete",
        "runs": args.runs,
        "rows_total": int(len(data)),
        "labeled_rows": int(data["human_label"].map(clean_label).ne("").sum()),
        "label_counts": label_counts,
        "original_003Y_rule": original,
        "top_rules": top_rules[:20],
        "interpretation": "For live rejection, require dangerous_hit_total=0 and unknown_hit_total=0 on reviewed hits before freeze.",
    }

    out_csv = out_dir / "cross_batch_label_features_003Y5.csv"
    out_json = out_dir / "cross_batch_score_summary_003Y5.json"
    out_html = out_dir / "cross_batch_score_003Y5.html"

    data.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, data, summary, top_rules)

    print("003Y5 status=OK")
    print("rows_total=", len(data))
    print("labeled_rows=", summary["labeled_rows"])
    print("label_counts=", label_counts)
    print("original_rule=", json.dumps(original, ensure_ascii=False))

    if top_rules:
        print("best_rule=", json.dumps(top_rules[0], ensure_ascii=False))
    else:
        print("best_rule=-")

    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
