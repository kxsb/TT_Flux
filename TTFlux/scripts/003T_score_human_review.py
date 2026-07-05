from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003T"

VALID_LABELS = {"reject", "keep", "partial", "unsure"}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def clean_label(x) -> str:
    s = str(x or "").strip().lower()
    return s if s in VALID_LABELS else ""


def pct(a: int, b: int) -> float:
    if b <= 0:
        return 0.0
    return round(100.0 * a / b, 2)


def write_html(path: Path, merged: pd.DataFrame, summary: dict) -> None:
    rows = []

    for _, r in merged.iterrows():
        label = clean_label(r.get("human_label_003S", ""))
        hit = boolish(r.get("003N_shadow_hit"))

        if label == "reject":
            cls = "good"
        elif label in {"keep", "partial"}:
            cls = "bad"
        elif label == "unsure":
            cls = "warn"
        else:
            cls = "miss"

        trs = []
        for c in [
            "review_id",
            "human_label_003S",
            "comment_003S",
            "003N_shadow_hit",
            "target_class_003G",
            "003N_player_motion_inside_value",
            "003N_old_table_distance_value",
            "003N_max_accel_value",
            "003N_failed_clauses",
            "mp4",
            "csv",
        ]:
            if c in r.index:
                trs.append(f"<tr><th>{esc(c)}</th><td>{esc(r.get(c, ''))}</td></tr>")

        rows.append(f"""
<div class="card {cls}">
<h2>{esc(r.get('review_id', ''))} · {esc(label or 'unlabeled')}</h2>
<table><tbody>{''.join(trs)}</tbody></table>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003T human score</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section,.card{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
.card.good{{border-color:rgba(116,217,159,.75);box-shadow:inset 4px 0 0 rgba(116,217,159,.85)}}
.card.warn{{border-color:rgba(255,200,80,.75);box-shadow:inset 4px 0 0 rgba(255,200,80,.85)}}
.card.bad{{border-color:rgba(255,80,80,.75);box-shadow:inset 4px 0 0 rgba(255,80,80,.85)}}
.card.miss{{border-color:#2b303b}}
h1,h2{{margin-top:0}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{width:280px;background:#20242e}}
</style>
</head>
<body>
<h1>TTFlux · 003T human score</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

{''.join(rows)}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_002A")
    ap.add_argument("--source", default="frozen_shadow_rule_run_003N.csv")
    ap.add_argument("--labels", default="human_review_003S_labels.csv")
    args = ap.parse_args()

    root = Path.cwd()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    source = Path(args.source)
    if not source.is_absolute():
        source = run_dir / source

    labels = Path(args.labels)
    if not labels.is_absolute():
        labels = run_dir / labels

    if not source.is_file():
        raise SystemExit(f"Source 003N introuvable: {source}")

    if not labels.is_file():
        raise SystemExit(f"Labels humains introuvables: {labels}")

    src = pd.read_csv(source)
    lab = pd.read_csv(labels)

    if "review_id" not in src.columns:
        raise SystemExit("review_id absent du CSV source.")

    if "review_id" not in lab.columns:
        raise SystemExit("review_id absent du CSV labels.")

    src["review_id"] = src["review_id"].astype(str)
    lab["review_id"] = lab["review_id"].astype(str)

    if "human_label_003S" not in lab.columns:
        raise SystemExit("human_label_003S absent du CSV labels.")

    lab["human_label_003S"] = lab["human_label_003S"].map(clean_label)

    merged = src.merge(
        lab[["review_id", "human_label_003S", "comment_003S", "updated_at"]],
        on="review_id",
        how="left",
    )

    merged["human_label_003S"] = merged["human_label_003S"].map(clean_label)

    hits = merged[merged["003N_shadow_hit"].map(boolish)].copy()
    annotated_hits = hits[hits["human_label_003S"] != ""].copy()

    label_counts = Counter(annotated_hits["human_label_003S"].tolist())

    reject = int(label_counts.get("reject", 0))
    keep = int(label_counts.get("keep", 0))
    partial = int(label_counts.get("partial", 0))
    unsure = int(label_counts.get("unsure", 0))

    dangerous_false_positive = keep + partial
    resolved = reject + keep + partial
    annotated_total = int(len(annotated_hits))
    hit_total = int(len(hits))

    if annotated_total < hit_total:
        status = "INCOMPLETE_ANNOTATION"
    elif dangerous_false_positive > 0:
        status = "RULE_TOO_AGGRESSIVE_ON_003R_PROXY"
    elif unsure > 0:
        status = "NEEDS_UNSURE_REVIEW"
    elif reject == hit_total:
        status = "ALL_HITS_CONFIRMED_REJECT_ON_003R_PROXY"
    else:
        status = "OK_REVIEWED"

    recommendation = (
        "Do not promote live. batch_002A used 003R proxy features, not historical 003D. "
        "Use this result to decide whether the proxy is too broad and which rows need deeper inspection."
    )

    if dangerous_false_positive > 0:
        recommendation = (
            "Do not promote. The shadow rule caught keep/partial rows on batch_002A. "
            "Next step: inspect false positives and compare with true 003C/003D feature pipeline, or harden the rule."
        )

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "policy": "human_score_only_no_live_filter_no_delete",
        "run_dir": str(run_dir),
        "source": str(source),
        "labels": str(labels),
        "rows_source": int(len(src)),
        "shadow_hit_total": hit_total,
        "annotated_hit_total": annotated_total,
        "missing_annotation_total": int(hit_total - annotated_total),
        "label_counts": dict(label_counts),
        "reject_true_positive_count": reject,
        "keep_false_positive_count": keep,
        "partial_false_positive_count": partial,
        "unsure_count": unsure,
        "dangerous_false_positive_count": dangerous_false_positive,
        "precision_reject_over_annotated_hits_pct": pct(reject, annotated_total),
        "precision_reject_over_resolved_hits_pct": pct(reject, resolved),
        "dangerous_false_positive_over_annotated_hits_pct": pct(dangerous_false_positive, annotated_total),
        "reject_ids": annotated_hits[annotated_hits["human_label_003S"] == "reject"]["review_id"].astype(str).tolist(),
        "keep_ids": annotated_hits[annotated_hits["human_label_003S"] == "keep"]["review_id"].astype(str).tolist(),
        "partial_ids": annotated_hits[annotated_hits["human_label_003S"] == "partial"]["review_id"].astype(str).tolist(),
        "unsure_ids": annotated_hits[annotated_hits["human_label_003S"] == "unsure"]["review_id"].astype(str).tolist(),
        "recommendation": recommendation,
        "warning": "batch_002A was scored through 003R proxy bridge, not the original historical 003D pipeline.",
    }

    out_csv = run_dir / "human_review_score_003T.csv"
    out_json = run_dir / "human_review_score_summary_003T.json"
    out_html = run_dir / "human_review_score_003T.html"

    merged.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, merged, summary)

    print("003T status=" + status)
    print("shadow_hit_total=", hit_total)
    print("annotated_hit_total=", annotated_total)
    print("label_counts=", dict(label_counts))
    print("reject_ids=" + (",".join(summary["reject_ids"]) or "-"))
    print("keep_ids=" + (",".join(summary["keep_ids"]) or "-"))
    print("partial_ids=" + (",".join(summary["partial_ids"]) or "-"))
    print("unsure_ids=" + (",".join(summary["unsure_ids"]) or "-"))
    print("precision_reject_over_annotated_hits_pct=", summary["precision_reject_over_annotated_hits_pct"])
    print("dangerous_false_positive_count=", dangerous_false_positive)
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
