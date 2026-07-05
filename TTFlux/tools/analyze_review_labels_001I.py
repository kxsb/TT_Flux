# PATCH 001I - Analyze human labels against TTFlux trajectory scores
#
# Inputs:
# - review_manifest_001G_trajectory.csv
# - review_labels_001H3_fr_export.csv
#
# Outputs:
# - review_analysis_001I.csv
# - review_analysis_001I.html
#
# Usage:
#   python tools\analyze_review_labels_001I.py ^
#     --manifest runs\batch_001E\review_manifest_001G_trajectory.csv ^
#     --labels runs\batch_001E\review_labels_001H3_fr_export.csv ^
#     --out-csv runs\batch_001E\review_analysis_001I.csv ^
#     --out-html runs\batch_001E\review_analysis_001I.html

from __future__ import annotations

import argparse
import csv
import html
import statistics
from pathlib import Path
from typing import Any


POSITIVE_LABELS = {"gold_keep", "ok_ball"}
PARTIAL_LABELS = {"partial_ball"}
NEGATIVE_LABELS = {"false_track"}
UNCLEAR_LABELS = {"unclear", ""}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in reader]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "review_id",
        "triage_rank",
        "clip_id",
        "segment_idx",
        "segment_name",
        "risk_score_001G",
        "smoothness_score_001G",
        "review_guess_001G",
        "manual_label",
        "manual_label_fr",
        "manual_bucket",
        "manual_notes",
        "density",
        "traj_density",
        "max_frame_gap",
        "speed_median",
        "speed_p95",
        "speed_max",
        "accel_p95",
        "hard_turn_count",
        "risk_flags_001G",
        "mp4",
        "csv",
    ]

    seen = set()
    columns = []

    for col in preferred:
        if col not in seen:
            seen.add(col)
            columns.append(col)

    for row in rows:
        for col in row.keys():
            if col not in seen:
                seen.add(col)
                columns.append(col)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def bucket_label(label: str) -> str:
    label = label.strip()

    if label in POSITIVE_LABELS:
        return "positive"
    if label in PARTIAL_LABELS:
        return "partial"
    if label in NEGATIVE_LABELS:
        return "negative"
    if label in UNCLEAR_LABELS:
        return "unclear"

    return "unknown"


def median_or_empty(values: list[float]) -> str:
    if not values:
        return ""
    return f"{statistics.median(values):.2f}".rstrip("0").rstrip(".")


def mean_or_empty(values: list[float]) -> str:
    if not values:
        return ""
    return f"{(sum(values) / len(values)):.2f}".rstrip("0").rstrip(".")


def analyze_thresholds(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    usable = [
        r for r in rows
        if r.get("manual_bucket") in {"positive", "partial", "negative"}
    ]

    thresholds = list(range(0, 101, 5))
    out = []

    for threshold in thresholds:
        predicted_bad = []
        actual_bad = []
        actual_good = []

        for row in usable:
            risk = safe_float(row.get("risk_score_001G"))
            bucket = row.get("manual_bucket")

            is_pred_bad = risk >= threshold
            is_actual_bad = bucket == "negative"
            is_actual_good = bucket in {"positive", "partial"}

            predicted_bad.append(is_pred_bad)
            actual_bad.append(is_actual_bad)
            actual_good.append(is_actual_good)

        tp = sum(1 for p, a in zip(predicted_bad, actual_bad) if p and a)
        fp = sum(1 for p, g in zip(predicted_bad, actual_good) if p and g)
        fn = sum(1 for p, a in zip(predicted_bad, actual_bad) if (not p) and a)
        tn = sum(1 for p, g in zip(predicted_bad, actual_good) if (not p) and g)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

        out.append(
            {
                "threshold": str(threshold),
                "tp_false_tracks_caught": str(tp),
                "fp_good_tracks_flagged": str(fp),
                "fn_false_tracks_missed": str(fn),
                "tn_good_tracks_kept": str(tn),
                "precision": f"{precision:.3f}",
                "recall": f"{recall:.3f}",
                "f1": f"{f1:.3f}",
            }
        )

    return out


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_bucket: dict[str, list[dict[str, str]]] = {}
    by_label: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        by_bucket.setdefault(row.get("manual_bucket", "unknown"), []).append(row)
        by_label.setdefault(row.get("manual_label", "unknown"), []).append(row)

    summary = {
        "total": len(rows),
        "by_bucket": {k: len(v) for k, v in sorted(by_bucket.items())},
        "by_label": {k: len(v) for k, v in sorted(by_label.items())},
        "risk_by_bucket": {},
    }

    for bucket, bucket_rows in sorted(by_bucket.items()):
        risks = [safe_float(r.get("risk_score_001G")) for r in bucket_rows]
        gaps = [safe_float(r.get("max_frame_gap")) for r in bucket_rows]
        density = [safe_float(r.get("traj_density")) for r in bucket_rows]

        summary["risk_by_bucket"][bucket] = {
            "count": len(bucket_rows),
            "risk_mean": mean_or_empty(risks),
            "risk_median": median_or_empty(risks),
            "gap_median": median_or_empty(gaps),
            "density_median": median_or_empty(density),
        }

    return summary


def html_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)

    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
        body_rows.append(f"<tr>{cells}</tr>")

    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_html(path: Path, rows: list[dict[str, str]], threshold_rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r.get("manual_bucket", ""),
            -safe_float(r.get("risk_score_001G")),
            r.get("clip_id", ""),
        ),
    )

    bucket_lines = []
    for bucket, count in summary["by_bucket"].items():
        stats = summary["risk_by_bucket"].get(bucket, {})
        bucket_lines.append(
            f"<li><b>{html.escape(bucket)}</b> : {count} "
            f"· risque moyen {html.escape(stats.get('risk_mean', ''))} "
            f"· médiane risque {html.escape(stats.get('risk_median', ''))} "
            f"· médiane gap {html.escape(stats.get('gap_median', ''))} "
            f"· médiane densité {html.escape(stats.get('density_median', ''))}</li>"
        )

    best_f1 = max(threshold_rows, key=lambda r: safe_float(r.get("f1"))) if threshold_rows else {}
    best_precision = max(threshold_rows, key=lambda r: safe_float(r.get("precision"))) if threshold_rows else {}
    best_recall = max(threshold_rows, key=lambda r: safe_float(r.get("recall"))) if threshold_rows else {}

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux review analysis 001I</title>
<style>
  :root {{
    --bg: #101218;
    --panel: #181b22;
    --line: #2b303b;
    --text: #edf0f7;
    --muted: #9ea7b8;
    --accent: #83d4ff;
  }}
  body {{
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  h1, h2 {{
    margin-top: 0;
  }}
  section {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 16px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
  }}
  th, td {{
    border-bottom: 1px solid var(--line);
    padding: 7px 8px;
    vertical-align: top;
    font-size: 13px;
  }}
  th {{
    text-align: left;
    background: #20242e;
    position: sticky;
    top: 0;
  }}
  .muted {{
    color: var(--muted);
  }}
  b {{
    color: #ffffff;
  }}
  code {{
    color: #d7e4ff;
  }}
</style>
</head>
<body>
  <h1>TTFlux review analysis 001I</h1>

  <section>
    <h2>Résumé humain</h2>
    <p class="muted">Total : {summary["total"]} segments jugés.</p>
    <ul>
      {''.join(bucket_lines)}
    </ul>
  </section>

  <section>
    <h2>Seuils de risque testés</h2>
    <p>
      Meilleur F1 : seuil <b>{html.escape(best_f1.get("threshold", ""))}</b>
      · precision {html.escape(best_f1.get("precision", ""))}
      · recall {html.escape(best_f1.get("recall", ""))}
      · F1 {html.escape(best_f1.get("f1", ""))}
    </p>
    <p class="muted">
      Precision = parmi les segments signalés faux, combien étaient vraiment faux.
      Recall = parmi les fausses pistes humaines, combien sont attrapées.
    </p>
    {html_table(threshold_rows, ["threshold", "tp_false_tracks_caught", "fp_good_tracks_flagged", "fn_false_tracks_missed", "tn_good_tracks_kept", "precision", "recall", "f1"])}
  </section>

  <section>
    <h2>Détail par segment</h2>
    {html_table(sorted_rows, ["review_id", "manual_bucket", "manual_label", "manual_label_fr", "risk_score_001G", "review_guess_001G", "clip_id", "segment_name", "max_frame_gap", "traj_density", "speed_max", "hard_turn_count", "manual_notes"])}
  </section>
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="runs/batch_001E/review_manifest_001G_trajectory.csv")
    parser.add_argument("--labels", default="runs/batch_001E/review_labels_001H3_fr_export.csv")
    parser.add_argument("--out-csv", default="runs/batch_001E/review_analysis_001I.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/review_analysis_001I.html")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    labels_path = Path(args.labels)
    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)

    manifest_rows = read_csv(manifest_path)
    label_rows = read_csv(labels_path)

    manifest_by_id = {row.get("review_id", ""): row for row in manifest_rows}

    merged = []
    missing = []

    for label_row in label_rows:
        review_id = label_row.get("review_id", "")
        base = manifest_by_id.get(review_id)

        if not base:
            missing.append(review_id)
            continue

        row = dict(base)
        row["manual_label"] = label_row.get("manual_label", "")
        row["manual_label_fr"] = label_row.get("manual_label_fr", "")
        row["manual_notes"] = label_row.get("manual_notes", "")
        row["manual_bucket"] = bucket_label(row["manual_label"])

        merged.append(row)

    threshold_rows = analyze_thresholds(merged)
    summary = summarize(merged)

    write_csv(out_csv, merged)
    write_html(out_html, merged, threshold_rows, summary)

    print(f"[001I] manifest rows : {len(manifest_rows)}")
    print(f"[001I] label rows    : {len(label_rows)}")
    print(f"[001I] merged        : {len(merged)}")
    print(f"[001I] missing ids   : {missing}")
    print(f"[001I] buckets       : {summary['by_bucket']}")
    print(f"[001I] labels        : {summary['by_label']}")
    print(f"[001I] wrote CSV     : {out_csv}")
    print(f"[001I] wrote HTML    : {out_html}")


if __name__ == "__main__":
    main()