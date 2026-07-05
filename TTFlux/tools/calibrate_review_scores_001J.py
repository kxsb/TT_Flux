# PATCH 001J - Calibrate TTFlux review scores from human labels
#
# Reads:
# - review_analysis_001I.csv
#
# Produces:
# - review_calibrated_001J.csv
# - review_calibrated_001J.html
# - review_calibration_001J.json
#
# Goal:
# Split the old single "risk" idea into:
# - fragmentation_score_001J : holes / sparse track / partial-ball risk
# - false_track_score_001J   : zigzag / unstable / wrong-object risk
# - training_value_001J      : how useful the segment is for later learning
#
# Usage:
#   python tools\calibrate_review_scores_001J.py ^
#     --analysis runs\batch_001E\review_analysis_001I.csv ^
#     --out-csv runs\batch_001E\review_calibrated_001J.csv ^
#     --out-html runs\batch_001E\review_calibrated_001J.html ^
#     --out-json runs\batch_001E\review_calibration_001J.json

from __future__ import annotations

import argparse
import csv
import html
import json
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


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def fmt(value: float, digits: int = 2) -> str:
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


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
        "manual_label",
        "manual_label_fr",
        "manual_bucket",
        "training_value_001J",
        "fragmentation_score_001J",
        "false_track_score_001J",
        "keep_score_001J",
        "calibrated_guess_001J",
        "calibrated_notes_001J",
        "risk_score_001G",
        "review_guess_001G",
        "max_frame_gap",
        "traj_density",
        "traj_n_points",
        "hard_turn_count",
        "speed_p95_over_median",
        "speed_median",
        "speed_p95",
        "speed_max",
        "accel_p95",
        "x_range",
        "y_range",
        "travel",
        "risk_flags_001G",
        "mp4",
        "csv",
        "manual_notes",
    ]

    seen = set()
    columns = []

    for col in preferred:
        if col not in seen:
            columns.append(col)
            seen.add(col)

    for row in rows:
        for col in row.keys():
            if col not in seen:
                columns.append(col)
                seen.add(col)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def bucket_label(label: str) -> str:
    label = str(label).strip()

    if label in POSITIVE_LABELS:
        return "positive"
    if label in PARTIAL_LABELS:
        return "partial"
    if label in NEGATIVE_LABELS:
        return "negative"
    if label in UNCLEAR_LABELS:
        return "unclear"

    return "unknown"


def score_fragmentation(row: dict[str, str]) -> tuple[float, list[str]]:
    max_gap = safe_float(row.get("max_frame_gap"))
    density = safe_float(row.get("traj_density"))
    n_points = safe_float(row.get("traj_n_points") or row.get("n_points"))
    big_gap_count = safe_float(row.get("big_gap_count"))
    gap_count = safe_float(row.get("gap_count"))

    score = 0.0
    notes: list[str] = []

    if max_gap >= 40:
        score += 38
        notes.append("trou énorme")
    elif max_gap >= 30:
        score += 30
        notes.append("trou très grand")
    elif max_gap >= 20:
        score += 22
        notes.append("grand trou")
    elif max_gap >= 10:
        score += 12
        notes.append("trou moyen")
    elif max_gap >= 4:
        score += 5
        notes.append("petit trou")

    if density < 0.24:
        score += 30
        notes.append("densité très faible")
    elif density < 0.32:
        score += 22
        notes.append("densité faible")
    elif density < 0.42:
        score += 10
        notes.append("densité moyenne")

    if n_points < 15:
        score += 12
        notes.append("peu de points")
    elif n_points < 18:
        score += 6
        notes.append("points limités")

    if big_gap_count >= 3:
        score += 12
        notes.append("plusieurs gros trous")
    elif big_gap_count >= 1:
        score += 6
        notes.append("gros trou isolé")

    if gap_count >= 10:
        score += 8
        notes.append("nombreux trous")

    return clamp(score), notes


def score_false_track(row: dict[str, str]) -> tuple[float, list[str]]:
    hard_turns = safe_float(row.get("hard_turn_count"))
    medium_turns = safe_float(row.get("medium_turn_count"))
    speed_ratio = safe_float(row.get("speed_p95_over_median"))
    speed_max = safe_float(row.get("speed_max"))
    accel_p95 = safe_float(row.get("accel_p95"))
    max_gap = safe_float(row.get("max_frame_gap"))
    density = safe_float(row.get("traj_density"))
    x_range = safe_float(row.get("traj_x_range") or row.get("x_range"))
    y_range = safe_float(row.get("traj_y_range") or row.get("y_range"))
    travel = safe_float(row.get("travel"))

    score = 0.0
    notes: list[str] = []

    # False tracks often wiggle or jump between wrong visual objects.
    if hard_turns >= 7:
        score += 30
        notes.append("zigzag très fort")
    elif hard_turns >= 5:
        score += 24
        notes.append("nombreux virages durs")
    elif hard_turns >= 3:
        score += 14
        notes.append("virages durs")
    elif hard_turns >= 1:
        score += 5
        notes.append("virage dur isolé")

    if speed_ratio >= 7:
        score += 28
        notes.append("ratio vitesse très instable")
    elif speed_ratio >= 5:
        score += 22
        notes.append("ratio vitesse instable")
    elif speed_ratio >= 3.5:
        score += 14
        notes.append("ratio vitesse suspect")
    elif speed_ratio >= 2.5:
        score += 7
        notes.append("ratio vitesse à surveiller")

    if accel_p95 >= 55:
        score += 16
        notes.append("accélération haute")
    elif accel_p95 >= 40:
        score += 9
        notes.append("accélération suspecte")

    # Compact + zigzag can mean the tracker is stuck on texture/reflet.
    compact = x_range <= 130 and y_range <= 130
    semi_compact = x_range <= 220 and y_range <= 180

    if compact and hard_turns >= 3:
        score += 18
        notes.append("zigzag compact")
    elif semi_compact and hard_turns >= 4:
        score += 12
        notes.append("zigzag semi-compact")

    # A high speed alone can be real in table tennis, so small contribution only.
    if speed_max >= 80:
        score += 7
        notes.append("vitesse max très haute")

    # Very fragmented tracks are often partial rather than false. Reduce false suspicion
    # when the problem is mostly missing frames and not zigzag.
    if max_gap >= 24 and hard_turns <= 2 and speed_ratio < 3.0:
        score -= 16
        notes.append("plutôt fragmenté que faux")

    if density < 0.28 and hard_turns <= 2:
        score -= 8
        notes.append("faible densité sans zigzag net")

    # Long travel with broad ranges can be a real ball arc, not necessarily false.
    if travel >= 700 and x_range >= 250 and hard_turns <= 2:
        score -= 8
        notes.append("grand déplacement plausible")

    return clamp(score), notes


def score_keep(row: dict[str, str], fragmentation: float, false_score: float) -> tuple[float, list[str]]:
    density = safe_float(row.get("traj_density"))
    hard_turns = safe_float(row.get("hard_turn_count"))
    max_gap = safe_float(row.get("max_frame_gap"))
    speed_ratio = safe_float(row.get("speed_p95_over_median"))
    n_points = safe_float(row.get("traj_n_points") or row.get("n_points"))

    score = 100.0
    notes: list[str] = []

    score -= fragmentation * 0.55
    score -= false_score * 0.85

    if density >= 0.48:
        score += 10
        notes.append("bonne densité")
    if max_gap <= 4:
        score += 8
        notes.append("trous faibles")
    if hard_turns <= 1:
        score += 8
        notes.append("peu de virages durs")
    if speed_ratio <= 2.0:
        score += 6
        notes.append("vitesse stable")
    if n_points >= 20:
        score += 5
        notes.append("assez de points")

    return clamp(score), notes


def training_value(label: str) -> str:
    if label == "gold_keep":
        return "gold_positive"
    if label == "ok_ball":
        return "positive"
    if label == "partial_ball":
        return "partial_diagnostic"
    if label == "false_track":
        return "negative"
    if label == "unclear":
        return "ignore_unclear"
    return "unlabeled"


def calibrated_guess(fragmentation: float, false_score: float, keep_score: float) -> str:
    if false_score >= 55:
        return "probable_false_track"
    if fragmentation >= 55 and false_score < 45:
        return "probably_partial_or_hard_ball"
    if keep_score >= 70 and false_score < 35:
        return "probably_keep"
    if false_score >= 40:
        return "check_false_track"
    if fragmentation >= 40:
        return "check_fragmented_ball"
    return "low_risk"


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched = []

    for row in rows:
        row = dict(row)

        label = row.get("manual_label", "")
        bucket = row.get("manual_bucket") or bucket_label(label)
        row["manual_bucket"] = bucket

        frag, frag_notes = score_fragmentation(row)
        false_score, false_notes = score_false_track(row)
        keep, keep_notes = score_keep(row, frag, false_score)

        row["fragmentation_score_001J"] = fmt(frag)
        row["false_track_score_001J"] = fmt(false_score)
        row["keep_score_001J"] = fmt(keep)
        row["training_value_001J"] = training_value(label)
        row["calibrated_guess_001J"] = calibrated_guess(frag, false_score, keep)

        notes = []
        if frag_notes:
            notes.append("fragmentation: " + ", ".join(frag_notes))
        if false_notes:
            notes.append("faux: " + ", ".join(false_notes))
        if keep_notes:
            notes.append("garde: " + ", ".join(keep_notes))

        row["calibrated_notes_001J"] = " | ".join(notes)

        enriched.append(row)

    return enriched


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_label: dict[str, int] = {}
    by_bucket: dict[str, int] = {}
    by_guess: dict[str, int] = {}
    by_training: dict[str, int] = {}

    for row in rows:
        by_label[row.get("manual_label", "")] = by_label.get(row.get("manual_label", ""), 0) + 1
        by_bucket[row.get("manual_bucket", "")] = by_bucket.get(row.get("manual_bucket", ""), 0) + 1
        by_guess[row.get("calibrated_guess_001J", "")] = by_guess.get(row.get("calibrated_guess_001J", ""), 0) + 1
        by_training[row.get("training_value_001J", "")] = by_training.get(row.get("training_value_001J", ""), 0) + 1

    def med(col: str, bucket: str) -> str:
        vals = [
            safe_float(r.get(col))
            for r in rows
            if r.get("manual_bucket") == bucket
        ]
        if not vals:
            return ""
        return fmt(statistics.median(vals))

    bucket_stats = {}
    for bucket in sorted(by_bucket):
        bucket_stats[bucket] = {
            "count": by_bucket[bucket],
            "old_risk_median": med("risk_score_001G", bucket),
            "fragmentation_median": med("fragmentation_score_001J", bucket),
            "false_track_median": med("false_track_score_001J", bucket),
            "keep_median": med("keep_score_001J", bucket),
        }

    return {
        "total": len(rows),
        "by_label": dict(sorted(by_label.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_guess": dict(sorted(by_guess.items())),
        "by_training": dict(sorted(by_training.items())),
        "bucket_stats": bucket_stats,
        "interpretation": [
            "risk_score_001G is mainly a fragmentation/difficulty score, not a false-track score.",
            "001J separates fragmentation_score from false_track_score.",
            "Segments labeled gold_keep or ok_ball should not be rejected only because of large frame gaps.",
            "false_track_score is driven more by hard turns, speed instability and compact zigzag patterns.",
        ],
    }


def html_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)

    body = []
    for row in rows:
        cls = html.escape(row.get("manual_bucket", ""))
        cells = "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
        body.append(f'<tr class="{cls}">{cells}</tr>')

    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            -safe_float(r.get("false_track_score_001J")),
            -safe_float(r.get("fragmentation_score_001J")),
            r.get("clip_id", ""),
        ),
    )

    bucket_items = []
    for bucket, stats in summary["bucket_stats"].items():
        bucket_items.append(
            f"<li><b>{html.escape(bucket)}</b> : {stats['count']} "
            f"· ancien risque médian {html.escape(stats['old_risk_median'])} "
            f"· fragmentation médiane {html.escape(stats['fragmentation_median'])} "
            f"· faux médian {html.escape(stats['false_track_median'])} "
            f"· keep médian {html.escape(stats['keep_median'])}</li>"
        )

    training_items = []
    for key, count in summary["by_training"].items():
        training_items.append(f"<li><b>{html.escape(key)}</b> : {count}</li>")

    guess_items = []
    for key, count in summary["by_guess"].items():
        guess_items.append(f"<li><b>{html.escape(key)}</b> : {count}</li>")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux calibration 001J</title>
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
  section {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 16px;
  }}
  h1, h2 {{
    margin-top: 0;
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
  tr.negative td {{
    background: rgba(255, 80, 80, 0.07);
  }}
  tr.partial td {{
    background: rgba(255, 200, 80, 0.06);
  }}
  tr.positive td {{
    background: rgba(100, 255, 150, 0.035);
  }}
  tr.unclear td {{
    background: rgba(150, 170, 255, 0.045);
  }}
  .muted {{
    color: var(--muted);
  }}
  b {{
    color: #ffffff;
  }}
</style>
</head>
<body>
  <h1>TTFlux calibration 001J</h1>

  <section>
    <h2>Diagnostic</h2>
    <p>
      001G mélangeait deux phénomènes : une vraie balle difficile/trouée, et une fausse piste.
      001J les sépare avec deux scores : <b>fragmentation</b> et <b>fausse piste</b>.
    </p>
    <ul>
      {''.join(bucket_items)}
    </ul>
  </section>

  <section>
    <h2>Valeur pour apprentissage</h2>
    <ul>
      {''.join(training_items)}
    </ul>
  </section>

  <section>
    <h2>Estimation recalibrée 001J</h2>
    <ul>
      {''.join(guess_items)}
    </ul>
  </section>

  <section>
    <h2>Détail par score de fausse piste décroissant</h2>
    {html_table(sorted_rows, [
        "review_id",
        "manual_label_fr",
        "manual_bucket",
        "training_value_001J",
        "risk_score_001G",
        "fragmentation_score_001J",
        "false_track_score_001J",
        "keep_score_001J",
        "calibrated_guess_001J",
        "clip_id",
        "segment_name",
        "max_frame_gap",
        "traj_density",
        "hard_turn_count",
        "speed_p95_over_median",
        "calibrated_notes_001J",
    ])}
  </section>
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="runs/batch_001E/review_analysis_001I.csv")
    parser.add_argument("--out-csv", default="runs/batch_001E/review_calibrated_001J.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/review_calibrated_001J.html")
    parser.add_argument("--out-json", default="runs/batch_001E/review_calibration_001J.json")
    args = parser.parse_args()

    analysis_path = Path(args.analysis)
    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)
    out_json = Path(args.out_json)

    rows = read_csv(analysis_path)
    enriched = enrich_rows(rows)
    summary = summarize(enriched)

    write_csv(out_csv, enriched)
    write_html(out_html, enriched, summary)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[001J] input rows     : {len(rows)}")
    print(f"[001J] labels         : {summary['by_label']}")
    print(f"[001J] buckets        : {summary['by_bucket']}")
    print(f"[001J] training value : {summary['by_training']}")
    print(f"[001J] calibrated     : {summary['by_guess']}")
    print(f"[001J] wrote CSV      : {out_csv}")
    print(f"[001J] wrote HTML     : {out_html}")
    print(f"[001J] wrote JSON     : {out_json}")


if __name__ == "__main__":
    main()