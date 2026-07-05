from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


def fnum(value: Any, default: float = 0.0) -> float:
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


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def target_from_label(label: str) -> tuple[str, str, float, str]:
    label = label.strip()

    if label == "gold_keep":
        return "positive", "keep", 1.0, "train_gold_positive"

    if label == "ok_ball":
        return "positive", "keep", 1.0, "train_positive"

    if label == "partial_ball":
        return "partial", "review", 0.7, "diagnostic_partial"

    if label == "false_track":
        return "negative", "reject", 1.0, "train_negative"

    if label == "unclear":
        return "ignore", "ignore", 0.0, "ignore_unclear"

    return "unknown", "ignore", 0.0, "unknown"


def build_goldset(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []

    for row in rows:
        label = row.get("manual_label", "")
        target, expected, weight, split = target_from_label(label)

        out.append(
            {
                "review_id": row.get("review_id", ""),
                "clip_id": row.get("clip_id", ""),
                "segment_idx": row.get("segment_idx", ""),
                "segment_name": row.get("segment_name", ""),
                "human_label": label,
                "human_label_fr": row.get("manual_label_fr", ""),
                "target_class": target,
                "expected_action": expected,
                "eval_weight": str(weight),
                "training_split": split,
                "risk_score_001G": row.get("risk_score_001G", ""),
                "fragmentation_score_001J": row.get("fragmentation_score_001J", ""),
                "false_track_score_001J": row.get("false_track_score_001J", ""),
                "keep_score_001J": row.get("keep_score_001J", ""),
                "max_frame_gap": row.get("max_frame_gap", ""),
                "traj_density": row.get("traj_density", ""),
                "hard_turn_count": row.get("hard_turn_count", ""),
                "speed_p95_over_median": row.get("speed_p95_over_median", ""),
                "mp4": row.get("mp4", ""),
                "csv": row.get("csv", ""),
                "manual_notes": row.get("manual_notes", ""),
            }
        )

    return out


def rule_old_001G(row: dict[str, str]) -> str:
    risk = fnum(row.get("risk_score_001G"))

    if risk >= 60:
        return "reject"
    if risk >= 35:
        return "review"
    return "keep"


def rule_001J_human_safe(row: dict[str, str]) -> str:
    false_score = fnum(row.get("false_track_score_001J"))
    frag = fnum(row.get("fragmentation_score_001J"))
    keep = fnum(row.get("keep_score_001J"))

    # Prudence : on ne rejette que si le faux est fort ET que le keep est faible.
    if false_score >= 58 and keep < 60:
        return "reject"

    if false_score >= 45:
        return "review"

    if frag >= 60:
        return "review"

    if keep >= 70:
        return "keep"

    return "review"


def rule_001J_keep_first(row: dict[str, str]) -> str:
    false_score = fnum(row.get("false_track_score_001J"))
    frag = fnum(row.get("fragmentation_score_001J"))
    keep = fnum(row.get("keep_score_001J"))

    if keep >= 80 and false_score < 50:
        return "keep"

    if false_score >= 60 and keep < 70:
        return "reject"

    if frag >= 50 or false_score >= 35:
        return "review"

    return "keep"


def rule_001J_strict_false(row: dict[str, str]) -> str:
    false_score = fnum(row.get("false_track_score_001J"))
    frag = fnum(row.get("fragmentation_score_001J"))
    keep = fnum(row.get("keep_score_001J"))

    if false_score >= 55:
        return "reject"

    if false_score >= 40:
        return "review"

    if frag >= 55:
        return "review"

    if keep >= 70:
        return "keep"

    return "review"


RULES = {
    "old_001G_risk": rule_old_001G,
    "001J_human_safe": rule_001J_human_safe,
    "001J_keep_first": rule_001J_keep_first,
    "001J_strict_false": rule_001J_strict_false,
}


def score_prediction(expected: str, predicted: str, target: str) -> tuple[float, str]:
    if expected == "ignore":
        return 0.0, "ignored"

    if predicted == expected:
        return 1.0, "exact"

    if predicted == "review":
        if target == "partial":
            return 0.9, "good_review_partial"
        return 0.5, "safe_review"

    if expected == "keep" and predicted == "reject":
        return 0.0, "bad_rejected_good"

    if expected == "reject" and predicted == "keep":
        return 0.0, "bad_kept_false"

    if expected == "review" and predicted == "keep":
        return 0.35, "partial_kept"

    if expected == "review" and predicted == "reject":
        return 0.25, "partial_rejected"

    return 0.0, "wrong"


def evaluate_rule(rows: list[dict[str, str]], name: str, fn) -> dict[str, Any]:
    details = []
    total_weight = 0.0
    weighted_score = 0.0
    counts: dict[str, int] = {}

    for row in rows:
        expected = row.get("expected_action", "")
        target = row.get("target_class", "")
        weight = fnum(row.get("eval_weight"))
        predicted = fn(row)

        points, reason = score_prediction(expected, predicted, target)

        total_weight += weight
        weighted_score += points * weight
        counts[reason] = counts.get(reason, 0) + 1

        detail = dict(row)
        detail["rule_name"] = name
        detail["predicted_action"] = predicted
        detail["score_points"] = str(points)
        detail["score_reason"] = reason
        details.append(detail)

    score = weighted_score / max(1e-9, total_weight)

    return {
        "rule": name,
        "score": round(score, 4),
        "counts": counts,
        "details": details,
    }


def evaluate_all(rows: list[dict[str, str]]) -> dict[str, Any]:
    results = []

    for name, fn in RULES.items():
        results.append(evaluate_rule(rows, name, fn))

    results.sort(key=lambda r: r["score"], reverse=True)

    return {
        "best_rule": results[0]["rule"] if results else "",
        "best_score": results[0]["score"] if results else 0.0,
        "rules": results,
    }


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)

    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
        body.append(f"<tr>{cells}</tr>")

    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, gold_rows: list[dict[str, str]], eval_result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    target_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}

    for row in gold_rows:
        target_counts[row["target_class"]] = target_counts.get(row["target_class"], 0) + 1
        split_counts[row["training_split"]] = split_counts.get(row["training_split"], 0) + 1

    target_text = " · ".join(f"{k}: {v}" for k, v in sorted(target_counts.items()))
    split_text = " · ".join(f"{k}: {v}" for k, v in sorted(split_counts.items()))

    summary_rows = []
    for result in eval_result["rules"]:
        c = result["counts"]
        summary_rows.append(
            {
                "rule": result["rule"],
                "score": result["score"],
                "exact": c.get("exact", 0),
                "safe_review": c.get("safe_review", 0),
                "good_review_partial": c.get("good_review_partial", 0),
                "bad_rejected_good": c.get("bad_rejected_good", 0),
                "bad_kept_false": c.get("bad_kept_false", 0),
                "partial_kept": c.get("partial_kept", 0),
                "partial_rejected": c.get("partial_rejected", 0),
                "ignored": c.get("ignored", 0),
            }
        )

    best = eval_result["rules"][0]
    detail_rows = []

    for d in best["details"]:
        detail_rows.append(
            {
                "review_id": d.get("review_id", ""),
                "human_label_fr": d.get("human_label_fr", ""),
                "target": d.get("target_class", ""),
                "expected": d.get("expected_action", ""),
                "predicted": d.get("predicted_action", ""),
                "reason": d.get("score_reason", ""),
                "risk_001G": d.get("risk_score_001G", ""),
                "frag_001J": d.get("fragmentation_score_001J", ""),
                "false_001J": d.get("false_track_score_001J", ""),
                "keep_001J": d.get("keep_score_001J", ""),
                "clip_id": d.get("clip_id", ""),
                "segment_name": d.get("segment_name", ""),
            }
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 001K goldset eval</title>
<style>
  :root {{
    --bg: #101218;
    --panel: #181b22;
    --line: #2b303b;
    --text: #edf0f7;
    --muted: #9ea7b8;
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
    background: #20242e;
    text-align: left;
    position: sticky;
    top: 0;
  }}
  .muted {{
    color: var(--muted);
  }}
  b {{
    color: #fff;
  }}
</style>
</head>
<body>
  <h1>TTFlux 001K goldset eval</h1>

  <section>
    <h2>Goldset figé</h2>
    <p class="muted">{len(gold_rows)} segments · {html.escape(target_text)}</p>
    <p class="muted">{html.escape(split_text)}</p>
    <p>
      Ce fichier devient le banc d'essai permanent : garder les positifs,
      rejeter les fausses pistes, envoyer les partiels en revue.
    </p>
  </section>

  <section>
    <h2>Comparaison des règles</h2>
    <p>Meilleure règle actuelle : <b>{html.escape(eval_result["best_rule"])}</b> · score {html.escape(str(eval_result["best_score"]))}</p>
    {table(summary_rows, ["rule", "score", "exact", "safe_review", "good_review_partial", "bad_rejected_good", "bad_kept_false", "partial_kept", "partial_rejected", "ignored"])}
  </section>

  <section>
    <h2>Détail de la meilleure règle</h2>
    {table(detail_rows, ["review_id", "human_label_fr", "target", "expected", "predicted", "reason", "risk_001G", "frag_001J", "false_001J", "keep_001J", "clip_id", "segment_name"])}
  </section>
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def write_json(path: Path, gold_rows: list[dict[str, str]], eval_result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    target_counts: dict[str, int] = {}

    for row in gold_rows:
        target_counts[row["target_class"]] = target_counts.get(row["target_class"], 0) + 1

    payload = {
        "name": "TTFlux goldset 001K",
        "count": len(gold_rows),
        "target_counts": dict(sorted(target_counts.items())),
        "best_rule": eval_result["best_rule"],
        "best_score": eval_result["best_score"],
        "rules": [
            {
                "rule": r["rule"],
                "score": r["score"],
                "counts": r["counts"],
            }
            for r in eval_result["rules"]
        ],
        "rows": gold_rows,
    }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated", default="runs/batch_001E/review_calibrated_001J.csv")
    parser.add_argument("--out-csv", default="runs/batch_001E/goldset_001K.csv")
    parser.add_argument("--out-json", default="runs/batch_001E/goldset_001K.json")
    parser.add_argument("--out-html", default="runs/batch_001E/eval_rules_001K.html")
    args = parser.parse_args()

    calibrated = Path(args.calibrated)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_html = Path(args.out_html)

    rows = read_csv(calibrated)
    gold_rows = build_goldset(rows)
    eval_result = evaluate_all(gold_rows)

    columns = [
        "review_id",
        "clip_id",
        "segment_idx",
        "segment_name",
        "human_label",
        "human_label_fr",
        "target_class",
        "expected_action",
        "eval_weight",
        "training_split",
        "risk_score_001G",
        "fragmentation_score_001J",
        "false_track_score_001J",
        "keep_score_001J",
        "max_frame_gap",
        "traj_density",
        "hard_turn_count",
        "speed_p95_over_median",
        "mp4",
        "csv",
        "manual_notes",
    ]

    write_csv(out_csv, gold_rows, columns)
    write_json(out_json, gold_rows, eval_result)
    write_html(out_html, gold_rows, eval_result)

    target_counts: dict[str, int] = {}
    for row in gold_rows:
        target_counts[row["target_class"]] = target_counts.get(row["target_class"], 0) + 1

    print(f"[001K] input rows : {len(rows)}")
    print(f"[001K] gold rows  : {len(gold_rows)}")
    print(f"[001K] targets    : {dict(sorted(target_counts.items()))}")
    print(f"[001K] best rule  : {eval_result['best_rule']} score={eval_result['best_score']}")

    for result in eval_result["rules"]:
        print(f"[001K] rule {result['rule']}: score={result['score']} counts={result['counts']}")

    print(f"[001K] wrote CSV  : {out_csv}")
    print(f"[001K] wrote JSON : {out_json}")
    print(f"[001K] wrote HTML : {out_html}")


if __name__ == "__main__":
    main()
