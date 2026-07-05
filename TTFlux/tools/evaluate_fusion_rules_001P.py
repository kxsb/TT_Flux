# PATCH 001P - Fusion rules evaluation
#
# Input:
# - micro_blob_features_001O.csv
#
# Output:
# - fusion_eval_001P.csv
# - fusion_eval_001P.html
# - fusion_eval_001P.json
#
# Goal:
# Compare final decision rules:
# - keep positive segments
# - reject false tracks
# - review partial / contradictory segments
#
# 001P does not modify tracking yet.
# It creates a safer final arbitration layer.

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


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "rule_name",
        "review_id",
        "human_label_fr",
        "target_class",
        "expected_action",
        "predicted_action",
        "score_reason",
        "score_points",
        "risk_score_001G",
        "fragmentation_score_001J",
        "false_track_score_001J",
        "keep_score_001J",
        "center_blob_false_score_001N",
        "center_blob_score_001N",
        "micro_keep_score_001O",
        "micro_false_score_001O",
        "micro_candidate_rate_001O",
        "micro_score_med_001O",
        "micro_distance_med_001O",
        "micro_guess_001O",
        "clip_id",
        "segment_name",
    ]

    seen = set()
    columns = []

    for col in preferred:
        if col not in seen:
            columns.append(col)
            seen.add(col)

    for row in rows:
        for col in row:
            if col not in seen:
                columns.append(col)
                seen.add(col)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def old_001g_risk(row: dict[str, str]) -> str:
    risk = fnum(row.get("risk_score_001G"))

    if risk >= 60:
        return "reject"

    if risk >= 35:
        return "review"

    return "keep"


def rule_001j_human_safe(row: dict[str, str]) -> str:
    false_j = fnum(row.get("false_track_score_001J"))
    frag_j = fnum(row.get("fragmentation_score_001J"))
    keep_j = fnum(row.get("keep_score_001J"))

    if false_j >= 58 and keep_j < 60:
        return "reject"

    if false_j >= 45:
        return "review"

    if frag_j >= 60:
        return "review"

    if keep_j >= 70:
        return "keep"

    return "review"


def rule_001p_fusion_guarded(row: dict[str, str]) -> str:
    false_j = fnum(row.get("false_track_score_001J"))
    frag_j = fnum(row.get("fragmentation_score_001J"))
    keep_j = fnum(row.get("keep_score_001J"))

    center_false = fnum(row.get("center_blob_false_score_001N"))
    center_keep = fnum(row.get("center_blob_score_001N"))

    micro_keep = fnum(row.get("micro_keep_score_001O"))
    micro_false = fnum(row.get("micro_false_score_001O"))
    micro_dist = fnum(row.get("micro_distance_med_001O"), 999.0)
    micro_rate = fnum(row.get("micro_candidate_rate_001O"))

    # Rescue case: true ball with aggressive geometry.
    # Example target: R0018.
    if (
        false_j >= 55
        and micro_keep >= 72
        and micro_false <= 10
        and micro_dist <= 7.0
        and center_false <= 35
    ):
        return "keep"

    # Strong visual micro evidence can keep a fragmented but real-looking track,
    # as long as center-blob did not scream "large object".
    if (
        micro_keep >= 75
        and micro_false <= 10
        and micro_dist <= 7.0
        and false_j < 45
        and center_false < 45
    ):
        return "keep"

    # Strong geometric false tracks.
    if false_j >= 62:
        if keep_j < 40 or micro_keep < 35 or center_false >= 85:
            return "reject"
        return "review"

    # Medium geometric false tracks.
    if false_j >= 50:
        if keep_j < 45 and micro_keep < 75:
            return "reject"
        return "review"

    # Large object masquerading as ball.
    # Do not auto-reject because this also catches some real fragmented positives.
    if center_false >= 85 and micro_keep < 50:
        return "review"

    # Weak micro evidence with strong center warning.
    if center_false >= 85 and micro_false >= 45:
        return "review"

    # Fragmented tracks are useful but not clean enough.
    if frag_j >= 60:
        return "review"

    # Clean keep.
    if micro_keep >= 70 and micro_false <= 15 and keep_j >= 70:
        return "keep"

    if keep_j >= 80 and false_j < 45:
        return "keep"

    if micro_false >= 50 and micro_keep < 45:
        return "review"

    return "review"


def rule_001p_stricter_reject(row: dict[str, str]) -> str:
    false_j = fnum(row.get("false_track_score_001J"))
    frag_j = fnum(row.get("fragmentation_score_001J"))
    keep_j = fnum(row.get("keep_score_001J"))

    center_false = fnum(row.get("center_blob_false_score_001N"))
    micro_keep = fnum(row.get("micro_keep_score_001O"))
    micro_false = fnum(row.get("micro_false_score_001O"))
    micro_dist = fnum(row.get("micro_distance_med_001O"), 999.0)

    if micro_keep >= 75 and micro_false <= 10 and micro_dist <= 7 and false_j < 70:
        return "keep"

    if false_j >= 62 and keep_j < 70:
        return "reject"

    if false_j >= 50 and keep_j < 45:
        return "reject"

    if center_false >= 90 and micro_keep < 45 and keep_j < 80:
        return "reject"

    if frag_j >= 55 or false_j >= 45 or center_false >= 85:
        return "review"

    if keep_j >= 75:
        return "keep"

    return "review"


def rule_001p_review_first(row: dict[str, str]) -> str:
    false_j = fnum(row.get("false_track_score_001J"))
    frag_j = fnum(row.get("fragmentation_score_001J"))
    keep_j = fnum(row.get("keep_score_001J"))

    center_false = fnum(row.get("center_blob_false_score_001N"))
    micro_keep = fnum(row.get("micro_keep_score_001O"))
    micro_false = fnum(row.get("micro_false_score_001O"))
    micro_dist = fnum(row.get("micro_distance_med_001O"), 999.0)

    if micro_keep >= 78 and micro_false <= 8 and micro_dist <= 7 and false_j < 70:
        return "keep"

    if false_j >= 65 and keep_j < 45:
        return "reject"

    if false_j >= 45 or frag_j >= 55 or center_false >= 85:
        return "review"

    if keep_j >= 80:
        return "keep"

    return "review"


RULES = {
    "old_001G_risk": old_001g_risk,
    "001J_human_safe": rule_001j_human_safe,
    "001P_fusion_guarded": rule_001p_fusion_guarded,
    "001P_stricter_reject": rule_001p_stricter_reject,
    "001P_review_first": rule_001p_review_first,
}


def score_action(expected: str, predicted: str, target: str) -> tuple[float, str]:
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


def evaluate_rule(rows: list[dict[str, str]], rule_name: str, fn) -> dict[str, Any]:
    details = []
    counts: dict[str, int] = {}

    weighted_score = 0.0
    total_weight = 0.0

    for row in rows:
        expected = row.get("expected_action", "")
        target = row.get("target_class", "")
        weight = fnum(row.get("eval_weight", "1"))

        predicted = fn(row)
        points, reason = score_action(expected, predicted, target)

        weighted_score += points * weight
        total_weight += weight
        counts[reason] = counts.get(reason, 0) + 1

        detail = dict(row)
        detail["rule_name"] = rule_name
        detail["predicted_action"] = predicted
        detail["score_reason"] = reason
        detail["score_points"] = str(points)
        details.append(detail)

    score = weighted_score / max(1e-9, total_weight)

    return {
        "rule_name": rule_name,
        "score": round(score, 4),
        "counts": dict(sorted(counts.items())),
        "details": details,
    }


def evaluate_all(rows: list[dict[str, str]]) -> dict[str, Any]:
    results = []

    for name, fn in RULES.items():
        results.append(evaluate_rule(rows, name, fn))

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "best_rule": results[0]["rule_name"] if results else "",
        "best_score": results[0]["score"] if results else 0.0,
        "rules": results,
    }


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)

    body = []
    for row in rows:
        cls = html.escape(str(row.get("score_reason", row.get("target_class", ""))))
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
        body.append(f'<tr class="{cls}">{cells}</tr>')

    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, rows: list[dict[str, str]], evaluation: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for result in evaluation["rules"]:
        c = result["counts"]
        summary_rows.append(
            {
                "rule": result["rule_name"],
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

    best = evaluation["rules"][0]
    best_details = sorted(
        best["details"],
        key=lambda r: (
            r.get("score_reason", ""),
            r.get("target_class", ""),
            r.get("review_id", ""),
        ),
    )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux fusion eval 001P</title>
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
    text-align: left;
    background: #20242e;
    position: sticky;
    top: 0;
  }}
  tr.bad_rejected_good td,
  tr.bad_kept_false td {{
    background: rgba(255, 60, 60, 0.16);
  }}
  tr.safe_review td,
  tr.good_review_partial td {{
    background: rgba(255, 200, 80, 0.06);
  }}
  tr.exact td {{
    background: rgba(100, 255, 150, 0.045);
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
  <h1>TTFlux fusion eval 001P</h1>

  <section>
    <h2>Résumé</h2>
    <p>
      Meilleure règle : <b>{html.escape(str(evaluation["best_rule"]))}</b>
      · score {html.escape(str(evaluation["best_score"]))}
    </p>
    <p class="muted">
      001P fusionne géométrie 001J, gros objet centré 001N, et micro-blob 001O.
      Les contradictions sont envoyées en revue plutôt que rejetées.
    </p>
    {table(summary_rows, [
        "rule",
        "score",
        "exact",
        "safe_review",
        "good_review_partial",
        "bad_rejected_good",
        "bad_kept_false",
        "partial_kept",
        "partial_rejected",
        "ignored",
    ])}
  </section>

  <section>
    <h2>Détail de la meilleure règle</h2>
    {table(best_details, [
        "review_id",
        "human_label_fr",
        "target_class",
        "expected_action",
        "predicted_action",
        "score_reason",
        "risk_score_001G",
        "fragmentation_score_001J",
        "false_track_score_001J",
        "keep_score_001J",
        "center_blob_false_score_001N",
        "micro_keep_score_001O",
        "micro_false_score_001O",
        "micro_distance_med_001O",
        "micro_guess_001O",
        "clip_id",
        "segment_name",
    ])}
  </section>
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def write_json(path: Path, evaluation: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "best_rule": evaluation["best_rule"],
        "best_score": evaluation["best_score"],
        "rules": [
            {
                "rule_name": r["rule_name"],
                "score": r["score"],
                "counts": r["counts"],
            }
            for r in evaluation["rules"]
        ],
    }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="runs/batch_001E/micro_blob_features_001O.csv")
    parser.add_argument("--out-csv", default="runs/batch_001E/fusion_eval_001P.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/fusion_eval_001P.html")
    parser.add_argument("--out-json", default="runs/batch_001E/fusion_eval_001P.json")
    args = parser.parse_args()

    features_path = Path(args.features)
    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)
    out_json = Path(args.out_json)

    rows = read_csv(features_path)
    evaluation = evaluate_all(rows)

    all_details = []
    for result in evaluation["rules"]:
        all_details.extend(result["details"])

    write_csv(out_csv, all_details)
    write_html(out_html, rows, evaluation)
    write_json(out_json, evaluation)

    print(f"[001P] rows      : {len(rows)}")
    print(f"[001P] best rule : {evaluation['best_rule']} score={evaluation['best_score']}")

    for result in evaluation["rules"]:
        print(f"[001P] {result['rule_name']}: score={result['score']} counts={result['counts']}")

    print(f"[001P] wrote CSV  : {out_csv}")
    print(f"[001P] wrote HTML : {out_html}")
    print(f"[001P] wrote JSON : {out_json}")


if __name__ == "__main__":
    main()