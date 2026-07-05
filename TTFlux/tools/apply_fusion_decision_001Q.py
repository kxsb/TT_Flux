# PATCH 001Q - Apply TTFlux fusion arbiter
#
# Reads:
# - micro_blob_features_001O.csv
#
# Produces:
# - decision_manifest_001Q.csv
# - decision_summary_001Q.json
# - decision_manifest_001Q.html
#
# Action:
# - keep   : segment accepted automatically
# - review : useful but uncertain, human/engine review
# - reject : likely false track
# - ignore : unclear / ignored labels when present
#
# This script applies the best 001P rule:
# 001P_fusion_guarded.

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
        "review_id",
        "clip_id",
        "segment_idx",
        "segment_name",
        "decision_001Q",
        "decision_confidence_001Q",
        "decision_reason_001Q",
        "decision_notes_001Q",
        "human_label_fr",
        "target_class",
        "expected_action",
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
        "mp4",
        "csv",
    ]

    seen = set()
    columns = []

    for col in preferred:
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


def fusion_guarded_decision(row: dict[str, str]) -> tuple[str, float, str, str]:
    false_j = fnum(row.get("false_track_score_001J"))
    frag_j = fnum(row.get("fragmentation_score_001J"))
    keep_j = fnum(row.get("keep_score_001J"))

    center_false = fnum(row.get("center_blob_false_score_001N"))
    micro_keep = fnum(row.get("micro_keep_score_001O"))
    micro_false = fnum(row.get("micro_false_score_001O"))
    micro_dist = fnum(row.get("micro_distance_med_001O"), 999.0)

    target = row.get("target_class", "")
    label = row.get("human_label_fr", "")

    if target == "ignore":
        return (
            "ignore",
            0.0,
            "human_ignore",
            "Segment marqué incertain dans le goldset.",
        )

    # Rescue: true ball with aggressive geometry.
    # Example: R0018.
    if (
        false_j >= 55
        and micro_keep >= 72
        and micro_false <= 10
        and micro_dist <= 7.0
        and center_false <= 35
    ):
        return (
            "keep",
            0.86,
            "rescue_fast_true_ball",
            "Géométrie agressive, mais micro-blob très compatible balle et centré.",
        )

    # Strong visual micro evidence can keep a real-looking track.
    if (
        micro_keep >= 75
        and micro_false <= 10
        and micro_dist <= 7.0
        and false_j < 45
        and center_false < 45
    ):
        return (
            "keep",
            0.84,
            "strong_micro_keep",
            "Micro-blob fort, proche du centre, peu de suspicion géométrique.",
        )

    # Strong geometric false track.
    if false_j >= 62:
        if keep_j < 40 or micro_keep < 35 or center_false >= 85:
            return (
                "reject",
                0.82,
                "strong_geometric_false",
                "Score faux géométrique élevé avec keep faible ou contexte visuel suspect.",
            )

        return (
            "review",
            0.62,
            "geometric_false_but_visual_conflict",
            "Score faux géométrique élevé, mais micro/keep contradictoire.",
        )

    # Medium geometric false track.
    if false_j >= 50:
        if keep_j < 45 and micro_keep < 75:
            return (
                "reject",
                0.72,
                "medium_geometric_false_low_keep",
                "Score faux moyen et keep faible.",
            )

        return (
            "review",
            0.58,
            "medium_false_visual_conflict",
            "Score faux moyen, mais micro ou keep empêche un rejet sûr.",
        )

    # Large object masquerading as ball.
    if center_false >= 85 and micro_keep < 50:
        return (
            "review",
            0.68,
            "large_object_warning",
            "Gros objet centré probable, mais pas assez sûr pour rejet automatique.",
        )

    # Weak micro evidence with strong center warning.
    if center_false >= 85 and micro_false >= 45:
        return (
            "review",
            0.64,
            "center_warning_micro_suspect",
            "Centre suspect et micro-blob faible.",
        )

    # Fragmented tracks are not clean enough for auto-keep.
    if frag_j >= 60:
        return (
            "review",
            0.66,
            "fragmented_track",
            "Trajectoire fragmentée : utile mais pas automatiquement propre.",
        )

    # Clean keep.
    if micro_keep >= 70 and micro_false <= 15 and keep_j >= 70:
        return (
            "keep",
            0.82,
            "clean_micro_and_keep",
            "Micro-blob favorable et score keep élevé.",
        )

    if keep_j >= 80 and false_j < 45:
        return (
            "keep",
            0.76,
            "high_keep_low_false",
            "Keep élevé et faible suspicion de fausse piste.",
        )

    if micro_false >= 50 and micro_keep < 45:
        return (
            "review",
            0.58,
            "micro_suspect",
            "Micro-blob suspect, mais rejet non sûr sans confirmation.",
        )

    return (
        "review",
        0.50,
        "default_review",
        "Cas ambigu par défaut.",
    )


def enrich(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []

    for row in rows:
        decision, confidence, reason, notes = fusion_guarded_decision(row)

        row = dict(row)
        row["decision_001Q"] = decision
        row["decision_confidence_001Q"] = f"{confidence:.2f}"
        row["decision_reason_001Q"] = reason
        row["decision_notes_001Q"] = notes

        out.append(row)

    return out


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_action: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_target_action: dict[str, dict[str, int]] = {}

    for row in rows:
        action = row.get("decision_001Q", "")
        reason = row.get("decision_reason_001Q", "")
        target = row.get("target_class", "unknown")

        by_action[action] = by_action.get(action, 0) + 1
        by_reason[reason] = by_reason.get(reason, 0) + 1

        by_target_action.setdefault(target, {})
        by_target_action[target][action] = by_target_action[target].get(action, 0) + 1

    return {
        "count": len(rows),
        "by_action": dict(sorted(by_action.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "by_target_action": {
            k: dict(sorted(v.items())) for k, v in sorted(by_target_action.items())
        },
        "rule": "001P_fusion_guarded",
    }


def html_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []

    for row in rows:
        cls = html.escape(row.get("decision_001Q", ""))
        cells = "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
        body.append(f'<tr class="{cls}">{cells}</tr>')

    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for action, count in summary["by_action"].items():
        summary_rows.append({"decision": action, "count": str(count)})

    reason_rows = []

    for reason, count in summary["by_reason"].items():
        reason_rows.append({"reason": reason, "count": str(count)})

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r.get("decision_001Q", ""),
            -fnum(r.get("decision_confidence_001Q")),
            r.get("review_id", ""),
        ),
    )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux decision manifest 001Q</title>
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
  tr.keep td {{
    background: rgba(100, 255, 150, 0.045);
  }}
  tr.review td {{
    background: rgba(255, 200, 80, 0.06);
  }}
  tr.reject td {{
    background: rgba(255, 80, 80, 0.08);
  }}
  tr.ignore td {{
    background: rgba(150, 170, 255, 0.045);
  }}
  .muted {{
    color: var(--muted);
  }}
</style>
</head>
<body>
  <h1>TTFlux decision manifest 001Q</h1>

  <section>
    <h2>Résumé</h2>
    <p class="muted">
      Règle appliquée : 001P_fusion_guarded. Les contradictions sont envoyées en review.
    </p>
    {html_table(summary_rows, ["decision", "count"])}
  </section>

  <section>
    <h2>Raisons</h2>
    {html_table(reason_rows, ["reason", "count"])}
  </section>

  <section>
    <h2>Détail</h2>
    {html_table(sorted_rows, [
        "review_id",
        "human_label_fr",
        "target_class",
        "decision_001Q",
        "decision_confidence_001Q",
        "decision_reason_001Q",
        "decision_notes_001Q",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="runs/batch_001E/micro_blob_features_001O.csv")
    parser.add_argument("--out-csv", default="runs/batch_001E/decision_manifest_001Q.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/decision_manifest_001Q.html")
    parser.add_argument("--out-json", default="runs/batch_001E/decision_summary_001Q.json")
    args = parser.parse_args()

    features_path = Path(args.features)
    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)
    out_json = Path(args.out_json)

    rows = read_csv(features_path)
    enriched = enrich(rows)
    summary = summarize(enriched)

    write_csv(out_csv, enriched)
    write_html(out_html, enriched, summary)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[001Q] rows      : {len(rows)}")
    print(f"[001Q] decisions : {summary['by_action']}")
    print(f"[001Q] reasons   : {summary['by_reason']}")
    print(f"[001Q] wrote CSV : {out_csv}")
    print(f"[001Q] wrote HTML: {out_html}")
    print(f"[001Q] wrote JSON: {out_json}")


if __name__ == "__main__":
    main()