from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


ACTIONS = ["keep", "review", "reject"]


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
        "decision_001U",
        "decision_confidence_001U",
        "decision_reason_001U",
        "decision_notes_001U",
        "risk_score_001G",
        "n_points",
        "center_blob_false_score_001N",
        "micro_keep_score_001O",
        "micro_false_score_001O",
        "micro_distance_med_001O",
        "micro_guess_001O",
        "csv",
        "mp4",
    ]

    seen = set()
    cols = []

    for col in preferred:
        if col not in seen:
            cols.append(col)
            seen.add(col)

    for row in rows:
        for col in row:
            if col not in seen:
                cols.append(col)
                seen.add(col)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def production_decision(row: dict[str, str]) -> tuple[str, float, str, str]:
    risk = fnum(row.get("risk_score_001G"))
    n_points = fnum(row.get("n_points"))

    center_false = fnum(row.get("center_blob_false_score_001N"))
    micro_keep = fnum(row.get("micro_keep_score_001O"))
    micro_false = fnum(row.get("micro_false_score_001O"))
    micro_dist = fnum(row.get("micro_distance_med_001O"), 999.0)

    # Très bon micro-blob, proche du centre, contexte non massif.
    if (
        micro_keep >= 75
        and micro_false <= 10
        and micro_dist <= 7.0
        and center_false <= 35
    ):
        return (
            "keep",
            0.86,
            "strong_centered_micro_ball",
            "Micro-blob fort, centré, et pas accroché à un gros objet.",
        )

    # Bon micro-blob même si le masque centre est large, mais seulement si la trajectoire est facile.
    # Cela récupère les cas propres où 001N attrape table/ligne autour de la balle.
    if (
        micro_keep >= 90
        and micro_false <= 8
        and micro_dist <= 7.0
        and risk <= 25
    ):
        return (
            "keep",
            0.82,
            "clean_micro_low_trajectory_risk",
            "Micro-blob très fort et trajectoire peu risquée.",
        )

    # Fausse piste évidente : pas de micro-balle et gros objet centré.
    # On évite de rejeter les segments longs/partiels : beaucoup de points => review.
    if (
        center_false >= 85
        and micro_keep <= 5
        and micro_false >= 50
        and n_points < 25
    ):
        return (
            "reject",
            0.80,
            "large_object_no_micro_ball",
            "Gros objet centré, micro-blob absent, segment court.",
        )

    # Fausse piste probable : trajectoire risquée + micro seulement moyen + pas de gros warning centre.
    # Typiquement les petites zones brillantes qui ressemblent à une balle mais bougent comme un artefact.
    if (
        risk >= 45
        and micro_keep < 75
        and micro_false <= 15
        and center_false <= 35
    ):
        return (
            "reject",
            0.72,
            "risky_trajectory_weak_micro",
            "Trajectoire risquée avec micro-blob insuffisant.",
        )

    # Très fort micro, mais trop loin du centre ou trop contradictoire : ne pas garder automatiquement.
    if micro_keep >= 85 and micro_false <= 10:
        return (
            "review",
            0.66,
            "strong_micro_but_context_uncertain",
            "Micro-blob fort, mais contexte ou trajectoire pas assez sûr.",
        )

    # Gros objet probable : review plutôt que rejet agressif.
    if center_false >= 85:
        return (
            "review",
            0.64,
            "large_object_warning",
            "Gros objet centré probable, review nécessaire.",
        )

    # Micro suspect : review si pas assez de signaux pour rejet.
    if micro_false >= 45 and micro_keep <= 40:
        return (
            "review",
            0.60,
            "micro_suspect_review",
            "Micro-blob suspect, mais rejet non certain.",
        )

    # Trajectoire risquée : review.
    if risk >= 55:
        return (
            "review",
            0.58,
            "trajectory_risk_review",
            "Risque trajectoire élevé, mais pas assez de preuve pour rejet.",
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
        action, confidence, reason, notes = production_decision(row)

        item = dict(row)
        item["decision_001U"] = action
        item["decision_confidence_001U"] = f"{confidence:.2f}"
        item["decision_reason_001U"] = reason
        item["decision_notes_001U"] = notes
        out.append(item)

    return out


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_action: dict[str, int] = {}
    by_reason: dict[str, int] = {}

    for row in rows:
        action = row.get("decision_001U", "")
        reason = row.get("decision_reason_001U", "")

        by_action[action] = by_action.get(action, 0) + 1
        by_reason[reason] = by_reason.get(reason, 0) + 1

    return {
        "count": len(rows),
        "by_action": dict(sorted(by_action.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "rule": "production_arbiter_001U",
        "inputs": [
            "risk_score_001G",
            "n_points",
            "center_blob_false_score_001N",
            "micro_keep_score_001O",
            "micro_false_score_001O",
            "micro_distance_med_001O",
        ],
    }


def html_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []

    for row in rows:
        cls = html.escape(row.get("decision_001U", ""))
        cells = "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
        body.append(f'<tr class="{cls}">{cells}</tr>')

    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {"decision": k, "count": str(v)}
        for k, v in summary["by_action"].items()
    ]

    sections = []

    for action in ACTIONS:
        action_rows = [
            row for row in rows
            if row.get("decision_001U") == action
        ]

        action_rows = sorted(
            action_rows,
            key=lambda r: (
                -fnum(r.get("decision_confidence_001U")),
                r.get("review_id", ""),
            ),
        )

        sections.append(
            f"""
<section>
<h2>{html.escape(action)} · {len(action_rows)}</h2>
{html_table(action_rows, [
    "review_id",
    "clip_id",
    "segment_name",
    "decision_confidence_001U",
    "decision_reason_001U",
    "risk_score_001G",
    "n_points",
    "center_blob_false_score_001N",
    "micro_keep_score_001O",
    "micro_false_score_001O",
    "micro_distance_med_001O",
    "micro_guess_001O",
])}
</section>
"""
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux production arbiter 001U</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
th,td{{border-bottom:1px solid #2b303b;padding:7px 8px;vertical-align:top;font-size:13px}}
th{{background:#20242e;text-align:left;position:sticky;top:0}}
.keep td{{background:rgba(100,255,150,.045)}}
.review td{{background:rgba(255,200,80,.06)}}
.reject td{{background:rgba(255,80,80,.08)}}
.muted{{color:#9ea7b8}}
</style>
</head>
<body>
<h1>TTFlux production arbiter 001U</h1>
<section>
<h2>Résumé</h2>
<p class="muted">
Arbitre production : ne dépend pas des labels humains ni des scores 001J.
</p>
{html_table(summary_rows, ["decision", "count"])}
</section>
{''.join(sections)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="runs/batch_001E/micro_blob_features_001T2.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/arbiter_001U")
    args = parser.parse_args()

    features = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(features)
    enriched = enrich(rows)
    summary = summarize(enriched)

    write_csv(out_dir / "arbiter_001U_all.csv", enriched)

    for action in ACTIONS:
        split = [
            row for row in enriched
            if row.get("decision_001U") == action
        ]
        write_csv(out_dir / f"arbiter_001U_{action}.csv", split)

    (out_dir / "arbiter_001U_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_html(out_dir / "arbiter_001U.html", enriched, summary)

    print(f"[001U] rows      : {len(enriched)}")
    print(f"[001U] decisions : {summary['by_action']}")
    print(f"[001U] out dir   : {out_dir}")
    print(f"[001U] wrote     : {out_dir / 'arbiter_001U.html'}")


if __name__ == "__main__":
    main()
