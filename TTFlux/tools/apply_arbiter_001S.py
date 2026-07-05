from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


ACTIONS = ["keep", "review", "reject", "ignore"]


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
        "decision_001S",
        "decision_confidence_001S",
        "decision_reason_001S",
        "decision_notes_001S",
        "human_label_fr",
        "target_class",
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


def fusion_guarded_decision(row: dict[str, str], respect_ignore: bool = False) -> tuple[str, float, str, str]:
    false_j = fnum(row.get("false_track_score_001J"))
    frag_j = fnum(row.get("fragmentation_score_001J"))
    keep_j = fnum(row.get("keep_score_001J"))

    center_false = fnum(row.get("center_blob_false_score_001N"))
    micro_keep = fnum(row.get("micro_keep_score_001O"))
    micro_false = fnum(row.get("micro_false_score_001O"))
    micro_dist = fnum(row.get("micro_distance_med_001O"), 999.0)

    target = row.get("target_class", "")

    if respect_ignore and target == "ignore":
        return (
            "ignore",
            0.0,
            "human_ignore",
            "Segment marqué incertain dans le goldset.",
        )

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

    if center_false >= 85 and micro_keep < 50:
        return (
            "review",
            0.68,
            "large_object_warning",
            "Gros objet centré probable, mais pas assez sûr pour rejet automatique.",
        )

    if center_false >= 85 and micro_false >= 45:
        return (
            "review",
            0.64,
            "center_warning_micro_suspect",
            "Centre suspect et micro-blob faible.",
        )

    if frag_j >= 60:
        return (
            "review",
            0.66,
            "fragmented_track",
            "Trajectoire fragmentée : utile mais pas automatiquement propre.",
        )

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


def enrich(rows: list[dict[str, str]], respect_ignore: bool) -> list[dict[str, str]]:
    out = []

    for row in rows:
        action, confidence, reason, notes = fusion_guarded_decision(row, respect_ignore=respect_ignore)

        item = dict(row)
        item["decision_001S"] = action
        item["decision_confidence_001S"] = f"{confidence:.2f}"
        item["decision_reason_001S"] = reason
        item["decision_notes_001S"] = notes
        out.append(item)

    return out


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_action: dict[str, int] = {}
    by_reason: dict[str, int] = {}

    for row in rows:
        action = row.get("decision_001S", "")
        reason = row.get("decision_reason_001S", "")

        by_action[action] = by_action.get(action, 0) + 1
        by_reason[reason] = by_reason.get(reason, 0) + 1

    return {
        "count": len(rows),
        "by_action": dict(sorted(by_action.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "rule": "001P_fusion_guarded",
        "version": "001S",
    }


def html_table(rows: list[dict[str, str]], columns: list[str], class_col: str = "decision_001S") -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []

    for row in rows:
        cls = html.escape(row.get(class_col, ""))
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
            if row.get("decision_001S") == action
        ]

        action_rows = sorted(
            action_rows,
            key=lambda r: (
                -fnum(r.get("decision_confidence_001S")),
                r.get("review_id", ""),
            ),
        )

        sections.append(
            f"""
<section>
<h2>{html.escape(action)} · {len(action_rows)}</h2>
{html_table(action_rows, [
    "review_id",
    "human_label_fr",
    "target_class",
    "decision_confidence_001S",
    "decision_reason_001S",
    "clip_id",
    "segment_name",
    "false_track_score_001J",
    "keep_score_001J",
    "center_blob_false_score_001N",
    "micro_keep_score_001O",
    "micro_false_score_001O",
    "micro_guess_001O",
])}
</section>
"""
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux arbiter 001S</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
th,td{{border-bottom:1px solid #2b303b;padding:7px 8px;vertical-align:top;font-size:13px}}
th{{background:#20242e;text-align:left;position:sticky;top:0}}
.keep td{{background:rgba(100,255,150,.045)}}
.review td{{background:rgba(255,200,80,.06)}}
.reject td{{background:rgba(255,80,80,.08)}}
.ignore td{{background:rgba(150,170,255,.045)}}
.muted{{color:#9ea7b8}}
</style>
</head>
<body>
<h1>TTFlux arbiter 001S</h1>
<section>
<h2>Résumé</h2>
<p class="muted">Application autonome de la règle 001P_fusion_guarded.</p>
{html_table(summary_rows, ["decision", "count"], class_col="decision")}
</section>
{''.join(sections)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="runs/batch_001E/micro_blob_features_001O.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/arbiter_001S")
    parser.add_argument("--respect-ignore", action="store_true")
    args = parser.parse_args()

    features = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(features)
    enriched = enrich(rows, respect_ignore=args.respect_ignore)
    summary = summarize(enriched)

    write_csv(out_dir / "arbiter_001S_all.csv", enriched)

    for action in ACTIONS:
        split = [
            row for row in enriched
            if row.get("decision_001S") == action
        ]

        split = sorted(
            split,
            key=lambda r: (
                -fnum(r.get("decision_confidence_001S")),
                r.get("review_id", ""),
            ),
        )

        write_csv(out_dir / f"arbiter_001S_{action}.csv", split)

    (out_dir / "arbiter_001S_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_html(out_dir / "arbiter_001S.html", enriched, summary)

    print(f"[001S] rows      : {len(enriched)}")
    print(f"[001S] decisions : {summary['by_action']}")
    print(f"[001S] out dir   : {out_dir}")
    print(f"[001S] wrote     : {out_dir / 'arbiter_001S.html'}")


if __name__ == "__main__":
    main()
