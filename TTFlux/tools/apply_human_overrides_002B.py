from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "review_id",
        "clip_id",
        "segment_idx",
        "segment_name",
        "final_decision_002B",
        "final_confidence_002B",
        "final_source_002B",
        "final_reason_002B",
        "decision_001U",
        "decision_reason_001U",
        "human_decision_002A",
        "human_reason_002A",
        "classification_001Z",
        "human_visible",
        "human_points_total",
        "best_offset_frames",
        "best_median_dist",
        "best_p80_dist",
        "best_hit50",
        "risk_score_001G",
        "center_blob_false_score_001N",
        "micro_keep_score_001O",
        "micro_false_score_001O",
        "micro_distance_med_001O",
        "csv",
        "mp4",
    ]

    cols: list[str] = []
    seen = set()

    for c in preferred:
        if c not in seen:
            cols.append(c)
            seen.add(c)

    for row in rows:
        for k in row:
            if k not in seen:
                cols.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def human_to_final(human_decision: str) -> tuple[str, float, str]:
    if human_decision == "reject_human_verified":
        return "reject", 0.95, "human_verified_reject"

    if human_decision == "keep_human_verified":
        return "keep", 0.95, "human_verified_keep"

    if human_decision == "review_partial_human_verified":
        return "review", 0.88, "human_verified_partial_review"

    if human_decision == "review_lag_or_usable_human_verified":
        return "review", 0.86, "human_verified_lag_or_usable"

    if human_decision == "keep_or_lag_review_human_verified":
        return "review", 0.82, "human_verified_possible_lag"

    if human_decision == "review_insufficient_human_points":
        return "review", 0.70, "human_insufficient_points"

    return "", 0.0, ""


def build_final_rows(
    arbiter_rows: list[dict[str, str]],
    human_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    human_by_id = {
        row.get("review_id", ""): row
        for row in human_rows
        if row.get("review_id", "")
    }

    out: list[dict[str, Any]] = []

    for row in arbiter_rows:
        rid = row.get("review_id", "")
        item: dict[str, Any] = dict(row)

        human = human_by_id.get(rid)

        if human is not None:
            human_decision = human.get("human_decision_002A", "")
            final_decision, conf, source = human_to_final(human_decision)

            item.update({
                "human_decision_002A": human_decision,
                "human_reason_002A": human.get("human_reason_002A", ""),
                "classification_001Z": human.get("classification_001Z", ""),
                "human_visible": human.get("human_visible", ""),
                "human_points_total": human.get("human_points_total", ""),
                "best_offset_frames": human.get("best_offset_frames", ""),
                "best_median_dist": human.get("best_median_dist", ""),
                "best_p80_dist": human.get("best_p80_dist", ""),
                "best_hit50": human.get("best_hit50", ""),
            })

            if final_decision:
                item["final_decision_002B"] = final_decision
                item["final_confidence_002B"] = f"{conf:.2f}"
                item["final_source_002B"] = "human_002A"
                item["final_reason_002B"] = source + " · " + human.get("human_reason_002A", "")
            else:
                item["final_decision_002B"] = row.get("decision_001U", "")
                item["final_confidence_002B"] = row.get("decision_confidence_001U", "")
                item["final_source_002B"] = "arbiter_001U_fallback"
                item["final_reason_002B"] = row.get("decision_reason_001U", "")
        else:
            item["human_decision_002A"] = ""
            item["human_reason_002A"] = ""
            item["classification_001Z"] = ""
            item["final_decision_002B"] = row.get("decision_001U", "")
            item["final_confidence_002B"] = row.get("decision_confidence_001U", "")
            item["final_source_002B"] = "arbiter_001U"
            item["final_reason_002B"] = row.get("decision_reason_001U", "")

        out.append(item)

    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_final: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_final_source: dict[str, int] = {}

    for row in rows:
        decision = str(row.get("final_decision_002B", ""))
        source = str(row.get("final_source_002B", ""))

        by_final[decision] = by_final.get(decision, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1

        k = f"{decision}__{source}"
        by_final_source[k] = by_final_source.get(k, 0) + 1

    return {
        "rows": len(rows),
        "by_final_decision_002B": dict(sorted(by_final.items())),
        "by_source_002B": dict(sorted(by_source.items())),
        "by_final_decision_and_source_002B": dict(sorted(by_final_source.items())),
    }


def write_splits(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    for action in ["keep", "review", "reject", "ignore"]:
        split = [
            row for row in rows
            if row.get("final_decision_002B", "") == action
        ]

        write_csv(out_dir / f"final_decisions_002B_{action}.csv", split)


def write_html(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_html = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["by_final_decision_002B"].items()
    )

    sections = []

    for action in ["keep", "review", "reject", "ignore"]:
        action_rows = [
            row for row in rows
            if row.get("final_decision_002B", "") == action
        ]

        body = []

        for row in action_rows:
            source = str(row.get("final_source_002B", ""))
            src_cls = "human" if source == "human_002A" else "auto"

            body.append(f"""
<tr class="{html.escape(action)} {src_cls}">
<td>{html.escape(str(row.get("review_id", "")))}</td>
<td><code>{html.escape(str(row.get("segment_name", "")))}</code></td>
<td>{html.escape(str(row.get("final_source_002B", "")))}</td>
<td>{html.escape(str(row.get("final_confidence_002B", "")))}</td>
<td>{html.escape(str(row.get("final_reason_002B", "")))}</td>
<td>{html.escape(str(row.get("decision_001U", "")))}</td>
<td>{html.escape(str(row.get("human_decision_002A", "")))}</td>
<td>{html.escape(str(row.get("best_median_dist", "")))}</td>
<td>{html.escape(str(row.get("best_hit50", "")))}</td>
</tr>
""")

        sections.append(f"""
<section>
<h2>{html.escape(action)} · {len(action_rows)}</h2>
<table>
<thead>
<tr>
<th>ID</th>
<th>segment</th>
<th>source finale</th>
<th>confiance</th>
<th>raison finale</th>
<th>001U</th>
<th>002A humain</th>
<th>median humain</th>
<th>hit50 humain</th>
</tr>
</thead>
<tbody>
{''.join(body)}
</tbody>
</table>
</section>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux final decisions 002B</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
th,td{{border-bottom:1px solid #2b303b;padding:7px 8px;vertical-align:top;font-size:13px}}
th{{background:#20242e;text-align:left;position:sticky;top:0}}
.keep td{{background:rgba(100,255,150,.055)}}
.review td{{background:rgba(255,200,80,.060)}}
.reject td{{background:rgba(255,80,80,.080)}}
.ignore td{{background:rgba(150,170,255,.050)}}
.human td{{box-shadow: inset 4px 0 0 rgba(190,120,255,.75)}}
code{{color:#dce4ff}}
.muted{{color:#9ea7b8}}
</style>
</head>
<body>
<h1>TTFlux final decisions 002B</h1>

<section>
<h2>Résumé</h2>
<p class="muted">
Décision finale = arbitre production 001U + surcouche humaine 002A quand disponible.
Les lignes humaines sont marquées par un liseré violet.
</p>
<table>
<thead><tr><th>décision finale 002B</th><th>count</th></tr></thead>
<tbody>{summary_html}</tbody>
</table>
</section>

{''.join(sections)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arbiter-csv", default="runs/batch_001E/arbiter_001V/arbiter_001U_all.csv")
    parser.add_argument("--human-decisions", default="runs/batch_001E/human_decisions_002A/human_segment_decisions_002A.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/final_decisions_002B")
    args = parser.parse_args()

    arbiter_csv = Path(args.arbiter_csv)
    human_decisions = Path(args.human_decisions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arbiter_rows = read_csv(arbiter_csv)
    human_rows = read_csv(human_decisions)

    final_rows = build_final_rows(arbiter_rows, human_rows)
    summary = summarize(final_rows)

    write_csv(out_dir / "final_decisions_002B_all.csv", final_rows)
    write_splits(out_dir, final_rows)

    (out_dir / "final_decisions_002B_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_html(out_dir / "final_decisions_002B.html", final_rows, summary)

    print(f"[002B] rows      : {summary['rows']}")
    print(f"[002B] decisions : {summary['by_final_decision_002B']}")
    print(f"[002B] sources   : {summary['by_source_002B']}")
    print(f"[002B] out dir   : {out_dir}")
    print(f"[002B] html      : {out_dir / 'final_decisions_002B.html'}")


if __name__ == "__main__":
    main()
