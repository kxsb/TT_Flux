from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
from pathlib import Path
from typing import Any


FEATURES = [
    "risk_score_001G",
    "center_blob_false_score_001N",
    "micro_keep_score_001O",
    "micro_false_score_001O",
    "micro_distance_med_001O",
]

FEATURE_LABELS = {
    "risk_score_001G": "risk",
    "center_blob_false_score_001N": "center_false",
    "micro_keep_score_001O": "micro_keep",
    "micro_false_score_001O": "micro_false",
    "micro_distance_med_001O": "micro_dist",
}

THRESHOLDS = {
    "risk_score_001G": [0, 10, 25, 35, 45, 55, 65, 70],
    "center_blob_false_score_001N": [20, 35, 50, 74, 85, 90],
    "micro_keep_score_001O": [0, 5, 20, 40, 55, 70, 80, 90],
    "micro_false_score_001O": [0, 6, 10, 18, 22, 30, 40, 50],
    "micro_distance_med_001O": [5, 7, 8, 9, 10, 11, 12],
}


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = []
    seen = set()

    preferred = [
        "rank",
        "score",
        "rule",
        "caught_total",
        "caught_ids",
        "human_reject_caught",
        "human_reject_ids",
        "human_partial_caught",
        "human_partial_ids",
        "keep_caught",
        "keep_ids",
        "auto_review_caught",
        "auto_review_ids",
        "recommendation",
    ]

    for c in preferred:
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


def target_class(row: dict[str, str]) -> str:
    human_decision = row.get("human_decision_002A", "")
    final_decision = row.get("final_decision_002B", "")
    source = row.get("final_source_002B", "")

    if human_decision == "reject_human_verified":
        return "human_reject"

    if human_decision == "review_partial_human_verified":
        return "human_partial"

    if final_decision == "keep":
        return "keep"

    if final_decision == "reject" and source != "human_002A":
        return "auto_reject"

    if final_decision == "review":
        return "auto_review"

    return "other"


def cond_text(feature: str, op: str, threshold: float) -> str:
    return f"{FEATURE_LABELS.get(feature, feature)} {op} {threshold:g}"


def pass_condition(row: dict[str, str], condition: tuple[str, str, float]) -> bool:
    feature, op, threshold = condition
    value = fnum(row.get(feature), 0.0)

    if op == ">=":
        return value >= threshold

    if op == "<=":
        return value <= threshold

    raise ValueError(op)


def pass_rule(row: dict[str, str], conditions: list[tuple[str, str, float]]) -> bool:
    return all(pass_condition(row, c) for c in conditions)


def rule_to_text(conditions: list[tuple[str, str, float]]) -> str:
    return " AND ".join(cond_text(*c) for c in conditions)


def build_candidate_conditions() -> list[tuple[str, str, float]]:
    conditions = []

    # Rejet potentiel : risque haut, gros objet, micro faible/suspect, distance micro élevée.
    for t in THRESHOLDS["risk_score_001G"]:
        conditions.append(("risk_score_001G", ">=", float(t)))

    for t in THRESHOLDS["center_blob_false_score_001N"]:
        conditions.append(("center_blob_false_score_001N", ">=", float(t)))

    for t in THRESHOLDS["micro_keep_score_001O"]:
        conditions.append(("micro_keep_score_001O", "<=", float(t)))

    for t in THRESHOLDS["micro_false_score_001O"]:
        conditions.append(("micro_false_score_001O", ">=", float(t)))

    for t in THRESHOLDS["micro_distance_med_001O"]:
        conditions.append(("micro_distance_med_001O", ">=", float(t)))

    return conditions


def evaluate_rule(rows: list[dict[str, str]], conditions: list[tuple[str, str, float]]) -> dict[str, Any]:
    caught = [r for r in rows if pass_rule(r, conditions)]

    buckets: dict[str, list[str]] = {
        "human_reject": [],
        "human_partial": [],
        "keep": [],
        "auto_reject": [],
        "auto_review": [],
        "other": [],
    }

    for row in caught:
        buckets.setdefault(target_class(row), []).append(row.get("review_id", ""))

    human_reject = len(buckets["human_reject"])
    human_partial = len(buckets["human_partial"])
    keep = len(buckets["keep"])
    auto_review = len(buckets["auto_review"])
    auto_reject = len(buckets["auto_reject"])

    # Score conservateur :
    # - attraper un rejet humain est bien
    # - attraper un keep est interdit
    # - attraper un partial humain est mauvais
    # - attraper beaucoup de review auto non vérifiées est risqué
    score = (
        human_reject * 5.0
        + auto_reject * 1.0
        - human_partial * 6.0
        - keep * 12.0
        - auto_review * 0.45
    )

    if keep > 0:
        recommendation = "unsafe_catches_keep"
    elif human_partial > 0:
        recommendation = "unsafe_catches_human_partial"
    elif human_reject >= 2 and auto_review <= 2:
        recommendation = "candidate_strict_reject_rule"
    elif human_reject >= 1 and auto_review <= 1:
        recommendation = "candidate_narrow_reject_rule"
    elif human_reject >= 1:
        recommendation = "interesting_but_too_broad"
    else:
        recommendation = "not_useful"

    return {
        "score": round(score, 4),
        "rule": rule_to_text(conditions),
        "caught_total": len(caught),
        "caught_ids": ",".join(r.get("review_id", "") for r in caught),
        "human_reject_caught": human_reject,
        "human_reject_ids": ",".join(buckets["human_reject"]),
        "human_partial_caught": human_partial,
        "human_partial_ids": ",".join(buckets["human_partial"]),
        "keep_caught": keep,
        "keep_ids": ",".join(buckets["keep"]),
        "auto_review_caught": auto_review,
        "auto_review_ids": ",".join(buckets["auto_review"]),
        "auto_reject_caught": auto_reject,
        "auto_reject_ids": ",".join(buckets["auto_reject"]),
        "recommendation": recommendation,
    }


def generate_rules(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    base_conditions = build_candidate_conditions()
    results = []

    # Règles à 2 et 3 conditions. Pas plus, sinon on sur-apprend 5 exemples.
    for k in [2, 3]:
        for combo in itertools.combinations(base_conditions, k):
            features = [c[0] for c in combo]

            # Évite deux seuils contradictoires sur la même feature.
            if len(set(features)) < len(features):
                continue

            res = evaluate_rule(rows, list(combo))

            if res["human_reject_caught"] <= 0:
                continue

            if res["recommendation"] == "not_useful":
                continue

            results.append(res)

    results = sorted(
        results,
        key=lambda r: (
            -float(r["score"]),
            int(r["keep_caught"]),
            int(r["human_partial_caught"]),
            int(r["auto_review_caught"]),
            -int(r["human_reject_caught"]),
            int(r["caught_total"]),
        ),
    )

    # Déduplique les règles identiques.
    out = []
    seen = set()

    for r in results:
        key = r["rule"]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    for i, r in enumerate(out, start=1):
        r["rank"] = i

    return out


def feature_vector(row: dict[str, str]) -> list[float]:
    values = [fnum(row.get(f), 0.0) for f in FEATURES]

    # Normalisation grossière pour distances comparables.
    scales = [100.0, 100.0, 100.0, 100.0, 20.0]
    return [v / s for v, s in zip(values, scales)]


def euclid(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest_neighbors(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    human_rows = [
        r for r in rows
        if target_class(r) in ("human_reject", "human_partial")
    ]

    out = []

    for src in human_rows:
        src_vec = feature_vector(src)
        neighbors = []

        for other in rows:
            if other.get("review_id") == src.get("review_id"):
                continue

            d = euclid(src_vec, feature_vector(other))
            neighbors.append((d, other))

        neighbors = sorted(neighbors, key=lambda x: x[0])[:5]

        for rank, (d, other) in enumerate(neighbors, start=1):
            out.append({
                "source_review_id": src.get("review_id", ""),
                "source_target": target_class(src),
                "source_segment": src.get("segment_name", ""),
                "neighbor_rank": rank,
                "neighbor_review_id": other.get("review_id", ""),
                "neighbor_target": target_class(other),
                "neighbor_segment": other.get("segment_name", ""),
                "feature_distance": round(d, 4),
                "risk_src": src.get("risk_score_001G", ""),
                "risk_neighbor": other.get("risk_score_001G", ""),
                "center_src": src.get("center_blob_false_score_001N", ""),
                "center_neighbor": other.get("center_blob_false_score_001N", ""),
                "micro_keep_src": src.get("micro_keep_score_001O", ""),
                "micro_keep_neighbor": other.get("micro_keep_score_001O", ""),
                "micro_false_src": src.get("micro_false_score_001O", ""),
                "micro_false_neighbor": other.get("micro_false_score_001O", ""),
                "micro_dist_src": src.get("micro_distance_med_001O", ""),
                "micro_dist_neighbor": other.get("micro_distance_med_001O", ""),
            })

    return out


def summarize(rows: list[dict[str, str]], rules: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, int] = {}

    for row in rows:
        t = target_class(row)
        by_target[t] = by_target.get(t, 0) + 1

    recs: dict[str, int] = {}

    for rule in rules:
        rec = str(rule.get("recommendation", ""))
        recs[rec] = recs.get(rec, 0) + 1

    top_rules = rules[:10]

    return {
        "rows": len(rows),
        "by_target_class": dict(sorted(by_target.items())),
        "rule_count": len(rules),
        "by_rule_recommendation": dict(sorted(recs.items())),
        "top_rules": top_rules,
        "warning": (
            "Small sample. Do not harden global arbiter unless a rule catches human rejects "
            "without catching human partials or keep segments."
        ),
    }


def write_html(
    path: Path,
    rows: list[dict[str, str]],
    rules: list[dict[str, Any]],
    neighbors: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    target_rows = []
    for row in rows:
        t = target_class(row)
        cls = t.replace("_", "-")

        target_rows.append(f"""
<tr class="{html.escape(cls)}">
<td>{html.escape(row.get("review_id", ""))}</td>
<td>{html.escape(t)}</td>
<td><code>{html.escape(row.get("segment_name", ""))}</code></td>
<td>{html.escape(row.get("final_decision_002B", ""))}</td>
<td>{html.escape(row.get("human_decision_002A", ""))}</td>
<td>{html.escape(row.get("risk_score_001G", ""))}</td>
<td>{html.escape(row.get("center_blob_false_score_001N", ""))}</td>
<td>{html.escape(row.get("micro_keep_score_001O", ""))}</td>
<td>{html.escape(row.get("micro_false_score_001O", ""))}</td>
<td>{html.escape(row.get("micro_distance_med_001O", ""))}</td>
</tr>
""")

    rule_rows = []
    for rule in rules[:40]:
        rec = str(rule.get("recommendation", ""))
        cls = rec.replace("_", "-")

        rule_rows.append(f"""
<tr class="{html.escape(cls)}">
<td>{html.escape(str(rule.get("rank", "")))}</td>
<td>{html.escape(str(rule.get("score", "")))}</td>
<td><code>{html.escape(str(rule.get("rule", "")))}</code></td>
<td>{html.escape(str(rule.get("recommendation", "")))}</td>
<td>{html.escape(str(rule.get("human_reject_ids", "")))}</td>
<td>{html.escape(str(rule.get("human_partial_ids", "")))}</td>
<td>{html.escape(str(rule.get("keep_ids", "")))}</td>
<td>{html.escape(str(rule.get("auto_review_ids", "")))}</td>
<td>{html.escape(str(rule.get("caught_ids", "")))}</td>
</tr>
""")

    neigh_rows = []
    for n in neighbors:
        conflict = (
            n["source_target"] != n["neighbor_target"]
            and n["neighbor_target"] in ("human_partial", "keep")
        )
        cls = "conflict" if conflict else ""

        neigh_rows.append(f"""
<tr class="{cls}">
<td>{html.escape(str(n.get("source_review_id", "")))}</td>
<td>{html.escape(str(n.get("source_target", "")))}</td>
<td>{html.escape(str(n.get("neighbor_rank", "")))}</td>
<td>{html.escape(str(n.get("neighbor_review_id", "")))}</td>
<td>{html.escape(str(n.get("neighbor_target", "")))}</td>
<td>{html.escape(str(n.get("feature_distance", "")))}</td>
<td>{html.escape(str(n.get("risk_src", "")))} / {html.escape(str(n.get("risk_neighbor", "")))}</td>
<td>{html.escape(str(n.get("center_src", "")))} / {html.escape(str(n.get("center_neighbor", "")))}</td>
<td>{html.escape(str(n.get("micro_keep_src", "")))} / {html.escape(str(n.get("micro_keep_neighbor", "")))}</td>
<td>{html.escape(str(n.get("micro_false_src", "")))} / {html.escape(str(n.get("micro_false_neighbor", "")))}</td>
<td>{html.escape(str(n.get("micro_dist_src", "")))} / {html.escape(str(n.get("micro_dist_neighbor", "")))}</td>
</tr>
""")

    summary_rules = "".join(
        f"<li><code>{html.escape(str(r.get('rule', '')))}</code> · {html.escape(str(r.get('recommendation', '')))}</li>"
        for r in summary.get("top_rules", [])[:5]
    )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux human false pattern audit 002C</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
th,td{{border-bottom:1px solid #2b303b;padding:7px 8px;vertical-align:top;font-size:13px}}
th{{background:#20242e;text-align:left;position:sticky;top:0}}
code{{color:#dce4ff}}
.human-reject td{{background:rgba(255,80,80,.08)}}
.human-partial td{{background:rgba(255,200,80,.07)}}
.keep td{{background:rgba(100,255,150,.055)}}
.candidate-strict-reject-rule td,.candidate-narrow-reject-rule td{{background:rgba(100,255,150,.045)}}
.unsafe-catches-human-partial td,.unsafe-catches-keep td,.conflict td{{background:rgba(255,80,80,.08)}}
.interesting-but-too-broad td{{background:rgba(255,200,80,.060)}}
.muted{{color:#9ea7b8}}
.warn{{color:#ffd37a}}
</style>
</head>
<body>
<h1>TTFlux human false pattern audit 002C</h1>

<section>
<h2>Résumé</h2>
<p>Rows : <b>{summary["rows"]}</b></p>
<p>Règles candidates testées : <b>{summary["rule_count"]}</b></p>
<p class="warn">{html.escape(summary["warning"])}</p>
<h3>Top règles brutes</h3>
<ul>{summary_rules}</ul>
</section>

<section>
<h2>Segments et classes utilisées</h2>
<table>
<thead>
<tr>
<th>ID</th><th>target</th><th>segment</th><th>final 002B</th><th>human 002A</th>
<th>risk</th><th>center_false</th><th>micro_keep</th><th>micro_false</th><th>micro_dist</th>
</tr>
</thead>
<tbody>{''.join(target_rows)}</tbody>
</table>
</section>

<section>
<h2>Règles candidates, top 40</h2>
<p class="muted">
Une règle n'est intéressante que si elle attrape des rejets humains sans attraper de keep ni de partial humain.
</p>
<table>
<thead>
<tr>
<th>rank</th><th>score</th><th>rule</th><th>reco</th>
<th>human reject</th><th>human partial</th><th>keep</th><th>auto review</th><th>all caught</th>
</tr>
</thead>
<tbody>{''.join(rule_rows)}</tbody>
</table>
</section>

<section>
<h2>Voisins numériques des segments humains</h2>
<p class="muted">
Si un rejet humain a comme voisin très proche un partial humain ou un keep, les features actuelles ne permettent probablement pas de durcir sans risque.
</p>
<table>
<thead>
<tr>
<th>source</th><th>source target</th><th>rank</th><th>neighbor</th><th>neighbor target</th><th>distance</th>
<th>risk src/neigh</th><th>center src/neigh</th><th>micro_keep src/neigh</th><th>micro_false src/neigh</th><th>micro_dist src/neigh</th>
</tr>
</thead>
<tbody>{''.join(neigh_rows)}</tbody>
</table>
</section>
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-csv", default="runs/batch_001E/final_decisions_002B/final_decisions_002B_all.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/human_pattern_audit_002C")
    args = parser.parse_args()

    final_csv = Path(args.final_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(final_csv)

    rules = generate_rules(rows)
    neighbors = nearest_neighbors(rows)
    summary = summarize(rows, rules)

    write_csv(out_dir / "candidate_reject_rules_002C.csv", rules)
    write_csv(out_dir / "nearest_neighbors_002C.csv", neighbors)

    (out_dir / "human_pattern_audit_002C_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_html(
        out_dir / "human_pattern_audit_002C.html",
        rows=rows,
        rules=rules,
        neighbors=neighbors,
        summary=summary,
    )

    print(f"[002C] rows       : {summary['rows']}")
    print(f"[002C] targets    : {summary['by_target_class']}")
    print(f"[002C] rules      : {summary['rule_count']}")
    print(f"[002C] rule recos : {summary['by_rule_recommendation']}")
    print(f"[002C] out dir    : {out_dir}")
    print(f"[002C] html       : {out_dir / 'human_pattern_audit_002C.html'}")


if __name__ == "__main__":
    main()
