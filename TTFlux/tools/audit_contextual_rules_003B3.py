from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
from pathlib import Path
from typing import Any


KIN_FEATURES = [
    "n_points",
    "duration_frames",
    "density_points_per_frame",
    "trajectory_length_px",
    "displacement_px",
    "tortuosity",
    "x_range_kin",
    "y_range_kin",
    "bbox_area_kin",
    "median_speed",
    "p90_speed",
    "max_speed",
    "speed_cv",
    "median_accel",
    "p90_accel",
    "max_accel",
    "median_turn_deg",
    "p90_turn_deg",
    "sharp_turn_ratio",
    "stop_ratio",
    "jump_ratio",
    "endpoint_gap_ratio",
]

TABLE_FEATURES = [
    "point_inside_table_quad_ratio",
    "point_distance_to_table_px_median",
    "point_distance_to_table_px_p90",
    "point_distance_to_net_px_median",
    "u_min",
    "u_max",
    "v_min",
    "v_max",
    "u_range",
    "v_range",
    "trajectory_crosses_net_line",
    "trajectory_crosses_table_bounds",
    "trajectory_side_flips",
    "trajectory_side_consistency",
    "below_table_risk",
    "near_table_edge_ratio",
    "far_from_table_ratio",
    "table_context_score_003B2",
]

ALL_FEATURES = KIN_FEATURES + TABLE_FEATURES


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value or "").strip().replace(",", ".")
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
        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in csv.DictReader(f, dialect=dialect)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "review_id",
        "target_class",
        "final_decision_002B",
        "human_decision_002A",
        "clip_id_003B2",
        "segment_name",
        *KIN_FEATURES,
        "has_table_model",
        *TABLE_FEATURES,
    ]

    cols = []
    seen = set()

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


def key_row(row: dict[str, str]) -> str:
    rid = str(row.get("review_id", "")).strip()
    if rid:
        return rid
    return str(row.get("segment_name", "")).strip()


def merge_rows(kin_rows: list[dict[str, str]], table_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    table_by_key = {key_row(r): r for r in table_rows if key_row(r)}
    out = []

    for krow in kin_rows:
        key = key_row(krow)
        trow = table_by_key.get(key, {})

        row: dict[str, Any] = dict(krow)

        for k, v in trow.items():
            if k not in row or row.get(k, "") == "":
                row[k] = v
            else:
                row[f"table_{k}"] = v

        if not row.get("target_class") and trow.get("target_class"):
            row["target_class"] = trow["target_class"]

        out.append(row)

    return out


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0

    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return xs[lo]

    t = pos - lo
    return xs[lo] * (1 - t) + xs[hi] * t


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def numeric_values(rows: list[dict[str, Any]], feature: str) -> list[float]:
    vals = []

    for row in rows:
        v = fnum(row.get(feature), None)
        if v is not None:
            vals.append(float(v))

    return vals


def make_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conditions = []
    quantiles = [0.20, 0.35, 0.50, 0.65, 0.80]

    for feature in ALL_FEATURES:
        vals = numeric_values(rows, feature)

        if len(vals) < 5 or len(set(round(x, 6) for x in vals)) < 3:
            continue

        thresholds = sorted(set(round(percentile(vals, q), 6) for q in quantiles))

        for t in thresholds:
            conditions.append({"feature": feature, "op": ">=", "threshold": t})
            conditions.append({"feature": feature, "op": "<=", "threshold": t})

    return conditions


def pass_condition(row: dict[str, Any], cond: dict[str, Any]) -> bool:
    value = fnum(row.get(cond["feature"]), None)

    if value is None:
        return False

    if cond["op"] == ">=":
        return value >= float(cond["threshold"])

    if cond["op"] == "<=":
        return value <= float(cond["threshold"])

    return False


def pass_rule(row: dict[str, Any], conditions: list[dict[str, Any]]) -> bool:
    return all(pass_condition(row, c) for c in conditions)


def condition_text(c: dict[str, Any]) -> str:
    return f'{c["feature"]} {c["op"]} {float(c["threshold"]):g}'


def rule_text(conditions: list[dict[str, Any]]) -> str:
    return " AND ".join(condition_text(c) for c in conditions)


def evaluate_rule(rows: list[dict[str, Any]], conditions: list[dict[str, Any]]) -> dict[str, Any]:
    caught = [r for r in rows if pass_rule(r, conditions)]

    buckets = {
        "human_reject": [],
        "human_partial": [],
        "keep": [],
        "auto_reject": [],
        "auto_review": [],
        "other": [],
    }

    for r in caught:
        target = str(r.get("target_class", "other"))
        buckets.setdefault(target, []).append(str(r.get("review_id", "")))

    hr = len(buckets["human_reject"])
    ar = len(buckets["auto_reject"])
    hp = len(buckets["human_partial"])
    keep = len(buckets["keep"])
    av = len(buckets["auto_review"])

    score = (
        7.0 * hr
        + 2.5 * ar
        - 14.0 * keep
        - 10.0 * hp
        - 0.45 * av
        - 0.15 * max(0, len(caught) - hr - ar)
    )

    if keep > 0:
        reco = "unsafe_catches_keep"
    elif hp > 0:
        reco = "unsafe_catches_human_partial"
    elif hr >= 2 and ar >= 1 and av <= 3:
        reco = "candidate_contextual_reject"
    elif hr >= 2 and av <= 3:
        reco = "candidate_human_reject_filter"
    elif hr >= 1 and ar >= 1 and av <= 4:
        reco = "interesting_reject_filter"
    elif hr >= 1:
        reco = "interesting_but_broad"
    else:
        reco = "not_useful"

    return {
        "score": round(score, 6),
        "rule": rule_text(conditions),
        "features": ",".join(c["feature"] for c in conditions),
        "condition_count": len(conditions),
        "caught_total": len(caught),
        "caught_ids": ",".join(str(r.get("review_id", "")) for r in caught),
        "human_reject_caught": hr,
        "human_reject_ids": ",".join(buckets["human_reject"]),
        "auto_reject_caught": ar,
        "auto_reject_ids": ",".join(buckets["auto_reject"]),
        "human_partial_caught": hp,
        "human_partial_ids": ",".join(buckets["human_partial"]),
        "keep_caught": keep,
        "keep_ids": ",".join(buckets["keep"]),
        "auto_review_caught": av,
        "auto_review_ids": ",".join(buckets["auto_review"]),
        "recommendation": reco,
    }


def search_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conditions = make_conditions(rows)

    # Étape 1 : conditions seules.
    singles = []

    for cond in conditions:
        res = evaluate_rule(rows, [cond])

        if res["recommendation"] == "not_useful":
            continue

        if res["human_reject_caught"] + res["auto_reject_caught"] <= 0:
            continue

        res["_conditions"] = [cond]
        singles.append(res)

    singles = sorted(
        singles,
        key=lambda r: (
            -float(r["score"]),
            int(r["keep_caught"]),
            int(r["human_partial_caught"]),
            int(r["auto_review_caught"]),
            -int(r["human_reject_caught"]),
            -int(r["auto_reject_caught"]),
        ),
    )

    # On limite le pool pour éviter l'explosion combinatoire.
    base_conditions = []
    seen_cond = set()

    for r in singles[:120]:
        c = r["_conditions"][0]
        txt = condition_text(c)

        if txt not in seen_cond:
            seen_cond.add(txt)
            base_conditions.append(c)

    results = []

    for r in singles:
        r.pop("_conditions", None)
        results.append(r)

    for k in [2, 3]:
        for combo in itertools.combinations(base_conditions, k):
            features = [c["feature"] for c in combo]

            # Pas deux seuils sur la même feature dans une règle candidate.
            if len(set(features)) != len(features):
                continue

            res = evaluate_rule(rows, list(combo))

            if res["recommendation"] == "not_useful":
                continue

            if res["human_reject_caught"] + res["auto_reject_caught"] <= 0:
                continue

            results.append(res)

    # Déduplication.
    dedup = []
    seen_rules = set()

    for r in results:
        if r["rule"] in seen_rules:
            continue
        seen_rules.add(r["rule"])
        dedup.append(r)

    dedup = sorted(
        dedup,
        key=lambda r: (
            -float(r["score"]),
            int(r["keep_caught"]),
            int(r["human_partial_caught"]),
            int(r["auto_review_caught"]),
            -int(r["human_reject_caught"]),
            -int(r["auto_reject_caught"]),
            int(r["caught_total"]),
            int(r["condition_count"]),
        ),
    )

    for i, r in enumerate(dedup, start=1):
        r["rank"] = i

    return dedup


def means_by_target(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    targets = sorted(set(str(r.get("target_class", "")) for r in rows))
    out: dict[str, dict[str, float]] = {}

    for target in targets:
        subset = [r for r in rows if str(r.get("target_class", "")) == target]
        out[target] = {}

        for feature in ALL_FEATURES:
            vals = numeric_values(subset, feature)
            out[target][feature] = round(mean(vals), 6) if vals else 0.0

    return out


def summarize(rows: list[dict[str, Any]], rules: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, int] = {}
    by_reco: dict[str, int] = {}

    for r in rows:
        target = str(r.get("target_class", ""))
        by_target[target] = by_target.get(target, 0) + 1

    for r in rules:
        reco = str(r.get("recommendation", ""))
        by_reco[reco] = by_reco.get(reco, 0) + 1

    safe_rules = [
        r for r in rules
        if int(r["keep_caught"]) == 0
        and int(r["human_partial_caught"]) == 0
        and int(r["human_reject_caught"]) + int(r["auto_reject_caught"]) > 0
    ]

    return {
        "rows": len(rows),
        "by_target_class": dict(sorted(by_target.items())),
        "rule_count": len(rules),
        "by_recommendation": dict(sorted(by_reco.items())),
        "safe_rule_count": len(safe_rules),
        "top_safe_rules": safe_rules[:30],
        "top_rules": rules[:50],
        "means_by_target": means_by_target(rows),
        "warning": "003B3 is diagnostic. Do not harden arbiter until visual review confirms caught IDs.",
    }


def write_rules_csv(path: Path, rules: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "rank",
        "score",
        "recommendation",
        "rule",
        "features",
        "condition_count",
        "caught_total",
        "caught_ids",
        "human_reject_caught",
        "human_reject_ids",
        "auto_reject_caught",
        "auto_reject_ids",
        "human_partial_caught",
        "human_partial_ids",
        "keep_caught",
        "keep_ids",
        "auto_review_caught",
        "auto_review_ids",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rules:
            writer.writerow({c: r.get(c, "") for c in cols})


def write_html(path: Path, rows: list[dict[str, Any]], rules: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    target_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["by_target_class"].items()
    )

    reco_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["by_recommendation"].items()
    )

    rule_rows = []

    for r in rules[:100]:
        cls = str(r.get("recommendation", "")).replace("_", "-")

        rule_rows.append(f"""
<tr class="{html.escape(cls)}">
<td>{html.escape(str(r.get("rank", "")))}</td>
<td>{html.escape(str(r.get("score", "")))}</td>
<td>{html.escape(str(r.get("recommendation", "")))}</td>
<td><code>{html.escape(str(r.get("rule", "")))}</code></td>
<td>{html.escape(str(r.get("human_reject_ids", "")))}</td>
<td>{html.escape(str(r.get("auto_reject_ids", "")))}</td>
<td>{html.escape(str(r.get("human_partial_ids", "")))}</td>
<td>{html.escape(str(r.get("keep_ids", "")))}</td>
<td>{html.escape(str(r.get("auto_review_ids", "")))}</td>
<td>{html.escape(str(r.get("caught_ids", "")))}</td>
</tr>
""")

    segment_rows = []

    for r in rows:
        cls = str(r.get("target_class", "")).replace("_", "-")
        table_model = str(r.get("has_table_model", ""))

        if table_model == "0":
            cls += " missing-table"

        segment_rows.append(f"""
<tr class="{html.escape(cls)}">
<td>{html.escape(str(r.get("review_id", "")))}</td>
<td>{html.escape(str(r.get("target_class", "")))}</td>
<td>{html.escape(str(r.get("clip_id_003B2", "")))}</td>
<td><code>{html.escape(str(r.get("segment_name", "")))}</code></td>
<td>{html.escape(str(r.get("n_points", "")))}</td>
<td>{html.escape(str(r.get("duration_frames", "")))}</td>
<td>{html.escape(str(r.get("median_accel", "")))}</td>
<td>{html.escape(str(r.get("sharp_turn_ratio", "")))}</td>
<td>{html.escape(str(r.get("has_table_model", "")))}</td>
<td>{html.escape(str(r.get("point_inside_table_quad_ratio", "")))}</td>
<td>{html.escape(str(r.get("far_from_table_ratio", "")))}</td>
<td>{html.escape(str(r.get("below_table_risk", "")))}</td>
<td>{html.escape(str(r.get("table_context_score_003B2", "")))}</td>
</tr>
""")

    mean_sections = []

    for target, feats in summary["means_by_target"].items():
        body = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in feats.items()
        )

        mean_sections.append(f"""
<section>
<h2>Moyennes · {html.escape(target)}</h2>
<table>
<thead><tr><th>feature</th><th>mean</th></tr></thead>
<tbody>{body}</tbody>
</table>
</section>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux contextual audit 003B3</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}}
th{{background:#20242e;position:sticky;top:0}}
code,pre{{color:#dce4ff}}
pre{{white-space:pre-wrap;word-break:break-word;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.human-reject td{{background:rgba(255,80,80,.09)}}
.human-partial td{{background:rgba(255,200,80,.08)}}
.keep td{{background:rgba(100,255,150,.055)}}
.auto-reject td{{background:rgba(255,80,80,.045)}}
.auto-review td{{background:rgba(150,170,255,.045)}}
.missing-table td{{opacity:.55}}
.candidate-contextual-reject td,.candidate-human-reject-filter td,.interesting-reject-filter td{{background:rgba(100,255,150,.055)}}
.unsafe-catches-keep td,.unsafe-catches-human-partial td{{background:rgba(255,80,80,.10)}}
.interesting-but-broad td{{background:rgba(255,200,80,.06)}}
.warn{{color:#ffd37a}}
</style>
</head>
<body>
<h1>TTFlux · contextual audit 003B3</h1>

<section>
<h2>Résumé</h2>
<table>
<tr><th>rows</th><td>{summary["rows"]}</td></tr>
<tr><th>rule_count</th><td>{summary["rule_count"]}</td></tr>
<tr><th>safe_rule_count</th><td>{summary["safe_rule_count"]}</td></tr>
<tr><th>warning</th><td class="warn">{html.escape(summary["warning"])}</td></tr>
</table>

<h3>Classes</h3>
<table>
<thead><tr><th>target</th><th>count</th></tr></thead>
<tbody>{target_rows}</tbody>
</table>

<h3>Recommandations règles</h3>
<table>
<thead><tr><th>recommendation</th><th>count</th></tr></thead>
<tbody>{reco_rows}</tbody>
</table>
</section>

<section>
<h2>Règles candidates · top 100</h2>
<table>
<thead>
<tr>
<th>rank</th><th>score</th><th>reco</th><th>rule</th>
<th>human reject</th><th>auto reject</th><th>human partial</th><th>keep</th><th>auto review</th><th>all caught</th>
</tr>
</thead>
<tbody>{''.join(rule_rows)}</tbody>
</table>
</section>

<section>
<h2>Segments fusionnés · cinématique + table</h2>
<table>
<thead>
<tr>
<th>ID</th><th>target</th><th>clip</th><th>segment</th>
<th>n</th><th>dur</th><th>med accel</th><th>sharp turn</th>
<th>table</th><th>inside</th><th>far</th><th>below</th><th>table score</th>
</tr>
</thead>
<tbody>{''.join(segment_rows)}</tbody>
</table>
</section>

{''.join(mean_sections)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kin-csv", default="runs/batch_001E/kinematic_audit_002G2/kinematic_features_002G2.csv")
    parser.add_argument("--table-csv", default="runs/batch_001E/table_context_003B2/table_context_features_003B2.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/contextual_audit_003B3")
    args = parser.parse_args()

    kin_csv = Path(args.kin_csv)
    table_csv = Path(args.table_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kin_rows = read_csv(kin_csv)
    table_rows = read_csv(table_csv)

    rows = merge_rows(kin_rows, table_rows)
    rules = search_rules(rows)
    summary = summarize(rows, rules)

    merged_path = out_dir / "contextual_features_003B3.csv"
    rules_path = out_dir / "contextual_candidate_rules_003B3.csv"
    summary_path = out_dir / "contextual_audit_summary_003B3.json"
    html_path = out_dir / "contextual_audit_003B3.html"

    write_csv(merged_path, rows)
    write_rules_csv(rules_path, rules)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html(html_path, rows, rules, summary)

    print(f"[003B3] rows            : {summary['rows']}")
    print(f"[003B3] rules           : {summary['rule_count']}")
    print(f"[003B3] safe rules      : {summary['safe_rule_count']}")
    print(f"[003B3] by target       : {summary['by_target_class']}")
    print(f"[003B3] by reco         : {summary['by_recommendation']}")
    print(f"[003B3] out dir         : {out_dir}")
    print(f"[003B3] html            : {html_path}")


if __name__ == "__main__":
    main()
