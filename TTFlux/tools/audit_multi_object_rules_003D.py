from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
from datetime import datetime
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
    "has_table_model",
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
    "contextual_reject_003B4",
]

PLAYER_FEATURES = [
    "has_player_context_003C",
    "player_motion_area_ratio_003C",
    "player_motion_component_count_003C",
    "player_motion_bbox_area_ratio_003C",
    "inside_player_motion_mask_ratio_003C",
    "near_player_motion_mask_ratio_003C",
    "inside_hand_or_racket_proxy_ratio_003C",
    "near_foot_or_floor_proxy_ratio_003C",
    "occlusion_entry_exit_count_003C",
    "candidate_on_body_risk_003C",
    "player_context_confidence_003C",
]

ALL_FEATURES = list(dict.fromkeys(KIN_FEATURES + TABLE_FEATURES + PLAYER_FEATURES))


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
        "proposed_decision_003B4",
        "contextual_reject_003B4",
        "clip_id_003B2",
        "segment_name",
        *KIN_FEATURES,
        *TABLE_FEATURES,
        *PLAYER_FEATURES,
    ]

    cols = []
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


def feature_group(feature: str) -> str:
    if feature in KIN_FEATURES:
        return "kinematic"

    if feature in TABLE_FEATURES:
        return "table"

    if feature in PLAYER_FEATURES:
        return "player"

    return "other"


def make_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conditions = []
    quantiles = [0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85]

    for feature in ALL_FEATURES:
        vals = numeric_values(rows, feature)

        if len(vals) < 4:
            continue

        rounded_unique = sorted(set(round(x, 6) for x in vals))

        if len(rounded_unique) <= 1:
            continue

        if all(x in [0, 1] for x in rounded_unique):
            thresholds = [0.5]
        else:
            thresholds = sorted(set(round(percentile(vals, q), 6) for q in quantiles))

        for t in thresholds:
            conditions.append({
                "feature": feature,
                "group": feature_group(feature),
                "op": ">=",
                "threshold": t,
            })
            conditions.append({
                "feature": feature,
                "group": feature_group(feature),
                "op": "<=",
                "threshold": t,
            })

    # Quelques règles de bon sens explicites.
    explicit = [
        ("contextual_reject_003B4", ">=", 0.5),
        ("candidate_on_body_risk_003C", ">=", 0.65),
        ("candidate_on_body_risk_003C", ">=", 0.75),
        ("inside_hand_or_racket_proxy_ratio_003C", ">=", 0.75),
        ("near_player_motion_mask_ratio_003C", ">=", 0.90),
        ("point_inside_table_quad_ratio", "<=", 0.10),
        ("table_context_score_003B2", "<=", 0.50),
        ("far_from_table_ratio", ">=", 0.50),
        ("median_accel", ">=", 5.50),
    ]

    for feature, op, threshold in explicit:
        if feature in ALL_FEATURES:
            conditions.append({
                "feature": feature,
                "group": feature_group(feature),
                "op": op,
                "threshold": threshold,
            })

    dedup = []
    seen = set()

    for c in conditions:
        key = (c["feature"], c["op"], float(c["threshold"]))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)

    return dedup


def pass_condition(row: dict[str, Any], cond: dict[str, Any]) -> bool:
    value = fnum(row.get(cond["feature"]), None)

    if value is None:
        return False

    threshold = float(cond["threshold"])

    if cond["op"] == ">=":
        return value >= threshold

    if cond["op"] == "<=":
        return value <= threshold

    return False


def pass_rule(row: dict[str, Any], conditions: list[dict[str, Any]]) -> bool:
    return all(pass_condition(row, c) for c in conditions)


def condition_text(c: dict[str, Any]) -> str:
    return f'{c["feature"]} {c["op"]} {float(c["threshold"]):g}'


def rule_text(conditions: list[dict[str, Any]]) -> str:
    return " AND ".join(condition_text(c) for c in conditions)


def target_class(row: dict[str, Any]) -> str:
    target = str(row.get("target_class", "")).strip()

    if target:
        return target

    human = str(row.get("human_decision_002A", "")).strip()
    final = str(row.get("final_decision_002B", "")).strip()
    source = str(row.get("final_source_002B", "")).strip()

    if human == "reject_human_verified":
        return "human_reject"

    if human == "review_partial_human_verified":
        return "human_partial"

    if final == "keep":
        return "keep"

    if final == "reject" and source != "human_002A":
        return "auto_reject"

    if final == "review":
        return "auto_review"

    return "other"


def enrich_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []

    for row in rows:
        r = dict(row)
        r["target_class"] = target_class(r)
        out.append(r)

    return out


def evaluate_rule(rows: list[dict[str, Any]], conditions: list[dict[str, Any]]) -> dict[str, Any]:
    caught = [r for r in rows if pass_rule(r, conditions)]

    buckets: dict[str, list[str]] = {
        "human_reject": [],
        "human_partial": [],
        "keep": [],
        "auto_reject": [],
        "auto_review": [],
        "other": [],
    }

    for r in caught:
        target = target_class(r)
        buckets.setdefault(target, []).append(str(r.get("review_id", "")))

    hr = len(buckets["human_reject"])
    hp = len(buckets["human_partial"])
    keep = len(buckets["keep"])
    ar = len(buckets["auto_reject"])
    av = len(buckets["auto_review"])

    groups = [feature_group(c["feature"]) for c in conditions]
    group_counts = {
        "kinematic": groups.count("kinematic"),
        "table": groups.count("table"),
        "player": groups.count("player"),
        "other": groups.count("other"),
    }

    score = (
        8.0 * hr
        + 3.0 * ar
        - 15.0 * keep
        - 12.0 * hp
        - 0.50 * av
        - 0.12 * len(conditions)
    )

    if keep > 0:
        reco = "unsafe_catches_keep"
    elif hp > 0:
        reco = "unsafe_catches_human_partial"
    elif hr >= 3 and ar >= 3 and av <= 1:
        reco = "candidate_strong_multi_object_reject"
    elif hr >= 3 and ar >= 2 and av <= 1:
        reco = "candidate_multi_object_reject"
    elif hr >= 3 and ar >= 1 and av <= 2:
        reco = "candidate_human_reject_plus"
    elif hr >= 2 and ar >= 2 and av <= 2:
        reco = "interesting_reject_filter"
    elif hr + ar >= 2 and keep == 0 and hp == 0:
        reco = "interesting_but_weak"
    else:
        reco = "not_useful"

    mixed = group_counts["table"] > 0 and group_counts["player"] > 0
    player_aware = group_counts["player"] > 0
    table_aware = group_counts["table"] > 0

    return {
        "score": round(score, 6),
        "rule": rule_text(conditions),
        "features": ",".join(c["feature"] for c in conditions),
        "groups": ",".join(groups),
        "condition_count": len(conditions),
        "kinematic_feature_count": group_counts["kinematic"],
        "table_feature_count": group_counts["table"],
        "player_feature_count": group_counts["player"],
        "mixed_table_player": int(mixed),
        "player_aware": int(player_aware),
        "table_aware": int(table_aware),
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


def beam_score(result: dict[str, Any]) -> float:
    return (
        5.0 * int(result["human_reject_caught"])
        + 2.0 * int(result["auto_reject_caught"])
        - 2.0 * int(result["keep_caught"])
        - 1.5 * int(result["human_partial_caught"])
        - 0.15 * int(result["auto_review_caught"])
        + 0.25 * int(result["mixed_table_player"])
        + 0.10 * int(result["player_aware"])
    )


def search_rules(rows: list[dict[str, Any]], max_depth: int = 4, beam_width: int = 160) -> list[dict[str, Any]]:
    conditions = make_conditions(rows)

    all_results: list[dict[str, Any]] = []
    beam: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    seen_rules = set()

    for c in conditions:
        res = evaluate_rule(rows, [c])

        if int(res["human_reject_caught"]) + int(res["auto_reject_caught"]) <= 0:
            continue

        key = res["rule"]
        if key in seen_rules:
            continue

        seen_rules.add(key)

        if res["recommendation"] != "not_useful":
            all_results.append(res)

        beam.append(([c], res))

    beam = sorted(
        beam,
        key=lambda x: (
            -beam_score(x[1]),
            int(x[1]["keep_caught"]),
            int(x[1]["human_partial_caught"]),
            int(x[1]["auto_review_caught"]),
            -int(x[1]["player_aware"]),
            -int(x[1]["mixed_table_player"]),
        ),
    )[:beam_width]

    for depth in range(2, max_depth + 1):
        next_beam: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        next_seen = set()

        for conds, _ in beam:
            used_features = {c["feature"] for c in conds}

            for c in conditions:
                if c["feature"] in used_features:
                    continue

                new_conds = conds + [c]
                rule_key = " && ".join(sorted(condition_text(x) for x in new_conds))

                if rule_key in next_seen:
                    continue

                next_seen.add(rule_key)

                res = evaluate_rule(rows, new_conds)

                if int(res["human_reject_caught"]) + int(res["auto_reject_caught"]) <= 0:
                    continue

                if res["rule"] not in seen_rules:
                    seen_rules.add(res["rule"])
                    if res["recommendation"] != "not_useful":
                        all_results.append(res)

                next_beam.append((new_conds, res))

        beam = sorted(
            next_beam,
            key=lambda x: (
                -beam_score(x[1]),
                int(x[1]["keep_caught"]),
                int(x[1]["human_partial_caught"]),
                int(x[1]["auto_review_caught"]),
                -int(x[1]["player_aware"]),
                -int(x[1]["mixed_table_player"]),
                int(x[1]["condition_count"]),
            ),
        )[:beam_width]

    # Dédup stricte.
    dedup = []
    seen = set()

    for r in all_results:
        if r["rule"] in seen:
            continue
        seen.add(r["rule"])
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
            -int(r["mixed_table_player"]),
            -int(r["player_aware"]),
            int(r["condition_count"]),
        ),
    )

    for idx, r in enumerate(dedup, start=1):
        r["rank"] = idx

    return dedup


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}

    for r in rows:
        v = str(r.get(key, "")).strip()
        out[v] = out.get(v, 0) + 1

    return dict(sorted(out.items()))


def means_by_target(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    targets = sorted(set(target_class(r) for r in rows))
    out: dict[str, dict[str, float]] = {}

    for target in targets:
        subset = [r for r in rows if target_class(r) == target]
        out[target] = {}

        for feature in ALL_FEATURES:
            vals = numeric_values(subset, feature)
            out[target][feature] = round(mean(vals), 6) if vals else 0.0

    return out


def summarize(rows: list[dict[str, Any]], rules: list[dict[str, Any]]) -> dict[str, Any]:
    by_reco: dict[str, int] = {}

    for r in rules:
        reco = str(r.get("recommendation", ""))
        by_reco[reco] = by_reco.get(reco, 0) + 1

    safe_rules = [
        r for r in rules
        if int(r["keep_caught"]) == 0
        and int(r["human_partial_caught"]) == 0
        and int(r["human_reject_caught"]) + int(r["auto_reject_caught"]) > 0
    ]

    player_aware_safe = [
        r for r in safe_rules
        if int(r.get("player_aware", 0)) == 1
    ]

    mixed_safe = [
        r for r in safe_rules
        if int(r.get("mixed_table_player", 0)) == 1
    ]

    strong = [
        r for r in rules
        if r["recommendation"] in [
            "candidate_strong_multi_object_reject",
            "candidate_multi_object_reject",
            "candidate_human_reject_plus",
        ]
    ]

    return {
        "version": "003D",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(rows),
        "by_target_class": count_by(rows, "target_class"),
        "rule_count": len(rules),
        "by_recommendation": dict(sorted(by_reco.items())),
        "safe_rule_count": len(safe_rules),
        "player_aware_safe_rule_count": len(player_aware_safe),
        "mixed_table_player_safe_rule_count": len(mixed_safe),
        "top_rules": rules[:50],
        "top_safe_rules": safe_rules[:50],
        "top_player_aware_safe_rules": player_aware_safe[:50],
        "top_mixed_table_player_safe_rules": mixed_safe[:50],
        "top_strong_rules": strong[:50],
        "means_by_target": means_by_target(rows),
        "warning": [
            "003D is still diagnostic.",
            "Rule search is overfitting-prone on 22 segments.",
            "Prefer rules already supported by 003B4 and visually validated overlays.",
            "A player-aware rule is useful only if it improves without catching keep/human_partial.",
        ],
    }


def write_rules_csv(path: Path, rules: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "rank",
        "score",
        "recommendation",
        "rule",
        "features",
        "groups",
        "condition_count",
        "kinematic_feature_count",
        "table_feature_count",
        "player_feature_count",
        "mixed_table_player",
        "player_aware",
        "table_aware",
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


def write_rule_table(rules: list[dict[str, Any]], max_rows: int = 80) -> str:
    rows = []

    for r in rules[:max_rows]:
        cls = str(r.get("recommendation", "")).replace("_", "-")

        if int(r.get("mixed_table_player", 0)) == 1:
            cls += " mixed"
        elif int(r.get("player_aware", 0)) == 1:
            cls += " player-aware"

        rows.append(f"""
<tr class="{html.escape(cls)}">
<td>{html.escape(str(r.get("rank", "")))}</td>
<td>{html.escape(str(r.get("score", "")))}</td>
<td>{html.escape(str(r.get("recommendation", "")))}</td>
<td>{html.escape(str(r.get("mixed_table_player", "")))}</td>
<td>{html.escape(str(r.get("player_aware", "")))}</td>
<td><code>{html.escape(str(r.get("rule", "")))}</code></td>
<td>{html.escape(str(r.get("human_reject_ids", "")))}</td>
<td>{html.escape(str(r.get("auto_reject_ids", "")))}</td>
<td>{html.escape(str(r.get("human_partial_ids", "")))}</td>
<td>{html.escape(str(r.get("keep_ids", "")))}</td>
<td>{html.escape(str(r.get("auto_review_ids", "")))}</td>
<td>{html.escape(str(r.get("caught_ids", "")))}</td>
</tr>
""")

    return "".join(rows)


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

    segment_rows = []

    for r in rows:
        cls = target_class(r).replace("_", "-")
        body_risk = fnum(r.get("candidate_on_body_risk_003C"), 0) or 0
        contextual = int(fnum(r.get("contextual_reject_003B4"), 0) or 0)

        if body_risk >= 0.65:
            cls += " high-body-risk"
        if contextual:
            cls += " contextual-hit"

        segment_rows.append(f"""
<tr class="{html.escape(cls)}">
<td>{html.escape(str(r.get("review_id", "")))}</td>
<td>{html.escape(target_class(r))}</td>
<td>{html.escape(str(r.get("contextual_reject_003B4", "")))}</td>
<td>{html.escape(str(r.get("clip_id_003B2", "")))}</td>
<td><code>{html.escape(str(r.get("segment_name", "")))}</code></td>
<td>{html.escape(str(r.get("point_inside_table_quad_ratio", "")))}</td>
<td>{html.escape(str(r.get("v_range", "")))}</td>
<td>{html.escape(str(r.get("below_table_risk", "")))}</td>
<td>{html.escape(str(r.get("table_context_score_003B2", "")))}</td>
<td>{html.escape(str(r.get("candidate_on_body_risk_003C", "")))}</td>
<td>{html.escape(str(r.get("near_player_motion_mask_ratio_003C", "")))}</td>
<td>{html.escape(str(r.get("inside_hand_or_racket_proxy_ratio_003C", "")))}</td>
<td>{html.escape(str(r.get("median_accel", "")))}</td>
<td>{html.escape(str(r.get("sharp_turn_ratio", "")))}</td>
</tr>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux multi-object audit 003D</title>
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
.high-body-risk td{{outline:1px solid rgba(255,211,122,.38)}}
.contextual-hit td{{box-shadow:inset 3px 0 0 rgba(116,217,159,.95)}}
.candidate-strong-multi-object-reject td,
.candidate-multi-object-reject td,
.candidate-human-reject-plus td,
.interesting-reject-filter td{{background:rgba(116,217,159,.06)}}
.unsafe-catches-keep td,
.unsafe-catches-human-partial td{{background:rgba(255,80,80,.11)}}
.mixed td{{outline:1px solid rgba(116,217,159,.35)}}
.player-aware td{{outline:1px solid rgba(255,211,122,.28)}}
.warn{{color:#ffd37a}}
</style>
</head>
<body>
<h1>TTFlux · multi-object audit 003D</h1>

<section>
<h2>Résumé</h2>
<table>
<tr><th>rows</th><td>{summary["rows"]}</td></tr>
<tr><th>rule_count</th><td>{summary["rule_count"]}</td></tr>
<tr><th>safe_rule_count</th><td>{summary["safe_rule_count"]}</td></tr>
<tr><th>player_aware_safe_rule_count</th><td>{summary["player_aware_safe_rule_count"]}</td></tr>
<tr><th>mixed_table_player_safe_rule_count</th><td>{summary["mixed_table_player_safe_rule_count"]}</td></tr>
<tr><th>warning</th><td class="warn"><pre>{html.escape(json.dumps(summary["warning"], indent=2, ensure_ascii=False))}</pre></td></tr>
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
<h2>Top règles fortes</h2>
<table>
<thead>
<tr>
<th>rank</th><th>score</th><th>reco</th><th>mixed</th><th>player</th><th>rule</th>
<th>human reject</th><th>auto reject</th><th>human partial</th><th>keep</th><th>auto review</th><th>all</th>
</tr>
</thead>
<tbody>{write_rule_table(summary["top_strong_rules"], 80)}</tbody>
</table>
</section>

<section>
<h2>Top règles safe avec joueur</h2>
<table>
<thead>
<tr>
<th>rank</th><th>score</th><th>reco</th><th>mixed</th><th>player</th><th>rule</th>
<th>human reject</th><th>auto reject</th><th>human partial</th><th>keep</th><th>auto review</th><th>all</th>
</tr>
</thead>
<tbody>{write_rule_table(summary["top_player_aware_safe_rules"], 80)}</tbody>
</table>
</section>

<section>
<h2>Top règles safe table + joueur</h2>
<table>
<thead>
<tr>
<th>rank</th><th>score</th><th>reco</th><th>mixed</th><th>player</th><th>rule</th>
<th>human reject</th><th>auto reject</th><th>human partial</th><th>keep</th><th>auto review</th><th>all</th>
</tr>
</thead>
<tbody>{write_rule_table(summary["top_mixed_table_player_safe_rules"], 80)}</tbody>
</table>
</section>

<section>
<h2>Top global</h2>
<table>
<thead>
<tr>
<th>rank</th><th>score</th><th>reco</th><th>mixed</th><th>player</th><th>rule</th>
<th>human reject</th><th>auto reject</th><th>human partial</th><th>keep</th><th>auto review</th><th>all</th>
</tr>
</thead>
<tbody>{write_rule_table(rules, 100)}</tbody>
</table>
</section>

<section>
<h2>Segments · objets fusionnés</h2>
<table>
<thead>
<tr>
<th>ID</th><th>target</th><th>ctx003B4</th><th>clip</th><th>segment</th>
<th>inside table</th><th>v range</th><th>below</th><th>table score</th>
<th>body risk</th><th>near player</th><th>hand/racket</th>
<th>med accel</th><th>sharp turn</th>
</tr>
</thead>
<tbody>{''.join(segment_rows)}</tbody>
</table>
</section>
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="runs/batch_001E/player_context_003C/player_context_features_003C.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/multi_object_audit_003D")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=180)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = enrich_targets(read_csv(input_csv))
    rules = search_rules(rows, max_depth=args.max_depth, beam_width=args.beam_width)
    summary = summarize(rows, rules)

    features_path = out_dir / "multi_object_features_003D.csv"
    rules_path = out_dir / "multi_object_candidate_rules_003D.csv"
    summary_path = out_dir / "multi_object_audit_summary_003D.json"
    html_path = out_dir / "multi_object_audit_003D.html"

    write_csv(features_path, rows)
    write_rules_csv(rules_path, rules)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html(html_path, rows, rules, summary)

    print(f"[003D] rows                         : {summary['rows']}")
    print(f"[003D] rules                        : {summary['rule_count']}")
    print(f"[003D] safe rules                   : {summary['safe_rule_count']}")
    print(f"[003D] player-aware safe rules      : {summary['player_aware_safe_rule_count']}")
    print(f"[003D] mixed table+player safe rules: {summary['mixed_table_player_safe_rule_count']}")
    print(f"[003D] by target                    : {summary['by_target_class']}")
    print(f"[003D] by recommendation            : {summary['by_recommendation']}")
    print(f"[003D] out dir                      : {out_dir}")
    print(f"[003D] html                         : {html_path}")


if __name__ == "__main__":
    main()
