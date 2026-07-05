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
    "x_range",
    "y_range",
    "bbox_area",
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
    "endpoint_jump_ratio",
]


def fnum(value: Any, default: float | None = None) -> float | None:
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

    cols: list[str] = []
    seen = set()

    preferred = [
        "review_id",
        "target_class",
        "final_decision_002B",
        "human_decision_002A",
        "segment_name",
        "n_points",
        "duration_frames",
        "density_points_per_frame",
        "trajectory_length_px",
        "displacement_px",
        "tortuosity",
        "x_range",
        "y_range",
        "bbox_area",
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
        "endpoint_jump_ratio",
        "decision_reason_001U",
        "final_reason_002B",
        "csv",
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


def resolve_path(value: str, root: Path) -> Path | None:
    value = str(value or "").strip()

    if not value:
        return None

    p = Path(value)

    if p.is_absolute() and p.exists():
        return p

    p1 = root / p
    if p1.exists():
        return p1

    p2 = Path.cwd() / p
    if p2.exists():
        return p2

    return None


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


def median(values: list[float]) -> float:
    return percentile(values, 0.5)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def target_class(row: dict[str, str]) -> str:
    human = row.get("human_decision_002A", "")
    final = row.get("final_decision_002B", "")
    source = row.get("final_source_002B", "")

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


def load_track_points(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.exists():
        return []

    rows = read_csv(path)
    pts = []

    for r in rows:
        frame = fnum(r.get("frame") or r.get("frame_idx") or r.get("f"), None)
        x = fnum(r.get("x") or r.get("cx") or r.get("ball_x"), None)
        y = fnum(r.get("y") or r.get("cy") or r.get("ball_y"), None)

        if frame is None or x is None or y is None:
            continue

        pts.append({
            "frame": float(frame),
            "x": float(x),
            "y": float(y),
        })

    pts = sorted(pts, key=lambda p: p["frame"])

    # Déduplication simple par frame : garde le dernier point.
    dedup: dict[int, dict[str, float]] = {}
    for p in pts:
        dedup[int(round(p["frame"]))] = p

    return [dedup[k] for k in sorted(dedup)]


def angle_deg(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    ax, ay = v1
    bx, by = v2

    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)

    if na <= 1e-9 or nb <= 1e-9:
        return 0.0

    c = (ax * bx + ay * by) / (na * nb)
    c = max(-1.0, min(1.0, c))

    return math.degrees(math.acos(c))


def compute_kinematics(points: list[dict[str, float]]) -> dict[str, Any]:
    if not points:
        return {
            "n_points": 0,
            "duration_frames": 0,
        }

    frames = [p["frame"] for p in points]
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]

    n = len(points)
    duration = max(0.0, frames[-1] - frames[0])
    density = n / max(1.0, duration + 1.0)

    x_range = max(xs) - min(xs) if xs else 0.0
    y_range = max(ys) - min(ys) if ys else 0.0
    bbox_area = x_range * y_range

    steps = []
    velocities = []
    speeds = []
    endpoint_jumps = 0

    for a, b in zip(points[:-1], points[1:]):
        dt = max(1e-6, b["frame"] - a["frame"])
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        dist = math.hypot(dx, dy)
        speed = dist / dt

        steps.append({
            "dt": dt,
            "dx": dx,
            "dy": dy,
            "dist": dist,
            "speed": speed,
        })

        velocities.append((dx / dt, dy / dt))
        speeds.append(speed)

        if dt > 4:
            endpoint_jumps += 1

    trajectory_length = sum(s["dist"] for s in steps)
    displacement = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) if n >= 2 else 0.0
    tortuosity = trajectory_length / max(1.0, displacement)

    accels = []
    for v1, v2 in zip(velocities[:-1], velocities[1:]):
        dvx = v2[0] - v1[0]
        dvy = v2[1] - v1[1]
        accels.append(math.hypot(dvx, dvy))

    turns = []
    for v1, v2 in zip(velocities[:-1], velocities[1:]):
        turns.append(angle_deg(v1, v2))

    med_speed = median(speeds)
    speed_sd = std(speeds)
    speed_cv = speed_sd / max(1e-6, med_speed)

    stop_ratio = sum(1 for s in speeds if s <= 1.5) / max(1, len(speeds))
    jump_ratio = sum(1 for s in speeds if s >= 35.0) / max(1, len(speeds))
    sharp_turn_ratio = sum(1 for a in turns if a >= 120.0) / max(1, len(turns))
    endpoint_jump_ratio = endpoint_jumps / max(1, len(steps))

    return {
        "n_points": n,
        "first_frame": int(round(frames[0])),
        "last_frame": int(round(frames[-1])),
        "duration_frames": round(duration, 3),
        "density_points_per_frame": round(density, 6),
        "trajectory_length_px": round(trajectory_length, 3),
        "displacement_px": round(displacement, 3),
        "tortuosity": round(tortuosity, 6),
        "x_range": round(x_range, 3),
        "y_range": round(y_range, 3),
        "bbox_area": round(bbox_area, 3),
        "median_speed": round(med_speed, 6),
        "p90_speed": round(percentile(speeds, 0.90), 6),
        "max_speed": round(max(speeds) if speeds else 0.0, 6),
        "speed_cv": round(speed_cv, 6),
        "median_accel": round(median(accels), 6),
        "p90_accel": round(percentile(accels, 0.90), 6),
        "max_accel": round(max(accels) if accels else 0.0, 6),
        "median_turn_deg": round(median(turns), 6),
        "p90_turn_deg": round(percentile(turns, 0.90), 6),
        "sharp_turn_ratio": round(sharp_turn_ratio, 6),
        "stop_ratio": round(stop_ratio, 6),
        "jump_ratio": round(jump_ratio, 6),
        "endpoint_jump_ratio": round(endpoint_jump_ratio, 6),
    }


def build_kinematic_rows(final_rows: list[dict[str, str]], root: Path) -> list[dict[str, Any]]:
    out = []

    for row in final_rows:
        csv_path = resolve_path(row.get("csv") or row.get("csv_path") or "", root)
        points = load_track_points(csv_path)
        kin = compute_kinematics(points)

        item: dict[str, Any] = dict(row)
        item.update(kin)

        item["target_class"] = target_class(row)
        item["csv_resolved"] = str(csv_path) if csv_path else ""

        out.append(item)

    return out


def feature_value(row: dict[str, Any], feature: str) -> float:
    return float(fnum(row.get(feature), 0.0) or 0.0)


def condition_text(feature: str, op: str, threshold: float) -> str:
    return f"{feature} {op} {threshold:g}"


def pass_condition(row: dict[str, Any], cond: tuple[str, str, float]) -> bool:
    feature, op, threshold = cond
    value = feature_value(row, feature)

    if op == ">=":
        return value >= threshold

    if op == "<=":
        return value <= threshold

    raise ValueError(op)


def pass_rule(row: dict[str, Any], conditions: list[tuple[str, str, float]]) -> bool:
    return all(pass_condition(row, c) for c in conditions)


def rule_text(conditions: list[tuple[str, str, float]]) -> str:
    return " AND ".join(condition_text(*c) for c in conditions)


def candidate_conditions(rows: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []

    quantiles = [0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85]

    for f in KIN_FEATURES:
        values = [feature_value(r, f) for r in rows if fnum(r.get(f), None) is not None]

        if len(values) < 3:
            continue

        thresholds = sorted(set(round(percentile(values, q), 6) for q in quantiles))

        for t in thresholds:
            out.append((f, ">=", float(t)))
            out.append((f, "<=", float(t)))

    return out


def evaluate_rule(rows: list[dict[str, Any]], conditions: list[tuple[str, str, float]]) -> dict[str, Any]:
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
        buckets.setdefault(str(r.get("target_class", "")), []).append(str(r.get("review_id", "")))

    hr = len(buckets["human_reject"])
    hp = len(buckets["human_partial"])
    keep = len(buckets["keep"])
    ar = len(buckets["auto_reject"])
    av = len(buckets["auto_review"])

    score = (
        hr * 6.0
        + ar * 1.0
        - hp * 7.0
        - keep * 14.0
        - av * 0.35
    )

    if keep > 0:
        reco = "unsafe_catches_keep"
    elif hp > 0:
        reco = "unsafe_catches_human_partial"
    elif hr >= 2 and av <= 3:
        reco = "candidate_strict_kinematic_reject"
    elif hr >= 1 and av <= 1:
        reco = "candidate_narrow_kinematic_reject"
    elif hr >= 1:
        reco = "interesting_but_broad"
    else:
        reco = "not_useful"

    return {
        "score": round(score, 4),
        "rule": rule_text(conditions),
        "caught_total": len(caught),
        "caught_ids": ",".join(str(r.get("review_id", "")) for r in caught),
        "human_reject_caught": hr,
        "human_reject_ids": ",".join(buckets["human_reject"]),
        "human_partial_caught": hp,
        "human_partial_ids": ",".join(buckets["human_partial"]),
        "keep_caught": keep,
        "keep_ids": ",".join(buckets["keep"]),
        "auto_reject_caught": ar,
        "auto_reject_ids": ",".join(buckets["auto_reject"]),
        "auto_review_caught": av,
        "auto_review_ids": ",".join(buckets["auto_review"]),
        "recommendation": reco,
    }


def search_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conds = candidate_conditions(rows)
    results = []

    for k in [1, 2, 3]:
        for combo in itertools.combinations(conds, k):
            features = [c[0] for c in combo]

            # Pas deux conditions sur la même feature dans ce premier audit.
            if len(set(features)) != len(features):
                continue

            r = evaluate_rule(rows, list(combo))

            if r["human_reject_caught"] <= 0:
                continue

            if r["recommendation"] == "not_useful":
                continue

            results.append(r)

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

    out = []
    seen = set()

    for r in results:
        if r["rule"] in seen:
            continue
        seen.add(r["rule"])
        r["rank"] = len(out) + 1
        out.append(r)

    return out


def summarize(rows: list[dict[str, Any]], rules: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, int] = {}
    by_reco: dict[str, int] = {}

    for r in rows:
        t = str(r.get("target_class", ""))
        by_target[t] = by_target.get(t, 0) + 1

    for r in rules:
        rec = str(r.get("recommendation", ""))
        by_reco[rec] = by_reco.get(rec, 0) + 1

    # Moyennes par cible pour lecture rapide.
    means_by_target: dict[str, dict[str, float]] = {}

    for target in sorted(by_target):
        subset = [r for r in rows if r.get("target_class") == target]
        means_by_target[target] = {}

        for f in KIN_FEATURES:
            vals = [feature_value(r, f) for r in subset if fnum(r.get(f), None) is not None]
            means_by_target[target][f] = round(mean(vals), 4) if vals else 0.0

    return {
        "rows": len(rows),
        "by_target_class": dict(sorted(by_target.items())),
        "rule_count": len(rules),
        "by_rule_recommendation": dict(sorted(by_reco.items())),
        "top_rules": rules[:20],
        "means_by_target": means_by_target,
        "warning": "Small sample. Kinematic rules are diagnostic until validated on more annotated segments.",
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_html(path: Path, rows: list[dict[str, Any]], rules: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["by_target_class"].items()
    )

    segment_rows = []

    for r in rows:
        cls = str(r.get("target_class", "")).replace("_", "-")

        segment_rows.append(f"""
<tr class="{html.escape(cls)}">
<td>{html.escape(str(r.get("review_id", "")))}</td>
<td>{html.escape(str(r.get("target_class", "")))}</td>
<td><code>{html.escape(str(r.get("segment_name", "")))}</code></td>
<td>{html.escape(str(r.get("final_decision_002B", "")))}</td>
<td>{html.escape(str(r.get("human_decision_002A", "")))}</td>
<td>{html.escape(str(r.get("n_points", "")))}</td>
<td>{html.escape(str(r.get("duration_frames", "")))}</td>
<td>{html.escape(str(r.get("density_points_per_frame", "")))}</td>
<td>{html.escape(str(r.get("trajectory_length_px", "")))}</td>
<td>{html.escape(str(r.get("displacement_px", "")))}</td>
<td>{html.escape(str(r.get("tortuosity", "")))}</td>
<td>{html.escape(str(r.get("median_speed", "")))}</td>
<td>{html.escape(str(r.get("p90_speed", "")))}</td>
<td>{html.escape(str(r.get("max_speed", "")))}</td>
<td>{html.escape(str(r.get("median_turn_deg", "")))}</td>
<td>{html.escape(str(r.get("sharp_turn_ratio", "")))}</td>
<td>{html.escape(str(r.get("jump_ratio", "")))}</td>
<td>{html.escape(str(r.get("stop_ratio", "")))}</td>
</tr>
""")

    rule_rows = []

    for r in rules[:60]:
        rec = str(r.get("recommendation", ""))
        cls = rec.replace("_", "-")

        rule_rows.append(f"""
<tr class="{html.escape(cls)}">
<td>{html.escape(str(r.get("rank", "")))}</td>
<td>{html.escape(str(r.get("score", "")))}</td>
<td><code>{html.escape(str(r.get("rule", "")))}</code></td>
<td>{html.escape(str(r.get("recommendation", "")))}</td>
<td>{html.escape(str(r.get("human_reject_ids", "")))}</td>
<td>{html.escape(str(r.get("human_partial_ids", "")))}</td>
<td>{html.escape(str(r.get("keep_ids", "")))}</td>
<td>{html.escape(str(r.get("auto_review_ids", "")))}</td>
<td>{html.escape(str(r.get("caught_ids", "")))}</td>
</tr>
""")

    mean_sections = []

    for target, feats in summary["means_by_target"].items():
        body = "".join(
            f"<tr><td>{html.escape(f)}</td><td>{html.escape(str(v))}</td></tr>"
            for f, v in feats.items()
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
<title>TTFlux kinematic audit 002G</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
th,td{{border-bottom:1px solid #2b303b;padding:7px 8px;vertical-align:top;font-size:12px}}
th{{background:#20242e;text-align:left;position:sticky;top:0}}
code{{color:#dce4ff}}
.human-reject td{{background:rgba(255,80,80,.09)}}
.human-partial td{{background:rgba(255,200,80,.08)}}
.keep td{{background:rgba(100,255,150,.055)}}
.auto-reject td{{background:rgba(255,80,80,.045)}}
.auto-review td{{background:rgba(150,170,255,.045)}}
.candidate-strict-kinematic-reject td,.candidate-narrow-kinematic-reject td{{background:rgba(100,255,150,.055)}}
.unsafe-catches-human-partial td,.unsafe-catches-keep td{{background:rgba(255,80,80,.09)}}
.interesting-but-broad td{{background:rgba(255,200,80,.065)}}
.warn{{color:#ffd37a}}
.muted{{color:#9ea7b8}}
</style>
</head>
<body>
<h1>TTFlux kinematic audit 002G</h1>

<section>
<h2>Résumé</h2>
<p>Segments : <b>{summary["rows"]}</b></p>
<p>Règles candidates : <b>{summary["rule_count"]}</b></p>
<p class="warn">{html.escape(summary["warning"])}</p>
<table>
<thead><tr><th>target</th><th>count</th></tr></thead>
<tbody>{summary_rows}</tbody>
</table>
</section>

<section>
<h2>Segments · features cinématiques</h2>
<table>
<thead>
<tr>
<th>ID</th><th>target</th><th>segment</th><th>final</th><th>human</th>
<th>n</th><th>dur</th><th>density</th><th>length</th><th>disp</th><th>tortuosity</th>
<th>med speed</th><th>p90 speed</th><th>max speed</th>
<th>med turn</th><th>sharp turn</th><th>jump</th><th>stop</th>
</tr>
</thead>
<tbody>{''.join(segment_rows)}</tbody>
</table>
</section>

<section>
<h2>Règles cinématiques candidates · top 60</h2>
<p class="muted">
Une règle est exploitable seulement si elle attrape des rejets humains sans attraper de keep ni de partial humain.
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

{''.join(mean_sections)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-csv", default="runs/batch_001E/final_decisions_002B/final_decisions_002B_all.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/kinematic_audit_002G")
    args = parser.parse_args()

    root = Path.cwd()
    final_csv = Path(args.final_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_rows = read_csv(final_csv)
    kin_rows = build_kinematic_rows(final_rows, root)
    rules = search_rules(kin_rows)
    summary = summarize(kin_rows, rules)

    write_csv(out_dir / "kinematic_features_002G.csv", kin_rows)
    write_csv(out_dir / "candidate_kinematic_rules_002G.csv", rules)
    write_json(out_dir / "kinematic_audit_summary_002G.json", summary)
    write_html(out_dir / "kinematic_audit_002G.html", kin_rows, rules, summary)

    print(f"[002G] rows       : {summary['rows']}")
    print(f"[002G] targets    : {summary['by_target_class']}")
    print(f"[002G] rules      : {summary['rule_count']}")
    print(f"[002G] recos      : {summary['by_rule_recommendation']}")
    print(f"[002G] out dir    : {out_dir}")
    print(f"[002G] html       : {out_dir / 'kinematic_audit_002G.html'}")


if __name__ == "__main__":
    main()
