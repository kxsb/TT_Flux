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
        "segment_name",
        "track_csv_resolved_002G2",
        "track_found_002G2",
        *KIN_FEATURES,
        "decision_reason_001U",
        "final_reason_002B",
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


def resolve_existing_path(value: str, roots: list[Path]) -> Path | None:
    value = str(value or "").strip().strip('"')

    if not value:
        return None

    p = Path(value)

    if p.is_absolute() and p.exists():
        return p

    for root in roots:
        q = root / p
        if q.exists():
            return q

    return None


def find_track_csv(row: dict[str, str], roots: list[Path], run_dir: Path) -> Path | None:
    candidate_cols = [
        "csv",
        "csv_path",
        "segment_csv",
        "validation_csv",
        "track_csv",
        "track_csv_resolved",
        "track_csv_resolved_001T2",
        "csv_resolved",
    ]

    for col in candidate_cols:
        p = resolve_existing_path(row.get(col, ""), roots)
        if p is not None:
            return p

    segment_name = row.get("segment_name", "").strip()
    if not segment_name:
        return None

    names = [
        f"{segment_name}.csv",
        segment_name,
    ]

    for name in names:
        hits = list(run_dir.rglob(name))
        hits = [p for p in hits if p.is_file()]
        if hits:
            return sorted(hits, key=lambda p: len(str(p)))[0]

    hits = [
        p for p in run_dir.rglob("*.csv")
        if segment_name in p.name
    ]

    if hits:
        return sorted(hits, key=lambda p: len(str(p)))[0]

    return None


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
    points = []

    for r in rows:
        frame = fnum(r.get("frame") or r.get("frame_idx") or r.get("f"), None)
        x = fnum(r.get("x") or r.get("cx") or r.get("ball_x"), None)
        y = fnum(r.get("y") or r.get("cy") or r.get("ball_y"), None)

        if frame is None or x is None or y is None:
            continue

        points.append({
            "frame": float(frame),
            "x": float(x),
            "y": float(y),
        })

    dedup: dict[int, dict[str, float]] = {}

    for p in points:
        dedup[int(round(p["frame"]))] = p

    return [dedup[k] for k in sorted(dedup)]


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
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if not values:
        return 0.0

    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


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
    if len(points) < 2:
        return {
            "n_points": len(points),
            "duration_frames": 0,
            "density_points_per_frame": 0,
            "trajectory_length_px": 0,
            "displacement_px": 0,
            "tortuosity": 0,
            "x_range_kin": 0,
            "y_range_kin": 0,
            "bbox_area_kin": 0,
            "median_speed": 0,
            "p90_speed": 0,
            "max_speed": 0,
            "speed_cv": 0,
            "median_accel": 0,
            "p90_accel": 0,
            "max_accel": 0,
            "median_turn_deg": 0,
            "p90_turn_deg": 0,
            "sharp_turn_ratio": 0,
            "stop_ratio": 0,
            "jump_ratio": 0,
            "endpoint_gap_ratio": 0,
        }

    frames = [p["frame"] for p in points]
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]

    velocities = []
    speeds = []
    dists = []
    gaps = 0

    for a, b in zip(points[:-1], points[1:]):
        dt = b["frame"] - a["frame"]

        if dt <= 0:
            continue

        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        dist = math.hypot(dx, dy)
        speed = dist / dt

        if dt > 4:
            gaps += 1

        dists.append(dist)
        speeds.append(speed)
        velocities.append((dx / dt, dy / dt))

    accels = []

    for v1, v2 in zip(velocities[:-1], velocities[1:]):
        accels.append(math.hypot(v2[0] - v1[0], v2[1] - v1[1]))

    turns = []

    for v1, v2 in zip(velocities[:-1], velocities[1:]):
        turns.append(angle_deg(v1, v2))

    duration = frames[-1] - frames[0]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    traj_len = sum(dists)
    disp = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])

    med_speed = median(speeds)

    return {
        "n_points": len(points),
        "first_frame": int(round(frames[0])),
        "last_frame": int(round(frames[-1])),
        "duration_frames": round(duration, 3),
        "density_points_per_frame": round(len(points) / max(1.0, duration + 1.0), 6),
        "trajectory_length_px": round(traj_len, 3),
        "displacement_px": round(disp, 3),
        "tortuosity": round(traj_len / max(1.0, disp), 6),
        "x_range_kin": round(x_range, 3),
        "y_range_kin": round(y_range, 3),
        "bbox_area_kin": round(x_range * y_range, 3),
        "median_speed": round(med_speed, 6),
        "p90_speed": round(percentile(speeds, 0.90), 6),
        "max_speed": round(max(speeds) if speeds else 0, 6),
        "speed_cv": round(std(speeds) / max(1e-6, med_speed), 6),
        "median_accel": round(median(accels), 6),
        "p90_accel": round(percentile(accels, 0.90), 6),
        "max_accel": round(max(accels) if accels else 0, 6),
        "median_turn_deg": round(median(turns), 6),
        "p90_turn_deg": round(percentile(turns, 0.90), 6),
        "sharp_turn_ratio": round(sum(1 for a in turns if a >= 120) / max(1, len(turns)), 6),
        "stop_ratio": round(sum(1 for s in speeds if s <= 1.5) / max(1, len(speeds)), 6),
        "jump_ratio": round(sum(1 for s in speeds if s >= 35.0) / max(1, len(speeds)), 6),
        "endpoint_gap_ratio": round(gaps / max(1, len(dists)), 6),
    }


def build_rows(final_rows: list[dict[str, str]], roots: list[Path], run_dir: Path) -> list[dict[str, Any]]:
    out = []

    for r in final_rows:
        p = find_track_csv(r, roots, run_dir)
        pts = load_track_points(p)
        kin = compute_kinematics(pts)

        item: dict[str, Any] = dict(r)
        item.update(kin)
        item["target_class"] = target_class(r)
        item["track_csv_resolved_002G2"] = str(p) if p else ""
        item["track_found_002G2"] = 1 if p and len(pts) else 0

        out.append(item)

    return out


def feature_value(row: dict[str, Any], feature: str) -> float:
    return float(fnum(row.get(feature), 0.0) or 0.0)


def candidate_conditions(rows: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
    conditions = []
    qs = [0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85]

    for feature in KIN_FEATURES:
        values = [
            feature_value(r, feature)
            for r in rows
            if feature not in ("n_points", "duration_frames") or feature_value(r, feature) > 0
        ]

        if len(set(values)) < 3:
            continue

        thresholds = sorted(set(round(percentile(values, q), 6) for q in qs))

        for t in thresholds:
            conditions.append((feature, ">=", t))
            conditions.append((feature, "<=", t))

    return conditions


def pass_condition(row: dict[str, Any], cond: tuple[str, str, float]) -> bool:
    feature, op, threshold = cond
    value = feature_value(row, feature)

    if op == ">=":
        return value >= threshold

    if op == "<=":
        return value <= threshold

    return False


def pass_rule(row: dict[str, Any], conditions: list[tuple[str, str, float]]) -> bool:
    return all(pass_condition(row, c) for c in conditions)


def rule_text(conditions: list[tuple[str, str, float]]) -> str:
    return " AND ".join(f"{f} {op} {t:g}" for f, op, t in conditions)


def evaluate_rule(rows: list[dict[str, Any]], conditions: list[tuple[str, str, float]]) -> dict[str, Any]:
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
        buckets.setdefault(str(r.get("target_class", "")), []).append(str(r.get("review_id", "")))

    hr = len(buckets["human_reject"])
    hp = len(buckets["human_partial"])
    keep = len(buckets["keep"])
    ar = len(buckets["auto_reject"])
    av = len(buckets["auto_review"])

    score = hr * 6.0 + ar * 1.0 - hp * 7.0 - keep * 14.0 - av * 0.35

    if keep:
        reco = "unsafe_catches_keep"
    elif hp:
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
    rows = [r for r in rows if int(fnum(r.get("track_found_002G2"), 0) or 0) == 1]
    conditions = candidate_conditions(rows)
    results = []

    for k in [1, 2, 3]:
        for combo in itertools.combinations(conditions, k):
            features = [c[0] for c in combo]

            if len(set(features)) != len(features):
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
    by_target = {}
    by_reco = {}

    for r in rows:
        t = str(r.get("target_class", ""))
        by_target[t] = by_target.get(t, 0) + 1

    for r in rules:
        rec = str(r.get("recommendation", ""))
        by_reco[rec] = by_reco.get(rec, 0) + 1

    track_found = sum(1 for r in rows if int(fnum(r.get("track_found_002G2"), 0) or 0) == 1)
    track_missing = len(rows) - track_found

    means_by_target = {}

    for target in sorted(by_target):
        subset = [r for r in rows if r.get("target_class") == target and int(fnum(r.get("track_found_002G2"), 0) or 0) == 1]
        means_by_target[target] = {}

        for f in KIN_FEATURES:
            vals = [feature_value(r, f) for r in subset]
            means_by_target[target][f] = round(mean(vals), 4) if vals else 0.0

    return {
        "rows": len(rows),
        "track_found": track_found,
        "track_missing": track_missing,
        "by_target_class": dict(sorted(by_target.items())),
        "rule_count": len(rules),
        "by_rule_recommendation": dict(sorted(by_reco.items())),
        "top_rules": rules[:30],
        "means_by_target": means_by_target,
    }


def write_html(path: Path, rows: list[dict[str, Any]], rules: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["by_target_class"].items()
    )

    segment_rows = []

    for r in rows:
        cls = str(r.get("target_class", "")).replace("_", "-")
        found = int(fnum(r.get("track_found_002G2"), 0) or 0)

        segment_rows.append(f"""
<tr class="{html.escape(cls)} {'missing' if not found else ''}">
<td>{html.escape(str(r.get("review_id", "")))}</td>
<td>{html.escape(str(r.get("target_class", "")))}</td>
<td><code>{html.escape(str(r.get("segment_name", "")))}</code></td>
<td>{html.escape(str(r.get("track_found_002G2", "")))}</td>
<td>{html.escape(str(r.get("n_points", "")))}</td>
<td>{html.escape(str(r.get("duration_frames", "")))}</td>
<td>{html.escape(str(r.get("density_points_per_frame", "")))}</td>
<td>{html.escape(str(r.get("trajectory_length_px", "")))}</td>
<td>{html.escape(str(r.get("displacement_px", "")))}</td>
<td>{html.escape(str(r.get("tortuosity", "")))}</td>
<td>{html.escape(str(r.get("x_range_kin", "")))}</td>
<td>{html.escape(str(r.get("y_range_kin", "")))}</td>
<td>{html.escape(str(r.get("median_speed", "")))}</td>
<td>{html.escape(str(r.get("p90_speed", "")))}</td>
<td>{html.escape(str(r.get("max_speed", "")))}</td>
<td>{html.escape(str(r.get("median_turn_deg", "")))}</td>
<td>{html.escape(str(r.get("sharp_turn_ratio", "")))}</td>
<td>{html.escape(str(r.get("jump_ratio", "")))}</td>
<td><code>{html.escape(str(r.get("track_csv_resolved_002G2", "")))}</code></td>
</tr>
""")

    rule_rows = []

    for r in rules[:80]:
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
<title>TTFlux kinematic audit 002G2</title>
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
.missing td{{opacity:.55}}
.candidate-strict-kinematic-reject td,.candidate-narrow-kinematic-reject td{{background:rgba(100,255,150,.055)}}
.unsafe-catches-human-partial td,.unsafe-catches-keep td{{background:rgba(255,80,80,.09)}}
.interesting-but-broad td{{background:rgba(255,200,80,.065)}}
.warn{{color:#ffd37a}}
.muted{{color:#9ea7b8}}
</style>
</head>
<body>
<h1>TTFlux kinematic audit 002G2</h1>

<section>
<h2>Résumé</h2>
<p>Segments : <b>{summary["rows"]}</b></p>
<p>Tracks trouvées : <b>{summary["track_found"]}</b></p>
<p>Tracks manquantes : <b>{summary["track_missing"]}</b></p>
<p>Règles candidates : <b>{summary["rule_count"]}</b></p>
<p class="warn">Ces règles restent diagnostiques tant qu'on n'a pas plus de segments humains.</p>
<table>
<thead><tr><th>target</th><th>count</th></tr></thead>
<tbody>{summary_rows}</tbody>
</table>
</section>

<section>
<h2>Segments · vraie cinématique recalculée</h2>
<table>
<thead>
<tr>
<th>ID</th><th>target</th><th>segment</th><th>track</th>
<th>n</th><th>dur</th><th>density</th><th>length</th><th>disp</th><th>tortuosity</th>
<th>x range</th><th>y range</th><th>med speed</th><th>p90 speed</th><th>max speed</th>
<th>med turn</th><th>sharp turn</th><th>jump</th><th>csv</th>
</tr>
</thead>
<tbody>{''.join(segment_rows)}</tbody>
</table>
</section>

<section>
<h2>Règles cinématiques candidates · top 80</h2>
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
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--out-dir", default="runs/batch_001E/kinematic_audit_002G2")
    args = parser.parse_args()

    root = Path.cwd()
    run_dir = Path(args.run_dir)
    final_csv = Path(args.final_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = [root, run_dir, run_dir.resolve()]

    final_rows = read_csv(final_csv)
    rows = build_rows(final_rows, roots, run_dir)
    rules = search_rules(rows)
    summary = summarize(rows, rules)

    write_csv(out_dir / "kinematic_features_002G2.csv", rows)
    write_csv(out_dir / "candidate_kinematic_rules_002G2.csv", rules)

    (out_dir / "kinematic_audit_summary_002G2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_html(out_dir / "kinematic_audit_002G2.html", rows, rules, summary)

    print(f"[002G2] rows          : {summary['rows']}")
    print(f"[002G2] track_found   : {summary['track_found']}")
    print(f"[002G2] track_missing : {summary['track_missing']}")
    print(f"[002G2] targets       : {summary['by_target_class']}")
    print(f"[002G2] rules         : {summary['rule_count']}")
    print(f"[002G2] recos         : {summary['by_rule_recommendation']}")
    print(f"[002G2] out dir       : {out_dir}")
    print(f"[002G2] html          : {out_dir / 'kinematic_audit_002G2.html'}")


if __name__ == "__main__":
    main()
