from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict


PATCH_ID = "006D1_candidate_human_label_training_rows"


LABEL_FILES = {
    "visibility": "runs/006A_human_frame_labels/006A_visibility_labels.csv",
    "click": "runs/006A_human_frame_labels/006A_click_labels.csv",
    "trajectory": "runs/006B_human_trajectory_labels/006B1_human_trajectory_merged.csv",
}

CANDIDATE_FILES = {
    "gap_005D7A": "runs/005D7A_gap_candidates/005D7A_gap_candidates.csv",
    "scored_005D2": "runs/rally_ball_table_score_005D2/ball_points_table_scored_005D2.csv",
    "safe_005D2": "runs/rally_ball_table_score_005D2/ball_points_table_safe_005D2.csv",
    "gated_005D1": "runs/rally_ball_table_gate_005D1/ball_points_table_gated_005D1.csv",
    "relinked_005D4": "runs/rally_ball_table_relink_005D4/ball_points_table_relinked_005D4.csv",
}

TARGET = {
    "video_id": "-0bM0t0qS8Q",
    "review_id": "RLY0074",
    "local_start": 0,
    "local_end": 1158,
}


def read_csv(path: Path):
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def safe_float(v, default=math.nan):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v, default=-1):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return int(float(v))
    except Exception:
        return default


def find_col(fields, names):
    low = {c.lower(): c for c in fields}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def row_text(row: dict):
    return " ".join(str(v) for v in row.values())


def is_target_row(row: dict, source_name: str):
    txt = row_text(row)

    if TARGET["review_id"] in txt:
        return True

    if source_name == "gap_005D7A":
        # 005D7A a été généré pour le gap focus courant.
        return True

    return False


def load_human_labels(root: Path):
    """
    Fusion par frame.
    Priorité :
    1. trajectoire avec xy
    2. click avec xy
    3. visibilité visible/invisible sans xy
    """
    labels = {}
    raw_rows = []

    priority_by_source = {
        "visibility": 1,
        "click": 2,
        "trajectory": 3,
    }

    for source, rel_path in LABEL_FILES.items():
        path = root / rel_path
        rows, fields = read_csv(path)

        frame_col = find_col(fields, ["frame", "video_frame"])
        visible_col = find_col(fields, ["visible"])
        x_col = find_col(fields, ["x"])
        y_col = find_col(fields, ["y"])
        label_col = find_col(fields, ["label"])
        point_type_col = find_col(fields, ["point_type"])

        for r in rows:
            fr = safe_int(r.get(frame_col)) if frame_col else -1
            if fr < TARGET["local_start"] or fr > TARGET["local_end"]:
                continue

            visible = str(r.get(visible_col, "")).strip() if visible_col else ""
            x = safe_float(r.get(x_col)) if x_col else math.nan
            y = safe_float(r.get(y_col)) if y_col else math.nan
            has_xy = math.isfinite(x) and math.isfinite(y)

            label_name = str(r.get(label_col, "")).strip() if label_col else ""
            point_type = str(r.get(point_type_col, "")).strip() if point_type_col else ""

            item = {
                "frame": fr,
                "visible": visible,
                "x": x,
                "y": y,
                "has_xy": has_xy,
                "label": label_name,
                "point_type": point_type,
                "source": source,
                "priority": priority_by_source[source],
            }

            raw_rows.append(item)

            old = labels.get(fr)
            if old is None:
                labels[fr] = item
                continue

            # XY humain prioritaire.
            if item["has_xy"] and not old["has_xy"]:
                labels[fr] = item
                continue

            if item["has_xy"] == old["has_xy"] and item["priority"] >= old["priority"]:
                labels[fr] = item
                continue

    return labels, raw_rows


def load_candidates(root: Path):
    candidates = []
    source_meta = {}

    for source_name, rel_path in CANDIDATE_FILES.items():
        path = root / rel_path
        rows, fields = read_csv(path)

        frame_col = find_col(fields, ["frame", "video_frame", "frame_idx", "local_frame"])
        x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
        y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])
        score_col = find_col(fields, ["score", "conf", "confidence", "prob", "table_score", "score_005D2"])

        kept = 0
        invalid = 0

        for i, r in enumerate(rows):
            if not is_target_row(r, source_name):
                continue

            fr = safe_int(r.get(frame_col)) if frame_col else -1
            x = safe_float(r.get(x_col)) if x_col else math.nan
            y = safe_float(r.get(y_col)) if y_col else math.nan

            if fr < TARGET["local_start"] or fr > TARGET["local_end"]:
                invalid += 1
                continue

            if not (math.isfinite(x) and math.isfinite(y)):
                invalid += 1
                continue

            score = safe_float(r.get(score_col), math.nan) if score_col else math.nan

            c = {
                "candidate_uid": f"{source_name}_{i:07d}",
                "candidate_source": source_name,
                "candidate_row_index": i,
                "frame": fr,
                "x": x,
                "y": y,
                "score": score,
                "raw": r,
            }
            candidates.append(c)
            kept += 1

        source_meta[source_name] = {
            "path": str(path),
            "exists": path.exists(),
            "rows": len(rows),
            "fields": fields,
            "frame_col": frame_col,
            "x_col": x_col,
            "y_col": y_col,
            "score_col": score_col,
            "kept_target_candidates": kept,
            "invalid_target_rows": invalid,
        }

    return candidates, source_meta


def dist(a, b, c, d):
    return math.hypot(a - c, b - d)


def build_training_rows(labels, candidates):
    by_frame = defaultdict(list)
    for c in candidates:
        by_frame[c["frame"]].append(c)

    training_rows = []
    coverage_rows = []

    recall_thresholds = [10, 20, 30, 50, 80, 120]

    for fr in sorted(labels):
        lab = labels[fr]
        frame_candidates = by_frame.get(fr, [])

        human_visible = lab["visible"]
        human_has_xy = lab["has_xy"]

        best_dist_all = math.nan
        best_source = ""
        best_uid = ""
        source_counts = Counter()
        source_best = {}

        if human_has_xy and frame_candidates:
            for c in frame_candidates:
                d = dist(c["x"], c["y"], lab["x"], lab["y"])
                source_counts[c["candidate_source"]] += 1

                if not math.isfinite(best_dist_all) or d < best_dist_all:
                    best_dist_all = d
                    best_source = c["candidate_source"]
                    best_uid = c["candidate_uid"]

                old = source_best.get(c["candidate_source"])
                if old is None or d < old:
                    source_best[c["candidate_source"]] = d

        elif frame_candidates:
            for c in frame_candidates:
                source_counts[c["candidate_source"]] += 1

        cov = {
            "frame": fr,
            "human_visible": human_visible,
            "human_has_xy": "1" if human_has_xy else "0",
            "human_x": f"{lab['x']:.3f}" if human_has_xy else "",
            "human_y": f"{lab['y']:.3f}" if human_has_xy else "",
            "label_source": lab["source"],
            "candidate_count": len(frame_candidates),
            "best_candidate_distance_px": f"{best_dist_all:.3f}" if math.isfinite(best_dist_all) else "",
            "best_candidate_source": best_source,
            "best_candidate_uid": best_uid,
        }

        for t in recall_thresholds:
            cov[f"hit_le_{t}px"] = "1" if math.isfinite(best_dist_all) and best_dist_all <= t else "0"

        for source_name in CANDIDATE_FILES:
            cov[f"{source_name}_count"] = source_counts.get(source_name, 0)
            d = source_best.get(source_name)
            cov[f"{source_name}_best_dist_px"] = f"{d:.3f}" if d is not None else ""

        coverage_rows.append(cov)

        if not frame_candidates:
            training_rows.append({
                "row_type": "no_candidate_on_labeled_frame",
                "frame": fr,
                "candidate_uid": "",
                "candidate_source": "",
                "candidate_x": "",
                "candidate_y": "",
                "candidate_score": "",
                "human_visible": human_visible,
                "human_has_xy": "1" if human_has_xy else "0",
                "human_x": f"{lab['x']:.3f}" if human_has_xy else "",
                "human_y": f"{lab['y']:.3f}" if human_has_xy else "",
                "distance_px": "",
                "target_class": "visible_no_candidate" if human_visible == "1" else "invisible_no_candidate",
                "label_source": lab["source"],
            })
            continue

        for c in frame_candidates:
            if human_has_xy:
                d = dist(c["x"], c["y"], lab["x"], lab["y"])

                if d <= 20:
                    target_class = "positive_ball"
                elif d <= 50:
                    target_class = "near_ball_soft_positive"
                elif d <= 100:
                    target_class = "hard_negative_near_ball"
                else:
                    target_class = "negative_not_ball"

                row_type = "candidate_with_xy_label"

            elif human_visible == "0":
                d = math.nan
                target_class = "negative_no_ball_visible"
                row_type = "candidate_on_invisible_frame"

            else:
                d = math.nan
                target_class = "unusable_visible_no_xy"
                row_type = "candidate_with_visibility_only"

            training_rows.append({
                "row_type": row_type,
                "frame": fr,
                "candidate_uid": c["candidate_uid"],
                "candidate_source": c["candidate_source"],
                "candidate_x": f"{c['x']:.3f}",
                "candidate_y": f"{c['y']:.3f}",
                "candidate_score": f"{c['score']:.6f}" if math.isfinite(c["score"]) else "",
                "human_visible": human_visible,
                "human_has_xy": "1" if human_has_xy else "0",
                "human_x": f"{lab['x']:.3f}" if human_has_xy else "",
                "human_y": f"{lab['y']:.3f}" if human_has_xy else "",
                "distance_px": f"{d:.3f}" if math.isfinite(d) else "",
                "target_class": target_class,
                "label_source": lab["source"],
            })

    return training_rows, coverage_rows


def summarize(training_rows, coverage_rows, labels, candidates, source_meta):
    class_counts = Counter(r["target_class"] for r in training_rows)
    row_type_counts = Counter(r["row_type"] for r in training_rows)
    cand_source_counts = Counter(c["candidate_source"] for c in candidates)

    xy_rows = [r for r in coverage_rows if r["human_has_xy"] == "1"]

    recall = {}
    for t in [10, 20, 30, 50, 80, 120]:
        hits = sum(1 for r in xy_rows if r.get(f"hit_le_{t}px") == "1")
        recall[f"candidate_recall_le_{t}px"] = {
            "hits": hits,
            "total_xy_labels": len(xy_rows),
            "ratio": hits / len(xy_rows) if xy_rows else 0.0,
        }

    no_candidate_visible_xy = [
        r for r in coverage_rows
        if r["human_visible"] == "1" and r["human_has_xy"] == "1" and int(r["candidate_count"]) == 0
    ]

    no_candidate_visible_any = [
        r for r in coverage_rows
        if r["human_visible"] == "1" and int(r["candidate_count"]) == 0
    ]

    by_source_recall = {}
    for source_name in CANDIDATE_FILES:
        vals = []
        for r in xy_rows:
            d = safe_float(r.get(f"{source_name}_best_dist_px"), math.nan)
            if math.isfinite(d):
                vals.append(d)

        by_source_recall[source_name] = {
            "frames_with_candidate": len(vals),
            "hit_le_20": sum(1 for d in vals if d <= 20),
            "hit_le_50": sum(1 for d in vals if d <= 50),
            "hit_le_100": sum(1 for d in vals if d <= 100),
            "median_best_dist": float(sorted(vals)[len(vals)//2]) if vals else None,
        }

    return {
        "patch": PATCH_ID,
        "target": TARGET,
        "label_count": len(labels),
        "candidate_count": len(candidates),
        "candidate_source_counts": dict(cand_source_counts),
        "training_row_count": len(training_rows),
        "coverage_row_count": len(coverage_rows),
        "target_class_counts": dict(class_counts),
        "row_type_counts": dict(row_type_counts),
        "recall": recall,
        "by_source_recall": by_source_recall,
        "visible_xy_frames_without_any_candidate": len(no_candidate_visible_xy),
        "visible_frames_without_any_candidate": len(no_candidate_visible_any),
        "source_meta": source_meta,
        "interpretation": [
            "Si recall <=50px est bas, le réservoir candidat reste le problème principal.",
            "Si recall est bon mais positives mal classées ensuite, alors le reranker/Viterbi est le problème.",
            "Les lignes positive_ball / near_ball_soft_positive sont les premières utiles pour un pré-training léger.",
        ],
    }


def main():
    root = Path(".").resolve()
    out_dir = root / "runs" / "006D1_candidate_human_training_rows"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, raw_label_rows = load_human_labels(root)
    candidates, source_meta = load_candidates(root)
    training_rows, coverage_rows = build_training_rows(labels, candidates)
    summary = summarize(training_rows, coverage_rows, labels, candidates, source_meta)

    training_csv = out_dir / "006D1_candidate_human_training_rows.csv"
    coverage_csv = out_dir / "006D1_human_frame_candidate_coverage.csv"
    summary_json = out_dir / "006D1_candidate_human_training_summary.json"

    training_fields = [
        "row_type",
        "frame",
        "candidate_uid",
        "candidate_source",
        "candidate_x",
        "candidate_y",
        "candidate_score",
        "human_visible",
        "human_has_xy",
        "human_x",
        "human_y",
        "distance_px",
        "target_class",
        "label_source",
    ]

    coverage_fields = [
        "frame",
        "human_visible",
        "human_has_xy",
        "human_x",
        "human_y",
        "label_source",
        "candidate_count",
        "best_candidate_distance_px",
        "best_candidate_source",
        "best_candidate_uid",
        "hit_le_10px",
        "hit_le_20px",
        "hit_le_30px",
        "hit_le_50px",
        "hit_le_80px",
        "hit_le_120px",
    ]

    for source_name in CANDIDATE_FILES:
        coverage_fields.append(f"{source_name}_count")
        coverage_fields.append(f"{source_name}_best_dist_px")

    write_csv(training_csv, training_rows, training_fields)
    write_csv(coverage_csv, coverage_rows, coverage_fields)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print("=" * 72)
    print("")
    print("OK 006D1")
    print(f"training_csv = {training_csv}")
    print(f"coverage_csv = {coverage_csv}")
    print(f"summary_json = {summary_json}")
    print("")
    print(f"label_count={summary['label_count']}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"training_row_count={summary['training_row_count']}")
    print("")
    print("TARGET CLASS COUNTS")
    for k, v in summary["target_class_counts"].items():
        print(f"  {k}: {v}")
    print("")
    print("RECALL")
    for k, v in summary["recall"].items():
        print(f"  {k}: {v['hits']}/{v['total_xy_labels']} = {v['ratio']:.3f}")
    print("")
    print("BY SOURCE")
    for k, v in summary["by_source_recall"].items():
        print(
            f"  {k}: frames={v['frames_with_candidate']} "
            f"hit20={v['hit_le_20']} hit50={v['hit_le_50']} "
            f"hit100={v['hit_le_100']} median={v['median_best_dist']}"
        )
    print("")
    print(f"visible_xy_frames_without_any_candidate={summary['visible_xy_frames_without_any_candidate']}")
    print(f"visible_frames_without_any_candidate={summary['visible_frames_without_any_candidate']}")


if __name__ == "__main__":
    main()
