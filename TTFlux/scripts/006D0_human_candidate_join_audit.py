from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict


PATCH_ID = "006D0_human_labels_candidate_join_audit"


INPUTS = {
    "align_summary": "runs/005F3_align_clean_video/005F3_align_clean_video_summary.json",
    "labels_visibility": "runs/006A_human_frame_labels/006A_visibility_labels.csv",
    "labels_click": "runs/006A_human_frame_labels/006A_click_labels.csv",
    "labels_traj": "runs/006B_human_trajectory_labels/006B1_human_trajectory_merged.csv",
    "table_context": "runs/rally_scene_table_objects_005C9B/scene_table_objects_005C9B.csv",
    "table_scored": "runs/rally_ball_table_score_005D2/ball_points_table_scored_005D2.csv",
    "table_safe": "runs/rally_ball_table_score_005D2/ball_points_table_safe_005D2.csv",
    "table_gated": "runs/rally_ball_table_gate_005D1/ball_points_table_gated_005D1.csv",
    "table_relinked": "runs/rally_ball_table_relink_005D4/ball_points_table_relinked_005D4.csv",
    "gap_candidates": "runs/005D7A_gap_candidates/005D7A_gap_candidates.csv",
}


def read_csv(path: Path):
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


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


def parse_clip_range_from_text(txt: str):
    m = re.search(r"_f(\d+)_to_f(\d+)", txt)
    if not m:
        m = re.search(r"f(\d+)_to_f(\d+)", txt)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_clip_range(row: dict):
    txt = " ".join(str(v) for v in row.values())
    return parse_clip_range_from_text(txt)


def row_text(row: dict):
    return " ".join(str(v) for v in row.values())


def has_video_id(row: dict, video_id: str):
    return bool(video_id and video_id in row_text(row))


def has_review_id(row: dict, review_id: str):
    return bool(review_id and review_id in row_text(row))


def overlap_len(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def load_align_summary(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_labels(name: str, path: Path):
    rows, fields = read_csv(path)

    frame_col = find_col(fields, ["frame", "video_frame"])
    visible_col = find_col(fields, ["visible"])
    x_col = find_col(fields, ["x"])
    y_col = find_col(fields, ["y"])
    label_col = find_col(fields, ["label"])

    frames = []
    visible = 0
    invisible = 0
    clicked = 0
    rows_clean = []

    for r in rows:
        fr = safe_int(r.get(frame_col)) if frame_col else -1
        if fr >= 0:
            frames.append(fr)

        vis = str(r.get(visible_col, "")).strip() if visible_col else ""
        if vis == "1":
            visible += 1
        elif vis == "0":
            invisible += 1

        x = safe_float(r.get(x_col)) if x_col else math.nan
        y = safe_float(r.get(y_col)) if y_col else math.nan
        if math.isfinite(x) and math.isfinite(y):
            clicked += 1

        rr = dict(r)
        rr["_source_label_file"] = name
        rr["_frame"] = fr
        rr["_visible"] = vis
        rr["_x"] = x
        rr["_y"] = y
        rr["_label"] = r.get(label_col, "") if label_col else ""
        rows_clean.append(rr)

    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "fields": fields,
        "row_count": len(rows),
        "frame_col": frame_col,
        "visible_col": visible_col,
        "x_col": x_col,
        "y_col": y_col,
        "label_col": label_col,
        "visible_count": visible,
        "invisible_count": invisible,
        "clicked_count": clicked,
        "min_frame": min(frames) if frames else None,
        "max_frame": max(frames) if frames else None,
        "unique_frame_count": len(set(frames)),
        "rows_clean": rows_clean,
    }


def candidate_source_audit(name: str, path: Path, target):
    rows, fields = read_csv(path)

    frame_col = find_col(fields, [
        "frame", "video_frame", "frame_idx", "frame_id", "local_frame"
    ])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])
    score_col = find_col(fields, ["score", "conf", "confidence", "prob", "table_score", "score_005D2"])

    video_id = target["video_id"]
    review_id = target["review_id"]
    full_start = target["full_start"]
    full_end = target["full_end"]

    counts = Counter()
    local_frames = set()
    strict_frames = set()
    samples = []

    for i, r in enumerate(rows):
        txt = row_text(r)
        cr = parse_clip_range(r)

        m_video = has_video_id(r, video_id)
        m_review = has_review_id(r, review_id)

        m_range = False
        ov = 0
        if cr:
            ov = overlap_len(cr[0], cr[1], full_start, full_end)
            m_range = ov > 0

        if m_video:
            counts["same_video_id"] += 1
        if m_review:
            counts["same_review_id"] += 1
        if cr:
            counts["has_clip_range"] += 1
        if m_range:
            counts["range_overlap"] += 1

        strict = False
        if m_review:
            strict = True
        elif m_video and m_range:
            strict = True
        elif name == "gap_candidates":
            # 005D7A a été produit spécifiquement pour le gap focus courant.
            strict = True

        if strict:
            counts["strict_segment_match"] += 1

            fr = safe_int(r.get(frame_col)) if frame_col else -1
            if fr >= 0:
                local_frames.add(fr)
                strict_frames.add(fr)

            if len(samples) < 8:
                samples.append({
                    "row_index": i,
                    "frame": fr,
                    "x": r.get(x_col, "") if x_col else "",
                    "y": r.get(y_col, "") if y_col else "",
                    "score": r.get(score_col, "") if score_col else "",
                    "clip_range": cr,
                    "match_review": m_review,
                    "match_video": m_video,
                    "range_overlap": ov,
                })

    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "fields": fields,
        "row_count": len(rows),
        "frame_col": frame_col,
        "x_col": x_col,
        "y_col": y_col,
        "score_col": score_col,
        "counts": dict(counts),
        "strict_unique_frame_count": len(strict_frames),
        "strict_min_frame": min(strict_frames) if strict_frames else None,
        "strict_max_frame": max(strict_frames) if strict_frames else None,
        "samples": samples,
    }


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def main():
    root = Path(".").resolve()
    out_dir = root / "runs" / "006D0_human_candidate_join_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    align_path = root / INPUTS["align_summary"]
    align = load_align_summary(align_path)

    best_offset = safe_int(align.get("best_offset"), 16000)
    aligned_frames = safe_int(align.get("aligned_written_frames"), 1159)
    full_start = best_offset
    full_end = best_offset + max(0, aligned_frames - 1)

    target = {
        "video_id": "-0bM0t0qS8Q",
        "review_id": "RLY0074",
        "full_start": full_start,
        "full_end": full_end,
        "local_start": 0,
        "local_end": max(0, aligned_frames - 1),
    }

    label_summaries = []
    all_label_rows = []

    for name in ["labels_visibility", "labels_click", "labels_traj"]:
        s = summarize_labels(name, root / INPUTS[name])
        label_summaries.append({k: v for k, v in s.items() if k != "rows_clean"})
        all_label_rows.extend(s["rows_clean"])

    label_frames_visible = set()
    label_frames_invisible = set()
    label_frames_clicked = set()

    for r in all_label_rows:
        fr = r["_frame"]
        if fr < 0:
            continue
        if r["_visible"] == "1":
            label_frames_visible.add(fr)
        elif r["_visible"] == "0":
            label_frames_invisible.add(fr)
        if math.isfinite(r["_x"]) and math.isfinite(r["_y"]):
            label_frames_clicked.add(fr)

    source_names = [
        "gap_candidates",
        "table_scored",
        "table_safe",
        "table_gated",
        "table_relinked",
        "table_context",
    ]

    source_audits = []
    for name in source_names:
        source_audits.append(candidate_source_audit(name, root / INPUTS[name], target))

    source_rows_csv = out_dir / "006D0_source_audit.csv"
    source_rows = []
    for a in source_audits:
        c = a["counts"]
        source_rows.append({
            "name": a["name"],
            "exists": "1" if a["exists"] else "0",
            "row_count": a["row_count"],
            "frame_col": a["frame_col"],
            "x_col": a["x_col"],
            "y_col": a["y_col"],
            "score_col": a["score_col"],
            "same_video_id": c.get("same_video_id", 0),
            "same_review_id": c.get("same_review_id", 0),
            "has_clip_range": c.get("has_clip_range", 0),
            "range_overlap": c.get("range_overlap", 0),
            "strict_segment_match": c.get("strict_segment_match", 0),
            "strict_unique_frame_count": a["strict_unique_frame_count"],
            "strict_min_frame": a["strict_min_frame"],
            "strict_max_frame": a["strict_max_frame"],
            "path": a["path"],
        })

    write_csv(
        source_rows_csv,
        source_rows,
        [
            "name", "exists", "row_count", "frame_col", "x_col", "y_col", "score_col",
            "same_video_id", "same_review_id", "has_clip_range", "range_overlap",
            "strict_segment_match", "strict_unique_frame_count",
            "strict_min_frame", "strict_max_frame", "path",
        ],
    )

    labels_csv = out_dir / "006D0_human_label_frames.csv"
    label_rows_out = []
    for fr in sorted(set(label_frames_visible) | set(label_frames_invisible) | set(label_frames_clicked)):
        label_rows_out.append({
            "frame": fr,
            "visible_label": "1" if fr in label_frames_visible else ("0" if fr in label_frames_invisible else ""),
            "has_click_or_traj_xy": "1" if fr in label_frames_clicked else "0",
        })
    write_csv(labels_csv, label_rows_out, ["frame", "visible_label", "has_click_or_traj_xy"])

    summary = {
        "patch": PATCH_ID,
        "target": target,
        "inputs": INPUTS,
        "align_summary": str(align_path),
        "label_summaries": label_summaries,
        "label_frame_counts": {
            "visible_unique_frames": len(label_frames_visible),
            "invisible_unique_frames": len(label_frames_invisible),
            "clicked_or_traj_unique_frames": len(label_frames_clicked),
            "all_unique_label_frames": len(set(label_frames_visible) | set(label_frames_invisible) | set(label_frames_clicked)),
        },
        "source_audits": source_audits,
        "outputs": {
            "source_audit_csv": str(source_rows_csv),
            "human_label_frames_csv": str(labels_csv),
        },
        "next": "Choisir la/les sources candidates avec strict_segment_match > 0 pour construire 006D1 training rows.",
    }

    summary_path = out_dir / "006D0_human_candidate_join_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print("=" * 72)
    print("")
    print("OK 006D0")
    print(f"summary          = {summary_path}")
    print(f"source_audit_csv = {source_rows_csv}")
    print(f"label_frames_csv = {labels_csv}")
    print("")
    print("TARGET")
    print(target)
    print("")
    print("LABELS")
    print(summary["label_frame_counts"])
    print("")
    print("SOURCES")
    for r in source_rows:
        print(
            f"{r['name']}: rows={r['row_count']} "
            f"strict={r['strict_segment_match']} "
            f"unique_frames={r['strict_unique_frame_count']} "
            f"review={r['same_review_id']} "
            f"range_overlap={r['range_overlap']} "
            f"frame_col={r['frame_col']} x={r['x_col']} y={r['y_col']}"
        )


if __name__ == "__main__":
    main()
