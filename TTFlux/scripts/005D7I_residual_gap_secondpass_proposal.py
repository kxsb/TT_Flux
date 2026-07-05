from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005D7I_residual_gap_secondpass_proposal"


def safe_float(v, default=np.nan):
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


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def source_class(src: str) -> str:
    s = str(src or "").lower()
    if "motion_bright" in s:
        return "motion_bright"
    if "bright" in s and "motion" in s:
        return "motion_bright"
    if "bright" in s:
        return "bright"
    if "motion" in s:
        return "motion"
    return "other"


def proposal_score(c: dict) -> float:
    dist = safe_float(c.get("diag_dist_interp"), 9999.0)
    raw = safe_float(c.get("score"), 0.0)
    area = safe_float(c.get("area"), 0.0)
    mean_motion = safe_float(c.get("mean_motion"), 0.0)
    max_gray = safe_float(c.get("max_gray"), 0.0)
    cls = source_class(c.get("source", ""))

    prox = max(0.0, 1.0 - min(dist, 140.0) / 140.0)
    area_bonus = max(0.0, 1.0 - abs(area - 18.0) / 130.0)
    bright_bonus = min(1.0, max_gray / 255.0)
    motion_bonus = min(1.0, mean_motion / 70.0)

    if cls == "motion_bright":
        src_bonus = 0.26
    elif cls == "bright":
        src_bonus = 0.10
    elif cls == "motion":
        src_bonus = -0.22
    else:
        src_bonus = -0.08

    return (
        0.45 * prox
        + 0.20 * raw
        + 0.10 * area_bonus
        + 0.08 * bright_bonus
        + 0.07 * motion_bonus
        + 0.10 * src_bonus
    )


def gate_candidate(c: dict, args):
    dist = safe_float(c.get("diag_dist_interp"), 9999.0)
    raw = safe_float(c.get("score"), 0.0)
    area = safe_float(c.get("area"), 0.0)
    cls = source_class(c.get("source", ""))
    score = proposal_score(c)

    if cls == "motion":
        return False, "reject_pure_motion", score

    if area < args.min_area or area > args.max_area:
        return False, "reject_area", score

    if dist <= args.strict_px and cls in {"motion_bright", "bright"} and score >= args.min_score_strict:
        return True, "accept_secondpass_strict", score

    if dist <= args.relaxed_px and cls == "motion_bright" and raw >= args.min_raw_motion_bright and score >= args.min_score_relaxed:
        return True, "accept_secondpass_motion_bright", score

    if dist <= args.bright_px and cls == "bright" and raw >= args.min_raw_bright and score >= args.min_score_bright:
        return True, "accept_secondpass_bright", score

    if dist > args.relaxed_px:
        return False, "reject_far_interp", score

    return False, "reject_low_confidence", score


def group_candidates(rows):
    by_frame = {}
    for r in rows:
        fr = safe_int(r.get("frame"))
        if fr < 0:
            continue
        by_frame.setdefault(fr, []).append(r)

    for fr in by_frame:
        by_frame[fr].sort(key=lambda r: safe_float(r.get("diag_dist_interp"), 9999.0))

    return by_frame


def load_frame_status(validation):
    still = set(int(x) for x in validation.get("still_missing_frames", []))
    filled = set(int(x) for x in validation.get("filled_frames", []))
    per_gap = validation.get("per_gap", [])

    gap_by_frame = {}
    for g in per_gap:
        gid = int(g["gap_id"])
        for fr in range(int(g["start"]), int(g["end"]) + 1):
            gap_by_frame[fr] = g

    return still, filled, gap_by_frame


def select_proposals(cands_by_frame, residual_frames, gap_by_frame, args):
    selected = []

    for fr in sorted(residual_frames):
        cands = cands_by_frame.get(fr, [])
        accepted = []
        rejected = []

        for c in cands:
            ok, reason, score = gate_candidate(c, args)
            cc = dict(c)
            cc["proposal_score"] = f"{score:.6f}"
            cc["proposal_gate"] = reason
            cc["proposal_source_class"] = source_class(cc.get("source", ""))

            if ok:
                accepted.append(cc)
            else:
                rejected.append(cc)

        accepted.sort(key=lambda r: safe_float(r.get("proposal_score"), -999), reverse=True)
        rejected.sort(key=lambda r: safe_float(r.get("proposal_score"), -999), reverse=True)

        g = gap_by_frame.get(fr, {})
        if accepted:
            best = accepted[0]
            status = "proposed"
        elif rejected:
            best = rejected[0]
            status = "rejected"
        else:
            best = {"frame": fr}
            status = "no_candidate"

        best = dict(best)
        best["proposal_status"] = status
        best["gap_id"] = g.get("gap_id", "")
        best["gap_start"] = g.get("start", "")
        best["gap_end"] = g.get("end", "")
        best["gap_len"] = g.get("len", "")
        best["accepted_count_this_frame"] = len(accepted)
        best["candidate_count_this_frame"] = len(cands)

        selected.append(best)

    return selected


def draw_overlay(video_path: Path, out_path: Path, selected_rows):
    if not video_path.exists():
        print(f"WARN: vidéo introuvable: {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARN: impossible d'ouvrir vidéo: {video_path}")
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        print(f"WARN: impossible d'écrire overlay: {out_path}")
        return False

    by_frame = {safe_int(r.get("frame")): r for r in selected_rows}
    target = set(by_frame)

    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        if frame_idx not in target:
            continue

        out = frame.copy()
        r = by_frame[frame_idx]

        px = safe_float(r.get("pred_x"))
        py = safe_float(r.get("pred_y"))
        if math.isfinite(px) and math.isfinite(py):
            cv2.drawMarker(out, (int(round(px)), int(round(py))), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.circle(out, (int(round(px)), int(round(py))), 55, (0, 0, 180), 1)
            cv2.circle(out, (int(round(px)), int(round(py))), 95, (0, 0, 120), 1)

        x = safe_float(r.get("x"))
        y = safe_float(r.get("y"))
        status = r.get("proposal_status", "")

        if math.isfinite(x) and math.isfinite(y):
            if status == "proposed":
                color = (255, 0, 255)
                label = "PROPOSED"
            elif status == "rejected":
                color = (100, 100, 100)
                label = "REJECT"
            else:
                color = (0, 0, 255)
                label = "NONE"

            cv2.circle(out, (int(round(x)), int(round(y))), 14, color, 2)
            cv2.putText(out, label, (int(round(x)) + 14, int(round(y)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        header = (
            f"{PATCH_ID} | frame={frame_idx} | {status} | "
            f"gate={r.get('proposal_gate','')} | d={r.get('diag_dist_interp','')} | score={r.get('proposal_score','')}"
        )
        cv2.rectangle(out, (8, 8), (min(w - 8, 1240), 42), (0, 0, 0), -1)
        cv2.putText(out, header, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(out)

    cap.release()
    writer.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-summary", default="runs/005D7F_target_gap_validation/005D7F_target_gap_validation_summary.json")
    ap.add_argument("--atlas-candidates", default="runs/005D7H_residual_gap_atlas/005D7H_residual_gap_candidates_ranked.csv")
    ap.add_argument("--out-dir", default="runs/005D7I_residual_secondpass")
    ap.add_argument("--strict-px", type=float, default=42.0)
    ap.add_argument("--relaxed-px", type=float, default=78.0)
    ap.add_argument("--bright-px", type=float, default=54.0)
    ap.add_argument("--min-area", type=float, default=2.0)
    ap.add_argument("--max-area", type=float, default=130.0)
    ap.add_argument("--min_raw_motion_bright", type=float, default=0.47)
    ap.add_argument("--min_raw_bright", type=float, default=0.55)
    ap.add_argument("--min_score_strict", type=float, default=0.36)
    ap.add_argument("--min_score_relaxed", type=float, default=0.39)
    ap.add_argument("--min_score_bright", type=float, default=0.41)
    args = ap.parse_args()

    validation_path = Path(args.validation_summary)
    atlas_candidates = Path(args.atlas_candidates)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    video_path = Path(validation["video"])

    rows, fields = read_csv(atlas_candidates)
    cands_by_frame = group_candidates(rows)

    residual_frames, filled_frames, gap_by_frame = load_frame_status(validation)
    selected = select_proposals(cands_by_frame, residual_frames, gap_by_frame, args)

    proposed = [r for r in selected if r.get("proposal_status") == "proposed"]
    rejected = [r for r in selected if r.get("proposal_status") != "proposed"]

    out_fields = [
        "proposal_status",
        "proposal_gate",
        "proposal_score",
        "proposal_source_class",
        "gap_id",
        "gap_start",
        "gap_end",
        "gap_len",
        "frame",
        "rank",
        "x",
        "y",
        "r",
        "area",
        "source",
        "score",
        "dist_pred",
        "diag_dist_interp",
        "mean_gray",
        "max_gray",
        "mean_motion",
        "pred_x",
        "pred_y",
        "left_frame",
        "right_frame",
        "accepted_count_this_frame",
        "candidate_count_this_frame",
    ]

    selected_csv = out_dir / "005D7I_residual_secondpass_selected_all.csv"
    proposed_csv = out_dir / "005D7I_residual_secondpass_proposed_only.csv"
    summary_path = out_dir / "005D7I_residual_secondpass_summary.json"
    overlay_path = out_dir / "005D7I_residual_secondpass_overlay.mp4"

    write_csv(selected_csv, selected, out_fields)
    write_csv(proposed_csv, proposed, out_fields)

    overlay_ok = draw_overlay(video_path, overlay_path, selected)

    gate_counts = {}
    source_counts = {}
    for r in selected:
        gate_counts[r.get("proposal_gate", "no_candidate")] = gate_counts.get(r.get("proposal_gate", "no_candidate"), 0) + 1
        source_counts[r.get("proposal_source_class", "")] = source_counts.get(r.get("proposal_source_class", ""), 0) + 1

    dists = [
        safe_float(r.get("diag_dist_interp"))
        for r in proposed
        if math.isfinite(safe_float(r.get("diag_dist_interp")))
    ]

    summary = {
        "patch": PATCH_ID,
        "validation_summary": str(validation_path),
        "atlas_candidates": str(atlas_candidates),
        "video": str(video_path),
        "sequence_key": validation.get("sequence_key"),
        "residual_frame_count": len(residual_frames),
        "proposed_count": len(proposed),
        "rejected_count": len(rejected),
        "proposal_ratio": len(proposed) / max(1, len(residual_frames)),
        "proposed_frames": [safe_int(r.get("frame")) for r in proposed],
        "still_rejected_frames": [safe_int(r.get("frame")) for r in rejected],
        "proposal_dist_median": float(np.median(dists)) if dists else None,
        "proposal_dist_p90": float(np.percentile(dists, 90)) if dists else None,
        "gate_counts": gate_counts,
        "source_counts": source_counts,
        "selected_csv": str(selected_csv),
        "proposed_csv": str(proposed_csv),
        "overlay": str(overlay_path) if overlay_ok else None,
        "params": vars(args),
        "note": "Proposition second-pass uniquement. Ne pas promouvoir sans validation overlay/contact sheet.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"validation       = {validation_path}")
    print(f"atlas_candidates = {atlas_candidates}")
    print(f"video            = {video_path}")
    print("=" * 72)
    print("")
    print("OK 005D7I")
    print(f"selected_csv = {selected_csv}")
    print(f"proposed_csv = {proposed_csv}")
    print(f"summary      = {summary_path}")
    print(f"overlay      = {overlay_path if overlay_ok else 'NON_ECRIT'}")
    print("")
    print(f"residual_frame_count={len(residual_frames)}")
    print(f"proposed_count={len(proposed)} rejected_count={len(rejected)}")
    print(f"proposed_frames={[safe_int(r.get('frame')) for r in proposed]}")
    if dists:
        print(f"proposal_dist_median={summary['proposal_dist_median']:.2f} p90={summary['proposal_dist_p90']:.2f}")
    print(f"gate_counts={gate_counts}")
    print(f"source_counts={source_counts}")


if __name__ == "__main__":
    main()
