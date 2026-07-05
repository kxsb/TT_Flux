from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005D7J_secondpass_proposal_audit"


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


def find_col(cols, names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def source_class(src):
    s = str(src or "").lower()
    if "motion_bright" in s:
        return "motion_bright"
    if "bright" in s:
        return "bright"
    if "motion" in s:
        return "motion"
    return "other"


def read_track(path: Path):
    rows, fields = read_csv(path)

    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])
    score_col = find_col(fields, ["score", "conf", "confidence", "prob"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError(f"Colonnes frame/x/y introuvables dans {path}")

    points = {}
    duplicates = []

    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        sc = safe_float(r.get(score_col), 1.0) if score_col else 1.0

        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            continue

        if fr in points:
            duplicates.append(fr)

        points[fr] = {
            "frame": fr,
            "x": x,
            "y": y,
            "score": sc,
            "row": dict(r),
            "point_source": str(r.get("point_source", "")),
        }

    meta = {
        "rows": len(rows),
        "valid_points": len(points),
        "duplicate_frames": sorted(set(duplicates)),
        "frame_col": frame_col,
        "x_col": x_col,
        "y_col": y_col,
        "score_col": score_col,
        "fields": fields,
    }

    return rows, fields, points, meta


def nearest_neighbors(points, fr):
    frames = sorted(points)
    left = None
    right = None

    for f in frames:
        if f < fr:
            left = f
        elif f > fr:
            right = f
            break

    return left, right


def interp(points, lf, rf, fr):
    if lf is None or rf is None or lf == rf:
        return None

    a = points[lf]
    b = points[rf]
    t = (fr - lf) / float(rf - lf)
    x = a["x"] * (1 - t) + b["x"] * t
    y = a["y"] * (1 - t) + b["y"] * t
    return x, y, t


def audit_proposals(proposed_rows, points, args):
    audited = []

    for r in proposed_rows:
        fr = safe_int(r.get("frame"))
        x = safe_float(r.get("x"))
        y = safe_float(r.get("y"))

        out = dict(r)
        out["audit_patch"] = PATCH_ID

        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            out["audit_status"] = "rejected"
            out["audit_reason"] = "invalid_xy"
            audited.append(out)
            continue

        if fr in points:
            out["audit_status"] = "rejected"
            out["audit_reason"] = "frame_already_present"
            audited.append(out)
            continue

        cls = source_class(r.get("source", ""))
        if cls == "motion":
            out["audit_status"] = "rejected"
            out["audit_reason"] = "pure_motion"
            audited.append(out)
            continue

        lf, rf = nearest_neighbors(points, fr)
        out["left_frame"] = "" if lf is None else lf
        out["right_frame"] = "" if rf is None else rf

        if lf is None or rf is None:
            out["audit_status"] = "rejected"
            out["audit_reason"] = "missing_bridge"
            audited.append(out)
            continue

        pred = interp(points, lf, rf, fr)
        if pred is None:
            out["audit_status"] = "rejected"
            out["audit_reason"] = "bad_bridge"
            audited.append(out)
            continue

        px, py, t = pred
        bridge_error = math.hypot(x - px, y - py)
        speed_left = math.hypot(x - points[lf]["x"], y - points[lf]["y"]) / max(1, fr - lf)
        speed_right = math.hypot(points[rf]["x"] - x, points[rf]["y"] - y) / max(1, rf - fr)

        diag_dist = safe_float(r.get("diag_dist_interp"), bridge_error)
        prop_score = safe_float(r.get("proposal_score"), 0.0)
        raw_score = safe_float(r.get("score"), 0.0)
        area = safe_float(r.get("area"), 0.0)

        out["bridge_interp_x"] = f"{px:.3f}"
        out["bridge_interp_y"] = f"{py:.3f}"
        out["bridge_t"] = f"{t:.6f}"
        out["bridge_error_px"] = f"{bridge_error:.3f}"
        out["speed_left_px_f"] = f"{speed_left:.3f}"
        out["speed_right_px_f"] = f"{speed_right:.3f}"
        out["audit_source_class"] = cls

        reasons = []

        if bridge_error > args.max_bridge_error_px:
            reasons.append("bridge_error_high")
        if diag_dist > args.max_diag_dist_px:
            reasons.append("diag_dist_high")
        if speed_left > args.max_speed_px_f:
            reasons.append("speed_left_high")
        if speed_right > args.max_speed_px_f:
            reasons.append("speed_right_high")
        if prop_score < args.min_proposal_score:
            reasons.append("proposal_score_low")
        if raw_score < args.min_raw_score:
            reasons.append("raw_score_low")
        if area < args.min_area or area > args.max_area:
            reasons.append("area_bad")

        if reasons:
            out["audit_status"] = "rejected"
            out["audit_reason"] = "|".join(reasons)
        else:
            out["audit_status"] = "accepted_safe"
            out["audit_reason"] = "ok"

        audited.append(out)

    return audited


def build_preview_rows(track_rows, track_fields, track_meta, safe_rows):
    frame_col = track_meta["frame_col"]
    x_col = track_meta["x_col"]
    y_col = track_meta["y_col"]
    score_col = track_meta["score_col"]

    extra_fields = [
        "point_source",
        "fill_patch",
        "fill_from_patch",
        "fill_safe_status",
        "fill_safe_reason",
        "fill_gap_id",
        "fill_bridge_error_px",
        "fill_proposal_score",
        "fill_source",
        "fill_dist_interp",
    ]

    fields = list(track_fields)
    for f in extra_fields:
        if f not in fields:
            fields.append(f)

    rows = []
    for r in track_rows:
        rr = dict(r)
        for f in extra_fields:
            rr.setdefault(f, "")
        rows.append(rr)

    template = dict(track_rows[0]) if track_rows else {}

    for r in safe_rows:
        fr = safe_int(r.get("frame"))
        x = safe_float(r.get("x"))
        y = safe_float(r.get("y"))
        sc = safe_float(r.get("proposal_score"), safe_float(r.get("score"), 0.0))

        rr = {k: template.get(k, "") for k in fields}
        rr[frame_col] = str(fr)
        rr[x_col] = f"{x:.6f}"
        rr[y_col] = f"{y:.6f}"
        if score_col:
            rr[score_col] = f"{sc:.6f}"

        rr["point_source"] = "gapfill_005D7J_SECOND_PASS_PREVIEW"
        rr["fill_patch"] = PATCH_ID
        rr["fill_from_patch"] = "005D7I"
        rr["fill_safe_status"] = r.get("audit_status", "")
        rr["fill_safe_reason"] = r.get("audit_reason", "")
        rr["fill_gap_id"] = r.get("gap_id", "")
        rr["fill_bridge_error_px"] = r.get("bridge_error_px", "")
        rr["fill_proposal_score"] = r.get("proposal_score", "")
        rr["fill_source"] = r.get("source", "")
        rr["fill_dist_interp"] = r.get("diag_dist_interp", "")

        rows.append(rr)

    rows.sort(key=lambda row: safe_int(row.get(frame_col)))
    return rows, fields


def frame_map_from_rows(rows, fields):
    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])

    out = {}
    dupes = []

    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            continue
        if fr in out:
            dupes.append(fr)
        out[fr] = (x, y, r)

    return out, sorted(set(dupes))


def continuity_stats(points):
    frames = sorted(points)
    speeds = []
    large = []

    for a, b in zip(frames[:-1], frames[1:]):
        dt = b - a
        if dt <= 0:
            continue
        ax, ay, _ = points[a]
        bx, by, _ = points[b]
        dist = math.hypot(bx - ax, by - ay)
        speed = dist / dt
        speeds.append(speed)

        if speed > 180 or dist > 260:
            large.append({
                "from": a,
                "to": b,
                "dt": dt,
                "dist": dist,
                "speed": speed,
            })

    if not speeds:
        return {
            "step_count": 0,
            "speed_median": None,
            "speed_p90": None,
            "speed_p95": None,
            "speed_max": None,
            "large_step_count": 0,
            "largest_steps": [],
        }

    return {
        "step_count": len(speeds),
        "speed_median": float(np.median(speeds)),
        "speed_p90": float(np.percentile(speeds, 90)),
        "speed_p95": float(np.percentile(speeds, 95)),
        "speed_max": float(np.max(speeds)),
        "large_step_count": len(large),
        "largest_steps": sorted(large, key=lambda x: x["speed"], reverse=True)[:20],
    }


def compute_target_missing(per_gap, points):
    present = set(points)
    missing = []
    for g in per_gap:
        for fr in range(int(g["start"]), int(g["end"]) + 1):
            if fr not in present:
                missing.append(fr)
    return missing


def draw_overlay(video_path: Path, out_path: Path, before_points, after_points, audited_rows):
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

    by_frame = {safe_int(r.get("frame")): r for r in audited_rows}
    target = set(by_frame)
    trail = 18
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        if frame_idx not in target:
            continue

        out = frame.copy()

        for fr in range(frame_idx - trail, frame_idx + trail + 1):
            p = before_points.get(fr)
            if p:
                x, y, _ = p
                cv2.circle(out, (int(round(x)), int(round(y))), 3, (150, 150, 150), -1)

        for fr in range(frame_idx - trail, frame_idx + trail + 1):
            p = after_points.get(fr)
            if p:
                x, y, row = p
                src = str(row.get("point_source", ""))
                if src.startswith("gapfill_005D7J"):
                    cv2.circle(out, (int(round(x)), int(round(y))), 8, (255, 0, 255), 2)
                elif src.startswith("gapfill"):
                    cv2.circle(out, (int(round(x)), int(round(y))), 5, (180, 0, 255), 1)
                else:
                    cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 220, 255), -1)

        r = by_frame[frame_idx]

        px = safe_float(r.get("bridge_interp_x"))
        py = safe_float(r.get("bridge_interp_y"))
        if math.isfinite(px) and math.isfinite(py):
            cv2.drawMarker(out, (int(round(px)), int(round(py))), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.circle(out, (int(round(px)), int(round(py))), 55, (0, 0, 180), 1)

        x = safe_float(r.get("x"))
        y = safe_float(r.get("y"))
        status = r.get("audit_status", "")

        if math.isfinite(x) and math.isfinite(y):
            color = (255, 0, 255) if status == "accepted_safe" else (100, 100, 100)
            label = "SAFE" if status == "accepted_safe" else "REJECT"
            cv2.circle(out, (int(round(x)), int(round(y))), 14, color, 2)
            cv2.putText(out, label, (int(round(x)) + 14, int(round(y)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        header = (
            f"{PATCH_ID} | frame={frame_idx} | {status} | "
            f"err={r.get('bridge_error_px','')} | {r.get('audit_reason','')}"
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
    ap.add_argument("--track-csv", default="runs/005D7G_gapfilled_canonical/005D7G_ball_points_gapfilled_canonical_unique.csv")
    ap.add_argument("--proposed-csv", default="runs/005D7I_residual_secondpass/005D7I_residual_secondpass_proposed_only.csv")
    ap.add_argument("--out-dir", default="runs/005D7J_secondpass_audit")

    # Seuils un peu stricts : c'est une 2e passe, on ne veut pas dégrader.
    ap.add_argument("--max-bridge-error-px", type=float, default=58.0)
    ap.add_argument("--max-diag-dist-px", type=float, default=55.0)
    ap.add_argument("--max-speed-px-f", type=float, default=125.0)
    ap.add_argument("--min-proposal-score", type=float, default=0.34)
    ap.add_argument("--min-raw-score", type=float, default=0.36)
    ap.add_argument("--min-area", type=float, default=2.0)
    ap.add_argument("--max-area", type=float, default=130.0)

    args = ap.parse_args()

    validation_path = Path(args.validation_summary)
    track_csv = Path(args.track_csv)
    proposed_csv = Path(args.proposed_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not validation_path.exists():
        raise FileNotFoundError(validation_path)
    if not track_csv.exists():
        raise FileNotFoundError(track_csv)
    if not proposed_csv.exists():
        raise FileNotFoundError(proposed_csv)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    video_path = Path(validation["video"])

    track_rows, track_fields, points, track_meta = read_track(track_csv)
    proposed_rows, _ = read_csv(proposed_csv)

    audited = audit_proposals(proposed_rows, points, args)
    safe = [r for r in audited if r.get("audit_status") == "accepted_safe"]
    rejected = [r for r in audited if r.get("audit_status") != "accepted_safe"]

    preview_rows, preview_fields = build_preview_rows(track_rows, track_fields, track_meta, safe)
    before_map, before_dupes = frame_map_from_rows(track_rows, track_fields)
    after_map, after_dupes = frame_map_from_rows(preview_rows, preview_fields)

    per_gap = validation.get("per_gap", [])
    missing_before = compute_target_missing(per_gap, before_map)
    missing_after = compute_target_missing(per_gap, after_map)

    cont_before = continuity_stats(before_map)
    cont_after = continuity_stats(after_map)

    audit_csv = out_dir / "005D7J_secondpass_audit_all.csv"
    safe_csv = out_dir / "005D7J_secondpass_safe_only.csv"
    preview_csv = out_dir / "005D7J_secondpass_preview_track.csv"
    overlay_path = out_dir / "005D7J_secondpass_audit_overlay.mp4"
    summary_path = out_dir / "005D7J_secondpass_audit_summary.json"

    audit_fields = [
        "audit_patch",
        "audit_status",
        "audit_reason",
        "frame",
        "gap_id",
        "gap_start",
        "gap_end",
        "gap_len",
        "x",
        "y",
        "source",
        "audit_source_class",
        "score",
        "proposal_score",
        "proposal_gate",
        "diag_dist_interp",
        "bridge_interp_x",
        "bridge_interp_y",
        "bridge_t",
        "bridge_error_px",
        "speed_left_px_f",
        "speed_right_px_f",
        "left_frame",
        "right_frame",
        "area",
        "mean_gray",
        "max_gray",
        "mean_motion",
    ]

    write_csv(audit_csv, audited, audit_fields)
    write_csv(safe_csv, safe, audit_fields)
    write_csv(preview_csv, preview_rows, preview_fields)

    overlay_ok = draw_overlay(video_path, overlay_path, before_map, after_map, audited)

    reason_counts = {}
    for r in audited:
        reason_counts[r.get("audit_reason", "")] = reason_counts.get(r.get("audit_reason", ""), 0) + 1

    bridge_errors = [
        safe_float(r.get("bridge_error_px"))
        for r in safe
        if math.isfinite(safe_float(r.get("bridge_error_px")))
    ]

    safe_frames = [safe_int(r.get("frame")) for r in safe]

    ok_for_next_promotion = (
        len(after_dupes) == 0
        and len(safe) > 0
        and all(fr not in missing_after for fr in safe_frames)
        and len(missing_after) == len(missing_before) - len(safe)
        and cont_after["large_step_count"] <= cont_before["large_step_count"] + 2
    )

    summary = {
        "patch": PATCH_ID,
        "validation_summary": str(validation_path),
        "track_csv": str(track_csv),
        "proposed_csv": str(proposed_csv),
        "video": str(video_path),
        "input_proposed_count": len(proposed_rows),
        "safe_secondpass_count": len(safe),
        "rejected_secondpass_count": len(rejected),
        "safe_frames": safe_frames,
        "rejected_frames": [safe_int(r.get("frame")) for r in rejected],
        "reason_counts": reason_counts,
        "bridge_error_median": float(np.median(bridge_errors)) if bridge_errors else None,
        "bridge_error_p90": float(np.percentile(bridge_errors, 90)) if bridge_errors else None,
        "target_missing_before": len(missing_before),
        "target_missing_after_preview": len(missing_after),
        "target_missing_after_frames": missing_after,
        "before_duplicate_frames": before_dupes,
        "after_duplicate_frames": after_dupes,
        "continuity_before": cont_before,
        "continuity_after_preview": cont_after,
        "ok_for_next_promotion": ok_for_next_promotion,
        "audit_csv": str(audit_csv),
        "safe_csv": str(safe_csv),
        "preview_csv": str(preview_csv),
        "overlay": str(overlay_path) if overlay_ok else None,
        "params": vars(args),
        "note": "Audit second-pass. Preview seulement. Promotion séparée si ok_for_next_promotion=true.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"validation = {validation_path}")
    print(f"track_csv  = {track_csv}")
    print(f"proposed   = {proposed_csv}")
    print(f"video      = {video_path}")
    print("=" * 72)
    print("")
    print("OK 005D7J")
    print(f"audit_csv   = {audit_csv}")
    print(f"safe_csv    = {safe_csv}")
    print(f"preview_csv = {preview_csv}")
    print(f"summary     = {summary_path}")
    print(f"overlay     = {overlay_path if overlay_ok else 'NON_ECRIT'}")
    print("")
    print(f"input_proposed_count={len(proposed_rows)}")
    print(f"safe_secondpass_count={len(safe)} rejected_secondpass_count={len(rejected)}")
    print(f"safe_frames={safe_frames}")
    if bridge_errors:
        print(f"bridge_error_median={summary['bridge_error_median']:.2f} p90={summary['bridge_error_p90']:.2f}")
    print(f"target_missing_before={len(missing_before)}")
    print(f"target_missing_after_preview={len(missing_after)}")
    print(f"continuity_before_large_step_count={cont_before['large_step_count']}")
    print(f"continuity_after_large_step_count={cont_after['large_step_count']}")
    print(f"after_duplicate_frames={after_dupes}")
    print(f"reason_counts={reason_counts}")
    print(f"ok_for_next_promotion={ok_for_next_promotion}")


if __name__ == "__main__":
    main()
