from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005D7D_safe_gapfill_injection_audit"


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


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def infer_from_accepted(rows: list[dict], args):
    video = Path(args.video) if args.video else None
    track = Path(args.track_csv) if args.track_csv else None
    sequence = args.sequence_key or ""

    if rows:
        r0 = rows[0]
        if video is None and r0.get("video"):
            video = Path(str(r0["video"]))
        if track is None and r0.get("track_csv"):
            track = Path(str(r0["track_csv"]))
        if not sequence and r0.get("chosen_sequence"):
            sequence = str(r0["chosen_sequence"])

    return video, track, sequence


def read_track_points(track_csv: Path, sequence_key: str | None) -> dict[int, dict]:
    rows = read_csv_rows(track_csv)
    if not rows:
        return {}

    cols = list(rows[0].keys())
    frame_col = find_col(cols, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(cols, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(cols, ["y", "cy", "ball_y", "center_y"])
    score_col = find_col(cols, ["score", "conf", "confidence", "prob"])
    seq_col = find_col(cols, ["sequence_key", "clip_id", "video_id", "segment_id", "source_id"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError(f"Colonnes track invalides dans {track_csv}")

    if seq_col and sequence_key:
        filtered = [r for r in rows if str(r.get(seq_col, "")).strip() == sequence_key]
        if filtered:
            rows = filtered

    out = {}
    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        sc = safe_float(r.get(score_col), 1.0) if score_col else 1.0

        if fr >= 0 and math.isfinite(x) and math.isfinite(y):
            out[fr] = {
                "frame": fr,
                "x": x,
                "y": y,
                "score": sc,
                "point_source": "base_005D4",
            }

    return out


def base_speed_stats(base: dict[int, dict]) -> dict:
    frames = sorted(base)
    speeds = []
    for a, b in zip(frames[:-1], frames[1:]):
        dt = b - a
        if dt <= 0 or dt > 3:
            continue
        dx = base[b]["x"] - base[a]["x"]
        dy = base[b]["y"] - base[a]["y"]
        speeds.append(math.hypot(dx, dy) / dt)

    if not speeds:
        return {
            "count": 0,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": len(speeds),
        "median": float(np.median(speeds)),
        "p90": float(np.percentile(speeds, 90)),
        "p95": float(np.percentile(speeds, 95)),
        "max": float(np.max(speeds)),
    }


def nearest_base_neighbors(base_frames: list[int], fr: int):
    left = None
    right = None
    for f in base_frames:
        if f < fr:
            left = f
        elif f > fr:
            right = f
            break
    return left, right


def interp_between(p0, f0, p1, f1, fr):
    if f1 == f0:
        return None
    t = (fr - f0) / float(f1 - f0)
    x = p0["x"] * (1 - t) + p1["x"] * t
    y = p0["y"] * (1 - t) + p1["y"] * t
    return x, y, t


def audit_accepted(accepted_rows: list[dict], base: dict[int, dict], args) -> list[dict]:
    base_frames = sorted(base)
    audited = []

    for r in accepted_rows:
        fr = safe_int(r.get("frame"))
        x = safe_float(r.get("x"))
        y = safe_float(r.get("y"))

        row = dict(r)
        row["patch"] = PATCH_ID

        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            row["safe_status"] = "reject_invalid_xy"
            row["safe_reason"] = "invalid_xy"
            audited.append(row)
            continue

        if fr in base:
            row["safe_status"] = "reject_already_base"
            row["safe_reason"] = "frame_already_has_base_point"
            audited.append(row)
            continue

        lf, rf = nearest_base_neighbors(base_frames, fr)
        row["left_base_frame"] = "" if lf is None else lf
        row["right_base_frame"] = "" if rf is None else rf

        if lf is None or rf is None:
            row["safe_status"] = "reject_no_bridge"
            row["safe_reason"] = "missing_left_or_right_base_neighbor"
            audited.append(row)
            continue

        pred = interp_between(base[lf], lf, base[rf], rf, fr)
        if pred is None:
            row["safe_status"] = "reject_bad_bridge"
            row["safe_reason"] = "invalid_bridge"
            audited.append(row)
            continue

        ix, iy, t = pred
        bridge_error = math.hypot(x - ix, y - iy)
        speed_left = math.hypot(x - base[lf]["x"], y - base[lf]["y"]) / max(1, fr - lf)
        speed_right = math.hypot(base[rf]["x"] - x, base[rf]["y"] - y) / max(1, rf - fr)

        row["bridge_interp_x"] = f"{ix:.3f}"
        row["bridge_interp_y"] = f"{iy:.3f}"
        row["bridge_t"] = f"{t:.6f}"
        row["bridge_error_px"] = f"{bridge_error:.3f}"
        row["speed_left_px_f"] = f"{speed_left:.3f}"
        row["speed_right_px_f"] = f"{speed_right:.3f}"

        reasons = []
        if bridge_error > args.max_bridge_error_px:
            reasons.append("bridge_error_high")

        if speed_left > args.max_speed_px_f:
            reasons.append("speed_left_high")

        if speed_right > args.max_speed_px_f:
            reasons.append("speed_right_high")

        strict_score = safe_float(row.get("strict_score"), 0.0)
        if strict_score < args.min_strict_score:
            reasons.append("strict_score_low")

        if reasons:
            row["safe_status"] = "rejected"
            row["safe_reason"] = "|".join(reasons)
        else:
            row["safe_status"] = "accepted_safe"
            row["safe_reason"] = "ok"

        audited.append(row)

    return audited


def build_merged(base: dict[int, dict], audited: list[dict], sequence: str) -> list[dict]:
    merged = []

    for fr in sorted(base):
        p = base[fr]
        merged.append({
            "sequence_key": sequence,
            "frame": fr,
            "x": f"{p['x']:.3f}",
            "y": f"{p['y']:.3f}",
            "score": f"{p['score']:.6f}",
            "point_source": "base_005D4",
            "gap_id": "",
            "safe_status": "",
            "safe_reason": "",
            "bridge_error_px": "",
        })

    for r in audited:
        if r.get("safe_status") != "accepted_safe":
            continue

        fr = safe_int(r.get("frame"))
        if fr in base:
            continue

        merged.append({
            "sequence_key": sequence,
            "frame": fr,
            "x": f"{safe_float(r.get('x')):.3f}",
            "y": f"{safe_float(r.get('y')):.3f}",
            "score": f"{safe_float(r.get('strict_score'), 0.0):.6f}",
            "point_source": "gapfill_005D7D_SAFE_PREVIEW",
            "gap_id": r.get("gap_id", ""),
            "safe_status": r.get("safe_status", ""),
            "safe_reason": r.get("safe_reason", ""),
            "bridge_error_px": r.get("bridge_error_px", ""),
        })

    merged.sort(key=lambda r: safe_int(r.get("frame")))
    return merged


def draw_overlay(frame, frame_idx, merged_by_frame, base, audited_by_frame, trail=18):
    out = frame.copy()

    # Trail local.
    for fr in range(frame_idx - trail, frame_idx + trail + 1):
        p = merged_by_frame.get(fr)
        if not p:
            continue

        x = safe_float(p.get("x"))
        y = safe_float(p.get("y"))
        if not math.isfinite(x) or not math.isfinite(y):
            continue

        src = p.get("point_source", "")
        if src.startswith("gapfill"):
            color = (255, 0, 255)
            radius = 5
        else:
            color = (180, 180, 180)
            radius = 3

        cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1)

    # Point courant si accepté/rejeté dans audit.
    audit = audited_by_frame.get(frame_idx)
    if audit:
        x = safe_float(audit.get("x"))
        y = safe_float(audit.get("y"))
        status = audit.get("safe_status", "")
        color = (255, 0, 255) if status == "accepted_safe" else (90, 90, 90)

        if math.isfinite(x) and math.isfinite(y):
            cv2.circle(out, (int(round(x)), int(round(y))), 14, color, 2)
            cv2.putText(
                out,
                "SAFE" if status == "accepted_safe" else "REJECT",
                (int(round(x)) + 14, int(round(y)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        ix = safe_float(audit.get("bridge_interp_x"))
        iy = safe_float(audit.get("bridge_interp_y"))
        if math.isfinite(ix) and math.isfinite(iy):
            cv2.drawMarker(
                out,
                (int(round(ix)), int(round(iy))),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                18,
                2,
            )

    label = f"{PATCH_ID} | frame={frame_idx}"
    if audit:
        label += f" | {audit.get('safe_status', '')} | err={audit.get('bridge_error_px', '')} | {audit.get('safe_reason', '')}"

    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 8, 1180), 42), (0, 0, 0), -1)
    cv2.putText(out, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def write_overlay(video_path: Path, out_path: Path, merged_rows, base, audited, full_video=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARN: impossible d'ouvrir la vidéo: {video_path}")
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"WARN: impossible d'écrire l'overlay: {out_path}")
        cap.release()
        return False

    merged_by_frame = {safe_int(r.get("frame")): r for r in merged_rows}
    audited_by_frame = {safe_int(r.get("frame")): r for r in audited}
    target_frames = set(audited_by_frame.keys())

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        if full_video or frame_idx in target_frames:
            out = draw_overlay(frame, frame_idx, merged_by_frame, base, audited_by_frame)
            writer.write(out)

    cap.release()
    writer.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted-csv", default="runs/005D7C_strict_gap_selected/005D7C_strict_accepted_only.csv")
    ap.add_argument("--video", default="")
    ap.add_argument("--track-csv", default="")
    ap.add_argument("--sequence-key", default="")
    ap.add_argument("--out-dir", default="runs/005D7D_safe_gapfill")
    ap.add_argument("--max-bridge-error-px", type=float, default=82.0)
    ap.add_argument("--max-speed-px-f", type=float, default=95.0)
    ap.add_argument("--min-strict-score", type=float, default=0.34)
    ap.add_argument("--full-video-overlay", action="store_true")
    args = ap.parse_args()

    accepted_csv = Path(args.accepted_csv)
    if not accepted_csv.exists():
        raise FileNotFoundError(f"accepted_csv introuvable: {accepted_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted_rows = read_csv_rows(accepted_csv)
    video_path, track_csv, sequence = infer_from_accepted(accepted_rows, args)

    if track_csv is None or not track_csv.exists():
        raise FileNotFoundError(f"track_csv introuvable: {track_csv}")

    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"video introuvable: {video_path}")

    base = read_track_points(track_csv, sequence)
    speed_stats = base_speed_stats(base)

    audited = audit_accepted(accepted_rows, base, args)
    safe = [r for r in audited if r.get("safe_status") == "accepted_safe"]
    rejected = [r for r in audited if r.get("safe_status") != "accepted_safe"]

    merged = build_merged(base, audited, sequence)

    audit_fields = [
        "patch",
        "safe_status",
        "safe_reason",
        "frame",
        "x",
        "y",
        "strict_score",
        "strict_gate_reason",
        "source",
        "source_class",
        "score",
        "dist_pred",
        "bridge_interp_x",
        "bridge_interp_y",
        "bridge_t",
        "bridge_error_px",
        "speed_left_px_f",
        "speed_right_px_f",
        "left_base_frame",
        "right_base_frame",
        "gap_id",
        "gap_start",
        "gap_end",
        "video",
        "track_csv",
        "chosen_sequence",
    ]

    audit_csv = out_dir / "005D7D_safe_gapfill_audit.csv"
    safe_csv = out_dir / "005D7D_safe_gapfill_accepted.csv"
    merged_csv = out_dir / "005D7D_safe_gapfill_preview_track.csv"
    overlay_path = out_dir / "005D7D_safe_gapfill_overlay.mp4"

    write_csv(audit_csv, audited, audit_fields)
    write_csv(safe_csv, safe, audit_fields)
    write_csv(
        merged_csv,
        merged,
        ["sequence_key", "frame", "x", "y", "score", "point_source", "gap_id", "safe_status", "safe_reason", "bridge_error_px"],
    )

    overlay_ok = write_overlay(
        video_path=video_path,
        out_path=overlay_path,
        merged_rows=merged,
        base=base,
        audited=audited,
        full_video=args.full_video_overlay,
    )

    bridge_errors = [
        safe_float(r.get("bridge_error_px"))
        for r in safe
        if math.isfinite(safe_float(r.get("bridge_error_px")))
    ]

    reason_counts = {}
    for r in audited:
        k = r.get("safe_reason", "")
        reason_counts[k] = reason_counts.get(k, 0) + 1

    summary = {
        "patch": PATCH_ID,
        "accepted_input_csv": str(accepted_csv),
        "video": str(video_path),
        "track_csv": str(track_csv),
        "sequence_key": sequence,
        "base_points": len(base),
        "input_accepted_005D7C": len(accepted_rows),
        "safe_accepted_count": len(safe),
        "safe_rejected_count": len(rejected),
        "safe_accept_ratio_from_005D7C": len(safe) / max(1, len(accepted_rows)),
        "final_preview_points": len(merged),
        "bridge_error_median": float(np.median(bridge_errors)) if bridge_errors else None,
        "bridge_error_p90": float(np.percentile(bridge_errors, 90)) if bridge_errors else None,
        "base_speed_stats_px_per_frame": speed_stats,
        "reason_counts": reason_counts,
        "audit_csv": str(audit_csv),
        "safe_csv": str(safe_csv),
        "merged_preview_csv": str(merged_csv),
        "overlay": str(overlay_path) if overlay_ok else None,
        "params": vars(args),
        "note": "Preview seulement. Aucun remplacement final du track source.",
    }

    summary_path = out_dir / "005D7D_safe_gapfill_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"accepted_csv = {accepted_csv}")
    print(f"video        = {video_path}")
    print(f"track_csv    = {track_csv}")
    print(f"sequence     = {sequence}")
    print("=" * 72)
    print("")
    print("OK 005D7D")
    print(f"audit_csv   = {audit_csv}")
    print(f"safe_csv    = {safe_csv}")
    print(f"preview_csv = {merged_csv}")
    print(f"summary     = {summary_path}")
    print(f"overlay     = {overlay_path if overlay_ok else 'NON_ECRIT'}")
    print("")
    print(f"input_accepted_005D7C={len(accepted_rows)}")
    print(f"safe_accepted_count={len(safe)} safe_rejected_count={len(rejected)}")
    if bridge_errors:
        print(f"bridge_error_median={summary['bridge_error_median']:.2f} p90={summary['bridge_error_p90']:.2f}")
    print(f"reason_counts={reason_counts}")
    print(f"base_speed_stats={speed_stats}")


if __name__ == "__main__":
    main()
