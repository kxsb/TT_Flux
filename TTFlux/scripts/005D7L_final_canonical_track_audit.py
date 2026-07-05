from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005D7L_final_canonical_track_audit"


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


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def read_track(path: Path):
    rows, fields = read_csv(path)

    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])
    score_col = find_col(fields, ["score", "conf", "confidence", "prob"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError(f"Colonnes frame/x/y introuvables dans {path}")

    points = {}
    dupes = []
    invalid = 0
    source_counts = {}

    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        sc = safe_float(r.get(score_col), 1.0) if score_col else 1.0

        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            invalid += 1
            continue

        if fr in points:
            dupes.append(fr)

        src = str(r.get("point_source", "source"))
        if not src:
            src = "source"
        source_counts[src] = source_counts.get(src, 0) + 1

        points[fr] = {
            "frame": fr,
            "x": x,
            "y": y,
            "score": sc,
            "point_source": src,
            "row": r,
        }

    return rows, fields, points, {
        "rows": len(rows),
        "valid_points": len(points),
        "invalid_rows": invalid,
        "duplicate_frames": sorted(set(dupes)),
        "source_counts": source_counts,
        "frame_col": frame_col,
        "x_col": x_col,
        "y_col": y_col,
        "score_col": score_col,
    }


def continuity_stats(points):
    frames = sorted(points)
    speeds = []
    jumps = []

    for a, b in zip(frames[:-1], frames[1:]):
        dt = b - a
        if dt <= 0:
            continue

        ax, ay = points[a]["x"], points[a]["y"]
        bx, by = points[b]["x"], points[b]["y"]
        dist = math.hypot(bx - ax, by - ay)
        speed = dist / dt
        speeds.append(speed)

        if speed > 180 or dist > 260:
            jumps.append({
                "from": a,
                "to": b,
                "dt": dt,
                "dist": float(dist),
                "speed": float(speed),
                "from_source": points[a].get("point_source", ""),
                "to_source": points[b].get("point_source", ""),
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
        "large_step_count": len(jumps),
        "largest_steps": sorted(jumps, key=lambda x: x["speed"], reverse=True)[:30],
    }


def compute_gaps(total_frames, points):
    present = set(points)
    gaps = []
    start = None

    for fr in range(total_frames):
        miss = fr not in present
        if miss and start is None:
            start = fr
        elif not miss and start is not None:
            end = fr - 1
            gaps.append({"start": start, "end": end, "len": end - start + 1})
            start = None

    if start is not None:
        end = total_frames - 1
        gaps.append({"start": start, "end": end, "len": end - start + 1})

    return gaps


def target_gap_status(per_gap, points):
    rows = []
    missing = []
    present = []

    for g in per_gap:
        gid = int(g["gap_id"])
        start = int(g["start"])
        end = int(g["end"])

        for fr in range(start, end + 1):
            p = points.get(fr)
            if p:
                present.append(fr)
                status = "present"
            else:
                missing.append(fr)
                status = "missing"

            rows.append({
                "gap_id": gid,
                "gap_start": start,
                "gap_end": end,
                "frame": fr,
                "status": status,
                "x": "" if not p else f"{p['x']:.3f}",
                "y": "" if not p else f"{p['y']:.3f}",
                "source": "" if not p else p.get("point_source", ""),
            })

    return rows, present, missing


def draw_overlay(video_path: Path, out_path: Path, points, target_rows):
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

    target_by_frame = {safe_int(r["frame"]): r for r in target_rows}
    target_frames = set(target_by_frame)
    trail = 20
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        if frame_idx not in target_frames:
            continue

        out = frame.copy()

        for fr in range(frame_idx - trail, frame_idx + trail + 1):
            p = points.get(fr)
            if not p:
                continue

            x = int(round(p["x"]))
            y = int(round(p["y"]))
            src = str(p.get("point_source", ""))

            if src.startswith("gapfill_005D7J"):
                color = (255, 0, 255)
                radius = 8
            elif src.startswith("gapfill_005D7E"):
                color = (180, 0, 255)
                radius = 6
            else:
                color = (0, 220, 255)
                radius = 3

            cv2.circle(out, (x, y), radius, color, 2 if radius >= 6 else -1)

        r = target_by_frame[frame_idx]
        status = r["status"]

        if status == "missing":
            label_status = "STILL_MISSING"
            color = (0, 0, 255)
        else:
            label_status = "FILLED/PRESENT"
            color = (255, 0, 255) if str(r.get("source", "")).startswith("gapfill") else (0, 220, 255)

        if r.get("x") and r.get("y"):
            x = int(round(float(r["x"])))
            y = int(round(float(r["y"])))
            cv2.circle(out, (x, y), 15, color, 2)
            cv2.putText(out, label_status, (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        header = f"{PATCH_ID} | frame={frame_idx} | gap={r['gap_id']} | {label_status} | src={r.get('source','')}"
        cv2.rectangle(out, (8, 8), (min(w - 8, 1240), 42), (0, 0, 0), -1)
        cv2.putText(out, header, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(out)

    cap.release()
    writer.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-csv", default="runs/005D7K_gapfilled_canonical_v2/005D7K_ball_points_gapfilled_canonical_unique.csv")
    ap.add_argument("--validation-summary", default="runs/005D7F_target_gap_validation/005D7F_target_gap_validation_summary.json")
    ap.add_argument("--out-dir", default="runs/005D7L_final_canonical_audit")
    args = ap.parse_args()

    track_csv = Path(args.track_csv)
    validation_path = Path(args.validation_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not track_csv.exists():
        raise FileNotFoundError(track_csv)
    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    video_path = Path(validation["video"])
    total_frames = 1159

    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or total_frames)
        cap.release()

    rows, fields, points, meta = read_track(track_csv)
    continuity = continuity_stats(points)
    all_gaps = compute_gaps(total_frames, points)

    target_rows, target_present, target_missing = target_gap_status(validation.get("per_gap", []), points)

    target_csv = out_dir / "005D7L_target_gap_final_status.csv"
    overlay_path = out_dir / "005D7L_final_target_gap_overlay.mp4"
    summary_path = out_dir / "005D7L_final_canonical_audit_summary.json"
    next_path = out_dir / "NEXT_TRACK_CSV.txt"
    locked_csv = out_dir / "CURRENT_CANONICAL_TRACK_005D7K.csv"

    write_csv(target_csv, target_rows, ["gap_id", "gap_start", "gap_end", "frame", "status", "x", "y", "source"])

    overlay_ok = draw_overlay(video_path, overlay_path, points, target_rows)

    shutil.copy2(track_csv, locked_csv)
    next_path.write_text(str(track_csv), encoding="utf-8")

    firstpass_count = sum(1 for p in points.values() if str(p.get("point_source", "")).startswith("gapfill_005D7E"))
    secondpass_count = sum(1 for p in points.values() if str(p.get("point_source", "")).startswith("gapfill_005D7J"))

    final_ok = (
        len(meta["duplicate_frames"]) == 0
        and len(points) == 1133
        and firstpass_count == 13
        and secondpass_count == 4
        and len(target_missing) == 30
    )

    summary = {
        "patch": PATCH_ID,
        "final_ok": final_ok,
        "track_csv": str(track_csv),
        "locked_copy": str(locked_csv),
        "validation_summary": str(validation_path),
        "video": str(video_path),
        "total_video_frames": total_frames,
        "point_count": len(points),
        "duplicate_frames": meta["duplicate_frames"],
        "source_counts": meta["source_counts"],
        "firstpass_gapfill_count": firstpass_count,
        "secondpass_gapfill_count": secondpass_count,
        "target_gap_total_frames": len(target_rows),
        "target_gap_present_count": len(target_present),
        "target_gap_missing_count": len(target_missing),
        "target_gap_missing_frames": target_missing,
        "all_missing_count_video_span": sum(g["len"] for g in all_gaps),
        "all_missing_runs_video_span": all_gaps,
        "continuity": continuity,
        "target_status_csv": str(target_csv),
        "overlay": str(overlay_path) if overlay_ok else None,
        "next_track_csv": str(track_csv),
        "note": "Audit final du track canonique 005D7K. Utiliser ce CSV comme référence pour les prochains audits.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"track_csv  = {track_csv}")
    print(f"validation = {validation_path}")
    print(f"video      = {video_path}")
    print("=" * 72)
    print("")
    print("OK 005D7L")
    print(f"summary    = {summary_path}")
    print(f"target_csv = {target_csv}")
    print(f"overlay    = {overlay_path if overlay_ok else 'NON_ECRIT'}")
    print(f"locked_csv = {locked_csv}")
    print(f"next_track = {next_path}")
    print("")
    print(f"final_ok={final_ok}")
    print(f"point_count={len(points)}")
    print(f"duplicate_frame_count={len(meta['duplicate_frames'])}")
    print(f"firstpass_gapfill_count={firstpass_count}")
    print(f"secondpass_gapfill_count={secondpass_count}")
    print(f"target_gap_present_count={len(target_present)}")
    print(f"target_gap_missing_count={len(target_missing)}")
    print(f"target_gap_missing_frames={target_missing}")
    print(f"continuity_large_step_count={continuity['large_step_count']}")
    print(f"continuity_speed_p95={continuity['speed_p95']}")


if __name__ == "__main__":
    main()
