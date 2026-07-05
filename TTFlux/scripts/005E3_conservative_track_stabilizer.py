from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005E3_conservative_track_stabilizer"


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


def read_track(path: Path):
    rows, fields = read_csv(path)

    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError("Colonnes frame/x/y introuvables.")

    points = {}

    for idx, r in enumerate(rows):
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))

        if fr >= 0 and math.isfinite(x) and math.isfinite(y):
            points[fr] = {
                "frame": fr,
                "x": x,
                "y": y,
                "row": r,
                "idx": idx,
                "source": str(r.get("point_source", "")) or "source",
            }

    return rows, fields, points, frame_col, x_col, y_col


def build_steps(points):
    frames = sorted(points)
    steps = []

    for a, b in zip(frames[:-1], frames[1:]):
        dt = b - a
        if dt <= 0:
            continue

        p0 = points[a]
        p1 = points[b]
        dx = p1["x"] - p0["x"]
        dy = p1["y"] - p0["y"]
        dist = math.hypot(dx, dy)
        speed = dist / dt

        steps.append({
            "from_frame": a,
            "to_frame": b,
            "dt": dt,
            "dist": dist,
            "speed": speed,
            "dx": dx,
            "dy": dy,
        })

    return steps


def detect_bad_frames(points, speed_cut, dist_cut, expand):
    bad_frames = set()
    bad_steps = []

    for s in build_steps(points):
        is_bad = (
            s["speed"] >= speed_cut
            or s["dist"] >= dist_cut
            or abs(s["dx"]) >= 300
            or abs(s["dy"]) >= 220
        )

        if not is_bad:
            continue

        a = s["from_frame"]
        b = s["to_frame"]
        bad_steps.append(s)

        for fr in range(a - expand, a + expand + 1):
            bad_frames.add(fr)
        for fr in range(b - expand, b + expand + 1):
            bad_frames.add(fr)

    return bad_frames, bad_steps


def stats(points):
    steps = build_steps(points)
    speeds = [s["speed"] for s in steps]

    bad = [
        s for s in steps
        if s["speed"] >= 180.0 or s["dist"] >= 260.0 or abs(s["dx"]) >= 300 or abs(s["dy"]) >= 220
    ]

    if not speeds:
        return {
            "point_count": len(points),
            "step_count": 0,
            "speed_median": None,
            "speed_p90": None,
            "speed_p95": None,
            "speed_max": None,
            "large_step_count": 0,
        }

    return {
        "point_count": len(points),
        "step_count": len(steps),
        "speed_median": float(np.median(speeds)),
        "speed_p90": float(np.percentile(speeds, 90)),
        "speed_p95": float(np.percentile(speeds, 95)),
        "speed_max": float(np.max(speeds)),
        "large_step_count": len(bad),
    }


def draw_simple(frame, frame_idx, points, trail=20):
    out = frame.copy()
    h, w = out.shape[:2]

    prev = None
    for fr in range(frame_idx - trail, frame_idx + 1):
        p = points.get(fr)

        if not p:
            prev = None
            continue

        x = int(round(p["x"]))
        y = int(round(p["y"]))

        if not (0 <= x < w and 0 <= y < h):
            prev = None
            continue

        age = frame_idx - fr

        if age == 0:
            color = (255, 255, 255)
            radius = 11
            thick = 2
        elif age <= 5:
            color = (0, 255, 255)
            radius = 5
            thick = -1
        else:
            color = (0, 155, 155)
            radius = 3
            thick = -1

        if prev is not None:
            cv2.line(out, prev, (x, y), (0, 150, 150), 2)

        cv2.circle(out, (x, y), radius, color, thick)
        prev = (x, y)

    cv2.rectangle(out, (8, 8), (96, 34), (0, 0, 0), -1)
    cv2.putText(out, f"f{frame_idx}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1, cv2.LINE_AA)

    return out


def make_video(video_path: Path, out_path: Path, points):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire vidéo: {out_path}")

    frame_idx = -1
    written = 0
    present = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        if frame_idx in points:
            present += 1

        writer.write(draw_simple(frame, frame_idx, points))
        written += 1

    cap.release()
    writer.release()

    return {
        "fps": fps,
        "width": w,
        "height": h,
        "source_frames": total,
        "written_frames": written,
        "present_frames": present,
        "missing_frames": written - present,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-csv", default="runs/005D7K_gapfilled_canonical_v2/005D7K_ball_points_gapfilled_canonical_unique.csv")
    ap.add_argument("--final-summary", default="runs/005D7L_final_canonical_audit/005D7L_final_canonical_audit_summary.json")
    ap.add_argument("--out-dir", default="runs/005E3_conservative_stabilizer")
    ap.add_argument("--speed-cut", type=float, default=180.0)
    ap.add_argument("--dist-cut", type=float, default=260.0)
    ap.add_argument("--expand", type=int, default=1)
    args = ap.parse_args()

    track_csv = Path(args.track_csv)
    final_summary_path = Path(args.final_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    video_path = Path(final_summary["video"])

    rows, fields, points, frame_col, x_col, y_col = read_track(track_csv)

    bad_frames, bad_steps = detect_bad_frames(
        points,
        speed_cut=args.speed_cut,
        dist_cut=args.dist_cut,
        expand=args.expand,
    )

    extra_fields = ["stability_patch", "stability_status", "stability_reason"]
    out_fields = list(fields)
    for f in extra_fields:
        if f not in out_fields:
            out_fields.append(f)

    kept_rows = []
    rejected_rows = []

    kept_points = {}

    for fr in sorted(points):
        p = points[fr]
        r = dict(p["row"])

        if fr in bad_frames:
            r["stability_patch"] = PATCH_ID
            r["stability_status"] = "rejected"
            r["stability_reason"] = "near_large_jump"
            rejected_rows.append(r)
        else:
            r["stability_patch"] = PATCH_ID
            r["stability_status"] = "kept"
            r["stability_reason"] = "smooth_local_context"
            kept_rows.append(r)
            kept_points[fr] = {
                "frame": fr,
                "x": p["x"],
                "y": p["y"],
                "source": p["source"],
            }

    stable_csv = out_dir / "005E3_conservative_stable_track.csv"
    rejected_csv = out_dir / "005E3_rejected_hyperactive_points.csv"
    video_out = out_dir / "005E3_conservative_stable_trajectory.mp4"
    summary_path = out_dir / "005E3_conservative_stabilizer_summary.json"

    write_csv(stable_csv, kept_rows, out_fields)
    write_csv(rejected_csv, rejected_rows, out_fields)

    video_meta = make_video(video_path, video_out, kept_points)

    before_stats = stats(points)
    after_stats = stats(kept_points)

    summary = {
        "patch": PATCH_ID,
        "track_csv": str(track_csv),
        "final_summary": str(final_summary_path),
        "video": str(video_path),
        "stable_csv": str(stable_csv),
        "rejected_csv": str(rejected_csv),
        "video_out": str(video_out),
        "params": vars(args),
        "input_point_count": len(points),
        "kept_point_count": len(kept_points),
        "rejected_point_count": len(rejected_rows),
        "bad_step_count": len(bad_steps),
        "bad_frame_count": len(bad_frames),
        "before_stats": before_stats,
        "after_stats": after_stats,
        "video_meta": video_meta,
        "note": "Stabilisation conservative : on retire, on ne remplit pas. Objectif lisibilité et base propre.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"track_csv = {track_csv}")
    print(f"video     = {video_path}")
    print("=" * 72)
    print("")
    print("OK 005E3")
    print(f"stable_csv   = {stable_csv}")
    print(f"rejected_csv = {rejected_csv}")
    print(f"video        = {video_out}")
    print(f"summary      = {summary_path}")
    print("")
    print(f"input_point_count={len(points)}")
    print(f"kept_point_count={len(kept_points)}")
    print(f"rejected_point_count={len(rejected_rows)}")
    print(f"bad_step_count={len(bad_steps)}")
    print(f"before_large_step_count={before_stats['large_step_count']}")
    print(f"after_large_step_count={after_stats['large_step_count']}")
    print(f"before_speed_p95={before_stats['speed_p95']}")
    print(f"after_speed_p95={after_stats['speed_p95']}")
    print(f"video_out={video_out}")


if __name__ == "__main__":
    main()
