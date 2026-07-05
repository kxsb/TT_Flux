from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005E2B_simple_tracking_trajectory_video"


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


def read_track(path: Path):
    rows, fields = read_csv(path)

    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError("Colonnes frame/x/y introuvables.")

    points = {}

    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))

        if fr >= 0 and math.isfinite(x) and math.isfinite(y):
            points[fr] = {
                "frame": fr,
                "x": x,
                "y": y,
                "source": str(r.get("point_source", "")) or "source",
            }

    return points


def draw_simple_trajectory(frame, frame_idx, points, trail=22, future=0, show_frame=True):
    out = frame.copy()
    h, w = out.shape[:2]

    # Trajectoire passée, simple et lisible.
    pts = []
    for fr in range(frame_idx - trail, frame_idx + future + 1):
        p = points.get(fr)
        if not p:
            if pts:
                pts.append(None)
            continue
        x = int(round(p["x"]))
        y = int(round(p["y"]))
        if 0 <= x < w and 0 <= y < h:
            pts.append((fr, x, y))

    prev_xy = None
    for item in pts:
        if item is None:
            prev_xy = None
            continue

        fr, x, y = item
        age = frame_idx - fr

        if age < 0:
            # Futur éventuel, désactivé par défaut.
            color = (80, 80, 80)
            radius = 3
            thick = 1
        elif age == 0:
            color = (255, 255, 255)
            radius = 10
            thick = 2
        elif age <= 5:
            color = (0, 255, 255)
            radius = 5
            thick = -1
        else:
            color = (0, 170, 170)
            radius = 3
            thick = -1

        if prev_xy is not None:
            cv2.line(out, prev_xy, (x, y), (0, 150, 150), 2)

        cv2.circle(out, (x, y), radius, color, thick)
        prev_xy = (x, y)

    # Point courant un peu plus visible.
    cur = points.get(frame_idx)
    if cur:
        x = int(round(cur["x"]))
        y = int(round(cur["y"]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), 16, (255, 255, 255), 2)
            cv2.circle(out, (x, y), 4, (255, 255, 255), -1)

    if show_frame:
        label = f"f{frame_idx}"
        cv2.rectangle(out, (8, 8), (92, 34), (0, 0, 0), -1)
        cv2.putText(out, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1, cv2.LINE_AA)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-csv", default="runs/005D7K_gapfilled_canonical_v2/005D7K_ball_points_gapfilled_canonical_unique.csv")
    ap.add_argument("--final-summary", default="runs/005D7L_final_canonical_audit/005D7L_final_canonical_audit_summary.json")
    ap.add_argument("--out-dir", default="runs/005E2B_simple_tracking_video")
    ap.add_argument("--trail", type=int, default=22)
    ap.add_argument("--future", type=int, default=0)
    ap.add_argument("--show-frame", type=int, default=1)
    ap.add_argument("--out-fps", type=float, default=0.0, help="0 = fps source")
    args = ap.parse_args()

    track_csv = Path(args.track_csv)
    final_summary_path = Path(args.final_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not track_csv.exists():
        raise FileNotFoundError(track_csv)
    if not final_summary_path.exists():
        raise FileNotFoundError(final_summary_path)

    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    video_path = Path(final_summary["video"])

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    points = read_track(track_csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    fps = src_fps if args.out_fps <= 0 else float(args.out_fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_video = out_dir / "005E2B_simple_tracking_trajectory_full.mp4"
    summary_path = out_dir / "005E2B_simple_tracking_trajectory_summary.json"

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire vidéo: {out_video}")

    frame_idx = -1
    written = 0
    missing_current = 0
    present_current = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1

        if frame_idx in points:
            present_current += 1
        else:
            missing_current += 1

        out = draw_simple_trajectory(
            frame,
            frame_idx,
            points,
            trail=args.trail,
            future=args.future,
            show_frame=bool(args.show_frame),
        )

        writer.write(out)
        written += 1

    cap.release()
    writer.release()

    summary = {
        "patch": PATCH_ID,
        "track_csv": str(track_csv),
        "final_summary": str(final_summary_path),
        "video": str(video_path),
        "out_video": str(out_video),
        "source_fps": src_fps,
        "out_fps": fps,
        "width": w,
        "height": h,
        "source_total_frames": total,
        "written_frames": written,
        "track_point_count": len(points),
        "present_current_frames_in_video": present_current,
        "missing_current_frames_in_video": missing_current,
        "duration_sec": written / fps if fps else None,
        "trail": args.trail,
        "future": args.future,
        "show_frame": bool(args.show_frame),
        "note": "Vidéo simple : trajectoire locale + point courant. Pas de zoom, pas de panel, pas de lignes de saut.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"track_csv = {track_csv}")
    print(f"video     = {video_path}")
    print("=" * 72)
    print("")
    print("OK 005E2B")
    print(f"video   = {out_video}")
    print(f"summary = {summary_path}")
    print("")
    print(f"source_frames={total}")
    print(f"written_frames={written}")
    print(f"source_fps={src_fps:.3f}")
    print(f"out_fps={fps:.3f}")
    print(f"duration_sec={written / fps:.2f}")
    print(f"track_point_count={len(points)}")
    print(f"present_current_frames_in_video={present_current}")
    print(f"missing_current_frames_in_video={missing_current}")


if __name__ == "__main__":
    main()
