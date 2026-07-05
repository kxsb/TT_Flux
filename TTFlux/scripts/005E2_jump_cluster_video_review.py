from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005E2_jump_cluster_video_review"


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


def read_track_points(path: Path):
    rows, fields = read_csv(path)

    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError("Colonnes frame/x/y introuvables dans le track.")

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


def read_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 50.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def cluster_related_jumps(cluster, jumps):
    start = safe_int(cluster.get("start_frame"))
    end = safe_int(cluster.get("end_frame"))

    out = []
    for j in jumps:
        a = safe_int(j.get("from_frame"))
        b = safe_int(j.get("to_frame"))
        if a <= end and b >= start:
            out.append(j)

    out.sort(key=lambda r: safe_float(r.get("speed_px_f")), reverse=True)
    return out


def draw_text_bg(img, text, org, scale=0.65, color=(255, 255, 255), thickness=1):
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 5, y - th - 7), (x + tw + 5, y + base + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def source_color(src):
    s = str(src)
    if s.startswith("gapfill_005D7J"):
        return (255, 0, 255), 8, 2
    if s.startswith("gapfill_005D7E"):
        return (180, 0, 255), 7, 2
    return (0, 220, 255), 4, -1


def draw_track_overlay(frame, frame_idx, points, related_jumps, trail=18):
    out = frame.copy()
    h, w = out.shape[:2]

    prev = None
    for fr in range(frame_idx - trail, frame_idx + trail + 1):
        p = points.get(fr)
        if not p:
            prev = None
            continue

        x = int(round(p["x"]))
        y = int(round(p["y"]))
        color, rad, thick = source_color(p.get("source", ""))

        if prev is not None:
            cv2.line(out, prev, (x, y), (80, 80, 80), 1)

        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), rad, color, thick)

        prev = (x, y)

    # Point courant.
    pcur = points.get(frame_idx)
    if pcur:
        x = int(round(pcur["x"]))
        y = int(round(pcur["y"]))
        cv2.circle(out, (x, y), 14, (255, 255, 255), 2)

    # Lignes rouges : sauts du cluster.
    for j in related_jumps[:5]:
        x0 = int(round(safe_float(j.get("from_x"))))
        y0 = int(round(safe_float(j.get("from_y"))))
        x1 = int(round(safe_float(j.get("to_x"))))
        y1 = int(round(safe_float(j.get("to_y"))))

        cv2.line(out, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.circle(out, (x0, y0), 12, (0, 0, 255), 2)
        cv2.circle(out, (x1, y1), 12, (0, 0, 255), 2)

    return out


def crop_square(img, cx, cy, size=360):
    h, w = img.shape[:2]
    half = size // 2

    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + size
    y2 = y1 + size

    crop = np.zeros((size, size, 3), dtype=np.uint8)

    sx1 = max(0, x1)
    sy1 = max(0, y1)
    sx2 = min(w, x2)
    sy2 = min(h, y2)

    dx1 = sx1 - x1
    dy1 = sy1 - y1
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)

    if sx2 > sx1 and sy2 > sy1:
        crop[dy1:dy2, dx1:dx2] = img[sy1:sy2, sx1:sx2]

    return crop


def jump_center(jump, points):
    if jump:
        x0 = safe_float(jump.get("from_x"))
        y0 = safe_float(jump.get("from_y"))
        x1 = safe_float(jump.get("to_x"))
        y1 = safe_float(jump.get("to_y"))

        if all(math.isfinite(v) for v in [x0, y0, x1, y1]):
            return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    if points:
        frames = sorted(points)
        mid = frames[len(frames) // 2]
        return points[mid]["x"], points[mid]["y"]

    return 640, 360


def fit_to_box(img, box_w, box_h):
    h, w = img.shape[:2]
    scale = min(box_w / w, box_h / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    x = (box_w - nw) // 2
    y = (box_h - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas


def compose_review_frame(raw, frame_idx, cluster, rank, total, related_jumps, points, out_w=1920, out_h=1080):
    drawn = draw_track_overlay(raw, frame_idx, points, related_jumps)

    main = fit_to_box(drawn, 1280, 720)

    max_jump = related_jumps[0] if related_jumps else None
    cx, cy = jump_center(max_jump, points)
    zoom_crop = crop_square(drawn, cx, cy, size=380)
    zoom = cv2.resize(zoom_crop, (560, 560), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Layout
    canvas[150:150+720, 30:30+1280] = main
    canvas[210:210+560, 1330:1330+560] = zoom

    cv2.rectangle(canvas, (1330, 210), (1330+560, 210+560), (90, 90, 90), 2)
    cv2.putText(canvas, "ZOOM", (1342, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

    cid = safe_int(cluster.get("cluster_id"))
    start = safe_int(cluster.get("start_frame"))
    end = safe_int(cluster.get("end_frame"))
    max_speed = safe_float(cluster.get("max_speed_px_f"))
    max_dist = safe_float(cluster.get("max_dist"))
    reasons = str(cluster.get("reasons", ""))

    if max_jump:
        jf = f"{safe_int(max_jump.get('from_frame'))}->{safe_int(max_jump.get('to_frame'))}"
        jspeed = safe_float(max_jump.get("speed_px_f"))
        jreason = str(max_jump.get("jump_reason", ""))
    else:
        jf = f"{start}->{end}"
        jspeed = max_speed
        jreason = reasons

    draw_text_bg(
        canvas,
        f"{PATCH_ID} | cluster {rank}/{total} | id={cid} | frames {start}->{end} | current f={frame_idx}",
        (30, 44),
        scale=0.72,
    )
    draw_text_bg(
        canvas,
        f"max jump {jf} | speed={jspeed:.1f}px/f | cluster_max={max_speed:.1f}px/f | dist={max_dist:.1f} | {jreason}",
        (30, 86),
        scale=0.62,
        color=(230, 230, 230),
    )
    draw_text_bg(
        canvas,
        "cyan=track source | violet=gapfill | red=line jump | white=current point",
        (30, 126),
        scale=0.56,
        color=(210, 210, 210),
    )

    # Progress bar.
    bar_x, bar_y, bar_w, bar_h = 30, 1010, 1860, 18
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    progress = rank / max(1, total)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (180, 180, 180), -1)

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-csv", default="runs/005D7K_gapfilled_canonical_v2/005D7K_ball_points_gapfilled_canonical_unique.csv")
    ap.add_argument("--jump-csv", default="runs/005E0_canonical_jump_audit/005E0_jump_steps_only.csv")
    ap.add_argument("--cluster-csv", default="runs/005E0_canonical_jump_audit/005E0_jump_clusters.csv")
    ap.add_argument("--final-summary", default="runs/005D7L_final_canonical_audit/005D7L_final_canonical_audit_summary.json")
    ap.add_argument("--out-dir", default="runs/005E2_jump_cluster_video_review")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--clusters", type=int, default=30)
    ap.add_argument("--out-fps", type=float, default=25.0)
    ap.add_argument("--pre", type=int, default=7)
    ap.add_argument("--post", type=int, default=9)
    args = ap.parse_args()

    track_csv = Path(args.track_csv)
    jump_csv = Path(args.jump_csv)
    cluster_csv = Path(args.cluster_csv)
    final_summary_path = Path(args.final_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    video_path = Path(final_summary["video"])

    vi = read_video_info(video_path)
    points = read_track_points(track_csv)
    jumps, _ = read_csv(jump_csv)
    clusters, _ = read_csv(cluster_csv)

    # Exclure les clusters hors vidéo, puis trier par vitesse max.
    inside = []
    for c in clusters:
        start = safe_int(c.get("start_frame"))
        end = safe_int(c.get("end_frame"))
        if 0 <= start < vi["frames"] and 0 <= end < vi["frames"]:
            inside.append(c)

    inside.sort(key=lambda r: safe_float(r.get("max_speed_px_f")), reverse=True)
    selected = inside[: max(1, args.clusters)]

    if not selected:
        raise RuntimeError("Aucun cluster exploitable dans la vidéo.")

    total_out_frames = int(round(args.seconds * args.out_fps))
    frames_per_cluster = max(8, total_out_frames // len(selected))
    actual_out_frames = frames_per_cluster * len(selected)
    actual_seconds = actual_out_frames / args.out_fps

    out_video = out_dir / "005E2_jump_cluster_review_top30_30s.mp4"
    manifest_csv = out_dir / "005E2_jump_cluster_review_manifest.csv"
    summary_path = out_dir / "005E2_jump_cluster_video_review_summary.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.out_fps,
        (1920, 1080),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire vidéo: {out_video}")

    manifest = []
    out_frame_idx = 0

    for rank, c in enumerate(selected, 1):
        cid = safe_int(c.get("cluster_id"))
        related = cluster_related_jumps(c, jumps)
        max_jump = related[0] if related else None

        if max_jump:
            center_a = safe_int(max_jump.get("from_frame"))
            center_b = safe_int(max_jump.get("to_frame"))
        else:
            center_a = safe_int(c.get("start_frame"))
            center_b = safe_int(c.get("end_frame"))

        src_start = max(0, center_a - args.pre)
        src_end = min(vi["frames"] - 1, center_b + args.post)
        source_window = list(range(src_start, src_end + 1))

        if len(source_window) <= 1:
            source_window = [max(0, min(vi["frames"] - 1, center_a))]

        chosen_frames = []
        for i in range(frames_per_cluster):
            if frames_per_cluster == 1:
                idx = 0
            else:
                idx = int(round(i * (len(source_window) - 1) / (frames_per_cluster - 1)))
            chosen_frames.append(source_window[idx])

        out_start_sec = out_frame_idx / args.out_fps

        for fr in chosen_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, raw = cap.read()

            if not ok or raw is None:
                raw = np.zeros((vi["height"], vi["width"], 3), dtype=np.uint8)
                draw_text_bg(raw, f"NO FRAME {fr}", (30, 60), scale=1.0, color=(0, 0, 255), thickness=2)

            canvas = compose_review_frame(
                raw,
                fr,
                c,
                rank,
                len(selected),
                related,
                points,
            )
            writer.write(canvas)
            out_frame_idx += 1

        out_end_sec = out_frame_idx / args.out_fps

        manifest.append({
            "review_rank": rank,
            "cluster_id": cid,
            "cluster_start": safe_int(c.get("start_frame")),
            "cluster_end": safe_int(c.get("end_frame")),
            "max_speed_px_f": f"{safe_float(c.get('max_speed_px_f')):.3f}",
            "max_dist": f"{safe_float(c.get('max_dist')):.3f}",
            "reasons": c.get("reasons", ""),
            "max_jump_from": "" if not max_jump else safe_int(max_jump.get("from_frame")),
            "max_jump_to": "" if not max_jump else safe_int(max_jump.get("to_frame")),
            "source_window_start": src_start,
            "source_window_end": src_end,
            "output_start_sec": f"{out_start_sec:.3f}",
            "output_end_sec": f"{out_end_sec:.3f}",
            "output_frames": frames_per_cluster,
            "related_jump_count": len(related),
        })

    cap.release()
    writer.release()

    write_csv(manifest_csv, manifest, [
        "review_rank",
        "cluster_id",
        "cluster_start",
        "cluster_end",
        "max_speed_px_f",
        "max_dist",
        "reasons",
        "max_jump_from",
        "max_jump_to",
        "source_window_start",
        "source_window_end",
        "output_start_sec",
        "output_end_sec",
        "output_frames",
        "related_jump_count",
    ])

    summary = {
        "patch": PATCH_ID,
        "track_csv": str(track_csv),
        "jump_csv": str(jump_csv),
        "cluster_csv": str(cluster_csv),
        "final_summary": str(final_summary_path),
        "video": str(video_path),
        "video_info": vi,
        "inside_cluster_count": len(inside),
        "selected_cluster_count": len(selected),
        "requested_seconds": args.seconds,
        "actual_seconds": actual_seconds,
        "out_fps": args.out_fps,
        "frames_per_cluster": frames_per_cluster,
        "out_video": str(out_video),
        "manifest_csv": str(manifest_csv),
        "selected_clusters": manifest,
        "note": "Vidéo review humaine. Ne modifie pas le track.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"track_csv   = {track_csv}")
    print(f"jump_csv    = {jump_csv}")
    print(f"cluster_csv = {cluster_csv}")
    print(f"video       = {video_path}")
    print("=" * 72)
    print("")
    print("OK 005E2")
    print(f"summary  = {summary_path}")
    print(f"video    = {out_video}")
    print(f"manifest = {manifest_csv}")
    print("")
    print(f"video_frames={vi['frames']}")
    print(f"inside_cluster_count={len(inside)}")
    print(f"selected_cluster_count={len(selected)}")
    print(f"frames_per_cluster={frames_per_cluster}")
    print(f"actual_seconds={actual_seconds:.2f}")
    print(f"out_fps={args.out_fps}")


if __name__ == "__main__":
    main()
