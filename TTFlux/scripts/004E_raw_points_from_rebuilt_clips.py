from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "004E"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def local_stats(gray, motion, x, y, radius=5):
    h, w = gray.shape[:2]
    x0 = max(0, int(x) - radius)
    x1 = min(w, int(x) + radius + 1)
    y0 = max(0, int(y) - radius)
    y1 = min(h, int(y) + radius + 1)

    g = gray[y0:y1, x0:x1]
    m = motion[y0:y1, x0:x1] if motion is not None else None

    bright = float(np.mean(g)) if g.size else 0.0
    mot = float(np.mean(m)) if m is not None and m.size else 0.0

    return bright, mot


def detect_candidates(frame, prev_gray):
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if prev_gray is None:
        motion = np.zeros_like(gray_blur)
        motion_mask = np.ones_like(gray_blur, dtype=np.uint8) * 255
    else:
        motion = cv2.absdiff(prev_gray, gray_blur)
        motion_mask = (motion >= 7).astype(np.uint8) * 255

    # Balle souvent blanche/jaune claire. On reste volontairement permissif.
    white_like = ((vv >= 155) & (ss <= 135))
    yellow_like = ((vv >= 145) & (hh >= 12) & (hh <= 45) & (ss >= 35) & (ss <= 210))
    bright_like = (white_like | yellow_like).astype(np.uint8) * 255

    roi = np.zeros_like(bright_like)
    roi[int(h * 0.06):int(h * 0.96), int(w * 0.03):int(w * 0.97)] = 255

    mask_primary = cv2.bitwise_and(bright_like, motion_mask)
    mask_primary = cv2.bitwise_and(mask_primary, roi)

    # Fallback : petit objet en mouvement pas forcément blanc.
    motion_fallback = ((motion >= 12) & (vv >= 120)).astype(np.uint8) * 255
    motion_fallback = cv2.bitwise_and(motion_fallback, roi)

    masks = [mask_primary, motion_fallback]

    candidates = []

    for pass_idx, mask in enumerate(masks):
        mask = cv2.medianBlur(mask, 3)
        num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)

        for i in range(1, num):
            x, y, bw, bh, area = stats[i]

            if area < 2 or area > 110:
                continue
            if bw < 1 or bh < 1 or bw > 26 or bh > 26:
                continue

            aspect = bw / max(1, bh)
            if aspect < 0.25 or aspect > 4.0:
                continue

            cx, cy = cents[i]

            bright, mot = local_stats(gray, motion, cx, cy, radius=5)

            fill = area / max(1, bw * bh)
            compact_bonus = 1.0 - min(1.0, abs(aspect - 1.0))
            size_penalty = max(0.0, area - 35) * 0.25

            score = (
                bright * 0.18
                + mot * 2.20
                + fill * 8.0
                + compact_bonus * 5.0
                - size_penalty
                - pass_idx * 4.0
            )

            candidates.append({
                "x": float(cx),
                "y": float(cy),
                "area": int(area),
                "bbox_w": int(bw),
                "bbox_h": int(bh),
                "brightness": round(bright, 3),
                "motion": round(mot, 3),
                "score": round(float(score), 6),
                "pass_idx": pass_idx,
            })

    return gray_blur, candidates


def track_clip(video_path: Path, clip_id: str, video_id: str, max_frames: int = 0):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return [], {
            "clip_id": clip_id,
            "video_id": video_id,
            "video_path": str(video_path),
            "status": "ERROR",
            "error": "opencv_open_failed",
            "frame_count": 0,
            "point_count": 0,
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    limit = frame_total
    if max_frames and max_frames > 0:
        limit = min(limit, max_frames)

    prev_gray = None
    last = None
    velocity = np.array([0.0, 0.0], dtype=np.float32)
    gap = 0

    points = []

    frame_idx = 0

    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray, candidates = detect_candidates(frame, prev_gray)

        best = None
        best_final = -1e9

        for c in candidates:
            xy = np.array([c["x"], c["y"]], dtype=np.float32)
            final = float(c["score"])

            if last is not None:
                pred = last + velocity
                dist = float(np.linalg.norm(xy - pred))

                # On garde un coût distance, mais on laisse le tracker se réinitialiser.
                if gap <= 5:
                    final -= dist * 0.33
                    if dist > 120:
                        final -= 40.0
                else:
                    final -= dist * 0.08

            if final > best_final:
                best_final = final
                best = c

        if best is not None and best_final > 8.0:
            xy = np.array([best["x"], best["y"]], dtype=np.float32)

            if last is not None:
                velocity = (xy - last) * 0.65 + velocity * 0.35

            last = xy
            gap = 0

            points.append({
                "clip_id": clip_id,
                "sequence_key": clip_id,
                "video_id": video_id,
                "frame": int(frame_idx),
                "x": round(float(best["x"]), 3),
                "y": round(float(best["y"]), 3),
                "score": round(float(best["score"]), 6),
                "final_score_004E": round(float(best_final), 6),
                "area": best["area"],
                "bbox_w": best["bbox_w"],
                "bbox_h": best["bbox_h"],
                "brightness": best["brightness"],
                "motion": best["motion"],
                "pass_idx": best["pass_idx"],
                "source_video": str(video_path),
                "source": "004E_raw_bright_motion",
            })

        else:
            gap += 1
            if gap > 12:
                last = None
                velocity[:] = 0.0

        prev_gray = gray
        frame_idx += 1

    cap.release()

    stats = summarize_points(points)
    stats.update({
        "clip_id": clip_id,
        "video_id": video_id,
        "video_path": str(video_path),
        "status": "OK",
        "error": "",
        "fps": round(fps, 3),
        "frame_count": frame_total,
        "frames_read": frame_idx,
    })

    return points, stats


def summarize_points(points):
    if not points:
        return {
            "point_count": 0,
            "density": 0.0,
            "first_frame": "",
            "last_frame": "",
            "travel": 0.0,
            "x_range": 0.0,
            "y_range": 0.0,
        }

    xs = np.array([p["x"] for p in points], dtype=np.float32)
    ys = np.array([p["y"] for p in points], dtype=np.float32)
    fs = np.array([p["frame"] for p in points], dtype=np.int32)

    dist = 0.0
    if len(points) >= 2:
        dx = np.diff(xs)
        dy = np.diff(ys)
        dist = float(np.sum(np.sqrt(dx * dx + dy * dy)))

    span = max(1, int(fs.max() - fs.min() + 1))

    return {
        "point_count": int(len(points)),
        "density": round(float(len(points) / span), 6),
        "first_frame": int(fs.min()),
        "last_frame": int(fs.max()),
        "travel": round(dist, 3),
        "x_range": round(float(xs.max() - xs.min()), 3),
        "y_range": round(float(ys.max() - ys.min()), 3),
    }


def write_html(path: Path, summary: dict, rows: list[dict]) -> None:
    trs = []

    for r in rows:
        cls = "ok" if r.get("point_count", 0) > 0 else "bad"
        trs.append(
            f"<tr class='{cls}'>"
            f"<td>{esc(r.get('clip_id'))}</td>"
            f"<td>{esc(r.get('video_id'))}</td>"
            f"<td>{esc(r.get('point_count'))}</td>"
            f"<td>{esc(r.get('density'))}</td>"
            f"<td>{esc(r.get('travel'))}</td>"
            f"<td>{esc(r.get('x_range'))}</td>"
            f"<td>{esc(r.get('y_range'))}</td>"
            f"<td>{esc(r.get('error'))}</td>"
            f"<td>{esc(r.get('video_path'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004E raw tracking points</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:78vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
tr.ok td{{background:rgba(116,217,159,.06)}}
tr.bad td{{background:rgba(255,80,80,.12)}}
</style>
</head>
<body>
<h1>TTFlux · 004E raw tracking points</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
<section>
<h2>Clips</h2>
<div class="wrap">
<table>
<thead>
<tr><th>clip</th><th>video</th><th>points</th><th>density</th><th>travel</th><th>x_range</th><th>y_range</th><th>error</th><th>path</th></tr>
</thead>
<tbody>
{''.join(trs)}
</tbody>
</table>
</div>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-manifest", default="runs/batch_004D_probe12/batch_config_manifest_004D.csv")
    ap.add_argument("--out-dir", default="runs/dataset_points_004E_probe12")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    root = Path.cwd()

    manifest = Path(args.batch_manifest)
    if not manifest.is_absolute():
        manifest = root / manifest

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest.is_file():
        raise SystemExit(f"Manifest absent: {manifest}")

    df = pd.read_csv(manifest)

    required = {"clip_id", "video_id", "video_path"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Colonnes absentes du manifest: {sorted(missing)}")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    all_points = []
    clip_stats = []

    print("004E clips=", len(df))

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        clip_id = str(r["clip_id"])
        video_id = str(r["video_id"])
        video_path = Path(str(r["video_path"]))

        if not video_path.is_absolute():
            video_path = root / video_path

        points, stats = track_clip(
            video_path=video_path,
            clip_id=clip_id,
            video_id=video_id,
            max_frames=args.max_frames,
        )

        all_points.extend(points)
        clip_stats.append(stats)

        print(
            f"  {i}/{len(df)} {clip_id} "
            f"points={stats.get('point_count')} density={stats.get('density')} travel={stats.get('travel')}"
        )

    out_points = out_dir / "raw_tracking_points_004E.csv"
    out_stats = out_dir / "raw_tracking_clip_stats_004E.csv"
    out_json = out_dir / "raw_tracking_summary_004E.json"
    out_html = out_dir / "raw_tracking_summary_004E.html"

    if all_points:
        fieldnames = list(all_points[0].keys())
        with out_points.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_points)
    else:
        out_points.write_text("clip_id,sequence_key,video_id,frame,x,y,score\n", encoding="utf-8")

    pd.DataFrame(clip_stats).to_csv(out_stats, index=False, encoding="utf-8")

    point_counts = [int(x.get("point_count", 0)) for x in clip_stats]

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "raw_points_only_no_segment_selection_no_delete",
        "batch_manifest": str(manifest),
        "clips": int(len(df)),
        "total_points": int(len(all_points)),
        "clips_with_points": int(sum(1 for x in point_counts if x > 0)),
        "min_points_per_clip": int(min(point_counts) if point_counts else 0),
        "median_points_per_clip": float(np.median(point_counts)) if point_counts else 0.0,
        "max_points_per_clip": int(max(point_counts) if point_counts else 0),
        "next_step": "004F patches configs so 003Q can consume raw_tracking_points_004E.csv.",
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, summary, clip_stats)

    print("004E status=OK")
    print("clips=", summary["clips"])
    print("total_points=", summary["total_points"])
    print("clips_with_points=", summary["clips_with_points"])
    print("median_points_per_clip=", summary["median_points_per_clip"])
    print("wrote", out_points)
    print("wrote", out_stats)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
