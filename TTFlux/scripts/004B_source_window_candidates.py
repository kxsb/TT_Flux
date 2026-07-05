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


VERSION = "004B"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "error": "opencv_open_failed"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    return {
        "ok": True,
        "fps": fps,
        "frames": frames,
        "width": w,
        "height": h,
        "duration_sec": frames / fps if fps else 0,
    }


def load_sources(root: Path, source_root: Path) -> list[dict]:
    rows = []

    for p in sorted(source_root.rglob("normalized_720p50.mp4")):
        info = probe_video(p)
        if not info.get("ok"):
            continue

        parent = p.parent.name
        # ex: 01_M2forQBQaZc
        parts = parent.split("_", 1)
        src_idx = parts[0] if parts else ""
        video_id = parts[1] if len(parts) > 1 else parent

        rows.append({
            "source_idx": src_idx,
            "video_id": video_id,
            "source_video": str(p),
            **info,
        })

    return rows


def activity_score_window(video_path: Path, start_frame: int, end_frame: int, sample_every: int = 10) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "score": 0.0,
            "samples": 0,
            "mean_motion": 0.0,
            "p90_motion": 0.0,
            "error": "opencv_open_failed",
        }

    motions = []
    prev = None

    try:
        for f in range(start_frame, end_frame, sample_every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            h, w = frame.shape[:2]

            # ROI centrale large : table + joueurs, en évitant bandeaux extrêmes.
            y0 = int(h * 0.12)
            y1 = int(h * 0.95)
            x0 = int(w * 0.04)
            x1 = int(w * 0.96)

            crop = frame[y0:y1, x0:x1]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev is not None:
                diff = cv2.absdiff(prev, gray)
                motions.append(float(np.mean(diff)))

            prev = gray

    finally:
        cap.release()

    if not motions:
        return {
            "score": 0.0,
            "samples": 0,
            "mean_motion": 0.0,
            "p90_motion": 0.0,
            "error": "no_motion_samples",
        }

    arr = np.array(motions, dtype=np.float32)

    mean_motion = float(np.mean(arr))
    p90_motion = float(np.percentile(arr, 90))

    # Score volontairement simple : favorise l’activité soutenue mais pas seulement les transitions/cuts.
    score = mean_motion * 0.65 + p90_motion * 0.35

    return {
        "score": round(score, 6),
        "samples": int(len(motions)),
        "mean_motion": round(mean_motion, 6),
        "p90_motion": round(p90_motion, 6),
        "error": "",
    }


def make_windows(sources: list[dict], window_sec: float, stride_sec: float, margin_sec: float, max_windows_per_video: int) -> list[dict]:
    rows = []

    for src in sources:
        fps = float(src["fps"])
        frames = int(src["frames"])
        duration = float(src["duration_sec"])

        start_sec = margin_sec
        last_start = max(margin_sec, duration - margin_sec - window_sec)

        starts = []
        t = start_sec
        while t <= last_start:
            starts.append(t)
            t += stride_sec

        if max_windows_per_video > 0:
            starts = starts[:max_windows_per_video]

        for idx, t0 in enumerate(starts, start=1):
            t1 = t0 + window_sec
            f0 = int(round(t0 * fps))
            f1 = min(frames - 1, int(round(t1 * fps)))

            rows.append({
                "candidate_id": f"C{len(rows)+1:06d}",
                "video_id": src["video_id"],
                "source_idx": src["source_idx"],
                "window_idx": idx,
                "source_video": src["source_video"],
                "fps": round(fps, 3),
                "width": src["width"],
                "height": src["height"],
                "duration_sec_video": round(duration, 3),
                "start_sec": round(t0, 3),
                "end_sec": round(t1, 3),
                "start_frame": f0,
                "end_frame": f1,
                "window_sec": round(window_sec, 3),
            })

    return rows


def write_html(path: Path, summary: dict, rows: list[dict]) -> None:
    trs = []

    for r in rows:
        trs.append(
            "<tr>"
            f"<td>{esc(r.get('rank'))}</td>"
            f"<td>{esc(r.get('candidate_id'))}</td>"
            f"<td>{esc(r.get('video_id'))}</td>"
            f"<td>{esc(r.get('start_sec'))}</td>"
            f"<td>{esc(r.get('end_sec'))}</td>"
            f"<td>{esc(r.get('activity_score'))}</td>"
            f"<td>{esc(r.get('mean_motion'))}</td>"
            f"<td>{esc(r.get('p90_motion'))}</td>"
            f"<td>{esc(r.get('source_video'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004B source window candidates</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:78vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
</style>
</head>
<body>
<h1>TTFlux · 004B source window candidates</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
<section>
<h2>Top candidates</h2>
<div class="wrap">
<table>
<thead>
<tr><th>rank</th><th>candidate</th><th>video</th><th>start</th><th>end</th><th>score</th><th>mean</th><th>p90</th><th>source</th></tr>
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
    ap.add_argument("--source-root", default="videos_dataset2_v60")
    ap.add_argument("--out-dir", default="runs/dataset_rebuild_004B")
    ap.add_argument("--window-sec", type=float, default=8.0)
    ap.add_argument("--stride-sec", type=float, default=6.0)
    ap.add_argument("--margin-sec", type=float, default=20.0)
    ap.add_argument("--sample-every-frames", type=int, default=10)
    ap.add_argument("--max-windows-per-video", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=240)
    args = ap.parse_args()

    root = Path.cwd()

    source_root = Path(args.source_root)
    if not source_root.is_absolute():
        source_root = root / source_root

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(root, source_root)

    if not sources:
        raise SystemExit(f"Aucune source normalized_720p50.mp4 trouvée dans {source_root}")

    rows = make_windows(
        sources=sources,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        margin_sec=args.margin_sec,
        max_windows_per_video=args.max_windows_per_video,
    )

    print("004B scoring windows=", len(rows))

    scored = []
    for i, row in enumerate(rows, start=1):
        score = activity_score_window(
            Path(row["source_video"]),
            int(row["start_frame"]),
            int(row["end_frame"]),
            sample_every=args.sample_every_frames,
        )

        out = {
            **row,
            "activity_score": score["score"],
            "motion_samples": score["samples"],
            "mean_motion": score["mean_motion"],
            "p90_motion": score["p90_motion"],
            "score_error": score["error"],
        }
        scored.append(out)

        if i % 50 == 0 or i == len(rows):
            print(f"  scored {i}/{len(rows)}")

    scored = sorted(scored, key=lambda r: float(r.get("activity_score") or 0), reverse=True)

    for rank, r in enumerate(scored, start=1):
        r["rank"] = rank

    selected = scored[:args.top_k]

    source_summary = []
    for src in sources:
        source_summary.append({
            "video_id": src["video_id"],
            "duration_min": round(float(src["duration_sec"]) / 60.0, 3),
            "fps": round(float(src["fps"]), 3),
            "width": src["width"],
            "height": src["height"],
            "source_video": src["source_video"],
        })

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "candidate_windows_only_no_video_cut_no_delete",
        "source_root": str(source_root),
        "source_count": len(sources),
        "source_total_duration_min": round(sum(float(s["duration_sec"]) for s in sources) / 60.0, 3),
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "margin_sec": args.margin_sec,
        "candidate_windows_total": len(scored),
        "top_k": args.top_k,
        "selected_count": len(selected),
        "sources": source_summary,
        "next_step": "004C cuts selected windows into MP4 clips and creates review/contact sheet.",
    }

    out_all = out_dir / "source_windows_all_004B.csv"
    out_top = out_dir / "source_windows_top_004B.csv"
    out_json = out_dir / "source_windows_summary_004B.json"
    out_html = out_dir / "source_windows_top_004B.html"

    fieldnames = list(scored[0].keys()) if scored else []

    with out_all.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored)

    with out_top.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, summary, selected)

    print("004B status=OK")
    print("source_count=", len(sources))
    print("source_total_duration_min=", summary["source_total_duration_min"])
    print("candidate_windows_total=", len(scored))
    print("selected_count=", len(selected))
    print("wrote", out_all)
    print("wrote", out_top)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
