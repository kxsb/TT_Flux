from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


VERSION = "004R"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False}

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


def load_sources(source_root: Path) -> list[dict]:
    rows = []

    for p in sorted(source_root.rglob("normalized_720p50.mp4")):
        info = probe_video(p)
        if not info.get("ok"):
            continue

        parent = p.parent.name
        parts = parent.split("_", 1)

        rows.append({
            "source_idx": parts[0] if parts else "",
            "video_id": parts[1] if len(parts) > 1 else parent,
            "source_video": str(p),
            **info,
        })

    return rows


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) == 0:
        return x
    k = max(1, int(k))
    if k <= 1:
        return x
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same")


def activity_series(video_path: Path, sample_every_frames: int = 10) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    rows = []
    prev = None

    for f in range(0, frame_count, sample_every_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]

        # ROI large : table + joueurs, on évite seulement les bordures extrêmes.
        y0 = int(h * 0.05)
        y1 = int(h * 0.97)
        x0 = int(w * 0.02)
        x1 = int(w * 0.98)

        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (360, 210), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        motion = 0.0
        p90 = 0.0

        if prev is not None:
            diff = cv2.absdiff(prev, gray)
            motion = float(np.mean(diff))
            p90 = float(np.percentile(diff, 90))

        rows.append({
            "frame": int(f),
            "time_sec": float(f / fps),
            "motion_mean": motion,
            "motion_p90": p90,
            "activity": motion * 0.65 + p90 * 0.35,
        })

        prev = gray

    cap.release()
    return rows


def spans_from_activity(
    series: list[dict],
    duration_sec: float,
    q: float,
    min_sec: float,
    max_sec: float,
    pad_sec: float,
    merge_gap_sec: float,
) -> list[dict]:
    if not series:
        return []

    times = np.array([r["time_sec"] for r in series], dtype=np.float32)
    scores = np.array([r["activity"] for r in series], dtype=np.float32)

    scores_s = smooth(scores, 5)

    positive_scores = scores_s[scores_s > 0]
    if len(positive_scores) == 0:
        return []

    threshold = float(np.quantile(positive_scores, q))

    active = scores_s >= threshold

    raw = []
    start = None

    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            raw.append((start, i - 1))
            start = None

    if start is not None:
        raw.append((start, len(active) - 1))

    spans = []
    for a, b in raw:
        t0 = max(0.0, float(times[a]) - pad_sec)
        t1 = min(duration_sec, float(times[b]) + pad_sec)

        if t1 - t0 >= min_sec * 0.45:
            spans.append([t0, t1])

    if not spans:
        return []

    # Fusion si gap court.
    spans = sorted(spans)
    merged = [spans[0]]

    for t0, t1 in spans[1:]:
        last = merged[-1]
        if t0 - last[1] <= merge_gap_sec:
            last[1] = max(last[1], t1)
        else:
            merged.append([t0, t1])

    final = []

    for t0, t1 in merged:
        dur = t1 - t0

        if dur < min_sec:
            center = (t0 + t1) / 2.0
            t0 = max(0.0, center - min_sec / 2.0)
            t1 = min(duration_sec, t0 + min_sec)

        # Split si trop long.
        dur = t1 - t0
        if dur <= max_sec:
            final.append([t0, t1])
        else:
            step = max_sec * 0.70
            s = t0
            while s < t1:
                e = min(t1, s + max_sec)
                if e - s >= min_sec:
                    final.append([s, e])
                s += step

    out = []

    for t0, t1 in final:
        mask = (times >= t0) & (times <= t1)
        sc = scores_s[mask]

        if len(sc):
            score = float(np.mean(sc) * 0.55 + np.percentile(sc, 90) * 0.45)
            mean_activity = float(np.mean(sc))
            p90_activity = float(np.percentile(sc, 90))
        else:
            score = 0.0
            mean_activity = 0.0
            p90_activity = 0.0

        out.append({
            "start_sec": round(float(t0), 3),
            "end_sec": round(float(t1), 3),
            "window_sec": round(float(t1 - t0), 3),
            "activity_score": round(score, 6),
            "mean_motion": round(mean_activity, 6),
            "p90_motion": round(p90_activity, 6),
            "threshold": round(threshold, 6),
        })

    out = sorted(out, key=lambda r: r["activity_score"], reverse=True)
    return out


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
            f"<td>{esc(r.get('window_sec'))}</td>"
            f"<td>{esc(r.get('activity_score'))}</td>"
            f"<td>{esc(r.get('source_video'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004R rally windows</title>
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
<h1>TTFlux · 004R rally-like windows</h1>
<section><h2>Résumé</h2><pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre></section>
<section>
<h2>Fenêtres</h2>
<div class="wrap">
<table>
<thead>
<tr><th>rank</th><th>id</th><th>video</th><th>start</th><th>end</th><th>sec</th><th>score</th><th>source</th></tr>
</thead>
<tbody>{''.join(trs)}</tbody>
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
    ap.add_argument("--out-dir", default="runs/dataset_rebuild_004R_rallies")
    ap.add_argument("--sample-every-frames", type=int, default=10)
    ap.add_argument("--activity-quantile", type=float, default=0.58)
    ap.add_argument("--min-sec", type=float, default=10.0)
    ap.add_argument("--max-sec", type=float, default=24.0)
    ap.add_argument("--pad-sec", type=float, default=2.0)
    ap.add_argument("--merge-gap-sec", type=float, default=2.5)
    ap.add_argument("--top-k", type=int, default=160)
    args = ap.parse_args()

    root = Path.cwd()

    source_root = Path(args.source_root)
    if not source_root.is_absolute():
        source_root = root / source_root

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(source_root)
    if not sources:
        raise SystemExit(f"Aucune source normalized_720p50.mp4 dans {source_root}")

    rows = []

    print("004R sources=", len(sources))

    for src_i, src in enumerate(sources, start=1):
        p = Path(src["source_video"])
        duration = float(src["duration_sec"])
        fps = float(src["fps"])
        frames = int(src["frames"])

        series = activity_series(p, sample_every_frames=args.sample_every_frames)

        spans = spans_from_activity(
            series=series,
            duration_sec=duration,
            q=args.activity_quantile,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            pad_sec=args.pad_sec,
            merge_gap_sec=args.merge_gap_sec,
        )

        for j, sp in enumerate(spans, start=1):
            start_sec = float(sp["start_sec"])
            end_sec = float(sp["end_sec"])

            row = {
                "candidate_id": f"RLY{len(rows)+1:06d}",
                "video_id": src["video_id"],
                "source_idx": src["source_idx"],
                "window_idx": j,
                "source_video": src["source_video"],
                "fps": round(fps, 3),
                "width": src["width"],
                "height": src["height"],
                "duration_sec_video": round(duration, 3),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "start_frame": int(round(start_sec * fps)),
                "end_frame": min(frames - 1, int(round(end_sec * fps))),
                "window_sec": round(end_sec - start_sec, 3),
                "activity_score": sp["activity_score"],
                "mean_motion": sp["mean_motion"],
                "p90_motion": sp["p90_motion"],
                "threshold": sp["threshold"],
            }
            rows.append(row)

        print(f"  {src_i}/{len(sources)} {src['video_id']} spans={len(spans)}")

    rows = sorted(rows, key=lambda r: float(r["activity_score"]), reverse=True)

    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank

    selected = rows[:args.top_k]

    by_video = {}
    for r in selected:
        by_video[str(r["video_id"])] = by_video.get(str(r["video_id"]), 0) + 1

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "rally_like_window_detection_no_tracking_no_delete",
        "source_root": str(source_root),
        "source_count": len(sources),
        "source_total_duration_min": round(sum(float(s["duration_sec"]) for s in sources) / 60.0, 3),
        "activity_quantile": args.activity_quantile,
        "min_sec": args.min_sec,
        "max_sec": args.max_sec,
        "pad_sec": args.pad_sec,
        "merge_gap_sec": args.merge_gap_sec,
        "windows_total": len(rows),
        "top_k": args.top_k,
        "selected_count": len(selected),
        "selected_by_video": by_video,
        "next_step": "Cut selected windows with 004C, then build long-segment manifest directly instead of 003Q mini-segments.",
    }

    out_all = out_dir / "rally_windows_all_004R.csv"
    out_top = out_dir / "rally_windows_top_004R.csv"
    out_json = out_dir / "rally_windows_summary_004R.json"
    out_html = out_dir / "rally_windows_top_004R.html"

    fieldnames = list(rows[0].keys()) if rows else []

    with out_all.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with out_top.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, summary, selected)

    print("004R status=OK")
    print("source_count=", len(sources))
    print("windows_total=", len(rows))
    print("selected_count=", len(selected))
    print("selected_by_video=", json.dumps(by_video, ensure_ascii=False))
    print("wrote", out_all)
    print("wrote", out_top)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
