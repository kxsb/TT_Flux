from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd


VERSION = "004T"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def rel(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    out = {
        "probe_ok": False,
        "fps": "",
        "frame_count": "",
        "width": "",
        "height": "",
        "duration_sec": "",
        "error": "",
    }

    if not path.is_file():
        out["error"] = "missing_file"
        return out

    if not cap.isOpened():
        out["error"] = "opencv_open_failed"
        return out

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        dur = frames / fps if fps else 0

        out.update({
            "probe_ok": True,
            "fps": round(fps, 3),
            "frame_count": frames,
            "width": w,
            "height": h,
            "duration_sec": round(dur, 3),
            "error": "",
        })
    except Exception as exc:
        out["error"] = repr(exc)
    finally:
        cap.release()

    return out


def write_html(path: Path, rows: list[dict], summary: dict):
    trs = []

    for r in rows:
        video = ""
        if r.get("probe_ok"):
            video = f"<video controls preload='metadata' width='360' src='{esc(r.get('mp4'))}'></video>"

        trs.append(
            "<tr>"
            f"<td>{esc(r.get('rally_id'))}</td>"
            f"<td>{esc(r.get('video_id'))}</td>"
            f"<td>{esc(r.get('duration_sec'))}</td>"
            f"<td>{esc(r.get('start_sec_source'))}</td>"
            f"<td>{esc(r.get('end_sec_source'))}</td>"
            f"<td>{esc(r.get('activity_score'))}</td>"
            f"<td>{video}</td>"
            f"<td>{esc(r.get('clip_path'))}</td>"
            f"<td>{esc(r.get('error'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004T rally manifest</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:78vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
video{{border-radius:8px;background:#000}}
</style>
</head>
<body>
<h1>TTFlux · 004T rally manifest</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
<section>
<h2>Rally clips</h2>
<div class="wrap">
<table>
<thead>
<tr>
<th>rally</th><th>video</th><th>duration</th><th>source start</th><th>source end</th><th>activity</th><th>preview</th><th>path</th><th>error</th>
</tr>
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
    ap.add_argument("--clips-csv", default="runs/dataset_rebuild_004S_rally_clips_balanced/rebuilt_clips_manifest_004C2.csv")
    ap.add_argument("--out-dir", default="runs/rally_dataset_004T")
    args = ap.parse_args()

    root = Path.cwd()

    clips_csv = Path(args.clips_csv)
    if not clips_csv.is_absolute():
        clips_csv = root / clips_csv

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    if not clips_csv.is_file():
        raise SystemExit(f"clips csv absent: {clips_csv}")

    df = pd.read_csv(clips_csv)

    rows = []

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        clip_path = Path(str(r.get("clip_path", "")))
        if not clip_path.is_absolute():
            clip_path = root / clip_path

        p = probe_video(clip_path)

        video_id = str(r.get("video_id", ""))
        rally_id = f"RLY{i:04d}_{video_id}"

        row = {
            "rally_id": rally_id,
            "review_id": f"RLY{i:04d}",
            "clip_id": str(r.get("clip_id", rally_id)),
            "video_id": video_id,
            "source_video": str(r.get("source_video", "")),
            "source_candidate_id": str(r.get("candidate_id", "")),
            "source_rank": str(r.get("rank", "")),
            "source_dataset_rank": str(r.get("dataset_rank_004C", "")),
            "start_sec_source": r.get("start_sec", ""),
            "end_sec_source": r.get("end_sec", ""),
            "start_frame_source": r.get("start_frame", ""),
            "end_frame_source": r.get("end_frame", ""),
            "activity_score": r.get("activity_score", ""),
            "mp4": rel(clip_path, out_dir) if clip_path.is_relative_to(out_dir) else rel(clip_path, root),
            "clip_path": str(clip_path),
            "segment_idx": 1,
            "first_frame": 0,
            "last_frame": int(p["frame_count"]) - 1 if p["probe_ok"] else "",
            "n_points": "",
            "density": "",
            "travel": "",
            "x_range": "",
            "y_range": "",
            "guess": "rally_candidate",
            "flags": "",
            **p,
        }

        rows.append(row)

    by_video = {}
    total_sec = 0.0

    for r in rows:
        by_video[r["video_id"]] = by_video.get(r["video_id"], 0) + 1
        if r.get("probe_ok"):
            total_sec += float(r.get("duration_sec") or 0)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "one_long_clip_equals_one_rally_sequence_no_003Q_recut",
        "input": str(clips_csv),
        "out_dir": str(out_dir),
        "rally_count": len(rows),
        "probe_ok": int(sum(1 for r in rows if r.get("probe_ok"))),
        "probe_failed": int(sum(1 for r in rows if not r.get("probe_ok"))),
        "total_duration_min": round(total_sec / 60.0, 3),
        "by_video": by_video,
        "next": "004U annotation/ranker on coherent long rally clips.",
    }

    out_csv = out_dir / "rally_manifest_004T.csv"
    out_json = out_dir / "rally_manifest_summary_004T.json"
    out_html = out_dir / "rally_manifest_004T.html"

    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, rows, summary)

    print("004T status=OK")
    print("rally_count=", summary["rally_count"])
    print("probe_ok=", summary["probe_ok"])
    print("probe_failed=", summary["probe_failed"])
    print("total_duration_min=", summary["total_duration_min"])
    print("by_video=", json.dumps(by_video, ensure_ascii=False))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
