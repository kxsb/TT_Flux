from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd


VERSION = "004C"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def safe_name(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in "-_":
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unknown"


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    out = {
        "clip_exists": path.is_file(),
        "clip_probe_ok": False,
        "clip_width": "",
        "clip_height": "",
        "clip_fps": "",
        "clip_frame_count": "",
        "clip_duration_sec": "",
        "clip_error": "",
    }

    if not path.is_file():
        out["clip_error"] = "missing_file"
        return out

    if not cap.isOpened():
        out["clip_error"] = "opencv_open_failed"
        return out

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        dur = frames / fps if fps else 0

        out.update({
            "clip_probe_ok": True,
            "clip_width": w,
            "clip_height": h,
            "clip_fps": round(fps, 3),
            "clip_frame_count": frames,
            "clip_duration_sec": round(dur, 3),
        })
    except Exception as exc:
        out["clip_error"] = repr(exc)
    finally:
        cap.release()

    return out


def select_balanced(df: pd.DataFrame, target: int, per_video_cap: int, min_gap_sec: float) -> pd.DataFrame:
    df = df.copy()
    df["activity_score"] = pd.to_numeric(df["activity_score"], errors="coerce").fillna(0.0)
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce").fillna(0.0)
    df["rank"] = pd.to_numeric(df.get("rank", 999999), errors="coerce").fillna(999999)

    df = df.sort_values(["activity_score", "rank"], ascending=[False, True])

    selected = []
    by_video_count = {}
    by_video_starts = {}

    for _, row in df.iterrows():
        vid = str(row["video_id"])

        if by_video_count.get(vid, 0) >= per_video_cap:
            continue

        t = float(row["start_sec"])
        starts = by_video_starts.get(vid, [])

        if any(abs(t - old) < min_gap_sec for old in starts):
            continue

        selected.append(row.to_dict())
        by_video_count[vid] = by_video_count.get(vid, 0) + 1
        by_video_starts.setdefault(vid, []).append(t)

        if len(selected) >= target:
            break

    # Si la contrainte min_gap bloque trop, on complète sans min_gap mais en gardant cap/video.
    if len(selected) < target:
        selected_ids = {str(r["candidate_id"]) for r in selected}

        for _, row in df.iterrows():
            cid = str(row["candidate_id"])
            if cid in selected_ids:
                continue

            vid = str(row["video_id"])
            if by_video_count.get(vid, 0) >= per_video_cap:
                continue

            selected.append(row.to_dict())
            selected_ids.add(cid)
            by_video_count[vid] = by_video_count.get(vid, 0) + 1

            if len(selected) >= target:
                break

    out = pd.DataFrame(selected)
    out.insert(0, "dataset_rank_004C", range(1, len(out) + 1))
    return out


def cut_with_ffmpeg(ffmpeg: str, src: Path, dst: Path, start_sec: float, duration_sec: float, mode: str) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(src),
            "-t", f"{duration_sec:.3f}",
            "-an",
            "-c:v", "copy",
            str(dst),
        ]
    else:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(src),
            "-t", f"{duration_sec:.3f}",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            str(dst),
        ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return False, (p.stderr or p.stdout or f"ffmpeg_failed_code_{p.returncode}").strip()

    if not dst.is_file() or dst.stat().st_size <= 0:
        return False, "ffmpeg_created_empty_file"

    return True, ""


def cut_with_opencv(src: Path, dst: Path, start_frame: int, end_frame: int, fps: float) -> tuple[bool, str]:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False, "opencv_open_failed"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 50.0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, src_fps, (w, h))

    if not writer.isOpened():
        cap.release()
        return False, "opencv_writer_failed"

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    n = 0
    ok_final = True

    try:
        for _frame_idx in range(int(start_frame), int(end_frame) + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            n += 1
    except Exception as exc:
        ok_final = False
        err = repr(exc)
    else:
        err = ""
    finally:
        writer.release()
        cap.release()

    if not ok_final:
        return False, err

    if n <= 0:
        return False, "no_frames_written"

    return True, ""


def write_html(path: Path, summary: dict, rows: list[dict], preview_limit: int) -> None:
    trs = []

    for r in rows:
        rel = r.get("clip_relpath", "")
        video_tag = ""
        if r.get("clip_probe_ok") and r.get("dataset_rank_004C", 999999) <= preview_limit:
            video_tag = f"<video controls preload='metadata' width='260' src='{esc(rel)}'></video>"

        cls = "ok" if r.get("clip_probe_ok") else "bad"

        trs.append(
            f"<tr class='{cls}'>"
            f"<td>{esc(r.get('dataset_rank_004C'))}</td>"
            f"<td>{esc(r.get('clip_id'))}</td>"
            f"<td>{esc(r.get('video_id'))}</td>"
            f"<td>{esc(r.get('start_sec'))}</td>"
            f"<td>{esc(r.get('end_sec'))}</td>"
            f"<td>{esc(r.get('activity_score'))}</td>"
            f"<td>{esc(r.get('clip_duration_sec'))}</td>"
            f"<td>{video_tag}</td>"
            f"<td>{esc(r.get('clip_path'))}</td>"
            f"<td>{esc(r.get('cut_error') or r.get('clip_error'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004C rebuilt dataset clips</title>
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
video{{border-radius:8px;background:#000}}
</style>
</head>
<body>
<h1>TTFlux · 004C rebuilt dataset clips</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
<section>
<h2>Clips</h2>
<div class="wrap">
<table>
<thead>
<tr>
<th>rank</th><th>clip</th><th>video</th><th>start</th><th>end</th><th>score</th><th>duration</th><th>preview</th><th>path</th><th>error</th>
</tr>
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
    ap.add_argument("--input", default="runs/dataset_rebuild_004B/source_windows_all_004B.csv")
    ap.add_argument("--out-dir", default="runs/dataset_rebuild_004C")
    ap.add_argument("--target", type=int, default=240)
    ap.add_argument("--per-video-cap", type=int, default=40)
    ap.add_argument("--min-gap-sec", type=float, default=10.0)
    ap.add_argument("--mode", choices=["copy", "reencode"], default="copy")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview-limit", type=int, default=80)
    args = ap.parse_args()

    root = Path.cwd()

    input_csv = Path(args.input)
    if not input_csv.is_absolute():
        input_csv = root / input_csv

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    clips_dir = out_dir / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.is_file():
        raise SystemExit(f"Input absent: {input_csv}")

    df = pd.read_csv(input_csv)
    selected = select_balanced(
        df,
        target=args.target,
        per_video_cap=args.per_video_cap,
        min_gap_sec=args.min_gap_sec,
    )

    if args.limit and args.limit > 0:
        selected = selected.head(args.limit).copy()

    ffmpeg = shutil.which("ffmpeg")

    rows = []
    errors = 0

    print("004C selected=", len(selected))
    print("004C ffmpeg=", ffmpeg or "NOT_FOUND")
    print("004C mode=", args.mode)

    for i, (_, r) in enumerate(selected.iterrows(), start=1):
        rank = int(r["dataset_rank_004C"])
        video_id = safe_name(r["video_id"])
        start_frame = int(r["start_frame"])
        end_frame = int(r["end_frame"])
        start_sec = float(r["start_sec"])
        end_sec = float(r["end_sec"])
        fps = float(r.get("fps", 50.0) or 50.0)
        duration_sec = max(0.1, end_sec - start_sec)

        clip_id = f"ds004_{rank:04d}_{video_id}_f{start_frame}_to_f{end_frame}"
        clip_path = clips_dir / f"{clip_id}.mp4"

        src = Path(str(r["source_video"]))

        cut_ok = False
        cut_error = ""

        if ffmpeg:
            cut_ok, cut_error = cut_with_ffmpeg(ffmpeg, src, clip_path, start_sec, duration_sec, args.mode)

        if not cut_ok:
            # fallback lent mais utile si ffmpeg/copy échoue.
            cut_ok, cut_error2 = cut_with_opencv(src, clip_path, start_frame, end_frame, fps)
            if not cut_ok:
                cut_error = f"{cut_error} | fallback={cut_error2}".strip(" |")

        probe = probe_video(clip_path)

        if not probe.get("clip_probe_ok"):
            errors += 1

        row = r.to_dict()
        row.update({
            "clip_id": clip_id,
            "clip_path": str(clip_path),
            "clip_relpath": "clips/" + clip_path.name,
            "cut_ok": bool(cut_ok),
            "cut_error": cut_error,
            **probe,
        })

        rows.append(row)

        if i % 25 == 0 or i == len(selected):
            print(f"  cut {i}/{len(selected)} errors={errors}")

    by_video = {}
    for r in rows:
        vid = str(r.get("video_id"))
        by_video.setdefault(vid, 0)
        by_video[vid] += 1

    ok_count = sum(1 for r in rows if r.get("clip_probe_ok"))

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "dataset_clip_cut_only_no_tracking_no_delete",
        "input": str(input_csv),
        "out_dir": str(out_dir),
        "clips_dir": str(clips_dir),
        "target": args.target,
        "actual_selected": len(rows),
        "clip_probe_ok": ok_count,
        "clip_probe_failed": len(rows) - ok_count,
        "mode": args.mode,
        "per_video_cap": args.per_video_cap,
        "min_gap_sec": args.min_gap_sec,
        "by_video_count": by_video,
        "next_step": "004D builds TTFlux batch configs from these clips.",
    }

    out_csv = out_dir / "rebuilt_clips_manifest_004C.csv"
    out_json = out_dir / "rebuilt_clips_summary_004C.json"
    out_html = out_dir / "rebuilt_clips_manifest_004C.html"

    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, summary, rows, args.preview_limit)

    print("004C status=OK")
    print("actual_selected=", len(rows))
    print("clip_probe_ok=", ok_count)
    print("clip_probe_failed=", len(rows) - ok_count)
    print("by_video_count=", json.dumps(by_video, ensure_ascii=False))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
