from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "004C2"


def load_004c_module(root: Path):
    p = root / "scripts" / "004C_cut_rebuilt_dataset_clips.py"
    if not p.is_file():
        raise RuntimeError(f"004C module absent: {p}")

    spec = importlib.util.spec_from_file_location("ttflux_004c", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible de charger 004C")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def far_enough(t: float, starts: list[float], min_gap: float) -> bool:
    return all(abs(t - old) >= min_gap for old in starts)


def pick_from_video(sub: pd.DataFrame, n: int, min_gap: float, selected_ids: set[str]) -> list[dict]:
    picked = []
    starts = []

    sub = sub.sort_values(["activity_score", "rank"], ascending=[False, True])

    # Phase stricte avec gap.
    for _, row in sub.iterrows():
        cid = str(row["candidate_id"])
        if cid in selected_ids:
            continue

        t = float(row["start_sec"])
        if not far_enough(t, starts, min_gap):
            continue

        d = row.to_dict()
        picked.append(d)
        starts.append(t)
        selected_ids.add(cid)

        if len(picked) >= n:
            return picked

    # Phase relaxée pour les vidéos courtes.
    for _, row in sub.iterrows():
        cid = str(row["candidate_id"])
        if cid in selected_ids:
            continue

        d = row.to_dict()
        picked.append(d)
        selected_ids.add(cid)

        if len(picked) >= n:
            return picked

    return picked


def select_floor_balanced(
    df: pd.DataFrame,
    target: int,
    min_per_video: int,
    max_per_video: int,
    min_gap_sec: float,
) -> pd.DataFrame:
    df = df.copy()

    df["activity_score"] = pd.to_numeric(df["activity_score"], errors="coerce").fillna(0.0)
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce").fillna(0.0)
    df["rank"] = pd.to_numeric(df.get("rank", 999999), errors="coerce").fillna(999999)

    selected = []
    selected_ids: set[str] = set()
    by_video_count: dict[str, int] = {}
    by_video_starts: dict[str, list[float]] = {}

    videos = sorted(df["video_id"].astype(str).unique().tolist())

    # 1) Floor par vidéo.
    for vid in videos:
        sub = df[df["video_id"].astype(str).eq(vid)].copy()
        wanted = min(min_per_video, max_per_video, len(sub))

        picked = pick_from_video(sub, wanted, min_gap_sec, selected_ids)

        for d in picked:
            selected.append(d)
            by_video_count[vid] = by_video_count.get(vid, 0) + 1
            by_video_starts.setdefault(vid, []).append(float(d["start_sec"]))

    # 2) Remplissage global avec cap + gap.
    ranked = df.sort_values(["activity_score", "rank"], ascending=[False, True])

    for _, row in ranked.iterrows():
        if len(selected) >= target:
            break

        cid = str(row["candidate_id"])
        if cid in selected_ids:
            continue

        vid = str(row["video_id"])
        if by_video_count.get(vid, 0) >= max_per_video:
            continue

        t = float(row["start_sec"])
        if not far_enough(t, by_video_starts.get(vid, []), min_gap_sec):
            continue

        d = row.to_dict()
        selected.append(d)
        selected_ids.add(cid)
        by_video_count[vid] = by_video_count.get(vid, 0) + 1
        by_video_starts.setdefault(vid, []).append(t)

    # 3) Remplissage relaxé si nécessaire.
    for _, row in ranked.iterrows():
        if len(selected) >= target:
            break

        cid = str(row["candidate_id"])
        if cid in selected_ids:
            continue

        vid = str(row["video_id"])
        if by_video_count.get(vid, 0) >= max_per_video:
            continue

        d = row.to_dict()
        selected.append(d)
        selected_ids.add(cid)
        by_video_count[vid] = by_video_count.get(vid, 0) + 1

    out = pd.DataFrame(selected)

    if out.empty:
        raise RuntimeError("Aucune fenêtre sélectionnée")

    out = out.head(target).copy()
    out.insert(0, "dataset_rank_004C", range(1, len(out) + 1))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="runs/dataset_rebuild_004B/source_windows_all_004B.csv")
    ap.add_argument("--out-dir", default="runs/dataset_rebuild_004C_balanced")
    ap.add_argument("--target", type=int, default=240)
    ap.add_argument("--min-per-video", type=int, default=12)
    ap.add_argument("--max-per-video", type=int, default=32)
    ap.add_argument("--min-gap-sec", type=float, default=8.0)
    ap.add_argument("--mode", choices=["copy", "reencode"], default="copy")
    ap.add_argument("--preview-limit", type=int, default=80)
    args = ap.parse_args()

    root = Path.cwd()
    mod004c = load_004c_module(root)

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

    selected = select_floor_balanced(
        df,
        target=args.target,
        min_per_video=args.min_per_video,
        max_per_video=args.max_per_video,
        min_gap_sec=args.min_gap_sec,
    )

    ffmpeg = shutil.which("ffmpeg")

    rows = []
    errors = 0

    print("004C2 selected=", len(selected))
    print("004C2 ffmpeg=", ffmpeg or "NOT_FOUND")
    print("004C2 mode=", args.mode)

    for i, (_, r) in enumerate(selected.iterrows(), start=1):
        rank = int(r["dataset_rank_004C"])
        video_id = mod004c.safe_name(r["video_id"])
        start_frame = int(r["start_frame"])
        end_frame = int(r["end_frame"])
        start_sec = float(r["start_sec"])
        end_sec = float(r["end_sec"])
        fps = float(r.get("fps", 50.0) or 50.0)
        duration_sec = max(0.1, end_sec - start_sec)

        clip_id = f"ds004b_{rank:04d}_{video_id}_f{start_frame}_to_f{end_frame}"
        clip_path = clips_dir / f"{clip_id}.mp4"
        src = Path(str(r["source_video"]))

        cut_ok = False
        cut_error = ""

        if ffmpeg:
            cut_ok, cut_error = mod004c.cut_with_ffmpeg(
                ffmpeg,
                src,
                clip_path,
                start_sec,
                duration_sec,
                args.mode,
            )

        if not cut_ok:
            cut_ok, cut_error2 = mod004c.cut_with_opencv(
                src,
                clip_path,
                start_frame,
                end_frame,
                fps,
            )
            if not cut_ok:
                cut_error = f"{cut_error} | fallback={cut_error2}".strip(" |")

        probe = mod004c.probe_video(clip_path)

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
        "policy": "balanced_dataset_clip_cut_only_no_tracking_no_delete",
        "input": str(input_csv),
        "out_dir": str(out_dir),
        "clips_dir": str(clips_dir),
        "target": args.target,
        "actual_selected": len(rows),
        "clip_probe_ok": ok_count,
        "clip_probe_failed": len(rows) - ok_count,
        "mode": args.mode,
        "min_per_video": args.min_per_video,
        "max_per_video": args.max_per_video,
        "min_gap_sec": args.min_gap_sec,
        "by_video_count": by_video,
        "source_videos_present": sorted(by_video.keys()),
        "next_step": "004D builds TTFlux batch configs from these balanced clips.",
    }

    out_csv = out_dir / "rebuilt_clips_manifest_004C2.csv"
    out_json = out_dir / "rebuilt_clips_summary_004C2.json"
    out_html = out_dir / "rebuilt_clips_manifest_004C2.html"

    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    mod004c.write_html(out_html, summary, rows, args.preview_limit)

    print("004C2 status=OK")
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
