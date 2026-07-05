from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.cwd()
REG = ROOT / "data_external" / "registry"
OUT = ROOT / "runs" / "ttnet_image_extract_006M"

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")

def imwrite_unicode(path: Path, img, jpg_quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    buf.tofile(str(path))

def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def as_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

def collect_ball_points(obj):
    points = []

    def visit(node, frame_hint=None):
        if isinstance(node, dict):
            local_frame = frame_hint

            for k in ["frame", "frame_id", "frame_index", "img_frame_id", "fid"]:
                if k in node:
                    try:
                        local_frame = int(float(node[k]))
                    except Exception:
                        pass

            for k, v in node.items():
                if isinstance(k, str) and k.isdigit():
                    visit(v, int(k))

            x = None
            y = None

            for kx in ["x", "cx", "center_x", "ball_x"]:
                if kx in node:
                    x = as_float(node[kx])
                    break

            for ky in ["y", "cy", "center_y", "ball_y"]:
                if ky in node:
                    y = as_float(node[ky])
                    break

            for pk in ["position", "pos", "ball", "center"]:
                if pk in node and isinstance(node[pk], (list, tuple)) and len(node[pk]) >= 2:
                    x = as_float(node[pk][0])
                    y = as_float(node[pk][1])

            if local_frame is not None and x is not None and y is not None:
                points.append({
                    "frame": int(local_frame),
                    "x": float(x),
                    "y": float(y),
                })

            for v in node.values():
                if isinstance(v, (dict, list)):
                    visit(v, local_frame)

        elif isinstance(node, list):
            for item in node:
                visit(item, frame_hint)

    visit(obj)

    by_frame = {}
    for p in points:
        f = int(p["frame"])
        if f >= 0 and f not in by_frame:
            by_frame[f] = p

    return [by_frame[k] for k in sorted(by_frame)]

def collect_mask_frames(mask_dir: Path):
    frames = {}
    if not mask_dir.exists():
        return frames

    for p in mask_dir.rglob("*"):
        if not p.is_file():
            continue
        nums = re.findall(r"\d+", p.stem)
        if not nums:
            continue
        # Heuristique : dernier nombre du nom = frame id.
        f = int(nums[-1])
        frames[f] = p

    return frames

def sample_even(values: list[int], limit: int):
    if limit <= 0 or len(values) <= limit:
        return values
    idxs = sorted(set(round(i * (len(values) - 1) / (limit - 1)) for i in range(limit)))
    return [values[i] for i in idxs]

def find_sample(sample_id: str):
    rows = read_csv(REG / "ttnet_layout_index.csv")
    for r in rows:
        if r.get("sample_id") == sample_id:
            return r
    raise RuntimeError(f"sample not found: {sample_id}")

def dir_size(path: Path):
    total = 0
    count = 0
    if not path.exists():
        return 0, 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            count += 1
    return total, count

def gb(n: int):
    return round(n / 1024 / 1024 / 1024, 3)

parser = argparse.ArgumentParser()
parser.add_argument("--sample", default="game_1")
parser.add_argument("--limit", type=int, default=256)
parser.add_argument("--all", action="store_true")
parser.add_argument("--write-layout", action="store_true")
parser.add_argument("--clean", action="store_true")
parser.add_argument("--jpg-quality", type=int, default=90)
args = parser.parse_args()

row = find_sample(args.sample)

video_path = ROOT / row["video_path"]
ann_path = ROOT / row["annotation_path"]
images_path = ROOT / row["images_path"]

ball_path = ann_path / "ball_markup.json"
events_path = ann_path / "events_markup.json"
masks_path = ann_path / "segmentation_masks"

ball_points = collect_ball_points(load_json(ball_path))
ball_by_frame = {p["frame"]: p for p in ball_points}
mask_by_frame = collect_mask_frames(masks_path)

candidate_frames = sorted(set(ball_by_frame.keys()) | set(mask_by_frame.keys()))

if args.all:
    selected_frames = candidate_frames
else:
    selected_frames = sample_even(candidate_frames, args.limit)

if args.write_layout:
    out_dir = images_path
else:
    out_dir = OUT / f"{args.sample}_probe" / "images"

if args.clean and out_dir.exists():
    shutil.rmtree(out_dir)

out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

manifest = []
thumbs = []
read_ok = 0

for idx, frame in enumerate(selected_frames, 1):
    out_img = out_dir / f"{frame:06d}.jpg"

    if out_img.exists():
        ok_read = True
        status = "exists"
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, img = cap.read()

        ok_read = bool(ok and img is not None)
        status = "written" if ok_read else "read_failed"

        if ok_read:
            imwrite_unicode(out_img, img, jpg_quality=args.jpg_quality)

            if len(thumbs) < 32:
                thumb = img.copy()
                p = ball_by_frame.get(frame)
                if p:
                    x = int(round(p["x"]))
                    y = int(round(p["y"]))
                    cv2.circle(thumb, (x, y), 12, (0, 0, 255), 2)
                    cv2.drawMarker(thumb, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
                thumb = cv2.resize(thumb, (320, 180), interpolation=cv2.INTER_AREA)
                thumbs.append(thumb)

    if ok_read:
        read_ok += 1

    p = ball_by_frame.get(frame)
    m = mask_by_frame.get(frame)

    manifest.append({
        "sample_id": args.sample,
        "frame": frame,
        "selected_index": idx,
        "out_image": rel(out_img),
        "read_ok": ok_read,
        "status": status,
        "has_ball_point": p is not None,
        "ball_x": round(p["x"], 2) if p else "",
        "ball_y": round(p["y"], 2) if p else "",
        "has_mask": m is not None,
        "mask_path": rel(m) if m else "",
    })

cap.release()

sheet_path = OUT / f"{args.sample}_probe" / "contact_sheet.jpg"
if thumbs:
    cols = 4
    rows_n = math.ceil(len(thumbs) / cols)
    sheet = np.zeros((rows_n * 180, cols * 320, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r * 180:(r + 1) * 180, c * 320:(c + 1) * 320] = t
    imwrite_unicode(sheet_path, sheet, jpg_quality=92)

manifest_path = OUT / f"{args.sample}_probe" / "extract_manifest.csv"
write_csv(manifest_path, manifest)

out_size, out_files = dir_size(out_dir)

summary = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "sample_id": args.sample,
    "mode": "write_layout" if args.write_layout else "probe",
    "video_path": rel(video_path),
    "annotation_path": rel(ann_path),
    "out_dir": rel(out_dir),
    "video_width": width,
    "video_height": height,
    "fps": fps,
    "frame_count": frame_count,
    "ball_points": len(ball_points),
    "mask_frames": len(mask_by_frame),
    "candidate_frames_union": len(candidate_frames),
    "selected_frames": len(selected_frames),
    "read_ok": read_ok,
    "out_files": out_files,
    "out_size_gb": gb(out_size),
    "contact_sheet": rel(sheet_path),
    "manifest": rel(manifest_path),
}

summary_dir = OUT / f"{args.sample}_probe"
summary_dir.mkdir(parents=True, exist_ok=True)
(summary_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

md = []
md.append("# TTNet controlled image extraction 006M")
md.append("")
md.append(f"- Sample: `{args.sample}`")
md.append(f"- Mode: `{summary['mode']}`")
md.append(f"- Video: `{summary['video_path']}`")
md.append(f"- Annotation: `{summary['annotation_path']}`")
md.append(f"- Output: `{summary['out_dir']}`")
md.append(f"- Resolution: **{width}x{height}**")
md.append(f"- FPS: **{fps}**")
md.append(f"- Video frames: **{frame_count}**")
md.append(f"- Ball points: **{len(ball_points)}**")
md.append(f"- Mask frames: **{len(mask_by_frame)}**")
md.append(f"- Candidate frame union: **{len(candidate_frames)}**")
md.append(f"- Selected frames: **{len(selected_frames)}**")
md.append(f"- Read OK: **{read_ok} / {len(selected_frames)}**")
md.append(f"- Output files: **{out_files}**")
md.append(f"- Output logical size: **{summary['out_size_gb']} GB**")
md.append("")
md.append("## Files")
md.append("")
md.append(f"- Contact sheet: `{summary['contact_sheet']}`")
md.append(f"- Manifest: `{summary['manifest']}`")
md.append(f"- Summary JSON: `{rel(summary_dir / 'summary.json')}`")

(summary_dir / "extract_summary.md").write_text("\n".join(md), encoding="utf-8")

print("")
print("=== IMAGE EXTRACT 006M DONE ===")
print("Sample:", args.sample)
print("Mode:", summary["mode"])
print("Output:", out_dir)
print("Selected:", len(selected_frames))
print("Read OK:", read_ok, "/", len(selected_frames))
print("Output size:", summary["out_size_gb"], "GB")
print("Summary:", summary_dir / "extract_summary.md")
print("Contact sheet:", sheet_path)
