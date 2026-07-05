from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

ROOT = Path.cwd()
REG = ROOT / "data_external" / "registry"
OUT = ROOT / "runs" / "ttnet_dataset_smoke_006J"

SAMPLE_ID = "game_1"
N_FRAMES = 32

OUT.mkdir(parents=True, exist_ok=True)

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")

def imwrite_unicode(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
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
    """
    Essaie de sortir frame/x/y depuis plusieurs formats JSON possibles.
    Robuste mais non destructif.
    """
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

            # Cas: {"12345": {"x":..., "y":...}}
            for k, v in node.items():
                if isinstance(k, str) and k.isdigit():
                    try:
                        visit(v, int(k))
                    except Exception:
                        pass

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

            # Cas possible: {"position": [x, y]}
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

    # Dédup frame, garde le premier point valide.
    by_frame = {}
    for p in points:
        f = p["frame"]
        if f >= 0 and f not in by_frame:
            by_frame[f] = p

    return [by_frame[k] for k in sorted(by_frame)]

def sample_even(points, n):
    if len(points) <= n:
        return points
    idxs = sorted(set(round(i * (len(points) - 1) / (n - 1)) for i in range(n)))
    return [points[i] for i in idxs]

def find_sample_row(sample_id: str):
    rows = read_csv(REG / "ttnet_layout_index.csv")
    for r in rows:
        if r.get("sample_id") == sample_id:
            return r
    raise RuntimeError(f"sample not found in ttnet_layout_index.csv: {sample_id}")

row = find_sample_row(SAMPLE_ID)

video_path = ROOT / row["video_path"]
ann_path = ROOT / row["annotation_path"]
ball_path = ann_path / "ball_markup.json"
events_path = ann_path / "events_markup.json"
masks_path = ann_path / "segmentation_masks"

ball_json = load_json(ball_path)
events_json = load_json(events_path)
points = collect_ball_points(ball_json)

if not points:
    raise RuntimeError(f"No ball points found in {ball_path}")

picked = sample_even(points, N_FRAMES)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

sample_out = OUT / SAMPLE_ID
frame_dir = sample_out / "frames"
frame_dir.mkdir(parents=True, exist_ok=True)

sample_rows = []
thumbs = []

for p in picked:
    f = int(p["frame"])
    x = int(round(p["x"]))
    y = int(round(p["y"]))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()

    out_img = frame_dir / f"{SAMPLE_ID}_frame_{f:06d}.jpg"

    if ok and img is not None:
        # Marqueur discret autour de la balle annotée.
        cv2.circle(img, (x, y), 12, (0, 0, 255), 2)
        cv2.drawMarker(img, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        cv2.putText(img, f"f={f} x={x} y={y}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        imwrite_unicode(out_img, img)

        thumb = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)

    sample_rows.append({
        "sample_id": SAMPLE_ID,
        "frame": f,
        "x": x,
        "y": y,
        "read_ok": bool(ok),
        "out_image": rel(out_img) if ok else "",
    })

cap.release()

# Contact sheet
if thumbs:
    cols = 4
    rows_n = math.ceil(len(thumbs) / cols)
    sheet = np.zeros((rows_n * 180, cols * 320, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r * 180:(r + 1) * 180, c * 320:(c + 1) * 320] = t
    imwrite_unicode(sample_out / "contact_sheet.jpg", sheet)

write_csv(sample_out / "sampled_frames.csv", sample_rows)

summary = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "sample_id": SAMPLE_ID,
    "video_path": rel(video_path),
    "annotation_path": rel(ann_path),
    "video_open_ok": True,
    "fps": fps,
    "frame_count_cv2": frame_count,
    "width": width,
    "height": height,
    "ball_points_collected": len(points),
    "events_top_level_count": len(events_json) if isinstance(events_json, (list, dict)) else 1,
    "mask_files": sum(1 for p in masks_path.rglob("*") if p.is_file()) if masks_path.exists() else 0,
    "sampled_frames": len(sample_rows),
    "sampled_read_ok": sum(1 for r in sample_rows if r["read_ok"]),
    "contact_sheet": rel(sample_out / "contact_sheet.jpg"),
    "csv": rel(sample_out / "sampled_frames.csv"),
}

(sample_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

md = []
md.append("# TTNet dataset smoke test 006J")
md.append("")
md.append(f"- Sample: `{SAMPLE_ID}`")
md.append(f"- Video: `{rel(video_path)}`")
md.append(f"- Annotation: `{rel(ann_path)}`")
md.append(f"- Resolution: **{width}x{height}**")
md.append(f"- FPS: **{fps}**")
md.append(f"- Video frames: **{frame_count}**")
md.append(f"- Ball points collected: **{len(points)}**")
md.append(f"- Events top-level count: **{summary['events_top_level_count']}**")
md.append(f"- Mask files: **{summary['mask_files']}**")
md.append(f"- Sampled frames: **{summary['sampled_frames']}**")
md.append(f"- Sampled read OK: **{summary['sampled_read_ok']}**")
md.append("")
md.append("## Outputs")
md.append("")
md.append(f"- Contact sheet: `{summary['contact_sheet']}`")
md.append(f"- CSV: `{summary['csv']}`")
md.append(f"- Frames dir: `{rel(frame_dir)}`")

(sample_out / "smoke_summary.md").write_text("\n".join(md), encoding="utf-8")

print("")
print("=== TTNET SMOKE 006J DONE ===")
print("Summary:", sample_out / "smoke_summary.md")
print("Contact sheet:", sample_out / "contact_sheet.jpg")
print("Frames read OK:", summary["sampled_read_ok"], "/", summary["sampled_frames"])
print("Ball points:", len(points))
print("Masks:", summary["mask_files"])
