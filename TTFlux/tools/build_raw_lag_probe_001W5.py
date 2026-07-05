from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


OFFSETS = [-6, -3, 0, 3, 6]


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in reader]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_json_values(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from iter_json_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_json_values(v)


def find_video_in_config(config_path: Path) -> Path | None:
    data = read_json(config_path)
    base = config_path.parent

    candidates: list[Path] = []

    for key, value in iter_json_values(data):
        if not isinstance(value, str):
            continue

        low_key = str(key).lower()
        low_val = value.lower()

        if (
            "video" not in low_key
            and "clip" not in low_key
            and not low_val.endswith((".mp4", ".avi", ".mov", ".mkv"))
        ):
            continue

        if not low_val.endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        p = Path(value)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((base / p).resolve())
            candidates.append((Path.cwd() / p).resolve())

    for p in candidates:
        if p.exists():
            return p

    return None


def build_clip_video_map(config_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}

    for cfg in sorted(config_dir.glob("*.json")):
        video = find_video_in_config(cfg)
        if video is None:
            continue

        stem = cfg.stem

        # Configs type batch_001E_01_<clip_id>
        m = re.search(r"(?:batch_001E_)?\d{2}_(.+)$", stem)
        if m:
            out[m.group(1)] = video

        out[stem] = video

    return out


def resolve_path(run_dir: Path, value: str) -> Path | None:
    value = str(value or "").strip()
    if not value:
        return None

    p = Path(value)
    if p.is_absolute():
        return p if p.exists() else None

    p1 = run_dir / p
    if p1.exists():
        return p1

    p2 = Path.cwd() / p
    if p2.exists():
        return p2

    return None


def load_points(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.exists():
        return []

    rows = read_csv(path)
    out = []

    for row in rows:
        frame = row.get("frame") or row.get("frame_idx") or row.get("f")
        x = row.get("x") or row.get("cx") or row.get("ball_x")
        y = row.get("y") or row.get("cy") or row.get("ball_y")

        if frame is None or x is None or y is None:
            continue

        out.append({
            "frame": fnum(frame),
            "x": fnum(x),
            "y": fnum(y),
        })

    return sorted(out, key=lambda r: r["frame"])


def pick_points(points: list[dict[str, float]], max_points: int) -> list[dict[str, float]]:
    if len(points) <= max_points:
        return points

    idxs = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    return [points[int(i)] for i in idxs]


def read_crop(cap: cv2.VideoCapture, frame_idx: int, x: float, y: float, size: int) -> np.ndarray:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total > 0:
        frame_idx = max(0, min(frame_idx, total - 1))
    else:
        frame_idx = max(0, frame_idx)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()

    if not ok or frame is None:
        return np.zeros((size, size, 3), dtype=np.uint8)

    h, w = frame.shape[:2]
    half = size // 2

    cx = int(round(x))
    cy = int(round(y))

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    crop = frame[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def mark_crop(raw: np.ndarray) -> np.ndarray:
    img = raw.copy()
    h, w = img.shape[:2]
    cx = w // 2
    cy = h // 2

    cv2.drawMarker(
        img,
        (cx, cy),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=1,
        line_type=cv2.LINE_AA,
    )
    cv2.circle(img, (cx, cy), 11, (0, 255, 255), 1, cv2.LINE_AA)

    return img


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    h, w = img.shape[:2]
    pad = 22
    out = np.zeros((h + pad, w, 3), dtype=np.uint8)
    out[:h, :, :] = img

    cv2.putText(
        out,
        text,
        (4, h + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    return out


def make_sheet(
    run_dir: Path,
    row: dict[str, str],
    source_video: Path,
    out_dir: Path,
    crop_size: int,
    max_points: int,
) -> Path | None:
    csv_path = resolve_path(run_dir, row.get("csv", ""))
    points = pick_points(load_points(csv_path), max_points=max_points)

    if not points:
        return None

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        return None

    rows_img = []

    header = []
    for off in OFFSETS:
        tile = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        header.append(add_label(tile, f"frame {off:+d}"))
    rows_img.append(np.hstack(header))

    for p in points:
        base_frame = int(round(p["frame"]))

        raw_tiles = []
        marked_tiles = []

        for off in OFFSETS:
            frame_idx = base_frame + off
            raw = read_crop(cap, frame_idx, p["x"], p["y"], crop_size)
            marked = mark_crop(raw)

            raw_tiles.append(add_label(raw, f"raw f{frame_idx}"))
            marked_tiles.append(add_label(marked, f"mark f{frame_idx}"))

        rows_img.append(np.hstack(raw_tiles))
        rows_img.append(np.hstack(marked_tiles))

    cap.release()

    sheet = np.vstack(rows_img)

    review_id = row.get("review_id", "RXXXX")
    segment_name = row.get("segment_name", "segment")
    safe_name = f"{review_id}_{segment_name}_lag_probe.jpg"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / safe_name

    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return out_path


def relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def write_html(out_html: Path, rows: list[dict[str, str]], image_map: dict[str, Path]) -> None:
    cards = []

    for row in rows:
        rid = row.get("review_id", "")
        img = image_map.get(rid)

        if img is not None:
            img_html = f'<img src="{html.escape(relpath(img, out_html.parent))}" loading="lazy">'
        else:
            img_html = '<p class="warn">Image non générée.</p>'

        cards.append(f"""
<section>
<h2>{html.escape(rid)} · {html.escape(row.get("segment_name", ""))}</h2>
<p class="muted">
clip : <code>{html.escape(row.get("clip_id", ""))}</code><br>
reason : <code>{html.escape(row.get("decision_reason_001U", ""))}</code><br>
risk={html.escape(row.get("risk_score_001G", ""))} ·
center_false={html.escape(row.get("center_blob_false_score_001N", ""))} ·
micro_keep={html.escape(row.get("micro_keep_score_001O", ""))} ·
micro_false={html.escape(row.get("micro_false_score_001O", ""))}
</p>
{img_html}
<p class="hint">
Ligne RAW = image propre. Ligne MARK = même image avec croix. Juge surtout sur RAW, puis vérifie MARK.
</p>
</section>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux raw lag probe 001W5</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
img{{max-width:100%;border-radius:12px;border:1px solid #2b303b;background:#000}}
code{{color:#dce4ff}}
.muted{{color:#9ea7b8}}
.warn{{color:#ffbc7a}}
.hint{{color:#d6c27a}}
</style>
</head>
<body>
<h1>TTFlux raw lag probe 001W5</h1>
<p class="muted">
Planche temporelle depuis les vidéos sources, sans trajectoire jaune.
Segments : {len(rows)}.
</p>
{''.join(cards)}
</body>
</html>
"""

    out_html.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--config-dir", default="configs/batch_001E")
    parser.add_argument("--review-csv", default="runs/batch_001E/arbiter_001V/arbiter_001U_review.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/lag_probe_001W5_raw")
    parser.add_argument("--out-html", default="runs/batch_001E/lag_probe_001W5_raw.html")
    parser.add_argument("--max-points", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=140)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_dir = Path(args.config_dir).resolve()
    review_csv = Path(args.review_csv)
    out_dir = Path(args.out_dir)
    out_html = Path(args.out_html)

    rows = read_csv(review_csv)
    video_map = build_clip_video_map(config_dir)

    image_map: dict[str, Path] = {}

    print(f"[001W5] review rows : {len(rows)}")
    print(f"[001W5] config videos: {len(video_map)}")

    for idx, row in enumerate(rows, start=1):
        rid = row.get("review_id", f"R{idx:04d}")
        clip_id = row.get("clip_id", "")
        source_video = video_map.get(clip_id)

        print(f"[001W5] {idx:02d}/{len(rows)} {rid} clip={clip_id} source={'ok' if source_video else 'missing'}")

        if source_video is None:
            continue

        img = make_sheet(
            run_dir=run_dir,
            row=row,
            source_video=source_video,
            out_dir=out_dir,
            crop_size=args.crop_size,
            max_points=args.max_points,
        )

        if img is not None:
            image_map[rid] = img

    write_html(out_html, rows, image_map)

    print(f"[001W5] sheets : {len(image_map)}/{len(rows)}")
    print(f"[001W5] html   : {out_html}")


if __name__ == "__main__":
    main()
