from __future__ import annotations

import argparse
import csv
import html
import math
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


def pick_points(points: list[dict[str, float]], max_points: int = 8) -> list[dict[str, float]]:
    if len(points) <= max_points:
        return points

    idxs = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    return [points[int(i)] for i in idxs]


def crop_frame(cap: cv2.VideoCapture, frame_idx: int, x: float, y: float, size: int = 80) -> np.ndarray:
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
        crop = np.zeros((size, size, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    # centre du point tracké
    cv2.drawMarker(
        crop,
        (size // 2, size // 2),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=1,
        line_type=cv2.LINE_AA,
    )

    return crop


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    h, w = img.shape[:2]
    pad = 20
    out = np.zeros((h + pad, w, 3), dtype=np.uint8)
    out[:h, :, :] = img

    cv2.putText(
        out,
        text,
        (4, h + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    return out


def make_sheet(
    run_dir: Path,
    row: dict[str, str],
    out_dir: Path,
    crop_size: int = 80,
    max_points: int = 8,
) -> Path | None:
    csv_path = resolve_path(run_dir, row.get("csv", ""))
    mp4_path = resolve_path(run_dir, row.get("mp4", ""))

    points = pick_points(load_points(csv_path), max_points=max_points)

    if mp4_path is None or not mp4_path.exists():
        return None
    if not points:
        return None

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None

    rows_img = []

    header_cells = []
    for off in OFFSETS:
        label = f"frame {off:+d}"
        tile = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        header_cells.append(add_label(tile, label))
    rows_img.append(np.hstack(header_cells))

    for p in points:
        base_frame = int(round(p["frame"]))
        tiles = []

        for off in OFFSETS:
            img = crop_frame(
                cap=cap,
                frame_idx=base_frame + off,
                x=p["x"],
                y=p["y"],
                size=crop_size,
            )
            tiles.append(add_label(img, f"f{base_frame + off}"))

        rows_img.append(np.hstack(tiles))

    cap.release()

    sheet = np.vstack(rows_img)

    review_id = row.get("review_id", "RXXXX")
    segment_name = row.get("segment_name", "segment")
    safe_name = f"{review_id}_{segment_name}_lag_probe.jpg"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)

    out_path = out_dir / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    return out_path


def relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def write_html(out_html: Path, rows: list[dict[str, str]], run_dir: Path, image_map: dict[str, Path]) -> None:
    cards = []

    for row in rows:
        rid = row.get("review_id", "")
        img = image_map.get(rid)

        img_html = ""
        if img is not None:
            img_html = f'<img src="{html.escape(relpath(img, out_html.parent))}" loading="lazy">'

        mp4 = row.get("mp4", "")
        csv_path = row.get("csv", "")

        cards.append(f"""
<section>
<h2>{html.escape(rid)} · {html.escape(row.get("segment_name", ""))}</h2>
<p class="muted">
clip : <code>{html.escape(row.get("clip_id", ""))}</code><br>
decision actuelle : <b>{html.escape(row.get("decision_001U", ""))}</b> ·
raison : <code>{html.escape(row.get("decision_reason_001U", ""))}</code><br>
risk={html.escape(row.get("risk_score_001G", ""))} ·
center_false={html.escape(row.get("center_blob_false_score_001N", ""))} ·
micro_keep={html.escape(row.get("micro_keep_score_001O", ""))} ·
micro_false={html.escape(row.get("micro_false_score_001O", ""))} ·
micro_dist={html.escape(row.get("micro_distance_med_001O", ""))}
</p>
<p>
<a href="{html.escape(mp4)}">mp4</a> ·
<a href="{html.escape(csv_path)}">csv</a>
</p>
{img_html}
<p class="hint">
Lecture : si la balle est mieux centrée en frame -3 ou -6, le tracking est probablement retardé.
Si elle est mieux centrée en +3 ou +6, le tracking est probablement en avance.
Si aucune colonne ne montre la balle au centre, c’est plutôt une fausse piste ou un cas trop compressé.
</p>
</section>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux lag probe 001W2</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
img{{max-width:100%;border-radius:12px;border:1px solid #2b303b;background:#000}}
code{{color:#dce4ff}}
.muted{{color:#9ea7b8}}
.hint{{color:#d6c27a}}
a{{color:#9bc2ff}}
</style>
</head>
<body>
<h1>TTFlux lag probe 001W2</h1>
<p class="muted">
Planche temporelle pour vérifier si le tracking est en retard ou en avance.
Chaque crop garde les mêmes coordonnées trackées, mais lit des frames voisines.
Segments analysés : {len(rows)}.
</p>
{''.join(cards)}
</body>
</html>
"""

    out_html.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--review-csv", default="runs/batch_001E/arbiter_001V/arbiter_001U_review.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/lag_probe_001W2")
    parser.add_argument("--out-html", default="runs/batch_001E/lag_probe_001W2.html")
    parser.add_argument("--max-points", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=80)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    review_csv = Path(args.review_csv)
    out_dir = Path(args.out_dir)
    out_html = Path(args.out_html)

    rows = read_csv(review_csv)
    image_map: dict[str, Path] = {}

    print(f"[001W2] review rows : {len(rows)}")

    for idx, row in enumerate(rows, start=1):
        rid = row.get("review_id", f"R{idx:04d}")
        print(f"[001W2] {idx:02d}/{len(rows)} {rid} {row.get('segment_name', '')}")

        img = make_sheet(
            run_dir=run_dir,
            row=row,
            out_dir=out_dir,
            crop_size=args.crop_size,
            max_points=args.max_points,
        )

        if img is not None:
            image_map[rid] = img

    write_html(out_html, rows, run_dir, image_map)

    print(f"[001W2] sheets : {len(image_map)}/{len(rows)}")
    print(f"[001W2] html   : {out_html}")


if __name__ == "__main__":
    main()
