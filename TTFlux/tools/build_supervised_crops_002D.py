from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def fnum(value: Any, default: float | None = None) -> float | None:
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = []
    seen = set()

    preferred = [
        "sample_id",
        "label",
        "split_hint",
        "review_id",
        "clip_id",
        "segment_name",
        "frame",
        "x",
        "y",
        "crop_path",
        "source_video",
        "dist_offset0",
        "dist_best",
        "best_offset_frames",
        "human_x",
        "human_y",
        "auto_x_offset0",
        "auto_y_offset0",
        "human_decision_002A",
        "classification_001Z",
    ]

    for c in preferred:
        cols.append(c)
        seen.add(c)

    for row in rows:
        for k in row:
            if k not in seen:
                cols.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def crop_frame(video_path: Path, frame: int, x: float, y: float, size: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total > 0:
        frame = max(0, min(frame, total - 1))
    else:
        frame = max(0, frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()

    if not ok or img is None:
        return None

    h, w = img.shape[:2]
    half = size // 2

    cx = int(round(x))
    cy = int(round(y))

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    crop = img[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return None

    if crop.shape[0] != size or crop.shape[1] != size:
        out = np.zeros((size, size, 3), dtype=np.uint8)

        ox = (size - crop.shape[1]) // 2
        oy = (size - crop.shape[0]) // 2

        out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        crop = out

    return crop


def make_marked_tile(crop_path: Path, label: str, text: str, size: int) -> np.ndarray:
    img = cv2.imread(str(crop_path))

    if img is None:
        img = np.zeros((size, size, 3), dtype=np.uint8)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    if label == "human_visible_positive":
        color = (80, 255, 140)
    elif label == "auto_far_negative":
        color = (80, 80, 255)
    else:
        color = (80, 200, 255)

    cv2.drawMarker(
        img,
        (size // 2, size // 2),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=1,
        line_type=cv2.LINE_AA,
    )

    pad = 28
    out = np.zeros((size + pad, size, 3), dtype=np.uint8)
    out[:size] = img

    cv2.putText(
        out,
        text[:22],
        (4, size + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    return out


def write_contact_sheet(
    out_path: Path,
    manifest_rows: list[dict[str, Any]],
    label: str,
    max_items: int,
    tile_size: int,
    cols: int = 8,
) -> None:
    rows = [r for r in manifest_rows if r.get("label") == label][:max_items]

    if not rows:
        return

    tiles = []

    for r in rows:
        text = f'{r.get("review_id","")} f{r.get("frame","")}'
        tiles.append(make_marked_tile(Path(str(r["crop_path"])), label, text, tile_size))

    tile_h, tile_w = tiles[0].shape[:2]
    n_cols = min(cols, len(tiles))
    n_rows = int(math.ceil(len(tiles) / n_cols))

    sheet = np.zeros((n_rows * tile_h, n_cols * tile_w, 3), dtype=np.uint8)

    for i, tile in enumerate(tiles):
        y = (i // n_cols) * tile_h
        x = (i % n_cols) * tile_w
        sheet[y:y + tile_h, x:x + tile_w] = tile

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


def build_video_map(merged_rows: list[dict[str, str]]) -> dict[tuple[str, int], Path]:
    out = {}

    for row in merged_rows:
        rid = row.get("review_id", "")
        frame = fnum(row.get("frame"), None)
        video_path = row.get("video_path", "")

        if not rid or frame is None or not video_path:
            continue

        p = Path(video_path)

        if p.exists():
            out[(rid, int(round(frame)))] = p

    return out


def build_human_decision_map(human_decisions: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        r.get("review_id", ""): r
        for r in human_decisions
        if r.get("review_id", "")
    }


def add_sample(
    out: list[dict[str, Any]],
    label: str,
    review_id: str,
    clip_id: str,
    segment_name: str,
    frame: int,
    x: float,
    y: float,
    video_path: Path,
    crop_dir: Path,
    crop_size: int,
    extra: dict[str, Any],
) -> None:
    sample_id = f"{label}_{review_id}_f{frame}_{len(out):06d}"
    filename = safe_name(sample_id) + ".jpg"
    label_dir = crop_dir / label
    crop_path = label_dir / filename

    crop = crop_frame(video_path, frame, x, y, crop_size)

    if crop is None:
        return

    label_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    item = {
        "sample_id": sample_id,
        "label": label,
        "split_hint": "train_candidate",
        "review_id": review_id,
        "clip_id": clip_id,
        "segment_name": segment_name,
        "frame": frame,
        "x": f"{x:.2f}",
        "y": f"{y:.2f}",
        "crop_path": str(crop_path),
        "source_video": str(video_path),
    }

    item.update(extra)
    out.append(item)


def build_dataset(
    merged_csv: Path,
    frame_errors_csv: Path,
    human_decisions_csv: Path,
    out_dir: Path,
    crop_size: int,
    negative_dist: float,
    near_dist: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged_rows = read_csv(merged_csv)
    frame_rows = read_csv(frame_errors_csv)
    human_decisions = read_csv(human_decisions_csv)

    video_map = build_video_map(merged_rows)
    human_map = build_human_decision_map(human_decisions)

    crop_dir = out_dir / "crops"
    samples: list[dict[str, Any]] = []

    for row in frame_rows:
        rid = row.get("review_id", "")
        frame_f = fnum(row.get("frame"), None)
        hx = fnum(row.get("human_x"), None)
        hy = fnum(row.get("human_y"), None)

        if not rid or frame_f is None or hx is None or hy is None:
            continue

        frame = int(round(frame_f))
        video_path = video_map.get((rid, frame))

        if video_path is None:
            continue

        human_dec = human_map.get(rid, {})

        extra_base = {
            "dist_offset0": row.get("dist_offset0", ""),
            "dist_best": row.get("dist_best", ""),
            "best_offset_frames": row.get("best_offset_frames", ""),
            "human_x": row.get("human_x", ""),
            "human_y": row.get("human_y", ""),
            "auto_x_offset0": row.get("auto_x_offset0", ""),
            "auto_y_offset0": row.get("auto_y_offset0", ""),
            "human_decision_002A": human_dec.get("human_decision_002A", ""),
            "classification_001Z": human_dec.get("classification_001Z", ""),
        }

        add_sample(
            out=samples,
            label="human_visible_positive",
            review_id=rid,
            clip_id=row.get("clip_id", ""),
            segment_name=row.get("segment_name", ""),
            frame=frame,
            x=hx,
            y=hy,
            video_path=video_path,
            crop_dir=crop_dir,
            crop_size=crop_size,
            extra=extra_base,
        )

        ax = fnum(row.get("auto_x_offset0"), None)
        ay = fnum(row.get("auto_y_offset0"), None)
        dist0 = fnum(row.get("dist_offset0"), None)

        if ax is None or ay is None or dist0 is None:
            continue

        if dist0 >= negative_dist:
            label = "auto_far_negative"
        elif dist0 >= near_dist:
            label = "auto_near_partial"
        else:
            continue

        add_sample(
            out=samples,
            label=label,
            review_id=rid,
            clip_id=row.get("clip_id", ""),
            segment_name=row.get("segment_name", ""),
            frame=frame,
            x=ax,
            y=ay,
            video_path=video_path,
            crop_dir=crop_dir,
            crop_size=crop_size,
            extra=extra_base,
        )

    by_label: dict[str, int] = {}
    by_review: dict[str, dict[str, int]] = {}

    for s in samples:
        label = str(s.get("label", ""))
        rid = str(s.get("review_id", ""))

        by_label[label] = by_label.get(label, 0) + 1

        if rid not in by_review:
            by_review[rid] = {}

        by_review[rid][label] = by_review[rid].get(label, 0) + 1

    summary = {
        "samples": len(samples),
        "crop_size": crop_size,
        "negative_dist_threshold": negative_dist,
        "near_dist_threshold": near_dist,
        "by_label": dict(sorted(by_label.items())),
        "by_review_id": by_review,
        "inputs": {
            "merged_csv": str(merged_csv),
            "frame_errors_csv": str(frame_errors_csv),
            "human_decisions_csv": str(human_decisions_csv),
        },
    }

    return samples, summary


def write_html(out_path: Path, summary: dict[str, Any], contact_sheets: dict[str, Path]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["by_label"].items()
    )

    review_rows = []

    for rid, counts in summary["by_review_id"].items():
        review_rows.append(f"""
<tr>
<td>{html.escape(rid)}</td>
<td>{html.escape(json.dumps(counts, ensure_ascii=False))}</td>
</tr>
""")

    sheet_sections = []

    for label, path in contact_sheets.items():
        rel = path.name
        sheet_sections.append(f"""
<section>
<h2>{html.escape(label)}</h2>
<img src="{html.escape(rel)}" loading="lazy">
</section>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux supervised crops 002D</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
th,td{{border-bottom:1px solid #2b303b;padding:7px 8px;vertical-align:top;font-size:13px}}
th{{background:#20242e;text-align:left}}
img{{max-width:100%;border-radius:12px;border:1px solid #2b303b;background:#000}}
code{{color:#dce4ff}}
.muted{{color:#9ea7b8}}
</style>
</head>
<body>
<h1>TTFlux supervised crops 002D</h1>

<section>
<h2>Résumé</h2>
<p>Samples : <b>{summary["samples"]}</b></p>
<p>Crop size : <b>{summary["crop_size"]}</b></p>
<p class="muted">
Positifs = crops centrés sur tes points humains visibles.
Négatifs = crops centrés sur le tracker auto quand il est loin de ton tracé humain.
Near/partial = crops auto moyennement éloignés, utiles pour diagnostic mais pas comme négatifs purs.
</p>
<table>
<thead><tr><th>label</th><th>count</th></tr></thead>
<tbody>{label_rows}</tbody>
</table>
</section>

<section>
<h2>Répartition par segment</h2>
<table>
<thead><tr><th>review_id</th><th>counts</th></tr></thead>
<tbody>{''.join(review_rows)}</tbody>
</table>
</section>

{''.join(sheet_sections)}
</body>
</html>
"""

    out_path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="runs/batch_001E/human_analysis_001Z")
    parser.add_argument("--human-decisions", default="runs/batch_001E/human_decisions_002A/human_segment_decisions_002A.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/supervised_crops_002D")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--negative-dist", type=float, default=90.0)
    parser.add_argument("--near-dist", type=float, default=35.0)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_csv = analysis_dir / "human_annotations_001Z_merged.csv"
    frame_errors_csv = analysis_dir / "human_tracker_frame_errors_001Z.csv"
    human_decisions_csv = Path(args.human_decisions)

    samples, summary = build_dataset(
        merged_csv=merged_csv,
        frame_errors_csv=frame_errors_csv,
        human_decisions_csv=human_decisions_csv,
        out_dir=out_dir,
        crop_size=args.crop_size,
        negative_dist=args.negative_dist,
        near_dist=args.near_dist,
    )

    manifest_csv = out_dir / "supervised_crops_manifest_002D.csv"
    write_csv(manifest_csv, samples)

    (out_dir / "supervised_crops_summary_002D.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    contact_sheets = {}

    for label in ["human_visible_positive", "auto_far_negative", "auto_near_partial"]:
        sheet_path = out_dir / f"contact_sheet_{label}_002D.jpg"
        write_contact_sheet(
            out_path=sheet_path,
            manifest_rows=samples,
            label=label,
            max_items=80,
            tile_size=args.crop_size,
            cols=8,
        )
        if sheet_path.exists():
            contact_sheets[label] = sheet_path

    write_html(out_dir / "supervised_crops_002D.html", summary, contact_sheets)

    print(f"[002D] samples  : {summary['samples']}")
    print(f"[002D] by_label : {summary['by_label']}")
    print(f"[002D] out dir  : {out_dir}")
    print(f"[002D] html     : {out_dir / 'supervised_crops_002D.html'}")


if __name__ == "__main__":
    main()
