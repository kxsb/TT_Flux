# PATCH 001M - TTFlux patch/context contact sheets
#
# Reads:
# - goldset_001K.csv
# - segment CSVs
# - source videos from configs/batch_001E
#
# Produces:
# - one PNG contact sheet per segment
# - patch_review_001M.html
#
# Goal:
# Human-readable audit of what the tracker is visually following.
# 001L showed local appearance features are too crude:
# this patch displays tight patches + wider context crops.

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FRAME_COLS = ["frame", "frame_idx", "frame_index", "f"]
X_COLS = ["x", "ball_x", "cx", "center_x"]
Y_COLS = ["y", "ball_y", "cy", "center_y"]

VIDEO_KEYS = [
    "video",
    "video_path",
    "input_video",
    "source_video",
    "source",
    "input",
    "clip",
    "clip_path",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return default
        out = float(text)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, default)))


def norm_col(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def first_existing_col(fieldnames: list[str], candidates: list[str]) -> str | None:
    normalized = {norm_col(c): c for c in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def clean_first_path(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.split("|")[0].strip()


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            return [], []

        rows = []
        for row in reader:
            rows.append({str(k): "" if v is None else str(v) for k, v in row.items()})

        return list(reader.fieldnames), rows


def resolve_path(run_dir: Path, value: Any) -> Path | None:
    text = clean_first_path(value)
    if not text:
        return None

    text = text.replace("\\", "/")
    path = Path(text)

    if path.is_absolute():
        return path

    return run_dir / path


def find_video_value(obj: Any) -> str:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in VIDEO_KEYS and isinstance(value, str):
                return value

        for value in obj.values():
            found = find_video_value(value)
            if found:
                return found

    if isinstance(obj, list):
        for item in obj:
            found = find_video_value(item)
            if found:
                return found

    return ""


def build_config_video_map(config_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}

    if not config_dir.exists():
        return out

    for path in config_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        video_text = find_video_value(data)
        if not video_text:
            continue

        video_path = Path(video_text)
        if not video_path.is_absolute():
            video_path = (path.parent / video_path).resolve()

        out[path.stem] = video_path

    return out


def load_points(segment_csv: Path | None) -> list[tuple[int, float, float]]:
    if segment_csv is None or not segment_csv.exists():
        return []

    try:
        fieldnames, rows = read_csv_rows(segment_csv)
    except Exception:
        return []

    frame_col = first_existing_col(fieldnames, FRAME_COLS)
    x_col = first_existing_col(fieldnames, X_COLS)
    y_col = first_existing_col(fieldnames, Y_COLS)

    if not frame_col or not x_col or not y_col:
        return []

    buckets: dict[int, list[tuple[float, float]]] = {}

    for row in rows:
        frame = safe_int(row.get(frame_col), -1)
        x = safe_float(row.get(x_col), float("nan"))
        y = safe_float(row.get(y_col), float("nan"))

        if frame < 0 or not math.isfinite(x) or not math.isfinite(y):
            continue

        buckets.setdefault(frame, []).append((x, y))

    out = []
    for frame in sorted(buckets):
        pts = buckets[frame]
        x = sum(p[0] for p in pts) / len(pts)
        y = sum(p[1] for p in pts) / len(pts)
        out.append((frame, x, y))

    return out


def sample_points(points: list[tuple[int, float, float]], n: int) -> list[tuple[int, float, float]]:
    if len(points) <= n:
        return points

    idxs = np.linspace(0, len(points) - 1, n).round().astype(int).tolist()
    return [points[i] for i in idxs]


def get_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()

    if not ok or frame is None:
        return None

    return frame


def crop_square(frame: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    r = size // 2

    xi = int(round(x))
    yi = int(round(y))

    x0 = xi - r
    y0 = yi - r
    x1 = xi + r
    y1 = yi + r

    out = np.zeros((size, size, 3), dtype=np.uint8)

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0

    if src_x1 > src_x0 and src_y1 > src_y0:
        out[
            dst_y0 : dst_y0 + (src_y1 - src_y0),
            dst_x0 : dst_x0 + (src_x1 - src_x0),
        ] = frame[src_y0:src_y1, src_x0:src_x1]

    return out


def draw_crosshair(img: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    cx = w // 2
    cy = h // 2

    cv2.line(out, (cx - 12, cy), (cx + 12, cy), color, thickness, cv2.LINE_AA)
    cv2.line(out, (cx, cy - 12), (cx, cy + 12), color, thickness, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), 8, color, thickness, cv2.LINE_AA)

    return out


def put_label(img: np.ndarray, text: str, y: int = 18) -> None:
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def make_sheet(
    row: dict[str, str],
    run_dir: Path,
    config_video_map: dict[str, Path],
    out_png: Path,
    samples: int,
    tight_size: int,
    context_size: int,
) -> tuple[bool, str]:
    clip_id = row.get("clip_id", "")

    segment_csv = resolve_path(run_dir, row.get("csv", ""))
    points = load_points(segment_csv)

    if not points:
        return False, "no_points"

    chosen = sample_points(points, samples)
    first_frame = min(p[0] for p in points)

    source_video = config_video_map.get(clip_id)
    source_kind = "source_video"

    if source_video is None or not source_video.exists():
        source_video = resolve_path(run_dir, row.get("mp4", ""))
        source_kind = "segment_mp4_fallback"

    if source_video is None or not source_video.exists():
        return False, "missing_video"

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        return False, "video_open_failed"

    tight_tiles = []
    context_tiles = []

    try:
        xs_all = [p[1] for p in points]
        ys_all = [p[2] for p in points]

        for original_frame, x, y in chosen:
            if source_kind == "segment_mp4_fallback":
                frame_idx = original_frame - first_frame
            else:
                frame_idx = original_frame

            frame = get_frame(cap, frame_idx)
            if frame is None:
                continue

            h, w = frame.shape[:2]

            max_x = max(xs_all) if xs_all else w
            max_y = max(ys_all) if ys_all else h

            sx = 1.0
            sy = 1.0

            if max_x > w * 1.05:
                sx = w / max(1.0, max_x + 12.0)

            if max_y > h * 1.05:
                sy = h / max(1.0, max_y + 12.0)

            xx = x * sx
            yy = y * sy

            tight = crop_square(frame, xx, yy, tight_size)
            context = crop_square(frame, xx, yy, context_size)

            tight = cv2.resize(tight, (96, 96), interpolation=cv2.INTER_NEAREST)
            context = cv2.resize(context, (160, 160), interpolation=cv2.INTER_AREA)

            tight = draw_crosshair(tight, (0, 255, 255), 1)
            context = draw_crosshair(context, (0, 255, 255), 1)

            put_label(tight, f"f{original_frame}", 16)
            put_label(context, f"f{original_frame}", 18)

            tight_tiles.append(tight)
            context_tiles.append(context)

    finally:
        cap.release()

    if not tight_tiles or not context_tiles:
        return False, "no_frames"

    tile_count = len(tight_tiles)
    margin = 12
    gap = 8

    header_h = 82
    tight_h = 96
    context_h = 160

    width = margin * 2 + tile_count * 160 + (tile_count - 1) * gap
    height = header_h + tight_h + gap + context_h + margin

    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    sheet[:, :] = (18, 21, 29)

    title = f"{row.get('review_id')} | {row.get('human_label_fr')} | {row.get('clip_id')}"
    subtitle = (
        f"{row.get('segment_name')} | target={row.get('target_class')} "
        f"| falseJ={row.get('false_track_score_001J')} keepJ={row.get('keep_score_001J')} "
        f"| appFalse={row.get('appearance_false_score_001L', '')} appKeep={row.get('appearance_keep_score_001L', '')}"
    )

    cv2.putText(sheet, title[:120], (margin, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 245, 255), 1, cv2.LINE_AA)
    cv2.putText(sheet, subtitle[:150], (margin, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 180, 200), 1, cv2.LINE_AA)

    y_tight = header_h
    y_context = header_h + tight_h + gap

    for i, tile in enumerate(tight_tiles):
        x0 = margin + i * (160 + gap)
        sheet[y_tight : y_tight + 96, x0 : x0 + 96] = tile

    for i, tile in enumerate(context_tiles):
        x0 = margin + i * (160 + gap)
        sheet[y_context : y_context + 160, x0 : x0 + 160] = tile

    out_png.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_png), sheet)

    if not ok:
        return False, "write_failed"

    return True, source_kind


def rel_from_html(out_html: Path, target: Path) -> str:
    try:
        rel = target.resolve().relative_to(out_html.parent.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(target.resolve()).replace("\\", "/")


def write_html(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cards = []

    def sort_key(row: dict[str, str]) -> tuple[str, float, str]:
        return (
            row.get("target_class", ""),
            -safe_float(row.get("false_track_score_001J")),
            row.get("review_id", ""),
        )

    for row in sorted(rows, key=sort_key):
        img = row.get("contact_sheet_rel", "")
        ok = row.get("contact_sheet_ok", "0") == "1"

        if ok:
            img_html = f'<img src="{html.escape(img)}" alt="{html.escape(row.get("review_id", ""))}">'
        else:
            img_html = f'<div class="missing">Contact sheet manquante : {html.escape(row.get("contact_sheet_msg", ""))}</div>'

        cls = html.escape(row.get("target_class", ""))

        cards.append(
            f"""
            <article class="card {cls}">
              <header>
                <div>
                  <h2>{html.escape(row.get("review_id", ""))} · {html.escape(row.get("human_label_fr", ""))}</h2>
                  <p>{html.escape(row.get("clip_id", ""))} · {html.escape(row.get("segment_name", ""))}</p>
                </div>
                <div class="scores">
                  <span>target <b>{html.escape(row.get("target_class", ""))}</b></span>
                  <span>falseJ <b>{html.escape(row.get("false_track_score_001J", ""))}</b></span>
                  <span>keepJ <b>{html.escape(row.get("keep_score_001J", ""))}</b></span>
                  <span>app <b>{html.escape(row.get("appearance_guess_001L", ""))}</b></span>
                </div>
              </header>
              {img_html}
            </article>
            """
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux patch review 001M</title>
<style>
  :root {{
    --bg: #101218;
    --panel: #181b22;
    --line: #2b303b;
    --text: #edf0f7;
    --muted: #9ea7b8;
  }}
  body {{
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  h1 {{
    margin-top: 0;
  }}
  .meta {{
    color: var(--muted);
    margin-bottom: 16px;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 18px;
  }}
  .card.negative {{
    border-color: rgba(255, 80, 80, 0.55);
  }}
  .card.partial {{
    border-color: rgba(255, 200, 80, 0.45);
  }}
  .card.positive {{
    border-color: rgba(100, 255, 150, 0.35);
  }}
  .card.ignore {{
    border-color: rgba(150, 170, 255, 0.35);
  }}
  header {{
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-start;
    margin-bottom: 10px;
  }}
  h2 {{
    margin: 0 0 4px;
    font-size: 18px;
  }}
  p {{
    margin: 0;
    color: var(--muted);
  }}
  .scores {{
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
  }}
  .scores span {{
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 4px 8px;
    color: var(--muted);
    font-size: 12px;
  }}
  .scores b {{
    color: var(--text);
  }}
  img {{
    max-width: 100%;
    height: auto;
    border: 1px solid var(--line);
    border-radius: 12px;
    display: block;
  }}
  .missing {{
    padding: 16px;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 12px;
  }}
</style>
</head>
<body>
  <h1>TTFlux patch review 001M</h1>
  <div class="meta">
    Rangée du haut : crop serré autour du point tracké. Rangée du bas : contexte large avec croix centrale.
    Si une fausse piste reste “ball-like”, le contexte doit révéler si c’est bord de table, raquette, reflet ou autre objet.
  </div>
  {''.join(cards)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--config-dir", default="configs/batch_001E")
    parser.add_argument("--features", default="runs/batch_001E/appearance_features_001L.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/contact_sheets_001M")
    parser.add_argument("--out-html", default="runs/batch_001E/patch_review_001M.html")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--tight-size", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=160)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_dir = Path(args.config_dir).resolve()
    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    out_html = Path(args.out_html)

    _, rows = read_csv_rows(features_path)
    video_map = build_config_video_map(config_dir)

    enriched = []

    print(f"[001M] rows          : {len(rows)}")
    print(f"[001M] config videos : {len(video_map)}")

    ok_count = 0

    for i, row in enumerate(rows, start=1):
        review_id = row.get("review_id", "")
        name = slugify(f"{review_id}_{row.get('clip_id', '')}_{row.get('segment_name', '')}")
        out_png = out_dir / f"{name}.png"

        ok, msg = make_sheet(
            row=row,
            run_dir=run_dir,
            config_video_map=video_map,
            out_png=out_png,
            samples=args.samples,
            tight_size=args.tight_size,
            context_size=args.context_size,
        )

        row = dict(row)
        row["contact_sheet_ok"] = "1" if ok else "0"
        row["contact_sheet_msg"] = msg
        row["contact_sheet_path"] = str(out_png)
        row["contact_sheet_rel"] = rel_from_html(out_html, out_png) if ok else ""

        enriched.append(row)

        if ok:
            ok_count += 1

        print(f"[001M] {i:02d}/{len(rows)} {review_id} ok={ok} msg={msg}")

    write_html(out_html, enriched)

    print(f"[001M] sheets ok  : {ok_count}/{len(rows)}")
    print(f"[001M] out_dir    : {out_dir}")
    print(f"[001M] wrote HTML : {out_html}")


if __name__ == "__main__":
    main()