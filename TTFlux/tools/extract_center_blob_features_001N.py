# PATCH 001N - Centered blob / isolation features
#
# Motivation:
# 001L local appearance was too naive. A player shirt, arm, table edge or shoe can look
# "ball-like" in a small patch.
#
# 001N focuses on the component centered under the tracked point:
# - is there a bright/orange blob near the center?
# - is it small enough?
# - is it compact?
# - does it touch patch borders?
# - is it isolated from a larger surrounding object?
#
# Inputs:
# - goldset_001K.csv
# - source videos from configs/batch_001E
# - segment CSVs
#
# Outputs:
# - center_blob_features_001N.csv
# - center_blob_features_001N.html
# - center_blob_summary_001N.json

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
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


def fnum(value: Any, default: float = 0.0) -> float:
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


def inum(value: Any, default: int = 0) -> int:
    return int(round(fnum(value, default)))


def fmt(value: float, digits: int = 5) -> str:
    if not math.isfinite(value):
        return ""
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


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


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "review_id",
        "human_label_fr",
        "target_class",
        "expected_action",
        "center_blob_points",
        "center_blob_ok_rate",
        "center_blob_area_med",
        "center_blob_area_ratio_med",
        "center_blob_roundness_med",
        "center_blob_fill_med",
        "center_blob_touch_border_rate",
        "center_blob_offset_med",
        "center_blob_isolation_med",
        "center_blob_large_object_rate",
        "center_blob_score_001N",
        "center_blob_false_score_001N",
        "center_blob_guess_001N",
        "center_blob_notes_001N",
        "risk_score_001G",
        "fragmentation_score_001J",
        "false_track_score_001J",
        "keep_score_001J",
        "appearance_false_score_001L",
        "appearance_keep_score_001L",
        "clip_id",
        "segment_name",
        "mp4",
        "csv",
    ]

    seen = set()
    columns = []

    for col in preferred:
        if col not in seen:
            columns.append(col)
            seen.add(col)

    for row in rows:
        for col in row:
            if col not in seen:
                columns.append(col)
                seen.add(col)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def resolve_path(run_dir: Path, value: Any) -> Path | None:
    text = clean_first_path(value)
    if not text:
        return None

    text = text.replace("\\", "/")
    p = Path(text)

    if p.is_absolute():
        return p

    return run_dir / p


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
        frame = inum(row.get(frame_col), -1)
        x = fnum(row.get(x_col), float("nan"))
        y = fnum(row.get(y_col), float("nan"))

        if frame < 0 or not math.isfinite(x) or not math.isfinite(y):
            continue

        buckets.setdefault(frame, []).append((x, y))

    out = []
    for frame in sorted(buckets):
        vals = buckets[frame]
        x = sum(v[0] for v in vals) / len(vals)
        y = sum(v[1] for v in vals) / len(vals)
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


def crop_square(frame: np.ndarray, x: float, y: float, size: int) -> np.ndarray | None:
    h, w = frame.shape[:2]
    r = size // 2
    xi = int(round(x))
    yi = int(round(y))

    x0 = xi - r
    y0 = yi - r
    x1 = xi + r + 1
    y1 = yi + r + 1

    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None

    out = np.zeros((size, size, 3), dtype=np.uint8)

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0

    out[
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = frame[src_y0:src_y1, src_x0:src_x1]

    return out


def make_ball_candidate_mask(patch_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # White/yellow/orange ping-ball candidates.
    white = (v >= 150) & (s <= 115)
    yellow_orange = (h >= 5) & (h <= 35) & (s >= 45) & (v >= 95)

    # Local high intensity can catch a white ball under changing light.
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    local_peak = (gray.astype(np.int16) - blur.astype(np.int16)) >= 18
    bright_peak = local_peak & (gray >= 95)

    mask = (white | yellow_orange | bright_peak).astype(np.uint8) * 255

    # Remove isolated one-pixel crumbs, but keep small balls.
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def centered_component_features(patch_bgr: np.ndarray) -> dict[str, float]:
    size = patch_bgr.shape[0]
    center = size // 2

    mask = make_ball_candidate_mask(patch_bgr)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    if n_labels <= 1:
        return {
            "ok": 0.0,
            "area": 0.0,
            "area_ratio": 0.0,
            "roundness": 0.0,
            "fill": 0.0,
            "touch_border": 0.0,
            "offset": 999.0,
            "isolation": 0.0,
            "large_object": 0.0,
        }

    # Pick component containing the center if possible, else nearest component centroid.
    center_label = int(labels[center, center])

    best_label = 0
    if center_label > 0:
        best_label = center_label
    else:
        best_dist = 999999.0
        for label in range(1, n_labels):
            cx, cy = centroids[label]
            dist = math.hypot(cx - center, cy - center)
            if dist < best_dist:
                best_dist = dist
                best_label = label

    if best_label <= 0:
        return {
            "ok": 0.0,
            "area": 0.0,
            "area_ratio": 0.0,
            "roundness": 0.0,
            "fill": 0.0,
            "touch_border": 0.0,
            "offset": 999.0,
            "isolation": 0.0,
            "large_object": 0.0,
        }

    area = float(stats[best_label, cv2.CC_STAT_AREA])
    x = float(stats[best_label, cv2.CC_STAT_LEFT])
    y = float(stats[best_label, cv2.CC_STAT_TOP])
    w = float(stats[best_label, cv2.CC_STAT_WIDTH])
    h = float(stats[best_label, cv2.CC_STAT_HEIGHT])
    cx, cy = centroids[best_label]

    area_ratio = area / float(size * size)
    fill = area / max(1.0, w * h)
    aspect = min(w, h) / max(1.0, max(w, h))
    roundness = aspect * fill

    touch_border = 1.0 if x <= 0 or y <= 0 or x + w >= size - 1 or y + h >= size - 1 else 0.0
    offset = math.hypot(float(cx) - center, float(cy) - center)

    total_area = float(np.sum(mask > 0))
    other_area = max(0.0, total_area - area)
    isolation = area / max(1.0, total_area)

    # Large centered component is probably player/table/line/hand, not a small ball.
    large_object = 1.0 if area_ratio >= 0.18 or touch_border > 0.5 else 0.0

    return {
        "ok": 1.0,
        "area": area,
        "area_ratio": area_ratio,
        "roundness": roundness,
        "fill": fill,
        "touch_border": touch_border,
        "offset": offset,
        "isolation": isolation,
        "large_object": large_object,
    }


def score_center_blob(row: dict[str, str]) -> tuple[str, str, str, str]:
    ok_rate = fnum(row.get("center_blob_ok_rate"))
    area = fnum(row.get("center_blob_area_med"))
    ratio = fnum(row.get("center_blob_area_ratio_med"))
    roundness = fnum(row.get("center_blob_roundness_med"))
    touch_rate = fnum(row.get("center_blob_touch_border_rate"))
    offset = fnum(row.get("center_blob_offset_med"))
    isolation = fnum(row.get("center_blob_isolation_med"))
    large_rate = fnum(row.get("center_blob_large_object_rate"))

    keep = 0.0
    false = 0.0
    notes = []

    if ok_rate < 0.25:
        false += 30
        notes.append("souvent aucun blob centré")
    elif ok_rate < 0.55:
        false += 15
        notes.append("blob centré intermittent")
    else:
        keep += 8
        notes.append("blob centré fréquent")

    # Ball should be small to medium in the crop.
    if 4 <= area <= 90:
        keep += 24
        notes.append("taille blob compatible balle")
    elif 90 < area <= 180:
        keep += 10
        false += 4
        notes.append("blob un peu gros")
    elif area > 180:
        false += 24
        notes.append("blob trop grand")
    elif area > 0:
        false += 8
        notes.append("blob très petit")

    if ratio > 0.18:
        false += 18
        notes.append("zone trop large dans le crop")
    elif 0.01 <= ratio <= 0.10:
        keep += 10
        notes.append("ratio surface plausible")

    if roundness >= 0.35:
        keep += 12
        notes.append("blob compact")
    elif roundness > 0:
        false += 6
        notes.append("blob peu compact")

    if touch_rate >= 0.35:
        false += 28
        notes.append("blob touche souvent le bord")
    elif touch_rate > 0:
        false += 10
        notes.append("blob touche parfois le bord")
    else:
        keep += 8
        notes.append("blob isolé du bord")

    if offset <= 4.0:
        keep += 10
        notes.append("blob centré")
    elif offset <= 8.0:
        keep += 4
        notes.append("blob proche centre")
    else:
        false += 10
        notes.append("blob décentré")

    if isolation >= 0.75:
        keep += 8
        notes.append("peu de pixels concurrents")
    elif isolation < 0.45:
        false += 10
        notes.append("beaucoup de pixels concurrents")

    if large_rate >= 0.35:
        false += 22
        notes.append("fréquent gros objet centré")

    keep = max(0.0, min(100.0, keep))
    false = max(0.0, min(100.0, false))

    if false >= 45 and keep < 45:
        guess = "center_blob_suspect"
    elif keep >= 45 and false < 35:
        guess = "center_blob_ball_like"
    elif keep >= 35 and false >= 35:
        guess = "center_blob_ambiguous"
    else:
        guess = "center_blob_weak"

    return fmt(keep, 2), fmt(false, 2), guess, ", ".join(notes)


def process_row(
    row: dict[str, str],
    run_dir: Path,
    config_map: dict[str, Path],
    patch_size: int,
    max_points: int,
) -> dict[str, str]:
    row = dict(row)

    segment_csv = resolve_path(run_dir, row.get("csv", ""))
    segment_mp4 = resolve_path(run_dir, row.get("mp4", ""))
    clip_id = row.get("clip_id", "")

    points = load_points(segment_csv)
    if not points:
        row["center_blob_points"] = "0"
        row["center_blob_guess_001N"] = "no_points"
        return row

    chosen = sample_points(points, max_points)
    first_frame = min(p[0] for p in points)

    source = config_map.get(clip_id)
    source_kind = "source_video"

    if source is None or not source.exists():
        source = segment_mp4
        source_kind = "segment_mp4_fallback"

    if source is None or not source.exists():
        row["center_blob_points"] = "0"
        row["center_blob_guess_001N"] = "missing_video"
        return row

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        row["center_blob_points"] = "0"
        row["center_blob_guess_001N"] = "video_open_failed"
        return row

    feats = []

    try:
        xs_all = [p[1] for p in points]
        ys_all = [p[2] for p in points]

        for frame_idx_original, x, y in chosen:
            if source_kind == "segment_mp4_fallback":
                frame_idx = frame_idx_original - first_frame
            else:
                frame_idx = frame_idx_original

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

            patch = crop_square(frame, x * sx, y * sy, patch_size)
            if patch is None:
                continue

            feats.append(centered_component_features(patch))

    finally:
        cap.release()

    if not feats:
        row["center_blob_points"] = "0"
        row["center_blob_guess_001N"] = "no_valid_patches"
        return row

    row["center_blob_source"] = source_kind
    row["center_blob_points"] = str(len(feats))
    row["center_blob_ok_rate"] = fmt(mean([f["ok"] for f in feats]), 4)
    row["center_blob_area_med"] = fmt(median([f["area"] for f in feats]), 3)
    row["center_blob_area_ratio_med"] = fmt(median([f["area_ratio"] for f in feats]), 5)
    row["center_blob_roundness_med"] = fmt(median([f["roundness"] for f in feats]), 5)
    row["center_blob_fill_med"] = fmt(median([f["fill"] for f in feats]), 5)
    row["center_blob_touch_border_rate"] = fmt(mean([f["touch_border"] for f in feats]), 4)
    row["center_blob_offset_med"] = fmt(median([f["offset"] for f in feats]), 3)
    row["center_blob_isolation_med"] = fmt(median([f["isolation"] for f in feats]), 5)
    row["center_blob_large_object_rate"] = fmt(mean([f["large_object"] for f in feats]), 4)

    keep, false, guess, notes = score_center_blob(row)

    row["center_blob_score_001N"] = keep
    row["center_blob_false_score_001N"] = false
    row["center_blob_guess_001N"] = guess
    row["center_blob_notes_001N"] = notes

    return row


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        by_target.setdefault(row.get("target_class", "unknown"), []).append(row)

    out: dict[str, Any] = {"count": len(rows), "by_target": {}}

    for target, target_rows in sorted(by_target.items()):
        out["by_target"][target] = {
            "count": len(target_rows),
            "keep_med": fmt(median([fnum(r.get("center_blob_score_001N")) for r in target_rows]), 2),
            "false_med": fmt(median([fnum(r.get("center_blob_false_score_001N")) for r in target_rows]), 2),
            "area_med": fmt(median([fnum(r.get("center_blob_area_med")) for r in target_rows]), 2),
            "touch_rate_med": fmt(median([fnum(r.get("center_blob_touch_border_rate")) for r in target_rows]), 4),
            "large_object_rate_med": fmt(median([fnum(r.get("center_blob_large_object_rate")) for r in target_rows]), 4),
            "offset_med": fmt(median([fnum(r.get("center_blob_offset_med")) for r in target_rows]), 2),
        }

    return out


def html_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []

    for row in rows:
        cls = html.escape(row.get("target_class", ""))
        cells = "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
        body.append(f'<tr class="{cls}">{cells}</tr>')

    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for target, stats in summary["by_target"].items():
        row = {"target": target}
        row.update(stats)
        summary_rows.append(row)

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r.get("target_class", ""),
            -fnum(r.get("center_blob_false_score_001N")),
            -fnum(r.get("false_track_score_001J")),
            r.get("review_id", ""),
        ),
    )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux center blob 001N</title>
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
  section {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 16px;
  }}
  h1, h2 {{
    margin-top: 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
  }}
  th, td {{
    border-bottom: 1px solid var(--line);
    padding: 7px 8px;
    vertical-align: top;
    font-size: 13px;
  }}
  th {{
    background: #20242e;
    text-align: left;
    position: sticky;
    top: 0;
  }}
  tr.negative td {{ background: rgba(255, 80, 80, 0.07); }}
  tr.partial td {{ background: rgba(255, 200, 80, 0.06); }}
  tr.positive td {{ background: rgba(100, 255, 150, 0.035); }}
  tr.ignore td {{ background: rgba(150, 170, 255, 0.045); }}
  .muted {{ color: var(--muted); }}
</style>
</head>
<body>
  <h1>TTFlux center blob 001N</h1>

  <section>
    <h2>Résumé</h2>
    <p class="muted">
      001N mesure le blob centré sous le point tracké : petit/isolé/compact ou gros objet collé au bord.
    </p>
    {html_table(summary_rows, [
        "target",
        "count",
        "keep_med",
        "false_med",
        "area_med",
        "touch_rate_med",
        "large_object_rate_med",
        "offset_med",
    ])}
  </section>

  <section>
    <h2>Détail</h2>
    {html_table(sorted_rows, [
        "review_id",
        "human_label_fr",
        "target_class",
        "center_blob_guess_001N",
        "center_blob_score_001N",
        "center_blob_false_score_001N",
        "center_blob_area_med",
        "center_blob_area_ratio_med",
        "center_blob_roundness_med",
        "center_blob_touch_border_rate",
        "center_blob_offset_med",
        "center_blob_isolation_med",
        "center_blob_large_object_rate",
        "center_blob_notes_001N",
        "false_track_score_001J",
        "keep_score_001J",
        "appearance_guess_001L",
        "clip_id",
        "segment_name",
    ])}
  </section>
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--config-dir", default="configs/batch_001E")
    parser.add_argument("--features", default="runs/batch_001E/appearance_features_001L.csv")
    parser.add_argument("--out-csv", default="runs/batch_001E/center_blob_features_001N.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/center_blob_features_001N.html")
    parser.add_argument("--out-json", default="runs/batch_001E/center_blob_summary_001N.json")
    parser.add_argument("--patch-size", type=int, default=33)
    parser.add_argument("--max-points", type=int, default=24)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_dir = Path(args.config_dir).resolve()
    features_path = Path(args.features)
    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)
    out_json = Path(args.out_json)

    _, rows = read_csv_rows(features_path)
    config_map = build_config_video_map(config_dir)

    print(f"[001N] rows          : {len(rows)}")
    print(f"[001N] config videos : {len(config_map)}")

    enriched = []

    for i, row in enumerate(rows, start=1):
        out = process_row(
            row=row,
            run_dir=run_dir,
            config_map=config_map,
            patch_size=args.patch_size,
            max_points=args.max_points,
        )
        enriched.append(out)

        print(
            f"[001N] {i:02d}/{len(rows)} {out.get('review_id')} "
            f"target={out.get('target_class')} "
            f"guess={out.get('center_blob_guess_001N')} "
            f"keep={out.get('center_blob_score_001N')} "
            f"false={out.get('center_blob_false_score_001N')} "
            f"area={out.get('center_blob_area_med')} "
            f"touch={out.get('center_blob_touch_border_rate')}"
        )

    summary = summarize(enriched)

    write_csv(out_csv, enriched)
    write_html(out_html, enriched, summary)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[001N] wrote CSV  : {out_csv}")
    print(f"[001N] wrote HTML : {out_html}")
    print(f"[001N] wrote JSON : {out_json}")


if __name__ == "__main__":
    main()