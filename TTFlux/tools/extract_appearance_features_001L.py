# PATCH 001L - TTFlux appearance features around tracked points
#
# Reads goldset_001K.csv and segment CSVs.
# For each tracked point, samples a small image patch around (x, y).
#
# Features:
# - brightness / saturation
# - white-ish score
# - orange-ish score
# - local contrast
# - edge density
# - blob area / roundness proxy
#
# It tries to use the original source video from configs/batch_001E/*.json.
# If not found, it falls back to the generated segment MP4.
#
# Usage:
#   python tools\extract_appearance_features_001L.py ^
#     --run-dir runs\batch_001E ^
#     --config-dir configs\batch_001E ^
#     --goldset runs\batch_001E\goldset_001K.csv ^
#     --out-csv runs\batch_001E\appearance_features_001L.csv ^
#     --out-html runs\batch_001E\appearance_features_001L.html ^
#     --out-json runs\batch_001E\appearance_summary_001L.json

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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return default
        value_f = float(text)
        if not math.isfinite(value_f):
            return default
        return value_f
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, default)))


def fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return ""
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


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
        "clip_id",
        "segment_idx",
        "segment_name",
        "human_label",
        "human_label_fr",
        "target_class",
        "expected_action",
        "training_split",
        "appearance_source",
        "appearance_points",
        "appearance_ok",
        "patch_size",
        "v_mean_med",
        "v_std_med",
        "s_mean_med",
        "contrast_med",
        "edge_density_med",
        "white_score_med",
        "orange_score_med",
        "color_peak_score_med",
        "blob_area_med",
        "blob_roundness_med",
        "blob_fill_med",
        "appearance_false_score_001L",
        "appearance_keep_score_001L",
        "appearance_guess_001L",
        "appearance_notes_001L",
        "risk_score_001G",
        "fragmentation_score_001J",
        "false_track_score_001J",
        "keep_score_001J",
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
    path = Path(text)

    if path.is_absolute():
        return path

    return run_dir / path


def find_video_value(obj: Any) -> str:
    if isinstance(obj, dict):
        for key, value in obj.items():
            low = str(key).lower()
            if low in VIDEO_KEYS and isinstance(value, str):
                return value

        for value in obj.values():
            found = find_video_value(value)
            if found:
                return found

    elif isinstance(obj, list):
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

        if not video_path.exists():
            # Keep it anyway only if the absolute path in JSON may exist on user machine.
            video_path = Path(video_text)

        clip_id = path.stem
        out[clip_id] = video_path

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

    points = []
    for frame in sorted(buckets):
        values = buckets[frame]
        x = sum(v[0] for v in values) / len(values)
        y = sum(v[1] for v in values) / len(values)
        points.append((frame, x, y))

    return points


def get_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    if frame_idx < 0:
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()

    if not ok or frame is None:
        return None

    return frame


def crop_patch(frame: np.ndarray, x: float, y: float, patch_size: int) -> np.ndarray | None:
    h, w = frame.shape[:2]
    r = patch_size // 2

    xi = int(round(x))
    yi = int(round(y))

    x0 = max(0, xi - r)
    y0 = max(0, yi - r)
    x1 = min(w, xi + r + 1)
    y1 = min(h, yi + r + 1)

    if x1 <= x0 or y1 <= y0:
        return None

    patch = frame[y0:y1, x0:x1]

    if patch.size == 0:
        return None

    return patch


def ring_contrast(hsv: np.ndarray, patch_size: int) -> float:
    h, w = hsv.shape[:2]
    cy = h // 2
    cx = w // 2

    inner_r = max(2, patch_size // 5)
    outer_r = max(inner_r + 1, patch_size // 2)

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    inner = dist <= inner_r
    ring = (dist > inner_r) & (dist <= outer_r)

    if not np.any(inner) or not np.any(ring):
        return 0.0

    v = hsv[:, :, 2].astype(np.float32)

    return float(abs(np.mean(v[inner]) - np.mean(v[ring])))


def blob_features(hsv: np.ndarray) -> tuple[float, float, float]:
    # Candidate "ball-like" pixels: bright/white-ish or saturated orange-ish.
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    white_mask = (v >= 165) & (s <= 90)
    orange_mask = (h >= 5) & (h <= 28) & (s >= 65) & (v >= 100)
    mask = (white_mask | orange_mask).astype(np.uint8) * 255

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    if n_labels <= 1:
        return 0.0, 0.0, 0.0

    best_i = 1
    best_area = 0

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_i = i

    area = float(best_area)
    x = float(stats[best_i, cv2.CC_STAT_LEFT])
    y = float(stats[best_i, cv2.CC_STAT_TOP])
    w = float(stats[best_i, cv2.CC_STAT_WIDTH])
    h = float(stats[best_i, cv2.CC_STAT_HEIGHT])

    fill = area / max(1.0, w * h)

    # Roundness proxy from bounding box aspect and fill.
    aspect = min(w, h) / max(1.0, max(w, h))
    roundness = aspect * fill

    return area, roundness, fill


def patch_features(patch_bgr: np.ndarray, patch_size: int) -> dict[str, float]:
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    v_mean = float(np.mean(v))
    v_std = float(np.std(v))
    s_mean = float(np.mean(s))

    white_score = float(np.mean((v >= 165) & (s <= 90)))
    orange_score = float(np.mean((h >= 5) & (h <= 28) & (s >= 65) & (v >= 100)))
    color_peak_score = max(white_score, orange_score)

    contrast = ring_contrast(hsv, patch_size)

    edges = cv2.Canny(gray, 50, 140)
    edge_density = float(np.mean(edges > 0))

    area, roundness, fill = blob_features(hsv)

    return {
        "v_mean": v_mean,
        "v_std": v_std,
        "s_mean": s_mean,
        "white_score": white_score,
        "orange_score": orange_score,
        "color_peak_score": color_peak_score,
        "contrast": contrast,
        "edge_density": edge_density,
        "blob_area": area,
        "blob_roundness": roundness,
        "blob_fill": fill,
    }


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def sample_segment_appearance(
    row: dict[str, str],
    run_dir: Path,
    config_video_map: dict[str, Path],
    patch_size: int,
    max_points: int,
) -> tuple[dict[str, str], list[dict[str, float]]]:
    clip_id = row.get("clip_id", "")
    segment_csv = resolve_path(run_dir, row.get("csv", ""))
    segment_mp4 = resolve_path(run_dir, row.get("mp4", ""))

    points = load_points(segment_csv)

    if not points:
        return {
            "appearance_ok": "0",
            "appearance_source": "none",
            "appearance_points": "0",
            "appearance_notes_001L": "no_points",
        }, []

    # Subsample evenly so we do not hammer random frame seeks.
    if len(points) > max_points:
        idxs = np.linspace(0, len(points) - 1, max_points).round().astype(int).tolist()
        points_sample = [points[i] for i in idxs]
    else:
        points_sample = points

    first_frame = min(p[0] for p in points)

    source_path = config_video_map.get(clip_id)
    source_kind = "source_video"

    if source_path is None or not source_path.exists():
        source_path = segment_mp4
        source_kind = "segment_mp4_fallback"

    if source_path is None or not source_path.exists():
        return {
            "appearance_ok": "0",
            "appearance_source": "missing_video",
            "appearance_points": "0",
            "appearance_notes_001L": "missing_video",
        }, []

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return {
            "appearance_ok": "0",
            "appearance_source": source_kind,
            "appearance_points": "0",
            "appearance_notes_001L": "video_open_failed",
        }, []

    per_point = []

    try:
        for original_frame, x, y in points_sample:
            if source_kind == "segment_mp4_fallback":
                video_frame_idx = original_frame - first_frame
            else:
                video_frame_idx = original_frame

            frame = get_frame(cap, video_frame_idx)
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # If coordinates are out of frame, try a light scaling fallback.
            xs_all = [p[1] for p in points]
            ys_all = [p[2] for p in points]
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

            patch = crop_patch(frame, xx, yy, patch_size)
            if patch is None:
                continue

            feats = patch_features(patch, patch_size)
            feats["frame"] = float(original_frame)
            feats["video_frame"] = float(video_frame_idx)
            feats["x"] = float(xx)
            feats["y"] = float(yy)
            per_point.append(feats)

    finally:
        cap.release()

    if not per_point:
        return {
            "appearance_ok": "0",
            "appearance_source": source_kind,
            "appearance_points": "0",
            "appearance_notes_001L": "no_valid_patches",
        }, []

    aggregate = {
        "appearance_ok": "1",
        "appearance_source": source_kind,
        "appearance_points": str(len(per_point)),
        "patch_size": str(patch_size),
    }

    for key in [
        "v_mean",
        "v_std",
        "s_mean",
        "white_score",
        "orange_score",
        "color_peak_score",
        "contrast",
        "edge_density",
        "blob_area",
        "blob_roundness",
        "blob_fill",
    ]:
        aggregate[f"{key}_med"] = fmt(median([p[key] for p in per_point]), 5)

    return aggregate, per_point


def score_appearance(row: dict[str, str]) -> tuple[str, str, str, str]:
    ok = row.get("appearance_ok") == "1"

    if not ok:
        return "0", "0", "appearance_missing", "appearance_missing"

    color_peak = safe_float(row.get("color_peak_score_med"))
    contrast = safe_float(row.get("contrast_med"))
    edge_density = safe_float(row.get("edge_density_med"))
    blob_area = safe_float(row.get("blob_area_med"))
    roundness = safe_float(row.get("blob_roundness_med"))
    fill = safe_float(row.get("blob_fill_med"))
    white_score = safe_float(row.get("white_score_med"))
    orange_score = safe_float(row.get("orange_score_med"))
    s_mean = safe_float(row.get("s_mean_med"))
    v_mean = safe_float(row.get("v_mean_med"))

    false_score = 0.0
    keep_score = 0.0
    notes = []

    # A ball candidate should usually have a small bright/color peak or a compact blob.
    if color_peak < 0.015 and blob_area < 2:
        false_score += 28
        notes.append("peu de pixels couleur/balle")
    elif color_peak < 0.035 and blob_area < 5:
        false_score += 14
        notes.append("signature balle faible")

    if contrast < 8:
        false_score += 12
        notes.append("faible contraste local")
    elif contrast > 18:
        keep_score += 8
        notes.append("contraste local correct")

    if edge_density > 0.28:
        false_score += 12
        notes.append("patch très texturé")
    elif edge_density < 0.18:
        keep_score += 5
        notes.append("patch peu texturé")

    if blob_area >= 3 and roundness >= 0.25:
        keep_score += 18
        notes.append("blob compact")
    elif blob_area >= 2 and fill >= 0.3:
        keep_score += 9
        notes.append("petit blob rempli")

    if white_score >= 0.04 or orange_score >= 0.04:
        keep_score += 12
        notes.append("signature blanc/orange")
    elif white_score < 0.01 and orange_score < 0.01:
        false_score += 10
        notes.append("pas de signature blanc/orange")

    # Saturated texture can be player/raquette/logo, not necessarily ball.
    if s_mean > 120 and color_peak < 0.04:
        false_score += 8
        notes.append("saturation diffuse sans pic balle")

    # Too dark, probably not the ball.
    if v_mean < 55:
        false_score += 12
        notes.append("patch sombre")

    false_score = max(0.0, min(100.0, false_score))
    keep_score = max(0.0, min(100.0, keep_score))

    if false_score >= 35 and keep_score < 18:
        guess = "appearance_suspect"
    elif keep_score >= 25 and false_score < 25:
        guess = "appearance_ball_like"
    else:
        guess = "appearance_mixed"

    return fmt(false_score), fmt(keep_score), guess, ", ".join(notes)


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        by_target.setdefault(row.get("target_class", "unknown"), []).append(row)

    out: dict[str, Any] = {
        "count": len(rows),
        "by_target": {},
    }

    for target, target_rows in sorted(by_target.items()):
        out["by_target"][target] = {
            "count": len(target_rows),
            "appearance_ok": sum(1 for r in target_rows if r.get("appearance_ok") == "1"),
            "appearance_false_med": fmt(median([safe_float(r.get("appearance_false_score_001L")) for r in target_rows])),
            "appearance_keep_med": fmt(median([safe_float(r.get("appearance_keep_score_001L")) for r in target_rows])),
            "white_med": fmt(median([safe_float(r.get("white_score_med")) for r in target_rows]), 5),
            "orange_med": fmt(median([safe_float(r.get("orange_score_med")) for r in target_rows]), 5),
            "contrast_med": fmt(median([safe_float(r.get("contrast_med")) for r in target_rows]), 5),
            "edge_med": fmt(median([safe_float(r.get("edge_density_med")) for r in target_rows]), 5),
            "blob_area_med": fmt(median([safe_float(r.get("blob_area_med")) for r in target_rows]), 5),
            "blob_roundness_med": fmt(median([safe_float(r.get("blob_roundness_med")) for r in target_rows]), 5),
        }

    return out


def html_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)

    body = []
    for row in rows:
        cls = html.escape(row.get("target_class", ""))
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
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
            -safe_float(r.get("appearance_false_score_001L")),
            r.get("review_id", ""),
        ),
    )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux appearance features 001L</title>
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
  tr.negative td {{
    background: rgba(255, 80, 80, 0.07);
  }}
  tr.partial td {{
    background: rgba(255, 200, 80, 0.06);
  }}
  tr.positive td {{
    background: rgba(100, 255, 150, 0.035);
  }}
  tr.ignore td {{
    background: rgba(150, 170, 255, 0.045);
  }}
  .muted {{
    color: var(--muted);
  }}
</style>
</head>
<body>
  <h1>TTFlux appearance features 001L</h1>

  <section>
    <h2>Résumé</h2>
    <p class="muted">
      On regarde maintenant l'apparence autour du point tracké : signature blanc/orange,
      contraste local, texture, blob compact. C'est le début du détecteur anti-reflets/raquettes.
    </p>
    {html_table(summary_rows, [
        "target",
        "count",
        "appearance_ok",
        "appearance_false_med",
        "appearance_keep_med",
        "white_med",
        "orange_med",
        "contrast_med",
        "edge_med",
        "blob_area_med",
        "blob_roundness_med",
    ])}
  </section>

  <section>
    <h2>Détail par segment</h2>
    {html_table(sorted_rows, [
        "review_id",
        "human_label_fr",
        "target_class",
        "appearance_source",
        "appearance_points",
        "appearance_false_score_001L",
        "appearance_keep_score_001L",
        "appearance_guess_001L",
        "white_score_med",
        "orange_score_med",
        "contrast_med",
        "edge_density_med",
        "blob_area_med",
        "blob_roundness_med",
        "appearance_notes_001L",
        "false_track_score_001J",
        "keep_score_001J",
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
    parser.add_argument("--goldset", default="runs/batch_001E/goldset_001K.csv")
    parser.add_argument("--out-csv", default="runs/batch_001E/appearance_features_001L.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/appearance_features_001L.html")
    parser.add_argument("--out-json", default="runs/batch_001E/appearance_summary_001L.json")
    parser.add_argument("--patch-size", type=int, default=25)
    parser.add_argument("--max-points", type=int, default=24)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_dir = Path(args.config_dir).resolve()
    goldset_path = Path(args.goldset)
    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)
    out_json = Path(args.out_json)

    _, gold_rows = read_csv_rows(goldset_path)
    config_video_map = build_config_video_map(config_dir)

    enriched = []

    print(f"[001L] run_dir        : {run_dir}")
    print(f"[001L] config_dir     : {config_dir}")
    print(f"[001L] config videos  : {len(config_video_map)}")
    print(f"[001L] gold rows      : {len(gold_rows)}")

    for i, row in enumerate(gold_rows, start=1):
        aggregate, _ = sample_segment_appearance(
            row=row,
            run_dir=run_dir,
            config_video_map=config_video_map,
            patch_size=args.patch_size,
            max_points=args.max_points,
        )

        merged = dict(row)
        merged.update(aggregate)

        false_score, keep_score, guess, notes = score_appearance(merged)
        merged["appearance_false_score_001L"] = false_score
        merged["appearance_keep_score_001L"] = keep_score
        merged["appearance_guess_001L"] = guess

        existing_notes = merged.get("appearance_notes_001L", "")
        merged["appearance_notes_001L"] = ", ".join(
            [x for x in [existing_notes, notes] if x]
        )

        enriched.append(merged)

        print(
            f"[001L] {i:02d}/{len(gold_rows)} "
            f"{merged.get('review_id')} "
            f"ok={merged.get('appearance_ok')} "
            f"src={merged.get('appearance_source')} "
            f"pts={merged.get('appearance_points')} "
            f"false={merged.get('appearance_false_score_001L')} "
            f"keep={merged.get('appearance_keep_score_001L')}"
        )

    summary = summarize(enriched)

    write_csv(out_csv, enriched)
    write_html(out_html, enriched, summary)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    ok_count = sum(1 for row in enriched if row.get("appearance_ok") == "1")

    print(f"[001L] appearance ok : {ok_count}/{len(enriched)}")
    print(f"[001L] wrote CSV     : {out_csv}")
    print(f"[001L] wrote HTML    : {out_html}")
    print(f"[001L] wrote JSON    : {out_json}")


if __name__ == "__main__":
    main()