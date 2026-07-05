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
        "micro_points",
        "micro_candidate_rate_001O",
        "micro_score_med_001O",
        "micro_area_med_001O",
        "micro_distance_med_001O",
        "micro_roundness_med_001O",
        "micro_fill_med_001O",
        "micro_contrast_med_001O",
        "micro_touch_rate_001O",
        "micro_elongated_rate_001O",
        "micro_keep_score_001O",
        "micro_false_score_001O",
        "micro_guess_001O",
        "micro_notes_001O",
        "center_blob_score_001N",
        "center_blob_false_score_001N",
        "center_blob_area_med",
        "center_blob_touch_border_rate",
        "center_blob_large_object_rate",
        "false_track_score_001J",
        "keep_score_001J",
        "appearance_guess_001L",
        "risk_score_001G",
        "fragmentation_score_001J",
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
        frame = inum(row.get(frame_col), -1)
        x = fnum(row.get(x_col), float("nan"))
        y = fnum(row.get(y_col), float("nan"))

        if frame < 0 or not math.isfinite(x) or not math.isfinite(y):
            continue

        buckets.setdefault(frame, []).append((x, y))

    out = []

    for frame in sorted(buckets):
        pts = buckets[frame]
        x = sum(v[0] for v in pts) / len(pts)
        y = sum(v[1] for v in pts) / len(pts)
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

    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)

    dx0 = sx0 - x0
    dy0 = sy0 - y0

    out[
        dy0 : dy0 + (sy1 - sy0),
        dx0 : dx0 + (sx1 - sx0),
    ] = frame[sy0:sy1, sx0:sx1]

    return out


def micro_blob_features(patch_bgr: np.ndarray) -> dict[str, float]:
    size = patch_bgr.shape[0]
    center = size // 2

    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    high = gray - blur

    high_pos = np.maximum(high, 0)
    high_thr = max(8.0, float(np.percentile(high_pos, 92)))

    local_peak = (high_pos >= high_thr) & (gray >= max(55.0, float(np.percentile(gray, 55))))

    white_local = (v >= 135) & (s <= 120) & (high_pos >= 5)
    orange_local = (h >= 5) & (h <= 35) & (s >= 45) & (v >= 90) & (high_pos >= 4)

    mask = (local_peak | white_local | orange_local).astype(np.uint8) * 255

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    if n_labels <= 1:
        return {
            "has_candidate": 0.0,
            "score": 0.0,
            "area": 0.0,
            "distance": 999.0,
            "roundness": 0.0,
            "fill": 0.0,
            "contrast": 0.0,
            "touch": 0.0,
            "elongated": 0.0,
        }

    best = None
    best_score = -999.0

    for label in range(1, n_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        x = float(stats[label, cv2.CC_STAT_LEFT])
        y = float(stats[label, cv2.CC_STAT_TOP])
        w = float(stats[label, cv2.CC_STAT_WIDTH])
        h_box = float(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]

        if area < 2 or area > 180:
            continue

        dist = math.hypot(float(cx) - center, float(cy) - center)

        if dist > 14:
            continue

        fill = area / max(1.0, w * h_box)
        aspect = min(w, h_box) / max(1.0, max(w, h_box))
        roundness = aspect * fill
        touch = 1.0 if x <= 0 or y <= 0 or x + w >= size - 1 or y + h_box >= size - 1 else 0.0
        elongated = 1.0 if aspect < 0.38 else 0.0

        component_mask = labels == label
        contrast = float(np.mean(high_pos[component_mask])) if np.any(component_mask) else 0.0

        area_score = 0.0
        if 4 <= area <= 80:
            area_score = 24.0
        elif 80 < area <= 130:
            area_score = 12.0
        elif 2 <= area < 4:
            area_score = 6.0

        center_score = max(0.0, 20.0 - dist * 1.6)
        round_score = min(20.0, roundness * 35.0)
        contrast_score = min(20.0, contrast * 1.2)

        score = area_score + center_score + round_score + contrast_score

        if touch:
            score -= 18.0

        if elongated:
            score -= 12.0

        if fill < 0.25:
            score -= 8.0

        if score > best_score:
            best_score = score
            best = {
                "has_candidate": 1.0,
                "score": max(0.0, min(100.0, score)),
                "area": area,
                "distance": dist,
                "roundness": roundness,
                "fill": fill,
                "contrast": contrast,
                "touch": touch,
                "elongated": elongated,
            }

    if best is None:
        return {
            "has_candidate": 0.0,
            "score": 0.0,
            "area": 0.0,
            "distance": 999.0,
            "roundness": 0.0,
            "fill": 0.0,
            "contrast": 0.0,
            "touch": 0.0,
            "elongated": 0.0,
        }

    return best


def score_micro(row: dict[str, str]) -> tuple[str, str, str, str]:
    rate = fnum(row.get("micro_candidate_rate_001O"))
    score = fnum(row.get("micro_score_med_001O"))
    area = fnum(row.get("micro_area_med_001O"))
    dist = fnum(row.get("micro_distance_med_001O"))
    roundness = fnum(row.get("micro_roundness_med_001O"))
    touch = fnum(row.get("micro_touch_rate_001O"))
    elongated = fnum(row.get("micro_elongated_rate_001O"))
    contrast = fnum(row.get("micro_contrast_med_001O"))

    keep = 0.0
    false = 0.0
    notes = []

    if rate >= 0.75:
        keep += 28
        notes.append("micro-candidat fréquent")
    elif rate >= 0.45:
        keep += 14
        false += 8
        notes.append("micro-candidat intermittent")
    else:
        false += 28
        notes.append("micro-candidat rare")

    if score >= 55:
        keep += 30
        notes.append("micro-score fort")
    elif score >= 35:
        keep += 16
        notes.append("micro-score moyen")
    else:
        false += 12
        notes.append("micro-score faible")

    if 4 <= area <= 90:
        keep += 16
        notes.append("taille petite compatible")
    elif area > 130:
        false += 18
        notes.append("micro trop grand")
    elif area > 0:
        keep += 4

    if dist <= 5:
        keep += 10
        notes.append("proche centre")
    elif dist > 10:
        false += 8
        notes.append("décentré")

    if roundness >= 0.35:
        keep += 10
        notes.append("compact")
    elif roundness > 0:
        false += 6
        notes.append("peu compact")

    if touch >= 0.35:
        false += 16
        notes.append("souvent collé au bord")

    if elongated >= 0.35:
        false += 16
        notes.append("souvent allongé")

    if contrast >= 8:
        keep += 8
        notes.append("pic local contrasté")
    elif contrast < 3:
        false += 6
        notes.append("contraste faible")

    keep = max(0.0, min(100.0, keep))
    false = max(0.0, min(100.0, false))

    if keep >= 60 and false < 35:
        guess = "micro_ball_like"
    elif false >= 50 and keep < 45:
        guess = "micro_suspect"
    elif keep >= 45:
        guess = "micro_mixed_keep"
    else:
        guess = "micro_uncertain"

    return fmt(keep, 2), fmt(false, 2), guess, ", ".join(notes)


def process_row(
    row: dict[str, str],
    run_dir: Path,
    config_map: dict[str, Path],
    patch_size: int,
    max_points: int,
) -> dict[str, str]:
    out = dict(row)

    points = load_points(resolve_path(run_dir, row.get("csv", "")))

    if not points:
        out["micro_points"] = "0"
        out["micro_guess_001O"] = "no_points"
        return out

    chosen = sample_points(points, max_points)
    first_frame = min(p[0] for p in points)

    source = config_map.get(row.get("clip_id", ""))
    source_kind = "source_video"

    if source is None or not source.exists():
        source = resolve_path(run_dir, row.get("mp4", ""))
        source_kind = "segment_mp4_fallback"

    if source is None or not source.exists():
        out["micro_points"] = "0"
        out["micro_guess_001O"] = "missing_video"
        return out

    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        out["micro_points"] = "0"
        out["micro_guess_001O"] = "video_open_failed"
        return out

    feats = []

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

            patch = crop_square(frame, x * sx, y * sy, patch_size)
            if patch is None:
                continue

            feats.append(micro_blob_features(patch))

    finally:
        cap.release()

    if not feats:
        out["micro_points"] = "0"
        out["micro_guess_001O"] = "no_valid_patches"
        return out

    out["micro_source"] = source_kind
    out["micro_points"] = str(len(feats))
    out["micro_candidate_rate_001O"] = fmt(mean([f["has_candidate"] for f in feats]), 4)
    out["micro_score_med_001O"] = fmt(median([f["score"] for f in feats]), 3)
    out["micro_area_med_001O"] = fmt(median([f["area"] for f in feats]), 3)
    out["micro_distance_med_001O"] = fmt(median([f["distance"] for f in feats]), 3)
    out["micro_roundness_med_001O"] = fmt(median([f["roundness"] for f in feats]), 5)
    out["micro_fill_med_001O"] = fmt(median([f["fill"] for f in feats]), 5)
    out["micro_contrast_med_001O"] = fmt(median([f["contrast"] for f in feats]), 3)
    out["micro_touch_rate_001O"] = fmt(mean([f["touch"] for f in feats]), 4)
    out["micro_elongated_rate_001O"] = fmt(mean([f["elongated"] for f in feats]), 4)

    keep, false, guess, notes = score_micro(out)

    out["micro_keep_score_001O"] = keep
    out["micro_false_score_001O"] = false
    out["micro_guess_001O"] = guess
    out["micro_notes_001O"] = notes

    return out


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        by_target.setdefault(row.get("target_class", "unknown"), []).append(row)

    out: dict[str, Any] = {"count": len(rows), "by_target": {}}

    for target, target_rows in sorted(by_target.items()):
        out["by_target"][target] = {
            "count": len(target_rows),
            "micro_keep_med": fmt(median([fnum(r.get("micro_keep_score_001O")) for r in target_rows]), 2),
            "micro_false_med": fmt(median([fnum(r.get("micro_false_score_001O")) for r in target_rows]), 2),
            "candidate_rate_med": fmt(median([fnum(r.get("micro_candidate_rate_001O")) for r in target_rows]), 4),
            "micro_score_med": fmt(median([fnum(r.get("micro_score_med_001O")) for r in target_rows]), 3),
            "area_med": fmt(median([fnum(r.get("micro_area_med_001O")) for r in target_rows]), 3),
            "distance_med": fmt(median([fnum(r.get("micro_distance_med_001O")) for r in target_rows]), 3),
            "touch_rate_med": fmt(median([fnum(r.get("micro_touch_rate_001O")) for r in target_rows]), 4),
            "elongated_rate_med": fmt(median([fnum(r.get("micro_elongated_rate_001O")) for r in target_rows]), 4),
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
            -fnum(r.get("micro_false_score_001O")),
            -fnum(r.get("false_track_score_001J")),
            r.get("review_id", ""),
        ),
    )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux micro blob 001O</title>
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
  <h1>TTFlux micro blob 001O</h1>

  <section>
    <h2>Résumé</h2>
    <p class="muted">
      001O cherche un petit pic local compact près du centre. C’est plus strict que 001N :
      une grosse zone lumineuse ne doit plus être considérée comme une balle.
    </p>
    {html_table(summary_rows, [
        "target",
        "count",
        "micro_keep_med",
        "micro_false_med",
        "candidate_rate_med",
        "micro_score_med",
        "area_med",
        "distance_med",
        "touch_rate_med",
        "elongated_rate_med",
    ])}
  </section>

  <section>
    <h2>Détail</h2>
    {html_table(sorted_rows, [
        "review_id",
        "human_label_fr",
        "target_class",
        "micro_guess_001O",
        "micro_keep_score_001O",
        "micro_false_score_001O",
        "micro_candidate_rate_001O",
        "micro_score_med_001O",
        "micro_area_med_001O",
        "micro_distance_med_001O",
        "micro_roundness_med_001O",
        "micro_touch_rate_001O",
        "micro_elongated_rate_001O",
        "micro_notes_001O",
        "center_blob_false_score_001N",
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
    parser.add_argument("--features", default="runs/batch_001E/center_blob_features_001N.csv")
    parser.add_argument("--out-csv", default="runs/batch_001E/micro_blob_features_001O.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/micro_blob_features_001O.html")
    parser.add_argument("--out-json", default="runs/batch_001E/micro_blob_summary_001O.json")
    parser.add_argument("--patch-size", type=int, default=41)
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

    print(f"[001O] rows          : {len(rows)}")
    print(f"[001O] config videos : {len(config_map)}")

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
            f"[001O] {i:02d}/{len(rows)} {out.get('review_id')} "
            f"target={out.get('target_class')} "
            f"guess={out.get('micro_guess_001O')} "
            f"keep={out.get('micro_keep_score_001O')} "
            f"false={out.get('micro_false_score_001O')} "
            f"rate={out.get('micro_candidate_rate_001O')} "
            f"score={out.get('micro_score_med_001O')}"
        )

    summary = summarize(enriched)

    write_csv(out_csv, enriched)
    write_html(out_html, enriched, summary)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[001O] wrote CSV  : {out_csv}")
    print(f"[001O] wrote HTML : {out_html}")
    print(f"[001O] wrote JSON : {out_json}")


if __name__ == "__main__":
    main()