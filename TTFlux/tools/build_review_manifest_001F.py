# PATCH 001F — TTFlux review manifest
# Build a human review manifest from a batch run directory.
#
# Usage:
#   python tools\build_review_manifest_001F.py ^
#     --run-dir runs\batch_001E ^
#     --out-csv runs\batch_001E\review_manifest_001F.csv ^
#     --out-html runs\batch_001E\review_manifest_001F.html

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CSV_EXTS = {".csv"}

ROLE_WORDS = {
    "overlay",
    "overlays",
    "segment",
    "segments",
    "track",
    "tracks",
    "selected",
    "review",
    "manifest",
    "summary",
    "debug",
    "points",
    "candidate",
    "candidates",
    "clip",
    "clips",
}

SEGMENT_PATTERNS = [
    re.compile(r"(?:segment|seg|segment_idx|idx)[_\- ]?(\d+)", re.IGNORECASE),
    re.compile(r"(?:^|[_\-/\\])s(\d{1,4})(?:$|[_\-.])", re.IGNORECASE),
]

FRAME_COLS = ["frame", "frame_idx", "frame_index", "f"]
X_COLS = ["x", "ball_x", "cx", "center_x"]
Y_COLS = ["y", "ball_y", "cy", "center_y"]
SCORE_COLS = ["score", "segment_score", "confidence", "conf", "prob"]


@dataclass
class SegmentReview:
    clip_id: str
    segment_idx: str

    first_frame: str = ""
    last_frame: str = ""
    n_points: str = ""
    score: str = ""
    density: str = ""
    travel: str = ""
    x_range: str = ""
    y_range: str = ""

    mp4: list[str] = field(default_factory=list)
    overlay_png: list[str] = field(default_factory=list)
    csv_files: list[str] = field(default_factory=list)

    review_status: str = ""
    notes: str = ""

    def key(self) -> tuple[str, str]:
        return self.clip_id, self.segment_idx


def norm_col(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def first_existing_col(fieldnames: list[str], candidates: list[str]) -> str | None:
    normalized = {norm_col(c): c for c in fieldnames}
    for c in candidates:
        if c in normalized:
            return normalized[c]
    return None


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip().replace(",", ".")
    if not text:
        return None

    try:
        val = float(text)
    except ValueError:
        return None

    if not math.isfinite(val):
        return None

    return val


def safe_int_text(value: Any) -> str:
    val = safe_float(value)
    if val is None:
        return ""
    return str(int(round(val)))


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = f"{value:.{digits}f}"
    text = text.rstrip("0").rstrip(".")
    return text


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
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
    except Exception as exc:
        print(f"[WARN] Could not read CSV {path}: {exc}")
        return [], []


def find_segment_idx_from_text(text: str) -> str:
    for pattern in SEGMENT_PATTERNS:
        match = pattern.search(text)
        if match:
            return str(int(match.group(1)))
    return ""


def clean_clip_id_from_stem(stem: str) -> str:
    text = stem

    text = re.sub(r"(?:segment|seg|segment_idx|idx)[_\- ]?\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:^|_)s\d{1,4}(?:$|_)", "_", text, flags=re.IGNORECASE)

    parts = re.split(r"[_\s]+", text)
    kept = []
    for part in parts:
        p = part.strip("-_ ")
        if not p:
            continue
        if p.lower() in ROLE_WORDS:
            continue
        kept.append(p)

    cleaned = "_".join(kept)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_- ")

    return cleaned or stem


def infer_key_from_path(path: Path, run_dir: Path) -> tuple[str, str]:
    rel = path.relative_to(run_dir)
    rel_text = str(rel)
    stem = path.stem

    segment_idx = find_segment_idx_from_text(rel_text)

    parent_candidates = [p.name for p in path.parents if p != run_dir and run_dir in p.parents]
    parent_candidates = list(reversed(parent_candidates))

    clip_id = ""

    # Prefer a parent folder that looks like a clip id.
    for parent in parent_candidates:
        if find_segment_idx_from_text(parent):
            continue
        low = parent.lower()
        if low in {"clips", "overlays", "segments", "csv", "png", "mp4", "debug", "selected"}:
            continue
        clip_id = clean_clip_id_from_stem(parent)
        break

    if not clip_id:
        clip_id = clean_clip_id_from_stem(stem)

    if not segment_idx:
        # Last fallback: if there is exactly one standalone number near the end.
        match = re.search(r"[_\-](\d{1,4})$", stem)
        if match:
            segment_idx = str(int(match.group(1)))

    if not segment_idx:
        segment_idx = "0"

    return clip_id, segment_idx


def update_if_empty(obj: SegmentReview, attr: str, value: Any) -> None:
    text = "" if value is None else str(value).strip()
    if text and not getattr(obj, attr):
        setattr(obj, attr, text)


def get_or_create(
    segments: dict[tuple[str, str], SegmentReview],
    clip_id: str,
    segment_idx: str,
) -> SegmentReview:
    key = (clip_id or "unknown_clip", segment_idx or "0")
    if key not in segments:
        segments[key] = SegmentReview(clip_id=key[0], segment_idx=key[1])
    return segments[key]


def compute_point_stats(rows: list[dict[str, str]], fieldnames: list[str]) -> dict[str, str]:
    frame_col = first_existing_col(fieldnames, FRAME_COLS)
    x_col = first_existing_col(fieldnames, X_COLS)
    y_col = first_existing_col(fieldnames, Y_COLS)
    score_col = first_existing_col(fieldnames, SCORE_COLS)

    points = []
    scores = []

    for row in rows:
        frame = safe_float(row.get(frame_col)) if frame_col else None
        x = safe_float(row.get(x_col)) if x_col else None
        y = safe_float(row.get(y_col)) if y_col else None

        if score_col:
            s = safe_float(row.get(score_col))
            if s is not None:
                scores.append(s)

        if frame is None or x is None or y is None:
            continue

        points.append((int(round(frame)), x, y))

    if not points:
        return {}

    points.sort(key=lambda p: p[0])
    frames = [p[0] for p in points]
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]

    first_frame = min(frames)
    last_frame = max(frames)
    duration = max(1, last_frame - first_frame + 1)
    unique_frames = len(set(frames))
    density = unique_frames / duration

    travel = 0.0
    prev = None
    for frame, x, y in points:
        if prev is not None:
            _, px, py = prev
            dist = math.hypot(x - px, y - py)
            if math.isfinite(dist):
                travel += dist
        prev = (frame, x, y)

    score = None
    if scores:
        score = sum(scores) / len(scores)

    return {
        "first_frame": str(first_frame),
        "last_frame": str(last_frame),
        "n_points": str(len(points)),
        "density": fmt_float(density, 4),
        "travel": fmt_float(travel, 2),
        "x_range": fmt_float(max(xs) - min(xs), 2),
        "y_range": fmt_float(max(ys) - min(ys), 2),
        "score": fmt_float(score, 4),
    }


def row_value(row: dict[str, str], names: list[str]) -> str:
    normalized = {norm_col(k): k for k in row.keys()}
    for name in names:
        key = normalized.get(name)
        if key:
            value = row.get(key, "").strip()
            if value:
                return value
    return ""


def ingest_summary_csv(
    path: Path,
    rel: str,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    segments: dict[tuple[str, str], SegmentReview],
    run_dir: Path,
) -> bool:
    normalized = {norm_col(c) for c in fieldnames}

    has_clip = any(c in normalized for c in ["clip_id", "clip", "sequence_key", "video_id"])
    has_segment = any(c in normalized for c in ["segment_idx", "segment", "seg_idx", "idx"])
    has_summary_signal = any(
        c in normalized
        for c in [
            "first_frame",
            "last_frame",
            "n_points",
            "density",
            "travel",
            "x_range",
            "y_range",
            "score",
            "segment_score",
            "overlay_mp4",
            "overlay_png",
            "csv",
            "csv_path",
        ]
    )

    if not (has_clip and has_summary_signal):
        return False

    for row in rows:
        clip_id = row_value(row, ["clip_id", "clip", "sequence_key", "video_id"])
        segment_idx = row_value(row, ["segment_idx", "segment", "seg_idx", "idx"])

        if not segment_idx:
            _, inferred_segment_idx = infer_key_from_path(path, run_dir)
            segment_idx = inferred_segment_idx

        seg = get_or_create(segments, clip_id, segment_idx)

        update_if_empty(seg, "first_frame", safe_int_text(row_value(row, ["first_frame", "start_frame", "frame_start"])))
        update_if_empty(seg, "last_frame", safe_int_text(row_value(row, ["last_frame", "end_frame", "frame_end"])))
        update_if_empty(seg, "n_points", safe_int_text(row_value(row, ["n_points", "points", "point_count", "track_len"])))
        update_if_empty(seg, "score", row_value(row, ["score", "segment_score", "confidence"]))
        update_if_empty(seg, "density", row_value(row, ["density", "track_density"]))
        update_if_empty(seg, "travel", row_value(row, ["travel", "path_length"]))
        update_if_empty(seg, "x_range", row_value(row, ["x_range", "range_x"]))
        update_if_empty(seg, "y_range", row_value(row, ["y_range", "range_y"]))

        mp4_path = row_value(row, ["mp4", "overlay_mp4", "video", "video_path"])
        png_path = row_value(row, ["png", "overlay_png", "image", "image_path"])
        csv_path = row_value(row, ["csv", "csv_path", "points_csv", "track_csv"])

        if mp4_path:
            seg.mp4.append(mp4_path)
        if png_path:
            seg.overlay_png.append(png_path)
        if csv_path:
            seg.csv_files.append(csv_path)

        if rel not in seg.csv_files:
            seg.csv_files.append(rel)

    return True


def ingest_point_csv(
    path: Path,
    rel: str,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    segments: dict[tuple[str, str], SegmentReview],
    run_dir: Path,
) -> None:
    stats = compute_point_stats(rows, fieldnames)
    if not stats:
        return

    clip_id, segment_idx = infer_key_from_path(path, run_dir)
    seg = get_or_create(segments, clip_id, segment_idx)

    for attr, value in stats.items():
        update_if_empty(seg, attr, value)

    if rel not in seg.csv_files:
        seg.csv_files.append(rel)


def append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def best_sort_key(seg: SegmentReview) -> tuple[str, int, int]:
    try:
        segment_idx = int(seg.segment_idx)
    except ValueError:
        segment_idx = 0

    try:
        first_frame = int(seg.first_frame)
    except ValueError:
        first_frame = 0

    return seg.clip_id, segment_idx, first_frame


def relpath_for_html(path_text: str) -> str:
    # Keep Windows paths clickable enough from a local HTML generated inside run_dir.
    return path_text.replace("\\", "/")


def write_manifest_csv(out_csv: Path, segments: list[SegmentReview]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "review_id",
        "clip_id",
        "segment_idx",
        "first_frame",
        "last_frame",
        "n_points",
        "score",
        "density",
        "travel",
        "x_range",
        "y_range",
        "mp4",
        "overlay_png",
        "csv",
        "review_status",
        "notes",
    ]

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for i, seg in enumerate(segments, start=1):
            writer.writerow(
                {
                    "review_id": f"R{i:04d}",
                    "clip_id": seg.clip_id,
                    "segment_idx": seg.segment_idx,
                    "first_frame": seg.first_frame,
                    "last_frame": seg.last_frame,
                    "n_points": seg.n_points,
                    "score": seg.score,
                    "density": seg.density,
                    "travel": seg.travel,
                    "x_range": seg.x_range,
                    "y_range": seg.y_range,
                    "mp4": " | ".join(seg.mp4),
                    "overlay_png": " | ".join(seg.overlay_png),
                    "csv": " | ".join(seg.csv_files),
                    "review_status": seg.review_status,
                    "notes": seg.notes,
                }
            )


def write_manifest_html(out_html: Path, segments: list[SegmentReview], out_csv: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    rows_html = []

    for i, seg in enumerate(segments, start=1):
        mp4_links = "<br>".join(
            f'<a href="{html.escape(relpath_for_html(p))}">mp4</a>' for p in seg.mp4
        )
        png_links = "<br>".join(
            f'<a href="{html.escape(relpath_for_html(p))}">png</a>' for p in seg.overlay_png
        )
        csv_links = "<br>".join(
            f'<a href="{html.escape(relpath_for_html(p))}">csv</a>' for p in seg.csv_files
        )

        preview = ""
        if seg.overlay_png:
            src = html.escape(relpath_for_html(seg.overlay_png[0]))
            preview = f'<a href="{src}"><img src="{src}" loading="lazy"></a>'

        rows_html.append(
            f"""
            <tr>
              <td>R{i:04d}</td>
              <td><code>{html.escape(seg.clip_id)}</code></td>
              <td>{html.escape(seg.segment_idx)}</td>
              <td>{html.escape(seg.first_frame)} → {html.escape(seg.last_frame)}</td>
              <td>{html.escape(seg.n_points)}</td>
              <td>{html.escape(seg.score)}</td>
              <td>{html.escape(seg.density)}</td>
              <td>{html.escape(seg.travel)}</td>
              <td>{html.escape(seg.x_range)} / {html.escape(seg.y_range)}</td>
              <td>{mp4_links}</td>
              <td>{png_links}</td>
              <td>{csv_links}</td>
              <td class="preview">{preview}</td>
              <td class="review">à revoir</td>
              <td></td>
            </tr>
            """
        )

    csv_link = html.escape(relpath_for_html(out_csv.name))

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux review manifest 001F</title>
<style>
  :root {{
    --bg: #111318;
    --panel: #181b22;
    --line: #2b303b;
    --text: #edf0f7;
    --muted: #9ea7b8;
    --accent: #83d4ff;
  }}
  body {{
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  h1 {{
    margin: 0 0 8px;
    font-size: 24px;
  }}
  .meta {{
    color: var(--muted);
    margin-bottom: 20px;
  }}
  a {{
    color: var(--accent);
    text-decoration: none;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    border: 1px solid var(--line);
  }}
  th, td {{
    border-bottom: 1px solid var(--line);
    padding: 8px 10px;
    vertical-align: top;
    font-size: 13px;
  }}
  th {{
    position: sticky;
    top: 0;
    background: #20242e;
    z-index: 2;
    text-align: left;
  }}
  code {{
    color: #d7e4ff;
    font-size: 12px;
  }}
  img {{
    max-width: 220px;
    max-height: 140px;
    border-radius: 8px;
    border: 1px solid var(--line);
  }}
  .preview {{
    min-width: 230px;
  }}
  .review {{
    color: #ffd37a;
    white-space: nowrap;
  }}
</style>
</head>
<body>
  <h1>TTFlux review manifest 001F</h1>
  <div class="meta">
    Segments détectés : {len(segments)} · CSV : <a href="{csv_link}">{html.escape(out_csv.name)}</a>
  </div>

  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>clip_id</th>
        <th>segment</th>
        <th>frames</th>
        <th>points</th>
        <th>score</th>
        <th>density</th>
        <th>travel</th>
        <th>x/y range</th>
        <th>mp4</th>
        <th>png</th>
        <th>csv</th>
        <th>preview</th>
        <th>review</th>
        <th>notes</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""

    out_html.write_text(doc, encoding="utf-8")


def build_review_manifest(run_dir: Path, out_csv: Path, out_html: Path | None) -> None:
    run_dir = run_dir.resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir does not exist: {run_dir}")

    segments: dict[tuple[str, str], SegmentReview] = {}

    all_files = [p for p in run_dir.rglob("*") if p.is_file()]

    csv_files = [p for p in all_files if p.suffix.lower() in CSV_EXTS]
    media_files = [p for p in all_files if p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS]

    print(f"[001F] run_dir={run_dir}")
    print(f"[001F] csv_files={len(csv_files)} media_files={len(media_files)}")

    for path in csv_files:
        rel = str(path.relative_to(run_dir))
        fieldnames, rows = read_csv_rows(path)

        if not fieldnames or not rows:
            continue

        was_summary = ingest_summary_csv(path, rel, fieldnames, rows, segments, run_dir)
        if not was_summary:
            ingest_point_csv(path, rel, fieldnames, rows, segments, run_dir)

    for path in media_files:
        rel = str(path.relative_to(run_dir))
        clip_id, segment_idx = infer_key_from_path(path, run_dir)
        seg = get_or_create(segments, clip_id, segment_idx)

        suffix = path.suffix.lower()
        name_low = path.name.lower()

        if suffix in VIDEO_EXTS:
            append_unique(seg.mp4, rel)
        elif suffix in IMAGE_EXTS:
            # Prefer overlay/contact style images, but include any image under the inferred segment.
            append_unique(seg.overlay_png, rel)

    ordered = sorted(segments.values(), key=best_sort_key)

    for seg in ordered:
        seg.mp4 = sorted(set(seg.mp4))
        seg.overlay_png = sorted(set(seg.overlay_png))
        seg.csv_files = sorted(set(seg.csv_files))

    write_manifest_csv(out_csv, ordered)

    if out_html:
        write_manifest_html(out_html, ordered, out_csv)

    print(f"[001F] segments={len(ordered)}")
    print(f"[001F] wrote CSV : {out_csv}")
    if out_html:
        print(f"[001F] wrote HTML: {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-html", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "review_manifest_001F.csv"
    out_html = Path(args.out_html) if args.out_html else run_dir / "review_manifest_001F.html"

    build_review_manifest(run_dir=run_dir, out_csv=out_csv, out_html=out_html)


if __name__ == "__main__":
    main()