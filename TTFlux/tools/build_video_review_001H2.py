# PATCH 001H2 - TTFlux video review interface
#
# Reads review_manifest_001G_trajectory.csv.
# Builds an HTML review interface with:
# - browser-compatible transcoded MP4 videos
# - trajectory SVG beside each video
# - human labeling controls
# - localStorage persistence
# - CSV export
#
# Requires ffmpeg available in PATH.
#
# Usage:
#   python tools\build_video_review_001H2.py ^
#     --run-dir runs\batch_001E ^
#     --manifest runs\batch_001E\review_manifest_001G_trajectory.csv ^
#     --out-html runs\batch_001E\video_review_001H2.html ^
#     --media-dir runs\batch_001E\review_media_001H2

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


FRAME_COLS = ["frame", "frame_idx", "frame_index", "f"]
X_COLS = ["x", "ball_x", "cx", "center_x"]
Y_COLS = ["y", "ball_y", "cy", "center_y"]


def norm_col(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def first_existing_col(fieldnames: list[str], candidates: list[str]) -> str | None:
    normalized = {norm_col(c): c for c in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip().replace(",", ".")
    if not text or text.lower() == "nan":
        return None

    try:
        val = float(text)
    except ValueError:
        return None

    if not math.isfinite(val):
        return None

    return val


def safe_int(value: Any, default: int = 0) -> int:
    val = safe_float(value)
    if val is None:
        return default
    return int(round(val))


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


def resolve_path(run_dir: Path, value: Any) -> Path | None:
    text = clean_first_path(value)
    if not text:
        return None

    text = text.replace("\\", "/")
    path = Path(text)

    if path.is_absolute():
        return path

    return run_dir / path


def slugify(value: str) -> str:
    text = value.strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def rel_from_html(out_html: Path, target: Path) -> str:
    try:
        rel = target.resolve().relative_to(out_html.parent.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(target.resolve()).replace("\\", "/")


def run_ffmpeg_transcode(src: Path, dst: Path, force: bool = False) -> tuple[bool, str]:
    if not src.exists():
        return False, f"missing source: {src}"

    if dst.exists() and not force:
        return True, "cached"

    dst.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg not found in PATH"

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        "scale='min(960,iw)':-2",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        return True, "transcoded"

    err = (result.stderr or result.stdout or "").strip()

    # Fallback for ffmpeg builds where libx264 is unavailable.
    cmd_fallback = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        "scale='min(960,iw)':-2",
        "-an",
        "-c:v",
        "h264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]

    result2 = subprocess.run(cmd_fallback, capture_output=True, text=True)

    if result2.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        return True, "transcoded_fallback_h264"

    err2 = (result2.stderr or result2.stdout or "").strip()

    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass

    return False, err2 or err or "ffmpeg failed"


def load_points(csv_path: Path | None) -> list[tuple[int, float, float]]:
    if csv_path is None or not csv_path.exists():
        return []

    try:
        fieldnames, rows = read_csv_rows(csv_path)
    except Exception:
        return []

    frame_col = first_existing_col(fieldnames, FRAME_COLS)
    x_col = first_existing_col(fieldnames, X_COLS)
    y_col = first_existing_col(fieldnames, Y_COLS)

    if not frame_col or not x_col or not y_col:
        return []

    buckets: dict[int, list[tuple[float, float]]] = {}

    for row in rows:
        frame = safe_float(row.get(frame_col))
        x = safe_float(row.get(x_col))
        y = safe_float(row.get(y_col))

        if frame is None or x is None or y is None:
            continue

        buckets.setdefault(int(round(frame)), []).append((x, y))

    points = []
    for frame in sorted(buckets):
        vals = buckets[frame]
        x = sum(v[0] for v in vals) / len(vals)
        y = sum(v[1] for v in vals) / len(vals)
        points.append((frame, x, y))

    return points


def local_bounds(points: list[tuple[int, float, float]]) -> tuple[float, float, float, float]:
    if not points:
        return 0.0, 1.0, 0.0, 1.0

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    if max_x - min_x < 1:
        max_x += 1

    if max_y - min_y < 1:
        max_y += 1

    pad_x = max(8.0, (max_x - min_x) * 0.18)
    pad_y = max(8.0, (max_y - min_y) * 0.18)

    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def make_svg(points: list[tuple[int, float, float]]) -> str:
    width = 420
    height = 235
    pad = 18

    if not points:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="traj">'
            f'<rect x="0" y="0" width="{width}" height="{height}" rx="14" class="svg-bg"/>'
            f'<text x="18" y="120" class="svg-caption">missing trajectory</text>'
            f"</svg>"
        )

    min_x, max_x, min_y, max_y = local_bounds(points)
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad

    def sx(x: float) -> float:
        return pad + ((x - min_x) / max(1e-9, max_x - min_x)) * usable_w

    def sy(y: float) -> float:
        return pad + ((y - min_y) / max(1e-9, max_y - min_y)) * usable_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="traj">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="14" class="svg-bg"/>',
        f'<rect x="{pad}" y="{pad}" width="{usable_w}" height="{usable_h}" rx="10" class="svg-frame"/>',
    ]

    for a, b in zip(points, points[1:]):
        f0, x0, y0 = a
        f1, x1, y1 = b

        gap = max(1, f1 - f0)
        cls = "seg"

        if gap >= 18:
            cls = "seg gap-huge"
        elif gap >= 8:
            cls = "seg gap-big"
        elif gap >= 4:
            cls = "seg gap-small"

        parts.append(
            '<line '
            f'x1="{sx(x0):.2f}" y1="{sy(y0):.2f}" '
            f'x2="{sx(x1):.2f}" y2="{sy(y1):.2f}" '
            f'class="{cls}">'
            f"<title>gap {gap} frames</title>"
            f"</line>"
        )

    for idx, (frame, x, y) in enumerate(points):
        cls = "pt"
        r = 2.5

        if idx == 0:
            cls = "pt start"
            r = 5.0
        elif idx == len(points) - 1:
            cls = "pt end"
            r = 5.0

        parts.append(
            f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="{r:.1f}" class="{cls}">'
            f'<title>frame {frame} | x={x:.1f} y={y:.1f}</title>'
            f"</circle>"
        )

    parts.append(
        f'<text x="18" y="{height - 12}" class="svg-caption">'
        f'f{points[0][0]} → f{points[-1][0]} · {len(points)} pts · bleu=début · vert=fin'
        f"</text>"
    )
    parts.append("</svg>")

    return "".join(parts)


def risk_class(row: dict[str, str]) -> str:
    return str(row.get("review_guess_001G", "unknown")).replace("_", "-")


def make_metric(label: str, value: Any) -> str:
    return f'<span>{html.escape(label)} <b>{html.escape(str(value))}</b></span>'


def make_video_tag(video_rel: str, original_rel: str, ok: bool, msg: str) -> str:
    if ok:
        return f"""
        <video controls muted loop preload="metadata" playsinline>
          <source src="{html.escape(video_rel)}" type="video/mp4">
          Vidéo non lisible par le navigateur.
        </video>
        """

    return f"""
    <div class="video-missing">
      <strong>Vidéo non transcodée</strong>
      <p>{html.escape(msg)}</p>
      <a href="{html.escape(original_rel)}">ouvrir source</a>
    </div>
    """


def build_html(
    rows: list[dict[str, str]],
    out_html: Path,
    media_dir: Path,
    video_status: dict[str, dict[str, str]],
    point_map: dict[str, list[tuple[int, float, float]]],
) -> None:
    cards = []

    state_rows = []

    for row in rows:
        review_id = row.get("review_id", "")
        points = point_map.get(review_id, [])
        svg = make_svg(points)

        status = video_status.get(review_id, {})
        video_rel = status.get("video_rel", "")
        original_rel = status.get("original_rel", "")
        video_ok = status.get("ok", "0") == "1"
        video_msg = status.get("msg", "")

        guess = row.get("review_guess_001G", "")
        cls = risk_class(row)

        card = f"""
        <article class="card {html.escape(cls)}" data-review-id="{html.escape(review_id)}" data-guess="{html.escape(guess)}">
          <header>
            <div>
              <div class="rank">#{html.escape(row.get("triage_rank", ""))} · {html.escape(review_id)}</div>
              <h2>{html.escape(row.get("clip_id", ""))}</h2>
              <p>{html.escape(row.get("segment_name", ""))}</p>
            </div>
            <div class="risk">
              <strong>{html.escape(row.get("risk_score_001G", ""))}</strong>
              <span>risk</span>
            </div>
          </header>

          <section class="review-pair">
            <div class="video-box">
              {make_video_tag(video_rel, original_rel, video_ok, video_msg)}
            </div>
            <div class="traj-box">
              {svg}
            </div>
          </section>

          <section class="metrics">
            {make_metric("guess", row.get("review_guess_001G", ""))}
            {make_metric("smooth", row.get("smoothness_score_001G", ""))}
            {make_metric("density", row.get("traj_density", ""))}
            {make_metric("gap max", row.get("max_frame_gap", ""))}
            {make_metric("speed med/p95/max", f'{row.get("speed_median", "")}/{row.get("speed_p95", "")}/{row.get("speed_max", "")}')}
            {make_metric("turns", row.get("hard_turn_count", ""))}
          </section>

          <p class="flags">{html.escape(row.get("risk_flags_001G", ""))}</p>

          <section class="guide">
            <b>À juger :</b>
            la courbe suit-elle la vraie balle visible dans la vidéo ?
            Un gros trou peut rester acceptable si la balle est retrouvée au bon endroit.
            C’est faux si la piste saute sur raquette, reflet, joueur, filet, logo ou bord de table.
          </section>

          <section class="manual">
            <button class="quick" data-label="gold_keep">gold_keep</button>
            <button class="quick" data-label="ok_ball">ok_ball</button>
            <button class="quick" data-label="partial_ball">partial_ball</button>
            <button class="quick danger" data-label="false_track">false_track</button>
            <button class="quick" data-label="unclear">unclear</button>

            <select class="manual-label">
              <option value="">à juger</option>
              <option value="gold_keep">gold_keep</option>
              <option value="ok_ball">ok_ball</option>
              <option value="partial_ball">partial_ball</option>
              <option value="false_track">false_track</option>
              <option value="unclear">unclear</option>
            </select>

            <textarea class="manual-notes" placeholder="notes : balle ok, trou acceptable, reflet, raquette, filet, trop court, etc."></textarea>
          </section>
        </article>
        """

        cards.append(card)

        state_rows.append(
            {
                "review_id": row.get("review_id", ""),
                "triage_rank": row.get("triage_rank", ""),
                "clip_id": row.get("clip_id", ""),
                "segment_idx": row.get("segment_idx", ""),
                "segment_name": row.get("segment_name", ""),
                "risk_score_001G": row.get("risk_score_001G", ""),
                "review_guess_001G": row.get("review_guess_001G", ""),
                "mp4": row.get("mp4", ""),
                "csv": row.get("csv", ""),
            }
        )

    counts: dict[str, int] = {}
    for row in rows:
        guess = row.get("review_guess_001G", "unknown")
        counts[guess] = counts.get(guess, 0) + 1

    counts_text = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    state_json = json.dumps(state_rows, ensure_ascii=False)

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux video review 001H2</title>
<style>
  :root {{
    --bg: #0f1117;
    --panel: #181b22;
    --panel2: #20242e;
    --line: #2b303b;
    --text: #eef2fa;
    --muted: #9ea7b8;
    --accent: #83d4ff;
    --risk: #ff8a8a;
    --medium: #ffd37a;
    --check: #99bcff;
    --ok: #9effb3;
  }}
  * {{
    box-sizing: border-box;
  }}
  body {{
    margin: 0;
    padding: 22px;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  h1 {{
    margin: 0 0 6px;
    font-size: 26px;
  }}
  .meta {{
    color: var(--muted);
    margin-bottom: 16px;
  }}
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 20;
    padding: 12px;
    margin-bottom: 16px;
    background: rgba(24, 27, 34, 0.96);
    border: 1px solid var(--line);
    border-radius: 14px;
    backdrop-filter: blur(8px);
  }}
  button, input, select, textarea {{
    background: #12151d;
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 8px 10px;
    font: inherit;
  }}
  button {{
    cursor: pointer;
  }}
  button:hover {{
    border-color: var(--accent);
  }}
  input {{
    min-width: 280px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 18px;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    overflow: hidden;
  }}
  .card.high-risk {{
    border-color: rgba(255, 138, 138, 0.75);
  }}
  .card.medium-risk {{
    border-color: rgba(255, 211, 122, 0.55);
  }}
  .card.check {{
    border-color: rgba(153, 188, 255, 0.55);
  }}
  .card.plausible {{
    border-color: rgba(158, 255, 179, 0.35);
  }}
  header {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 14px;
    background: var(--panel2);
    border-bottom: 1px solid var(--line);
  }}
  .rank {{
    color: var(--muted);
    font-size: 12px;
  }}
  h2 {{
    margin: 4px 0 2px;
    font-size: 17px;
  }}
  header p {{
    margin: 0;
    color: var(--muted);
    font-size: 13px;
  }}
  .risk {{
    min-width: 70px;
    text-align: center;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 6px 8px;
    background: #11141b;
  }}
  .risk strong {{
    display: block;
    font-size: 28px;
  }}
  .risk span {{
    color: var(--muted);
    font-size: 11px;
  }}
  .review-pair {{
    display: grid;
    grid-template-columns: minmax(420px, 1.35fr) minmax(360px, 0.85fr);
    gap: 12px;
    padding: 12px;
  }}
  video {{
    width: 100%;
    max-height: 420px;
    background: #080a0f;
    border: 1px solid var(--line);
    border-radius: 14px;
    display: block;
  }}
  .video-missing {{
    min-height: 260px;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: #11141b;
    padding: 18px;
    color: var(--muted);
  }}
  .traj {{
    width: 100%;
    height: auto;
    display: block;
    border: 1px solid var(--line);
    border-radius: 14px;
  }}
  .svg-bg {{
    fill: #10131b;
  }}
  .svg-frame {{
    fill: rgba(255,255,255,0.025);
    stroke: rgba(255,255,255,0.08);
  }}
  .seg {{
    stroke: #dfe8ff;
    stroke-width: 2.5;
    stroke-linecap: round;
    opacity: 0.9;
  }}
  .gap-small {{
    stroke: #ffd37a;
    stroke-dasharray: 4 4;
  }}
  .gap-big {{
    stroke: #ffac7a;
    stroke-dasharray: 7 5;
    stroke-width: 3;
  }}
  .gap-huge {{
    stroke: #ff7070;
    stroke-dasharray: 9 6;
    stroke-width: 3.4;
  }}
  .pt {{
    fill: #edf0f7;
    opacity: 0.95;
  }}
  .start {{
    fill: #8ee8ff;
  }}
  .end {{
    fill: #9effb3;
  }}
  .svg-caption {{
    fill: #9ea7b8;
    font-size: 12px;
  }}
  .metrics {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 0 12px 10px;
  }}
  .metrics span {{
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 4px 8px;
    color: var(--muted);
    font-size: 12px;
  }}
  .metrics b {{
    color: var(--text);
  }}
  .flags {{
    margin: 0 12px 10px;
    color: var(--muted);
    font-size: 12px;
  }}
  .guide {{
    margin: 0 12px 12px;
    padding: 10px 12px;
    color: #d6deee;
    background: rgba(131, 212, 255, 0.06);
    border: 1px solid rgba(131, 212, 255, 0.18);
    border-radius: 12px;
    font-size: 13px;
    line-height: 1.35;
  }}
  .manual {{
    display: grid;
    grid-template-columns: repeat(5, minmax(110px, auto));
    gap: 8px;
    padding: 12px;
    background: rgba(255,255,255,0.025);
    border-top: 1px solid var(--line);
  }}
  .manual select,
  .manual textarea {{
    grid-column: 1 / -1;
  }}
  textarea {{
    width: 100%;
    min-height: 66px;
    resize: vertical;
  }}
  .quick.danger {{
    border-color: rgba(255, 138, 138, 0.6);
  }}
  .hidden {{
    display: none;
  }}
  a {{
    color: var(--accent);
    text-decoration: none;
  }}
  @media (max-width: 980px) {{
    .review-pair {{
      grid-template-columns: 1fr;
    }}
    .manual {{
      grid-template-columns: 1fr 1fr;
    }}
  }}
</style>
</head>
<body>
  <h1>TTFlux video review 001H2</h1>
  <div class="meta">
    Segments : {len(rows)} · {html.escape(counts_text)} · vidéos web dans
    <code>{html.escape(media_dir.name)}</code>
  </div>

  <div class="toolbar">
    <button data-filter="all">tout</button>
    <button data-filter="high_risk">high</button>
    <button data-filter="medium_risk">medium</button>
    <button data-filter="check">check</button>
    <button data-filter="plausible">plausible</button>
    <button id="only-unlabeled">non jugés</button>
    <input id="search" placeholder="filtrer clip, segment, flags, label..." />
    <button id="export">export labels CSV</button>
    <button id="clear">vider labels locaux</button>
  </div>

  <main class="grid">
    {''.join(cards)}
  </main>

<script>
const reviewRows = {state_json};
const storageKey = "ttflux_review_001H2_labels";
let activeFilter = "all";
let onlyUnlabeled = false;

function loadState() {{
  try {{
    return JSON.parse(localStorage.getItem(storageKey) || "{{}}");
  }} catch (err) {{
    return {{}};
  }}
}}

function saveState(state) {{
  localStorage.setItem(storageKey, JSON.stringify(state));
}}

function syncFromStorage() {{
  const state = loadState();
  document.querySelectorAll(".card").forEach(card => {{
    const id = card.dataset.reviewId;
    const item = state[id] || {{}};
    card.querySelector(".manual-label").value = item.manual_label || "";
    card.querySelector(".manual-notes").value = item.manual_notes || "";
  }});
}}

function setCardLabel(card, labelValue) {{
  const id = card.dataset.reviewId;
  const state = loadState();
  const notes = card.querySelector(".manual-notes").value || "";
  state[id] = {{
    manual_label: labelValue,
    manual_notes: notes
  }};
  saveState(state);
  card.querySelector(".manual-label").value = labelValue;
  applyFilter();
}}

function bindInputs() {{
  document.querySelectorAll(".card").forEach(card => {{
    const id = card.dataset.reviewId;
    const label = card.querySelector(".manual-label");
    const notes = card.querySelector(".manual-notes");

    function update() {{
      const state = loadState();
      state[id] = {{
        manual_label: label.value,
        manual_notes: notes.value
      }};
      saveState(state);
      applyFilter();
    }}

    label.addEventListener("change", update);
    notes.addEventListener("input", update);

    card.querySelectorAll(".quick").forEach(button => {{
      button.addEventListener("click", () => {{
        setCardLabel(card, button.dataset.label || "");
      }});
    }});
  }});
}}

function applyFilter() {{
  const q = document.getElementById("search").value.trim().toLowerCase();
  const state = loadState();

  document.querySelectorAll(".card").forEach(card => {{
    const guess = (card.dataset.guess || "").toLowerCase();
    const text = card.innerText.toLowerCase();
    const label = (state[card.dataset.reviewId]?.manual_label || "").toLowerCase();

    const okKind = activeFilter === "all" || guess === activeFilter;
    const okSearch = !q || text.includes(q) || label.includes(q);
    const okLabel = !onlyUnlabeled || !label;

    card.classList.toggle("hidden", !(okKind && okSearch && okLabel));
  }});
}}

function csvEscape(value) {{
  const text = String(value ?? "");
  if (/[",\\n\\r;]/.test(text)) {{
    return '"' + text.replaceAll('"', '""') + '"';
  }}
  return text;
}}

function exportCsv() {{
  const state = loadState();
  const columns = [
    "review_id",
    "triage_rank",
    "clip_id",
    "segment_idx",
    "segment_name",
    "risk_score_001G",
    "review_guess_001G",
    "manual_label",
    "manual_notes",
    "mp4",
    "csv"
  ];

  const lines = [columns.join(",")];

  for (const row of reviewRows) {{
    const item = state[row.review_id] || {{}};
    const out = {{
      ...row,
      manual_label: item.manual_label || "",
      manual_notes: item.manual_notes || ""
    }};
    lines.push(columns.map(col => csvEscape(out[col] || "")).join(","));
  }}

  const blob = new Blob([lines.join("\\n")], {{type: "text/csv;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "review_labels_001H2_export.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}

document.querySelectorAll("[data-filter]").forEach(button => {{
  button.addEventListener("click", () => {{
    activeFilter = button.dataset.filter;
    applyFilter();
  }});
}});

document.getElementById("only-unlabeled").addEventListener("click", () => {{
  onlyUnlabeled = !onlyUnlabeled;
  applyFilter();
}});

document.getElementById("search").addEventListener("input", applyFilter);
document.getElementById("export").addEventListener("click", exportCsv);
document.getElementById("clear").addEventListener("click", () => {{
  localStorage.removeItem(storageKey);
  syncFromStorage();
  applyFilter();
}});

bindInputs();
syncFromStorage();
applyFilter();
</script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")


def build_review(run_dir: Path, manifest: Path, out_html: Path, media_dir: Path, force: bool) -> None:
    run_dir = run_dir.resolve()
    manifest = manifest.resolve()
    out_html = out_html.resolve()
    media_dir = media_dir.resolve()

    _, rows = read_csv_rows(manifest)

    def sort_key(row: dict[str, str]) -> tuple[int, int, str]:
        rank = safe_int(row.get("triage_rank"), 9999)
        risk = safe_int(row.get("risk_score_001G"), 0)
        return rank, -risk, row.get("clip_id", "")

    rows = sorted(rows, key=sort_key)

    video_status: dict[str, dict[str, str]] = {}
    point_map: dict[str, list[tuple[int, float, float]]] = {}

    ok_count = 0
    fail_count = 0

    for row in rows:
        review_id = row.get("review_id", "")
        segment_name = row.get("segment_name", "")
        safe_name = slugify(f"{review_id}_{segment_name}")
        dst_video = media_dir / f"{safe_name}.mp4"

        src_video = resolve_path(run_dir, row.get("mp4", ""))
        src_csv = resolve_path(run_dir, row.get("csv", ""))

        points = load_points(src_csv)
        point_map[review_id] = points

        if src_video is None:
            video_status[review_id] = {
                "ok": "0",
                "msg": "missing mp4 path",
                "video_rel": "",
                "original_rel": "",
            }
            fail_count += 1
            continue

        ok, msg = run_ffmpeg_transcode(src_video, dst_video, force=force)

        if ok:
            ok_count += 1
        else:
            fail_count += 1

        video_status[review_id] = {
            "ok": "1" if ok else "0",
            "msg": msg,
            "video_rel": rel_from_html(out_html, dst_video),
            "original_rel": rel_from_html(out_html, src_video),
        }

    build_html(
        rows=rows,
        out_html=out_html,
        media_dir=media_dir,
        video_status=video_status,
        point_map=point_map,
    )

    print(f"[001H2] run_dir       : {run_dir}")
    print(f"[001H2] manifest      : {manifest}")
    print(f"[001H2] segments      : {len(rows)}")
    print(f"[001H2] videos ok     : {ok_count}")
    print(f"[001H2] videos failed : {fail_count}")
    print(f"[001H2] media_dir     : {media_dir}")
    print(f"[001H2] wrote HTML    : {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--manifest", default="runs/batch_001E/review_manifest_001G_trajectory.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/video_review_001H2.html")
    parser.add_argument("--media-dir", default="runs/batch_001E/review_media_001H2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    build_review(
        run_dir=Path(args.run_dir),
        manifest=Path(args.manifest),
        out_html=Path(args.out_html),
        media_dir=Path(args.media_dir),
        force=args.force,
    )


if __name__ == "__main__":
    main()