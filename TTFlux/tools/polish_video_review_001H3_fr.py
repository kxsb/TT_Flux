# PATCH 001H3 - Polish TTFlux review HTML in French
#
# This does not re-run tracking.
# It rebuilds a friendlier French HTML review interface from:
# - review_manifest_001G_trajectory.csv
# - already transcoded videos in review_media_001H2
#
# Usage:
#   python tools\polish_video_review_001H3_fr.py ^
#     --run-dir runs\batch_001E ^
#     --manifest runs\batch_001E\review_manifest_001G_trajectory.csv ^
#     --media-dir runs\batch_001E\review_media_001H2 ^
#     --out-html runs\batch_001E\video_review_001H3_fr.html

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any


FRAME_COLS = ["frame", "frame_idx", "frame_index", "f"]
X_COLS = ["x", "ball_x", "cx", "center_x"]
Y_COLS = ["y", "ball_y", "cy", "center_y"]


LABELS = {
    "gold_keep": {
        "fr": "Exemple excellent",
        "short": "excellent",
        "help": "La balle est clairement suivie, la trajectoire est propre. À garder comme exemple positif de référence.",
    },
    "ok_ball": {
        "fr": "Balle correcte",
        "short": "correct",
        "help": "C’est bien la balle. Il peut y avoir un petit bruit, mais le segment reste utilisable.",
    },
    "partial_ball": {
        "fr": "Balle partielle",
        "short": "partiel",
        "help": "La piste suit parfois la balle, mais avec trous, pertes ou reprises douteuses. Utilisable pour diagnostic, pas comme vérité propre.",
    },
    "false_track": {
        "fr": "Fausse piste",
        "short": "faux",
        "help": "La piste suit autre chose : reflet, raquette, main, joueur, filet, logo, bord de table, bruit vidéo.",
    },
    "unclear": {
        "fr": "Impossible à trancher",
        "short": "incertain",
        "help": "La vidéo ne permet pas de décider honnêtement. Balle trop invisible, compression, ambiguïté forte.",
    },
}


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
    p = Path(text)
    if p.is_absolute():
        return p
    return run_dir / p


def rel_from_html(out_html: Path, target: Path) -> str:
    try:
        rel = target.resolve().relative_to(out_html.parent.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(target.resolve()).replace("\\", "/")


def slugify(value: str) -> str:
    text = value.strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


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
            f'<text x="18" y="120" class="svg-caption">trajectoire absente</text>'
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
            f"<title>trou de {gap} frames</title>"
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
        f'bleu = début · vert = fin · pointillé = trou entre frames'
        f"</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def metric(label: str, value: Any) -> str:
    return f'<span>{html.escape(label)} <b>{html.escape(str(value))}</b></span>'


def guess_fr(value: str) -> str:
    mapping = {
        "high_risk": "risque fort",
        "medium_risk": "risque moyen",
        "check": "à vérifier",
        "plausible": "plausible",
        "missing": "manquant",
    }
    return mapping.get(value, value)


def make_video_tag(video_rel: str) -> str:
    if not video_rel:
        return """
        <div class="video-missing">
          <strong>Vidéo absente</strong>
          <p>Relance d’abord 001H2 pour générer les vidéos compatibles navigateur.</p>
        </div>
        """

    return f"""
    <video controls muted loop preload="metadata" playsinline>
      <source src="{html.escape(video_rel)}" type="video/mp4">
      Vidéo non lisible par le navigateur.
    </video>
    """


def build_review(
    run_dir: Path,
    manifest: Path,
    media_dir: Path,
    out_html: Path,
) -> None:
    run_dir = run_dir.resolve()
    manifest = manifest.resolve()
    media_dir = media_dir.resolve()
    out_html = out_html.resolve()

    _, rows = read_csv_rows(manifest)

    def sort_key(row: dict[str, str]) -> tuple[int, int, str]:
        return (
            safe_int(row.get("triage_rank"), 9999),
            -safe_int(row.get("risk_score_001G"), 0),
            row.get("clip_id", ""),
        )

    rows = sorted(rows, key=sort_key)

    cards = []
    state_rows = []

    for row in rows:
        review_id = row.get("review_id", "")
        segment_name = row.get("segment_name", "")
        safe_name = slugify(f"{review_id}_{segment_name}")
        video_path = media_dir / f"{safe_name}.mp4"
        video_rel = rel_from_html(out_html, video_path) if video_path.exists() else ""

        csv_path = resolve_path(run_dir, row.get("csv", ""))
        points = load_points(csv_path)
        svg = make_svg(points)

        guess = row.get("review_guess_001G", "")
        cls = guess.replace("_", "-")

        label_buttons = []
        for key, cfg in LABELS.items():
            danger = " danger" if key == "false_track" else ""
            label_buttons.append(
                f'<button class="quick{danger}" data-label="{key}" title="{html.escape(cfg["help"])}">'
                f'{html.escape(cfg["fr"])}</button>'
            )

        label_options = ['<option value="">Non jugé</option>']
        for key, cfg in LABELS.items():
            label_options.append(
                f'<option value="{key}">{html.escape(cfg["fr"])} ({html.escape(key)})</option>'
            )

        cards.append(
            f"""
            <article class="card {html.escape(cls)}" data-review-id="{html.escape(review_id)}" data-guess="{html.escape(guess)}">
              <header>
                <div>
                  <div class="rank">Priorité #{html.escape(row.get("triage_rank", ""))} · {html.escape(review_id)}</div>
                  <h2>{html.escape(row.get("clip_id", ""))}</h2>
                  <p>{html.escape(segment_name)}</p>
                </div>
                <div class="risk">
                  <strong>{html.escape(row.get("risk_score_001G", ""))}</strong>
                  <span>risque</span>
                </div>
              </header>

              <section class="decision-help">
                <div>
                  <b>Question à te poser :</b>
                  est-ce que la piste dessinée suit la vraie balle visible dans la vidéo ?
                </div>
                <div class="legend">
                  <span><i class="dot blue"></i> début</span>
                  <span><i class="dot green"></i> fin</span>
                  <span><i class="line red"></i> gros trou</span>
                </div>
              </section>

              <section class="review-pair">
                <div class="video-box">
                  {make_video_tag(video_rel)}
                </div>
                <div class="traj-box">
                  {svg}
                </div>
              </section>

              <section class="metrics">
                {metric("estimation moteur", guess_fr(guess))}
                {metric("fluidité", row.get("smoothness_score_001G", ""))}
                {metric("densité", row.get("traj_density", ""))}
                {metric("plus gros trou", row.get("max_frame_gap", ""))}
                {metric("vitesse med/p95/max", f'{row.get("speed_median", "")}/{row.get("speed_p95", "")}/{row.get("speed_max", "")}')}
                {metric("virages durs", row.get("hard_turn_count", ""))}
              </section>

              <details class="flags-box">
                <summary>Pourquoi le moteur doute ?</summary>
                <p>{html.escape(row.get("risk_flags_001G", ""))}</p>
              </details>

              <section class="label-guide">
                <h3>Choisir un label</h3>
                <ul>
                  <li><b>Exemple excellent</b> : balle nette, trajectoire propre, bon positif de référence.</li>
                  <li><b>Balle correcte</b> : c’est bien la balle, même avec un petit bruit.</li>
                  <li><b>Balle partielle</b> : utile mais imparfait, trous ou reprise douteuse.</li>
                  <li><b>Fausse piste</b> : reflet, raquette, main, joueur, filet, logo, bord de table.</li>
                  <li><b>Impossible à trancher</b> : vidéo trop ambiguë pour décider.</li>
                </ul>
              </section>

              <section class="manual">
                <div class="quick-row">
                  {''.join(label_buttons)}
                </div>

                <select class="manual-label">
                  {''.join(label_options)}
                </select>

                <textarea class="manual-notes" placeholder="Note courte, exemple : balle ok malgré trou, reflet sur table, suit la raquette, trop compressé, bon positif, etc."></textarea>
              </section>
            </article>
            """
        )

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
        g = row.get("review_guess_001G", "unknown")
        counts[g] = counts.get(g, 0) + 1

    counts_text = " · ".join(f"{guess_fr(k)} : {v}" for k, v in sorted(counts.items()))
    state_json = json.dumps(state_rows, ensure_ascii=False)
    labels_json = json.dumps(LABELS, ensure_ascii=False)

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux revue vidéo 001H3</title>
<style>
  :root {{
    --bg: #0f1117;
    --panel: #181b22;
    --panel2: #20242e;
    --line: #2b303b;
    --text: #eef2fa;
    --muted: #9ea7b8;
    --accent: #83d4ff;
    --danger: #ff8a8a;
    --warn: #ffd37a;
    --ok: #9effb3;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 22px;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  h1 {{ margin: 0 0 6px; font-size: 26px; }}
  .meta {{ color: var(--muted); margin-bottom: 16px; }}
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
  button {{ cursor: pointer; }}
  button:hover {{ border-color: var(--accent); }}
  input {{ min-width: 280px; }}
  .progress {{
    margin-left: auto;
    color: var(--muted);
    font-size: 13px;
  }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    overflow: hidden;
  }}
  .card.high-risk {{ border-color: rgba(255, 138, 138, 0.75); }}
  .card.medium-risk {{ border-color: rgba(255, 211, 122, 0.55); }}
  .card.check {{ border-color: rgba(153, 188, 255, 0.55); }}
  .card.plausible {{ border-color: rgba(158, 255, 179, 0.35); }}
  header {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 14px;
    background: var(--panel2);
    border-bottom: 1px solid var(--line);
  }}
  .rank {{ color: var(--muted); font-size: 12px; }}
  h2 {{ margin: 4px 0 2px; font-size: 17px; }}
  header p {{ margin: 0; color: var(--muted); font-size: 13px; }}
  .risk {{
    min-width: 70px;
    text-align: center;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 6px 8px;
    background: #11141b;
  }}
  .risk strong {{ display: block; font-size: 28px; }}
  .risk span {{ color: var(--muted); font-size: 11px; }}
  .decision-help {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 10px;
    margin: 12px;
    padding: 10px 12px;
    color: #d6deee;
    background: rgba(131, 212, 255, 0.06);
    border: 1px solid rgba(131, 212, 255, 0.18);
    border-radius: 12px;
    font-size: 13px;
  }}
  .legend {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 999px;
    margin-right: 5px;
  }}
  .blue {{ background: #8ee8ff; }}
  .green {{ background: #9effb3; }}
  .line {{
    display: inline-block;
    width: 24px;
    border-top: 3px dashed #ff7070;
    margin-right: 5px;
    transform: translateY(-3px);
  }}
  .review-pair {{
    display: grid;
    grid-template-columns: minmax(420px, 1.35fr) minmax(360px, 0.85fr);
    gap: 12px;
    padding: 0 12px 12px;
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
  .svg-bg {{ fill: #10131b; }}
  .svg-frame {{ fill: rgba(255,255,255,0.025); stroke: rgba(255,255,255,0.08); }}
  .seg {{ stroke: #dfe8ff; stroke-width: 2.5; stroke-linecap: round; opacity: 0.9; }}
  .gap-small {{ stroke: #ffd37a; stroke-dasharray: 4 4; }}
  .gap-big {{ stroke: #ffac7a; stroke-dasharray: 7 5; stroke-width: 3; }}
  .gap-huge {{ stroke: #ff7070; stroke-dasharray: 9 6; stroke-width: 3.4; }}
  .pt {{ fill: #edf0f7; opacity: 0.95; }}
  .start {{ fill: #8ee8ff; }}
  .end {{ fill: #9effb3; }}
  .svg-caption {{ fill: #9ea7b8; font-size: 12px; }}
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
  .metrics b {{ color: var(--text); }}
  .flags-box {{
    margin: 0 12px 12px;
    color: var(--muted);
    font-size: 12px;
  }}
  .flags-box summary {{ cursor: pointer; color: #d6deee; }}
  .label-guide {{
    margin: 0 12px 12px;
    padding: 10px 12px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: rgba(255,255,255,0.025);
    font-size: 13px;
    color: #d6deee;
  }}
  .label-guide h3 {{ margin: 0 0 6px; font-size: 14px; }}
  .label-guide ul {{ margin: 0; padding-left: 18px; }}
  .label-guide li {{ margin: 3px 0; }}
  .manual {{
    display: grid;
    gap: 8px;
    padding: 12px;
    background: rgba(255,255,255,0.025);
    border-top: 1px solid var(--line);
  }}
  .quick-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .quick.danger {{ border-color: rgba(255, 138, 138, 0.6); }}
  textarea {{ width: 100%; min-height: 66px; resize: vertical; }}
  .hidden {{ display: none; }}
  code {{ color: #d7e4ff; }}
  @media (max-width: 980px) {{
    .review-pair {{ grid-template-columns: 1fr; }}
    .progress {{ margin-left: 0; width: 100%; }}
  }}
</style>
</head>
<body>
  <h1>TTFlux revue vidéo 001H3</h1>
  <div class="meta">
    Segments : {len(rows)} · {html.escape(counts_text)} · labels exportables en CSV
  </div>

  <div class="toolbar">
    <button data-filter="all">Tous</button>
    <button data-filter="high_risk">Risque fort</button>
    <button data-filter="medium_risk">Risque moyen</button>
    <button data-filter="check">À vérifier</button>
    <button data-filter="plausible">Plausibles</button>
    <button id="only-unlabeled">Non jugés</button>
    <input id="search" placeholder="Filtrer : clip, segment, label, note..." />
    <button id="export">Exporter les labels CSV</button>
    <button id="clear">Effacer mes labels locaux</button>
    <span class="progress" id="progress">0 / {len(rows)} jugés</span>
  </div>

  <main class="grid">
    {''.join(cards)}
  </main>

<script>
const reviewRows = {state_json};
const labelInfo = {labels_json};
const storageKey = "ttflux_review_001H3_fr_labels";
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

function updateProgress() {{
  const state = loadState();
  let done = 0;
  const counts = {{}};

  for (const row of reviewRows) {{
    const label = state[row.review_id]?.manual_label || "";
    if (label) {{
      done += 1;
      counts[label] = (counts[label] || 0) + 1;
    }}
  }}

  const detail = Object.entries(counts)
    .map(([k, v]) => `${{labelInfo[k]?.short || k}}: ${{v}}`)
    .join(" · ");

  document.getElementById("progress").textContent =
    `${{done}} / ${{reviewRows.length}} jugés` + (detail ? ` · ${{detail}}` : "");
}}

function syncFromStorage() {{
  const state = loadState();

  document.querySelectorAll(".card").forEach(card => {{
    const id = card.dataset.reviewId;
    const item = state[id] || {{}};
    card.querySelector(".manual-label").value = item.manual_label || "";
    card.querySelector(".manual-notes").value = item.manual_notes || "";
  }});

  updateProgress();
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
  updateProgress();
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
      updateProgress();
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
    const label = (state[card.dataset.reviewId]?.manual_label || "").toLowerCase();
    const notes = (state[card.dataset.reviewId]?.manual_notes || "").toLowerCase();
    const text = card.innerText.toLowerCase();

    const okKind = activeFilter === "all" || guess === activeFilter;
    const okSearch = !q || text.includes(q) || label.includes(q) || notes.includes(q);
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
    "manual_label_fr",
    "manual_notes",
    "mp4",
    "csv"
  ];

  const lines = [columns.join(",")];

  for (const row of reviewRows) {{
    const item = state[row.review_id] || {{}};
    const label = item.manual_label || "";
    const out = {{
      ...row,
      manual_label: label,
      manual_label_fr: labelInfo[label]?.fr || "",
      manual_notes: item.manual_notes || ""
    }};
    lines.push(columns.map(col => csvEscape(out[col] || "")).join(","));
  }}

  const blob = new Blob([lines.join("\\n")], {{type: "text/csv;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "review_labels_001H3_fr_export.csv";
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

    print(f"[001H3] segments   : {len(rows)}")
    print(f"[001H3] media_dir  : {media_dir}")
    print(f"[001H3] wrote HTML : {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--manifest", default="runs/batch_001E/review_manifest_001G_trajectory.csv")
    parser.add_argument("--media-dir", default="runs/batch_001E/review_media_001H2")
    parser.add_argument("--out-html", default="runs/batch_001E/video_review_001H3_fr.html")
    args = parser.parse_args()

    build_review(
        run_dir=Path(args.run_dir),
        manifest=Path(args.manifest),
        media_dir=Path(args.media_dir),
        out_html=Path(args.out_html),
    )


if __name__ == "__main__":
    main()