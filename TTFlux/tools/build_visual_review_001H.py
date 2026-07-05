# PATCH 001H - TTFlux visual trajectory review
#
# Reads review_manifest_001G_trajectory.csv and each segment CSV.
# Generates:
# - visual_review_001H.html with trajectory mini-cards, local labels, notes, export CSV
# - review_labels_001H_template.csv for manual annotation
#
# Usage:
#   python tools\build_visual_review_001H.py ^
#     --run-dir runs\batch_001E ^
#     --manifest runs\batch_001E\review_manifest_001G_trajectory.csv ^
#     --out-html runs\batch_001E\visual_review_001H.html ^
#     --out-labels runs\batch_001E\review_labels_001H_template.csv

from __future__ import annotations

import argparse
import csv
import html
import json
import math
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


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def resolve_segment_csv(run_dir: Path, row: dict[str, str]) -> Path | None:
    text = clean_first_path(row.get("csv", ""))
    if not text:
        return None

    text = text.replace("\\", "/")
    path = Path(text)

    if path.is_absolute():
        return path

    return run_dir / path


def load_points(path: Path | None) -> list[tuple[int, float, float]]:
    if path is None or not path.exists():
        return []

    try:
        fieldnames, rows = read_csv_rows(path)
    except Exception as exc:
        print(f"[WARN] cannot read segment CSV {path}: {exc}")
        return []

    frame_col = first_existing_col(fieldnames, FRAME_COLS)
    x_col = first_existing_col(fieldnames, X_COLS)
    y_col = first_existing_col(fieldnames, Y_COLS)

    if not frame_col or not x_col or not y_col:
        print(f"[WARN] missing frame/x/y columns in {path}")
        return []

    buckets: dict[int, list[tuple[float, float]]] = {}

    for row in rows:
        frame = safe_float(row.get(frame_col))
        x = safe_float(row.get(x_col))
        y = safe_float(row.get(y_col))

        if frame is None or x is None or y is None:
            continue

        frame_i = int(round(frame))
        buckets.setdefault(frame_i, []).append((x, y))

    points = []
    for frame in sorted(buckets):
        vals = buckets[frame]
        x = sum(v[0] for v in vals) / len(vals)
        y = sum(v[1] for v in vals) / len(vals)
        points.append((frame, x, y))

    return points


def compute_global_bounds(all_points: list[list[tuple[int, float, float]]]) -> tuple[float, float, float, float]:
    xs = []
    ys = []

    for points in all_points:
        for _, x, y in points:
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return 0.0, 1280.0, 0.0, 720.0

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    if max_x - min_x < 1:
        max_x = min_x + 1

    if max_y - min_y < 1:
        max_y = min_y + 1

    pad_x = max(8.0, (max_x - min_x) * 0.08)
    pad_y = max(8.0, (max_y - min_y) * 0.08)

    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def make_svg(
    points: list[tuple[int, float, float]],
    bounds: tuple[float, float, float, float],
    width: int = 340,
    height: int = 190,
) -> str:
    if not points:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="traj">'
            f'<rect x="0" y="0" width="{width}" height="{height}" rx="12" class="svg-bg"/>'
            f'<text x="18" y="98" class="svg-missing">missing trajectory</text>'
            f"</svg>"
        )

    min_x, max_x, min_y, max_y = bounds
    pad = 16

    usable_w = width - 2 * pad
    usable_h = height - 2 * pad

    def sx(x: float) -> float:
        return pad + ((x - min_x) / max(1e-9, max_x - min_x)) * usable_w

    def sy(y: float) -> float:
        # Keep image-like coordinates: y grows downward.
        return pad + ((y - min_y) / max(1e-9, max_y - min_y)) * usable_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="traj">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="12" class="svg-bg"/>',
        f'<rect x="{pad}" y="{pad}" width="{usable_w}" height="{usable_h}" rx="8" class="svg-frame"/>',
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
            f'class="{cls}" data-gap="{gap}"/>'
        )

    # Draw points after lines.
    for idx, (frame, x, y) in enumerate(points):
        cls = "pt"
        r = 2.0

        if idx == 0:
            cls = "pt start"
            r = 4.2
        elif idx == len(points) - 1:
            cls = "pt end"
            r = 4.2

        parts.append(
            f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="{r:.1f}" class="{cls}">'
            f'<title>frame {frame} | x={x:.1f} y={y:.1f}</title>'
            f"</circle>"
        )

    first_f = points[0][0]
    last_f = points[-1][0]
    parts.append(f'<text x="16" y="{height - 12}" class="svg-caption">f{first_f} → f{last_f} · {len(points)} pts</text>')
    parts.append("</svg>")

    return "".join(parts)


def href(path_text: Any, label: str) -> str:
    text = clean_first_path(path_text)
    if not text:
        return ""

    url = html.escape(text.replace("\\", "/"))
    return f'<a href="{url}">{html.escape(label)}</a>'


def risk_class(row: dict[str, str]) -> str:
    guess = str(row.get("review_guess_001G", "")).strip().replace("_", "-")
    return guess or "unknown"


def row_sort_key(row: dict[str, str]) -> tuple[int, int, str]:
    rank = safe_int(row.get("triage_rank"), 9999)
    risk = safe_int(row.get("risk_score_001G"), 0)
    return rank, -risk, row.get("clip_id", "")


def build_label_template(rows: list[dict[str, str]], out_labels: Path) -> None:
    columns = [
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
        "csv",
    ]

    label_rows = []
    for row in rows:
        label_rows.append(
            {
                "review_id": row.get("review_id", ""),
                "triage_rank": row.get("triage_rank", ""),
                "clip_id": row.get("clip_id", ""),
                "segment_idx": row.get("segment_idx", ""),
                "segment_name": row.get("segment_name", ""),
                "risk_score_001G": row.get("risk_score_001G", ""),
                "review_guess_001G": row.get("review_guess_001G", ""),
                "manual_label": "",
                "manual_notes": "",
                "mp4": row.get("mp4", ""),
                "csv": row.get("csv", ""),
            }
        )

    write_csv(out_labels, label_rows, columns)


def build_html(
    rows: list[dict[str, str]],
    point_map: dict[str, list[tuple[int, float, float]]],
    bounds: tuple[float, float, float, float],
    out_html: Path,
    out_labels: Path,
) -> None:
    cards = []

    for row in rows:
        review_id = row.get("review_id", "")
        points = point_map.get(review_id, [])

        svg = make_svg(points, bounds)
        cls = risk_class(row)

        risk = html.escape(row.get("risk_score_001G", ""))
        smooth = html.escape(row.get("smoothness_score_001G", ""))
        guess = html.escape(row.get("review_guess_001G", ""))
        flags = html.escape(row.get("risk_flags_001G", ""))

        card = f"""
        <article class="card {html.escape(cls)}" data-review-id="{html.escape(review_id)}" data-guess="{html.escape(guess)}">
          <header>
            <div>
              <div class="rank">#{html.escape(row.get("triage_rank", ""))} · {html.escape(review_id)}</div>
              <h2>{html.escape(row.get("clip_id", ""))}</h2>
              <p class="segment">{html.escape(row.get("segment_name", ""))}</p>
            </div>
            <div class="score">
              <strong>{risk}</strong>
              <span>risk</span>
            </div>
          </header>

          <div class="viz">{svg}</div>

          <div class="metrics">
            <span>guess <b>{guess}</b></span>
            <span>smooth <b>{smooth}</b></span>
            <span>density <b>{html.escape(row.get("traj_density", ""))}</b></span>
            <span>max gap <b>{html.escape(row.get("max_frame_gap", ""))}</b></span>
            <span>speed med/p95/max <b>{html.escape(row.get("speed_median", ""))} / {html.escape(row.get("speed_p95", ""))} / {html.escape(row.get("speed_max", ""))}</b></span>
            <span>turns <b>{html.escape(row.get("hard_turn_count", ""))}</b></span>
          </div>

          <p class="flags">{flags}</p>

          <div class="links">
            {href(row.get("mp4", ""), "ouvrir MP4")}
            {href(row.get("csv", ""), "ouvrir CSV")}
          </div>

          <div class="manual-box">
            <select class="manual-label">
              <option value="">à juger</option>
              <option value="ok_ball">ok_ball</option>
              <option value="partial_ball">partial_ball</option>
              <option value="false_track">false_track</option>
              <option value="unclear">unclear</option>
              <option value="gold_keep">gold_keep</option>
            </select>
            <textarea class="manual-notes" placeholder="notes rapides : bonne balle, trou, reflet, joueur, filet, etc."></textarea>
          </div>
        </article>
        """
        cards.append(card)

    counts: dict[str, int] = {}
    for row in rows:
        guess = row.get("review_guess_001G", "unknown")
        counts[guess] = counts.get(guess, 0) + 1

    counts_text = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items()))

    state_rows = [
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
        for row in rows
    ]

    state_json = json.dumps(state_rows, ensure_ascii=False)

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux visual review 001H</title>
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
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  h1 {{
    margin: 0 0 8px;
    font-size: 26px;
  }}
  .meta {{
    color: var(--muted);
    margin-bottom: 18px;
  }}
  .toolbar {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 18px;
    padding: 12px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    position: sticky;
    top: 0;
    z-index: 10;
  }}
  button, select, textarea, input {{
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
    min-width: 240px;
  }}
  a {{
    color: var(--accent);
    text-decoration: none;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(390px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    overflow: hidden;
  }}
  .card.high-risk {{
    border-color: rgba(255, 138, 138, 0.65);
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
    padding: 14px 14px 10px;
    border-bottom: 1px solid var(--line);
    background: var(--panel2);
  }}
  .rank {{
    color: var(--muted);
    font-size: 12px;
  }}
  h2 {{
    margin: 4px 0 3px;
    font-size: 15px;
    line-height: 1.25;
    word-break: break-word;
  }}
  .segment {{
    margin: 0;
    color: var(--muted);
    font-size: 12px;
    word-break: break-word;
  }}
  .score {{
    min-width: 62px;
    text-align: center;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 6px 8px;
    background: #11141b;
  }}
  .score strong {{
    display: block;
    font-size: 24px;
  }}
  .score span {{
    color: var(--muted);
    font-size: 11px;
  }}
  .viz {{
    padding: 12px;
  }}
  .traj {{
    width: 100%;
    height: auto;
    display: block;
  }}
  .svg-bg {{
    fill: #10131b;
  }}
  .svg-frame {{
    fill: rgba(255, 255, 255, 0.025);
    stroke: rgba(255, 255, 255, 0.08);
  }}
  .seg {{
    stroke: #dfe8ff;
    stroke-width: 2.2;
    stroke-linecap: round;
    opacity: 0.85;
  }}
  .gap-small {{
    stroke: #ffd37a;
    stroke-dasharray: 4 4;
  }}
  .gap-big {{
    stroke: #ffac7a;
    stroke-dasharray: 7 5;
    stroke-width: 2.7;
  }}
  .gap-huge {{
    stroke: #ff7070;
    stroke-dasharray: 9 6;
    stroke-width: 3.1;
  }}
  .pt {{
    fill: #edf0f7;
    opacity: 0.9;
  }}
  .start {{
    fill: #8ee8ff;
  }}
  .end {{
    fill: #9effb3;
  }}
  .svg-caption, .svg-missing {{
    fill: #9ea7b8;
    font-size: 12px;
  }}
  .metrics {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
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
    font-weight: 650;
  }}
  .flags {{
    margin: 0 12px 10px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.35;
  }}
  .links {{
    display: flex;
    gap: 12px;
    padding: 0 12px 12px;
    font-size: 13px;
  }}
  .manual-box {{
    display: grid;
    gap: 8px;
    padding: 12px;
    border-top: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.025);
  }}
  textarea {{
    width: 100%;
    min-height: 58px;
    resize: vertical;
  }}
  .hidden {{
    display: none;
  }}
</style>
</head>
<body>
  <h1>TTFlux visual review 001H</h1>
  <div class="meta">
    Segments : {len(rows)} · {html.escape(counts_text)} · labels template :
    <a href="{html.escape(out_labels.name)}">{html.escape(out_labels.name)}</a>
  </div>

  <div class="toolbar">
    <button data-filter="all">tout</button>
    <button data-filter="high_risk">high</button>
    <button data-filter="medium_risk">medium</button>
    <button data-filter="check">check</button>
    <button data-filter="plausible">plausible</button>
    <input id="search" placeholder="filtrer clip, segment, flags..." />
    <button id="export">export labels CSV</button>
    <button id="clear">vider labels locaux</button>
  </div>

  <main class="grid">
    {''.join(cards)}
  </main>

<script>
const reviewRows = {state_json};
const storageKey = "ttflux_review_001H_labels";

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
    }}

    label.addEventListener("change", update);
    notes.addEventListener("input", update);
  }});
}}

function applyFilter(kind) {{
  const q = document.getElementById("search").value.trim().toLowerCase();

  document.querySelectorAll(".card").forEach(card => {{
    const guess = (card.dataset.guess || "").toLowerCase();
    const text = card.innerText.toLowerCase();

    const okKind = kind === "all" || guess === kind;
    const okSearch = !q || text.includes(q);

    card.classList.toggle("hidden", !(okKind && okSearch));
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
  a.download = "review_labels_001H_export.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}

let activeFilter = "all";

document.querySelectorAll("[data-filter]").forEach(button => {{
  button.addEventListener("click", () => {{
    activeFilter = button.dataset.filter;
    applyFilter(activeFilter);
  }});
}});

document.getElementById("search").addEventListener("input", () => applyFilter(activeFilter));
document.getElementById("export").addEventListener("click", exportCsv);
document.getElementById("clear").addEventListener("click", () => {{
  localStorage.removeItem(storageKey);
  syncFromStorage();
}});

bindInputs();
syncFromStorage();
applyFilter("all");
</script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")


def build_visual_review(run_dir: Path, manifest: Path, out_html: Path, out_labels: Path) -> None:
    run_dir = run_dir.resolve()
    manifest = manifest.resolve()

    _, rows = read_csv_rows(manifest)
    rows = sorted(rows, key=row_sort_key)

    point_map: dict[str, list[tuple[int, float, float]]] = {}
    all_points: list[list[tuple[int, float, float]]] = []

    for row in rows:
        review_id = row.get("review_id", "")
        segment_csv = resolve_segment_csv(run_dir, row)
        points = load_points(segment_csv)

        point_map[review_id] = points
        all_points.append(points)

    bounds = compute_global_bounds(all_points)

    build_label_template(rows, out_labels)
    build_html(rows, point_map, bounds, out_html, out_labels)

    found = sum(1 for pts in point_map.values() if pts)
    print(f"[001H] run_dir      : {run_dir}")
    print(f"[001H] manifest     : {manifest}")
    print(f"[001H] segments     : {len(rows)}")
    print(f"[001H] trajectories : {found}/{len(rows)}")
    print(f"[001H] bounds       : x={bounds[0]:.1f}..{bounds[1]:.1f} y={bounds[2]:.1f}..{bounds[3]:.1f}")
    print(f"[001H] wrote HTML   : {out_html}")
    print(f"[001H] wrote labels : {out_labels}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--manifest", default="runs/batch_001E/review_manifest_001G_trajectory.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/visual_review_001H.html")
    parser.add_argument("--out-labels", default="runs/batch_001E/review_labels_001H_template.csv")
    args = parser.parse_args()

    build_visual_review(
        run_dir=Path(args.run_dir),
        manifest=Path(args.manifest),
        out_html=Path(args.out_html),
        out_labels=Path(args.out_labels),
    )


if __name__ == "__main__":
    main()