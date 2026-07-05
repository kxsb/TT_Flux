from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


LABELS = [
    {
        "key": "ok_ball",
        "text": "OK balle",
        "shortcut": "1",
        "offset": "0",
        "class": "ok",
    },
    {
        "key": "lagged_ball_minus6",
        "text": "Lag -6",
        "shortcut": "2",
        "offset": "-6",
        "class": "lag",
    },
    {
        "key": "lagged_ball_minus3",
        "text": "Lag -3",
        "shortcut": "3",
        "offset": "-3",
        "class": "lag",
    },
    {
        "key": "lagged_ball_plus3",
        "text": "Lag +3",
        "shortcut": "4",
        "offset": "+3",
        "class": "lag",
    },
    {
        "key": "lagged_ball_plus6",
        "text": "Lag +6",
        "shortcut": "5",
        "offset": "+6",
        "class": "lag",
    },
    {
        "key": "partial_ball",
        "text": "Partielle",
        "shortcut": "6",
        "offset": "",
        "class": "partial",
    },
    {
        "key": "false_track",
        "text": "Fausse piste",
        "shortcut": "7",
        "offset": "",
        "class": "false",
    },
    {
        "key": "unclear",
        "text": "Incertain",
        "shortcut": "8",
        "offset": "",
        "class": "unclear",
    },
]


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


def find_lag_image(lag_dir: Path, review_id: str, segment_name: str) -> Path | None:
    if not lag_dir.exists():
        return None

    candidates = list(lag_dir.glob(f"{review_id}_*_lag_probe.jpg"))
    if candidates:
        return sorted(candidates)[0]

    safe_segment = "".join(c if c.isalnum() or c in "._-" else "_" for c in segment_name)
    candidates = list(lag_dir.glob(f"*{safe_segment}*_lag_probe.jpg"))
    if candidates:
        return sorted(candidates)[0]

    return None


def relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def js_string(value: Any) -> str:
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def build_buttons(review_id: str) -> str:
    buttons = []

    for label in LABELS:
        buttons.append(
            f"""
<button
  class="label-btn {html.escape(label["class"])}"
  data-rid="{html.escape(review_id)}"
  data-label="{html.escape(label["key"])}"
  data-offset="{html.escape(label["offset"])}"
  onclick="setLabel(this.dataset.rid, this.dataset.label, this.dataset.offset)">
  {html.escape(label["shortcut"])} · {html.escape(label["text"])}
</button>
"""
        )

    return "\n".join(buttons)


def build_cards(rows: list[dict[str, str]], lag_dir: Path, out_html: Path) -> str:
    cards = []

    for i, row in enumerate(rows, start=1):
        rid = row.get("review_id", f"R{i:04d}")
        segment_name = row.get("segment_name", "")
        clip_id = row.get("clip_id", "")

        img = find_lag_image(lag_dir, rid, segment_name)
        img_html = ""

        if img is not None:
            img_html = f"""
<div class="image-wrap">
  <img src="{html.escape(relpath(img, out_html.parent))}" loading="lazy">
</div>
"""
        else:
            img_html = '<p class="warn">Image lag probe introuvable. Relance 001W2 si besoin.</p>'

        cards.append(
            f"""
<section class="card" id="{html.escape(rid)}" data-rid="{html.escape(rid)}">
  <div class="card-head">
    <div>
      <h2>{i:02d}. {html.escape(rid)} · {html.escape(segment_name)}</h2>
      <p class="muted">
        clip : <code>{html.escape(clip_id)}</code><br>
        décision 001U : <b>{html.escape(row.get("decision_001U", ""))}</b> ·
        raison : <code>{html.escape(row.get("decision_reason_001U", ""))}</code><br>
        risk={html.escape(row.get("risk_score_001G", ""))} ·
        center_false={html.escape(row.get("center_blob_false_score_001N", ""))} ·
        micro_keep={html.escape(row.get("micro_keep_score_001O", ""))} ·
        micro_false={html.escape(row.get("micro_false_score_001O", ""))} ·
        micro_dist={html.escape(row.get("micro_distance_med_001O", ""))}
      </p>
    </div>
    <div class="status" id="status_{html.escape(rid)}">non jugé</div>
  </div>

  {img_html}

  <div class="buttons">
    {build_buttons(rid)}
  </div>

  <label class="note-label">
    Note libre :
    <input
      id="note_{html.escape(rid)}"
      type="text"
      placeholder="ex: balle visible en -3 mais trajectoire table ensuite"
      oninput="setNote('{html.escape(rid)}', this.value)"
    >
  </label>
</section>
"""
        )

    return "\n".join(cards)


def write_html(out_html: Path, rows: list[dict[str, str]], lag_dir: Path, review_csv: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    labels_json = json.dumps(LABELS, ensure_ascii=False)
    rows_json = json.dumps(rows, ensure_ascii=False)

    cards = build_cards(rows, lag_dir, out_html)

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux lag labels 001W3</title>
<style>
body {{
  margin:0;
  padding:24px;
  background:#101218;
  color:#edf0f7;
  font-family:system-ui,Segoe UI,sans-serif;
}}
header, section {{
  background:#181b22;
  border:1px solid #2b303b;
  border-radius:16px;
  padding:16px;
  margin-bottom:16px;
}}
h1, h2 {{ margin-top:0; }}
a {{ color:#9bc2ff; }}
code {{ color:#dce4ff; }}
img {{
  max-width:100%;
  border-radius:12px;
  border:1px solid #2b303b;
  background:#000;
}}
.card-head {{
  display:flex;
  justify-content:space-between;
  gap:16px;
  align-items:flex-start;
}}
.status {{
  min-width:170px;
  text-align:center;
  padding:8px 10px;
  border-radius:999px;
  background:#252a35;
  color:#9ea7b8;
  font-weight:700;
}}
.status.done.ok {{ background:#123820; color:#9dffbc; }}
.status.done.lag {{ background:#35300e; color:#ffe27a; }}
.status.done.partial {{ background:#352912; color:#ffc36e; }}
.status.done.false {{ background:#3a1518; color:#ff9aa4; }}
.status.done.unclear {{ background:#222b42; color:#b8c8ff; }}
.buttons {{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  margin-top:12px;
}}
.label-btn {{
  border:1px solid #3a4150;
  background:#242936;
  color:#edf0f7;
  padding:9px 12px;
  border-radius:10px;
  cursor:pointer;
  font-weight:700;
}}
.label-btn:hover {{ filter:brightness(1.18); }}
.label-btn.selected {{ outline:2px solid #ffffff; }}
.label-btn.ok {{ background:#16331f; }}
.label-btn.lag {{ background:#363010; }}
.label-btn.partial {{ background:#382a12; }}
.label-btn.false {{ background:#3a1518; }}
.label-btn.unclear {{ background:#222b42; }}
.note-label {{
  display:block;
  margin-top:12px;
  color:#9ea7b8;
}}
.note-label input {{
  width:100%;
  margin-top:6px;
  box-sizing:border-box;
  border:1px solid #3a4150;
  background:#101218;
  color:#edf0f7;
  border-radius:10px;
  padding:10px;
}}
.toolbar {{
  display:flex;
  flex-wrap:wrap;
  gap:10px;
  align-items:center;
}}
.toolbar button {{
  border:1px solid #3a4150;
  background:#242936;
  color:#edf0f7;
  padding:9px 12px;
  border-radius:10px;
  cursor:pointer;
  font-weight:700;
}}
textarea {{
  width:100%;
  min-height:190px;
  box-sizing:border-box;
  background:#0b0d12;
  color:#edf0f7;
  border:1px solid #3a4150;
  border-radius:12px;
  padding:12px;
  margin-top:12px;
  font-family:Consolas, monospace;
}}
.muted {{ color:#9ea7b8; }}
.warn {{ color:#ffbc7a; }}
.hint {{ color:#d6c27a; }}
.progress {{
  font-weight:700;
  color:#9dffbc;
}}
</style>
</head>
<body>
<header>
<h1>TTFlux lag labels 001W3</h1>
<p class="muted">
Page de labellisation pour les segments <code>review</code> de l'arbitre 001V.
Source : <code>{html.escape(str(review_csv))}</code><br>
Segments : <b>{len(rows)}</b>
</p>

<p class="hint">
Règle : si la balle est mieux centrée dans la colonne <b>frame -3</b>, clique <b>Lag -3</b>.
On encode la meilleure colonne observée, pas une interprétation définitive.
</p>

<div class="toolbar">
  <button onclick="exportCsv()">Exporter CSV</button>
  <button onclick="downloadCsv()">Télécharger CSV</button>
  <button onclick="clearLabels()">Effacer les labels locaux</button>
  <span class="progress" id="progress">0/{len(rows)} jugés</span>
</div>

<textarea id="csvBox" placeholder="Clique Exporter CSV quand tu as fini."></textarea>
</header>

{cards}

<script>
const LABELS = {labels_json};
const ROWS = {rows_json};
const STORAGE_KEY = "ttflux_001W3_lag_labels_v1";

function loadState() {{
  try {{
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
  }} catch (e) {{
    return {{}};
  }}
}}

function saveState(state) {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}}

function labelMeta(labelKey) {{
  return LABELS.find(x => x.key === labelKey) || null;
}}

function setLabel(rid, labelKey, offset) {{
  const state = loadState();
  if (!state[rid]) state[rid] = {{}};

  state[rid].label_001W3 = labelKey;
  state[rid].best_offset_frames_001W3 = offset || "";
  state[rid].updated_at = new Date().toISOString();

  saveState(state);
  refresh();
}}

function setNote(rid, note) {{
  const state = loadState();
  if (!state[rid]) state[rid] = {{}};

  state[rid].note_001W3 = note || "";
  state[rid].updated_at = new Date().toISOString();

  saveState(state);
}}

function classFor(labelKey) {{
  const meta = labelMeta(labelKey);
  return meta ? meta.class : "";
}}

function refresh() {{
  const state = loadState();
  let done = 0;

  document.querySelectorAll(".card").forEach(card => {{
    const rid = card.dataset.rid;
    const item = state[rid] || {{}};
    const label = item.label_001W3 || "";

    const status = document.getElementById("status_" + rid);
    if (label) {{
      done += 1;
      const meta = labelMeta(label);
      status.textContent = meta ? meta.text : label;
      status.className = "status done " + classFor(label);
    }} else {{
      status.textContent = "non jugé";
      status.className = "status";
    }}

    card.querySelectorAll(".label-btn").forEach(btn => {{
      btn.classList.toggle("selected", btn.dataset.label === label);
    }});

    const noteInput = document.getElementById("note_" + rid);
    if (noteInput && document.activeElement !== noteInput) {{
      noteInput.value = item.note_001W3 || "";
    }}
  }});

  document.getElementById("progress").textContent = done + "/" + ROWS.length + " jugés";
}}

function csvEscape(value) {{
  value = String(value ?? "");
  if (value.includes('"') || value.includes(",") || value.includes("\\n") || value.includes("\\r")) {{
    return '"' + value.replaceAll('"', '""') + '"';
  }}
  return value;
}}

function buildCsv() {{
  const state = loadState();
  const cols = [
    "review_id",
    "clip_id",
    "segment_name",
    "decision_001U",
    "decision_reason_001U",
    "risk_score_001G",
    "center_blob_false_score_001N",
    "micro_keep_score_001O",
    "micro_false_score_001O",
    "micro_distance_med_001O",
    "label_001W3",
    "best_offset_frames_001W3",
    "note_001W3",
    "updated_at"
  ];

  const lines = [cols.join(",")];

  ROWS.forEach(row => {{
    const rid = row.review_id;
    const item = state[rid] || {{}};

    const out = {{
      ...row,
      label_001W3: item.label_001W3 || "",
      best_offset_frames_001W3: item.best_offset_frames_001W3 || "",
      note_001W3: item.note_001W3 || "",
      updated_at: item.updated_at || ""
    }};

    lines.push(cols.map(c => csvEscape(out[c] || "")).join(","));
  }});

  return lines.join("\\n");
}}

function exportCsv() {{
  const csv = buildCsv();
  document.getElementById("csvBox").value = csv;
  return csv;
}}

function downloadCsv() {{
  const csv = exportCsv();
  const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "review_labels_001W3_lag_export.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}

function clearLabels() {{
  if (!confirm("Effacer tous les labels locaux 001W3 ?")) return;
  localStorage.removeItem(STORAGE_KEY);
  document.getElementById("csvBox").value = "";
  refresh();
}}

document.addEventListener("keydown", ev => {{
  if (ev.target && ["INPUT", "TEXTAREA"].includes(ev.target.tagName)) return;

  const keys = LABELS.map(x => x.shortcut);
  if (!keys.includes(ev.key)) return;

  const visible = [...document.querySelectorAll(".card")].find(card => {{
    const rect = card.getBoundingClientRect();
    return rect.top >= 0 && rect.top < window.innerHeight * 0.55;
  }});

  if (!visible) return;

  const label = LABELS.find(x => x.shortcut === ev.key);
  setLabel(visible.dataset.rid, label.key, label.offset);
}});

refresh();
</script>
</body>
</html>
"""

    out_html.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", default="runs/batch_001E/arbiter_001V/arbiter_001U_review.csv")
    parser.add_argument("--lag-dir", default="runs/batch_001E/lag_probe_001W2")
    parser.add_argument("--out-html", default="runs/batch_001E/lag_label_review_001W3.html")
    args = parser.parse_args()

    review_csv = Path(args.review_csv)
    lag_dir = Path(args.lag_dir)
    out_html = Path(args.out_html)

    rows = read_csv(review_csv)
    write_html(out_html=out_html, rows=rows, lag_dir=lag_dir, review_csv=review_csv)

    print(f"[001W3] review rows : {len(rows)}")
    print(f"[001W3] lag dir     : {lag_dir}")
    print(f"[001W3] wrote HTML  : {out_html}")


if __name__ == "__main__":
    main()
