from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


LABELS = [
    {
        "key": "good_ball",
        "title": "✅ Balle suivie",
        "text": "La croix rouge est sur la balle, ou très proche, la plupart du temps.",
        "shortcut": "1",
        "class": "good",
        "offset": "0",
    },
    {
        "key": "lag_or_lead",
        "title": "🟡 Balle suivie mais décalée",
        "text": "C’est bien la balle, mais elle est mieux alignée dans une autre colonne temporelle.",
        "shortcut": "2",
        "class": "lag",
        "offset": "",
    },
    {
        "key": "partial_ball",
        "title": "🟠 Balle partielle",
        "text": "On voit la balle par moments, mais le suivi décroche ou devient instable.",
        "shortcut": "3",
        "class": "partial",
        "offset": "",
    },
    {
        "key": "false_track",
        "title": "🔴 Fausse piste",
        "text": "La croix suit surtout une ligne, un bras, un maillot, une chaussure, un panneau, etc.",
        "shortcut": "4",
        "class": "false",
        "offset": "",
    },
    {
        "key": "unclear",
        "title": "⚪ Impossible à trancher",
        "text": "La vidéo est trop compressée ou l’objet est trop ambigu.",
        "shortcut": "5",
        "class": "unclear",
        "offset": "",
    },
]


OFFSETS = [
    {"key": "-6", "text": "Balle mieux en frame -6"},
    {"key": "-3", "text": "Balle mieux en frame -3"},
    {"key": "0", "text": "Pas de décalage clair"},
    {"key": "+3", "text": "Balle mieux en frame +3"},
    {"key": "+6", "text": "Balle mieux en frame +6"},
    {"key": "unknown", "text": "Je ne sais pas"},
]


OBJECTS = [
    {"key": "", "text": "Non précisé"},
    {"key": "table_line", "text": "ligne / bord de table"},
    {"key": "player_body", "text": "joueur / bras / maillot"},
    {"key": "racket", "text": "raquette"},
    {"key": "shoe_floor", "text": "chaussure / sol"},
    {"key": "background_panel", "text": "panneau / fond"},
    {"key": "unknown_object", "text": "autre / pas sûr"},
]


CONFIDENCE = [
    {"key": "high", "text": "Sûr"},
    {"key": "medium", "text": "Moyen"},
    {"key": "low", "text": "Pas sûr"},
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

    candidates = sorted(lag_dir.glob(f"{review_id}_*_lag_probe.jpg"))
    if candidates:
        return candidates[0]

    safe_segment = "".join(c if c.isalnum() or c in "._-" else "_" for c in segment_name)
    candidates = sorted(lag_dir.glob(f"*{safe_segment}*_lag_probe.jpg"))
    if candidates:
        return candidates[0]

    return None


def relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def build_label_buttons(rid: str) -> str:
    chunks = []

    for label in LABELS:
        chunks.append(
            f"""
<button
  class="choice label {html.escape(label["class"])}"
  data-rid="{html.escape(rid)}"
  data-label="{html.escape(label["key"])}"
  onclick="setMainLabel(this.dataset.rid, this.dataset.label)">
  <b>{html.escape(label["shortcut"])} · {html.escape(label["title"])}</b>
  <span>{html.escape(label["text"])}</span>
</button>
"""
        )

    return "\n".join(chunks)


def build_offset_buttons(rid: str) -> str:
    return "\n".join(
        f"""
<button
  class="mini offset"
  data-rid="{html.escape(rid)}"
  data-offset="{html.escape(item["key"])}"
  onclick="setOffset(this.dataset.rid, this.dataset.offset)">
  {html.escape(item["text"])}
</button>
"""
        for item in OFFSETS
    )


def build_object_buttons(rid: str) -> str:
    return "\n".join(
        f"""
<button
  class="mini object"
  data-rid="{html.escape(rid)}"
  data-object="{html.escape(item["key"])}"
  onclick="setObject(this.dataset.rid, this.dataset.object)">
  {html.escape(item["text"])}
</button>
"""
        for item in OBJECTS
    )


def build_confidence_buttons(rid: str) -> str:
    return "\n".join(
        f"""
<button
  class="mini confidence"
  data-rid="{html.escape(rid)}"
  data-confidence="{html.escape(item["key"])}"
  onclick="setConfidence(this.dataset.rid, this.dataset.confidence)">
  {html.escape(item["text"])}
</button>
"""
        for item in CONFIDENCE
    )


def build_cards(rows: list[dict[str, str]], lag_dir: Path, out_html: Path) -> str:
    cards = []

    for i, row in enumerate(rows, start=1):
        rid = row.get("review_id", f"R{i:04d}")
        segment_name = row.get("segment_name", "")
        clip_id = row.get("clip_id", "")

        img = find_lag_image(lag_dir, rid, segment_name)
        if img:
            img_html = f'<img src="{html.escape(relpath(img, out_html.parent))}" loading="lazy">'
        else:
            img_html = '<p class="warn">Image lag probe introuvable. Relance 001W2 si besoin.</p>'

        cards.append(
            f"""
<section class="card" id="{html.escape(rid)}" data-rid="{html.escape(rid)}">
  <div class="card-top">
    <div>
      <h2>{i:02d}. {html.escape(rid)} · {html.escape(segment_name)}</h2>
      <p class="muted">
        clip : <code>{html.escape(clip_id)}</code><br>
        décision actuelle : <b>{html.escape(row.get("decision_001U", ""))}</b> ·
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

  <div class="instructions">
    <b>Comment lire l’image :</b>
    <ol>
      <li>Regarde d’abord la colonne <b>frame +0</b>.</li>
      <li>Si la croix rouge n’est pas sur la balle, regarde <b>-3</b> et <b>+3</b>.</li>
      <li>Si une autre colonne est clairement meilleure, choisis “balle suivie mais décalée”.</li>
      <li>Si la croix suit un autre objet, choisis “fausse piste”.</li>
    </ol>
  </div>

  <div class="image-wrap">
    {img_html}
  </div>

  <div class="block">
    <h3>1. Choisis le jugement principal</h3>
    <div class="choices">
      {build_label_buttons(rid)}
    </div>
  </div>

  <div class="block secondary">
    <h3>2. Si c’est décalé, quelle colonne semble la meilleure ?</h3>
    <div class="mini-row">
      {build_offset_buttons(rid)}
    </div>
  </div>

  <div class="block secondary">
    <h3>3. Si c’est une fausse piste, quel objet est suivi ?</h3>
    <div class="mini-row">
      {build_object_buttons(rid)}
    </div>
  </div>

  <div class="block secondary">
    <h3>4. Niveau de confiance</h3>
    <div class="mini-row">
      {build_confidence_buttons(rid)}
    </div>
  </div>

  <label class="note-label">
    Note libre :
    <input
      id="note_{html.escape(rid)}"
      type="text"
      placeholder="ex: plutôt balle en -3, mais décrochage sur ligne ensuite"
      oninput="setNote('{html.escape(rid)}', this.value)"
    >
  </label>
</section>
"""
        )

    return "\n".join(cards)


def write_html(out_html: Path, rows: list[dict[str, str]], lag_dir: Path, review_csv: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    rows_json = json.dumps(rows, ensure_ascii=False)
    labels_json = json.dumps(LABELS, ensure_ascii=False)
    offsets_json = json.dumps(OFFSETS, ensure_ascii=False)
    objects_json = json.dumps(OBJECTS, ensure_ascii=False)
    confidence_json = json.dumps(CONFIDENCE, ensure_ascii=False)

    cards = build_cards(rows, lag_dir, out_html)

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux human review 001W4</title>
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
header {{
  position:sticky;
  top:0;
  z-index:10;
}}
h1, h2, h3 {{ margin-top:0; }}
code {{ color:#dce4ff; }}
img {{
  max-width:100%;
  border-radius:12px;
  border:1px solid #2b303b;
  background:#000;
}}
.card-top {{
  display:flex;
  justify-content:space-between;
  gap:16px;
  align-items:flex-start;
}}
.status {{
  min-width:180px;
  text-align:center;
  padding:8px 10px;
  border-radius:999px;
  background:#252a35;
  color:#9ea7b8;
  font-weight:800;
}}
.status.done.good {{ background:#123820; color:#9dffbc; }}
.status.done.lag {{ background:#35300e; color:#ffe27a; }}
.status.done.partial {{ background:#382a12; color:#ffc36e; }}
.status.done.false {{ background:#3a1518; color:#ff9aa4; }}
.status.done.unclear {{ background:#222b42; color:#b8c8ff; }}
.instructions {{
  border:1px solid #3a4150;
  background:#121722;
  border-radius:12px;
  padding:12px 14px;
  margin:12px 0;
  color:#cbd4e7;
}}
.instructions ol {{
  margin:8px 0 0 22px;
  padding:0;
}}
.block {{
  margin-top:14px;
  border-top:1px solid #2b303b;
  padding-top:14px;
}}
.choices {{
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));
  gap:10px;
}}
.choice {{
  border:1px solid #3a4150;
  color:#edf0f7;
  padding:12px;
  border-radius:12px;
  cursor:pointer;
  text-align:left;
}}
.choice b {{
  display:block;
  margin-bottom:6px;
}}
.choice span {{
  display:block;
  color:#d0d7e8;
  font-size:13px;
  line-height:1.35;
}}
.choice.good {{ background:#16331f; }}
.choice.lag {{ background:#363010; }}
.choice.partial {{ background:#382a12; }}
.choice.false {{ background:#3a1518; }}
.choice.unclear {{ background:#222b42; }}
.choice.selected,
.mini.selected {{
  outline:3px solid #ffffff;
}}
.mini-row {{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
}}
.mini {{
  border:1px solid #3a4150;
  background:#242936;
  color:#edf0f7;
  padding:9px 11px;
  border-radius:999px;
  cursor:pointer;
  font-weight:700;
}}
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
  font-weight:800;
}}
textarea {{
  width:100%;
  min-height:150px;
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
.progress {{
  font-weight:800;
  color:#9dffbc;
}}
kbd {{
  background:#0b0d12;
  border:1px solid #3a4150;
  padding:2px 6px;
  border-radius:6px;
}}
</style>
</head>
<body>
<header>
<h1>TTFlux human review 001W4</h1>
<p class="muted">
Interface simplifiée pour juger les <b>{len(rows)}</b> segments en review.
Source : <code>{html.escape(str(review_csv))}</code>
</p>

<p>
Touches rapides :
<kbd>1</kbd> balle suivie ·
<kbd>2</kbd> balle décalée ·
<kbd>3</kbd> partielle ·
<kbd>4</kbd> fausse piste ·
<kbd>5</kbd> incertain
</p>

<div class="toolbar">
  <button onclick="exportCsv()">Exporter CSV</button>
  <button onclick="downloadCsv()">Télécharger CSV</button>
  <button onclick="clearLabels()">Effacer labels locaux</button>
  <span class="progress" id="progress">0/{len(rows)} jugés</span>
</div>

<textarea id="csvBox" placeholder="Clique Exporter CSV ou Télécharger CSV quand tu as fini."></textarea>
</header>

{cards}

<script>
const ROWS = {rows_json};
const LABELS = {labels_json};
const OFFSETS = {offsets_json};
const OBJECTS = {objects_json};
const CONFIDENCE = {confidence_json};
const STORAGE_KEY = "ttflux_001W4_human_review_v1";

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

function byKey(arr, key) {{
  return arr.find(x => x.key === key) || null;
}}

function setMainLabel(rid, label) {{
  const state = loadState();
  if (!state[rid]) state[rid] = {{}};
  state[rid].label_001W4 = label;
  state[rid].updated_at = new Date().toISOString();

  if (label !== "lag_or_lead") {{
    state[rid].best_offset_frames_001W4 = "";
  }}
  if (label !== "false_track") {{
    state[rid].false_object_001W4 = "";
  }}

  saveState(state);
  refresh();
}}

function setOffset(rid, offset) {{
  const state = loadState();
  if (!state[rid]) state[rid] = {{}};
  state[rid].best_offset_frames_001W4 = offset;
  state[rid].updated_at = new Date().toISOString();

  if (!state[rid].label_001W4) {{
    state[rid].label_001W4 = "lag_or_lead";
  }}

  saveState(state);
  refresh();
}}

function setObject(rid, objectKey) {{
  const state = loadState();
  if (!state[rid]) state[rid] = {{}};
  state[rid].false_object_001W4 = objectKey;
  state[rid].updated_at = new Date().toISOString();

  if (!state[rid].label_001W4) {{
    state[rid].label_001W4 = "false_track";
  }}

  saveState(state);
  refresh();
}}

function setConfidence(rid, confidence) {{
  const state = loadState();
  if (!state[rid]) state[rid] = {{}};
  state[rid].confidence_001W4 = confidence;
  state[rid].updated_at = new Date().toISOString();
  saveState(state);
  refresh();
}}

function setNote(rid, note) {{
  const state = loadState();
  if (!state[rid]) state[rid] = {{}};
  state[rid].note_001W4 = note || "";
  state[rid].updated_at = new Date().toISOString();
  saveState(state);
}}

function classFor(label) {{
  const meta = byKey(LABELS, label);
  return meta ? meta.class : "";
}}

function titleFor(label) {{
  const meta = byKey(LABELS, label);
  return meta ? meta.title : "non jugé";
}}

function refresh() {{
  const state = loadState();
  let done = 0;

  document.querySelectorAll(".card").forEach(card => {{
    const rid = card.dataset.rid;
    const item = state[rid] || {{}};
    const label = item.label_001W4 || "";

    const status = document.getElementById("status_" + rid);
    if (label) {{
      done++;
      status.textContent = titleFor(label);
      status.className = "status done " + classFor(label);
    }} else {{
      status.textContent = "non jugé";
      status.className = "status";
    }}

    card.querySelectorAll(".choice.label").forEach(btn => {{
      btn.classList.toggle("selected", btn.dataset.label === label);
    }});

    card.querySelectorAll(".mini.offset").forEach(btn => {{
      btn.classList.toggle("selected", btn.dataset.offset === (item.best_offset_frames_001W4 || ""));
    }});

    card.querySelectorAll(".mini.object").forEach(btn => {{
      btn.classList.toggle("selected", btn.dataset.object === (item.false_object_001W4 || ""));
    }});

    card.querySelectorAll(".mini.confidence").forEach(btn => {{
      btn.classList.toggle("selected", btn.dataset.confidence === (item.confidence_001W4 || ""));
    }});

    const noteInput = document.getElementById("note_" + rid);
    if (noteInput && document.activeElement !== noteInput) {{
      noteInput.value = item.note_001W4 || "";
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
    "label_001W4",
    "best_offset_frames_001W4",
    "false_object_001W4",
    "confidence_001W4",
    "note_001W4",
    "updated_at"
  ];

  const lines = [cols.join(",")];

  ROWS.forEach(row => {{
    const rid = row.review_id;
    const item = state[rid] || {{}};

    const out = {{
      ...row,
      label_001W4: item.label_001W4 || "",
      best_offset_frames_001W4: item.best_offset_frames_001W4 || "",
      false_object_001W4: item.false_object_001W4 || "",
      confidence_001W4: item.confidence_001W4 || "",
      note_001W4: item.note_001W4 || "",
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
  a.download = "review_labels_001W4_human_export.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}

function clearLabels() {{
  if (!confirm("Effacer tous les labels locaux 001W4 ?")) return;
  localStorage.removeItem(STORAGE_KEY);
  document.getElementById("csvBox").value = "";
  refresh();
}}

document.addEventListener("keydown", ev => {{
  if (ev.target && ["INPUT", "TEXTAREA"].includes(ev.target.tagName)) return;

  const map = {{
    "1": "good_ball",
    "2": "lag_or_lead",
    "3": "partial_ball",
    "4": "false_track",
    "5": "unclear",
  }};

  if (!map[ev.key]) return;

  const visible = [...document.querySelectorAll(".card")].find(card => {{
    const rect = card.getBoundingClientRect();
    return rect.top >= 0 && rect.top < window.innerHeight * 0.55;
  }});

  if (!visible) return;

  setMainLabel(visible.dataset.rid, map[ev.key]);
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
    parser.add_argument("--out-html", default="runs/batch_001E/human_review_001W4.html")
    args = parser.parse_args()

    review_csv = Path(args.review_csv)
    lag_dir = Path(args.lag_dir)
    out_html = Path(args.out_html)

    rows = read_csv(review_csv)
    write_html(out_html=out_html, rows=rows, lag_dir=lag_dir, review_csv=review_csv)

    print(f"[001W4] review rows : {len(rows)}")
    print(f"[001W4] lag dir     : {lag_dir}")
    print(f"[001W4] wrote HTML  : {out_html}")


if __name__ == "__main__":
    main()
