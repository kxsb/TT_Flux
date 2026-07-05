from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in csv.DictReader(f, dialect=dialect)]


def valid_point(p: Any) -> bool:
    return isinstance(p, dict) and p.get("x") is not None and p.get("y") is not None


def normalize_quad(q: Any) -> list[Any]:
    labels = ["front_left", "front_right", "back_right", "back_left"]
    out = []

    if not isinstance(q, list):
        q = []

    for i in range(4):
        if i < len(q) and valid_point(q[i]):
            p = q[i]
            out.append({
                "label": p.get("label", labels[i]),
                "x": float(p["x"]),
                "y": float(p["y"]),
            })
        else:
            out.append(None)

    return out


def convert_seed(seed: dict[str, Any] | None, clips: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "version": "003A3",
        "clips": {},
    }

    if not seed or not isinstance(seed.get("clips"), dict):
        return out

    for clip in clips:
        cid = clip["clip_id"]
        old = seed["clips"].get(cid)

        if not isinstance(old, dict):
            continue

        if isinstance(old.get("table_keyframes"), list):
            keyframes = old["table_keyframes"]
        else:
            q = normalize_quad(old.get("table_quad"))
            frame_ref = int(float(old.get("frame_ref") or 0))
            keyframes = [{
                "frame": frame_ref,
                "table_quad": q,
                "updated_at": old.get("updated_at"),
                "source": "converted_from_003A",
            }]

        fixed = []

        for kf in keyframes:
            if not isinstance(kf, dict):
                continue

            fixed.append({
                "frame": int(float(kf.get("frame") or 0)),
                "table_quad": normalize_quad(kf.get("table_quad")),
                "updated_at": kf.get("updated_at"),
                "source": kf.get("source", "003A3"),
            })

        out["clips"][cid] = {
            "clip_id": cid,
            "filename": clip.get("filename", ""),
            "video_path": clip.get("video_path", ""),
            "width": clip.get("width", 0),
            "height": clip.get("height", 0),
            "fps": clip.get("fps", 0),
            "frame_count": clip.get("frame_count", 0),
            "table_keyframes": fixed,
        }

    return out


def infer_frame_range(row: dict[str, str]) -> tuple[int | None, int | None]:
    for a, b in [
        ("first_frame", "last_frame"),
        ("frame_first", "frame_last"),
        ("start_frame", "end_frame"),
    ]:
        if row.get(a) and row.get(b):
            try:
                return int(float(row[a])), int(float(row[b]))
            except Exception:
                pass

    blob = " ".join(str(v) for v in row.values())
    m = re.search(r"f(\d+)_to_f(\d+)", blob)

    if m:
        return int(m.group(1)), int(m.group(2))

    return None, None


def collect_segments(final_csv: Path, clips: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows = read_csv(final_csv)
    out = {c["clip_id"]: [] for c in clips}

    for row in rows:
        blob = " ".join(str(v) for v in row.values())
        matched_clip = None

        for clip in clips:
            stem = Path(clip["filename"]).stem
            if stem in blob:
                matched_clip = clip
                break

        if matched_clip is None:
            continue

        f1, f2 = infer_frame_range(row)

        if f1 is None or f2 is None:
            continue

        mid = int(round((f1 + f2) / 2))

        out[matched_clip["clip_id"]].append({
            "review_id": row.get("review_id", ""),
            "segment_name": row.get("segment_name", ""),
            "first_frame": f1,
            "last_frame": f2,
            "mid_frame": mid,
            "final_decision_002B": row.get("final_decision_002B", ""),
            "human_decision_002A": row.get("human_decision_002A", ""),
        })

    for cid in out:
        out[cid] = sorted(out[cid], key=lambda s: s["mid_frame"])

    return out


def build_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)

    return r'''<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux table keyframes 003A3</title>
<style>
:root{
  --bg:#101218;
  --panel:#181b22;
  --line:#2b303b;
  --text:#edf0f7;
  --muted:#9ea7b8;
  --accent:#74d99f;
  --warn:#ffd37a;
  --bad:#ff7a7a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,Segoe UI,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
h1{margin:0;font-size:18px}
main{display:grid;grid-template-columns:minmax(0,1fr) 390px;gap:14px;padding:14px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px}
.viewer{position:relative;width:100%;min-height:320px;background:#000;border-radius:14px;overflow:hidden}
video{display:block;width:100%;max-height:78vh;background:#000}
canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
button,select,input{background:#20242e;color:var(--text);border:1px solid var(--line);border-radius:10px;padding:7px 9px}
button{cursor:pointer}
button.active{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
button.warn{border-color:#7b6040;color:var(--warn)}
button.bad{border-color:#7d4444;color:var(--bad)}
.row{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:8px 0}
.corner-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.small{font-size:12px;color:var(--muted)}
.kbd{font-family:ui-monospace,Consolas,monospace;background:#101218;border:1px solid var(--line);border-radius:6px;padding:1px 5px}
pre{white-space:pre-wrap;word-break:break-word;background:#101218;border:1px solid var(--line);border-radius:10px;padding:10px;max-height:220px;overflow:auto;font-size:12px}
.segment-list{max-height:170px;overflow:auto;border:1px solid var(--line);border-radius:10px;padding:6px;background:#101218}
.segment-btn{display:block;width:100%;margin:4px 0;text-align:left}
.ok{color:var(--accent)}
.warnText{color:var(--warn)}
.badText{color:var(--bad)}
@media(max-width:1050px){main{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>TTFlux · table keyframes 003A3</h1>
  <select id="clipSelect"></select>
  <button id="prevClipBtn">← clip</button>
  <button id="nextClipBtn">clip →</button>
  <button id="playBtn">play/pause</button>
  <button id="nearestKfBtn">keyframe proche</button>
</header>

<main>
  <section class="panel">
    <div class="viewer" id="viewer">
      <video id="video" controls preload="metadata"></video>
      <canvas id="canvas"></canvas>
    </div>
    <p class="small">
      Principe : ajoute une keyframe de table aux moments où la caméra change.
      Pour caméra fixe : une keyframe au milieu suffit. Pour caméra mobile : début / milieu / fin ou autour des segments.
    </p>
  </section>

  <aside class="panel">
    <h2>Keyframes</h2>

    <div class="row">
      <select id="keyframeSelect"></select>
    </div>

    <div class="row">
      <button id="addKfBtn">Ajouter ici</button>
      <button id="copyNearestBtn">Copier proche ici</button>
      <button id="deleteKfBtn" class="bad">Supprimer</button>
    </div>

    <div class="row">
      <button id="startBtn">début</button>
      <button id="midBtn">milieu</button>
      <button id="endBtn">fin</button>
    </div>

    <h3>Coins de la table</h3>
    <div class="corner-grid">
      <button data-corner="0" class="cornerBtn active">1 · avant-gauche</button>
      <button data-corner="1" class="cornerBtn">2 · avant-droite</button>
      <button data-corner="2" class="cornerBtn">3 · arrière-droite</button>
      <button data-corner="3" class="cornerBtn">4 · arrière-gauche</button>
    </div>

    <div class="row">
      <button id="undoPointBtn">Annuler point</button>
      <button id="clearKfBtn" class="warn">Effacer keyframe</button>
    </div>

    <h3>Segments du clip</h3>
    <div id="segmentList" class="segment-list small"></div>

    <h3>Export</h3>
    <div class="row">
      <button id="exportBtn">Exporter JSON</button>
      <label>
        <input id="importInput" type="file" accept=".json" style="display:none">
        <button id="importBtn" type="button">Importer JSON</button>
      </label>
    </div>

    <h3>État</h3>
    <div id="status"></div>

    <h3>Raccourcis</h3>
    <p class="small">
      <span class="kbd">1–4</span> coin ·
      <span class="kbd">A</span> ajouter keyframe ·
      <span class="kbd">K</span> keyframe proche ·
      <span class="kbd">Z</span> annuler point ·
      <span class="kbd">N/P</span> clip suivant/précédent ·
      <span class="kbd">Espace</span> play/pause
    </p>

    <h3>JSON courant</h3>
    <pre id="jsonPreview"></pre>
  </aside>
</main>

<script>
const payload = __PAYLOAD__;
const clips = payload.clips || [];
const segmentsByClip = payload.segments_by_clip || {};
const seedAnnotations = payload.seed_annotations || {version:"003A3", clips:{}};
const storageKey = "ttflux_003A3_table_keyframes_v1";

const cornerLabels = ["front_left","front_right","back_right","back_left"];
const cornerNames = ["avant-gauche","avant-droite","arrière-droite","arrière-gauche"];

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const viewer = document.getElementById("viewer");
const clipSelect = document.getElementById("clipSelect");
const keyframeSelect = document.getElementById("keyframeSelect");
const statusEl = document.getElementById("status");
const jsonPreviewEl = document.getElementById("jsonPreview");
const segmentListEl = document.getElementById("segmentList");

let currentClipIndex = 0;
let currentCorner = 0;
let currentKeyframeIndex = 0;
let annotations = loadAnnotations();

function nowIso(){ return new Date().toISOString(); }

function clone(obj){ return JSON.parse(JSON.stringify(obj)); }

function loadAnnotations(){
  try{
    const raw = localStorage.getItem(storageKey);
    if(raw) return JSON.parse(raw);
  }catch(e){}
  return clone(seedAnnotations);
}

function saveAnnotations(){
  annotations.updated_at = nowIso();
  localStorage.setItem(storageKey, JSON.stringify(annotations));
  updateAll();
}

function currentClip(){ return clips[currentClipIndex]; }

function currentAnn(){
  const c = currentClip();
  if(!c) return null;

  if(!annotations.clips) annotations.clips = {};

  if(!annotations.clips[c.clip_id]){
    annotations.clips[c.clip_id] = {
      clip_id: c.clip_id,
      filename: c.filename,
      video_path: c.video_path,
      width: c.width,
      height: c.height,
      fps: c.fps,
      frame_count: c.frame_count,
      table_keyframes: []
    };
  }

  return annotations.clips[c.clip_id];
}

function emptyQuad(){ return [null,null,null,null]; }

function validPoint(p){ return p && typeof p.x === "number" && typeof p.y === "number"; }

function quadCount(q){ return (q || []).filter(validPoint).length; }

function currentFrame(){
  const c = currentClip();
  const fps = Number(c?.fps || 0);
  return fps > 0 ? Math.round(video.currentTime * fps) : 0;
}

function frameToTime(frame){
  const c = currentClip();
  const fps = Number(c?.fps || 0);
  return fps > 0 ? frame / fps : 0;
}

function sortedKeyframes(){
  const ann = currentAnn();
  if(!ann) return [];
  if(!Array.isArray(ann.table_keyframes)) ann.table_keyframes = [];
  ann.table_keyframes.sort((a,b) => Number(a.frame || 0) - Number(b.frame || 0));
  return ann.table_keyframes;
}

function nearestKeyframeIndex(frame){
  const kfs = sortedKeyframes();
  if(!kfs.length) return -1;

  let best = 0;
  let bestD = Infinity;

  kfs.forEach((kf, i) => {
    const d = Math.abs(Number(kf.frame || 0) - frame);
    if(d < bestD){
      bestD = d;
      best = i;
    }
  });

  return best;
}

function currentKeyframe(){
  const kfs = sortedKeyframes();
  if(!kfs.length) return null;

  currentKeyframeIndex = Math.max(0, Math.min(currentKeyframeIndex, kfs.length - 1));
  return kfs[currentKeyframeIndex];
}

function ensureKeyframeAt(frame, copyFromNearest=true){
  const ann = currentAnn();
  const kfs = sortedKeyframes();

  let idx = kfs.findIndex(kf => Number(kf.frame || 0) === frame);
  if(idx >= 0){
    currentKeyframeIndex = idx;
    return kfs[idx];
  }

  let q = emptyQuad();

  if(copyFromNearest && kfs.length){
    const ni = nearestKeyframeIndex(frame);
    q = clone(kfs[ni].table_quad || emptyQuad());
  }

  const kf = {
    frame,
    table_quad: q,
    updated_at: nowIso(),
    source: "003A3_manual"
  };

  ann.table_keyframes.push(kf);
  sortedKeyframes();
  currentKeyframeIndex = ann.table_keyframes.findIndex(x => x === kf);
  saveAnnotations();
  return kf;
}

function initSelect(){
  clipSelect.innerHTML = "";
  clips.forEach((clip, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `${idx + 1}/${clips.length} · ${clip.filename}`;
    clipSelect.appendChild(opt);
  });

  clipSelect.addEventListener("change", () => loadClip(Number(clipSelect.value || 0)));
}

function loadClip(index){
  if(!clips.length) return;

  currentClipIndex = Math.max(0, Math.min(index, clips.length - 1));
  currentKeyframeIndex = 0;
  clipSelect.value = String(currentClipIndex);

  const c = currentClip();
  video.src = c.video_uri;
  video.load();

  video.onloadedmetadata = () => {
    resizeCanvas();
    chooseNearestKeyframe();
    updateAll();
  };

  updateAll();
}

function resizeCanvas(){
  const rect = viewer.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
  draw();
}

function getVideoGeom(){
  const c = currentClip();
  const vw = video.videoWidth || c?.width || 1;
  const vh = video.videoHeight || c?.height || 1;
  const cw = canvas.width;
  const ch = canvas.height;
  const scale = Math.min(cw / vw, ch / vh);
  const drawW = vw * scale;
  const drawH = vh * scale;
  const ox = (cw - drawW) / 2;
  const oy = (ch - drawH) / 2;
  return {vw, vh, cw, ch, scale, drawW, drawH, ox, oy};
}

function videoToCanvas(pt){
  const g = getVideoGeom();
  return {x:g.ox + pt.x * g.scale, y:g.oy + pt.y * g.scale};
}

function eventToVideo(evt){
  const rect = canvas.getBoundingClientRect();
  const cx = (evt.clientX - rect.left) * (canvas.width / rect.width);
  const cy = (evt.clientY - rect.top) * (canvas.height / rect.height);
  const g = getVideoGeom();
  const x = (cx - g.ox) / g.scale;
  const y = (cy - g.oy) / g.scale;
  return {
    x: Math.max(0, Math.min(g.vw, x)),
    y: Math.max(0, Math.min(g.vh, y))
  };
}

function setCorner(idx){
  currentCorner = Math.max(0, Math.min(3, idx));
  document.querySelectorAll(".cornerBtn").forEach(btn => {
    btn.classList.toggle("active", Number(btn.dataset.corner) === currentCorner);
  });
  updateStatus();
}

function chooseNearestKeyframe(){
  const idx = nearestKeyframeIndex(currentFrame());
  if(idx >= 0){
    currentKeyframeIndex = idx;
    updateAll();
  }
}

function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);

  const g = getVideoGeom();
  ctx.strokeStyle = "rgba(255,255,255,.28)";
  ctx.lineWidth = 1;
  ctx.strokeRect(g.ox, g.oy, g.drawW, g.drawH);

  const kf = currentKeyframe();
  if(!kf) return;

  const pts = kf.table_quad || [];
  const valid = pts.map((p,i) => p ? {...p, idx:i} : null).filter(Boolean);

  if(valid.length >= 2){
    ctx.beginPath();
    valid.forEach((p, i) => {
      const q = videoToCanvas(p);
      if(i === 0) ctx.moveTo(q.x, q.y);
      else ctx.lineTo(q.x, q.y);
    });

    if(valid.length === 4) ctx.closePath();

    ctx.strokeStyle = "rgba(116,217,159,.95)";
    ctx.lineWidth = 3;
    ctx.stroke();

    if(valid.length === 4){
      ctx.fillStyle = "rgba(116,217,159,.13)";
      ctx.fill();
    }
  }

  pts.forEach((p, i) => {
    if(!p) return;

    const q = videoToCanvas(p);

    ctx.beginPath();
    ctx.arc(q.x, q.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = i === currentCorner ? "#ffd37a" : "#74d99f";
    ctx.fill();
    ctx.strokeStyle = "#101218";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#fff";
    ctx.font = "13px system-ui";
    ctx.fillText(`${i + 1} · ${cornerNames[i]}`, q.x + 10, q.y - 10);
  });
}

function clickCanvas(evt){
  let kf = currentKeyframe();

  if(!kf){
    kf = ensureKeyframeAt(currentFrame(), true);
  }

  const p = eventToVideo(evt);

  kf.table_quad[currentCorner] = {
    label: cornerLabels[currentCorner],
    x: Math.round(p.x * 100) / 100,
    y: Math.round(p.y * 100) / 100
  };

  kf.updated_at = nowIso();

  if(currentCorner < 3) setCorner(currentCorner + 1);

  saveAnnotations();
  draw();
}

function undoPoint(){
  const kf = currentKeyframe();
  if(!kf) return;

  for(let i = 3; i >= 0; i--){
    if(kf.table_quad[i]){
      kf.table_quad[i] = null;
      setCorner(i);
      break;
    }
  }

  kf.updated_at = nowIso();
  saveAnnotations();
  draw();
}

function clearKeyframe(){
  const kf = currentKeyframe();
  if(!kf) return;

  kf.table_quad = emptyQuad();
  kf.updated_at = nowIso();
  setCorner(0);
  saveAnnotations();
  draw();
}

function deleteKeyframe(){
  const ann = currentAnn();
  const kfs = sortedKeyframes();
  if(!kfs.length) return;

  kfs.splice(currentKeyframeIndex, 1);
  currentKeyframeIndex = Math.max(0, currentKeyframeIndex - 1);
  ann.table_keyframes = kfs;
  saveAnnotations();
  draw();
}

function addKeyframeHere(){
  ensureKeyframeAt(currentFrame(), false);
  updateAll();
  draw();
}

function copyNearestHere(){
  ensureKeyframeAt(currentFrame(), true);
  updateAll();
  draw();
}

function updateKeyframeSelect(){
  const kfs = sortedKeyframes();
  keyframeSelect.innerHTML = "";

  if(!kfs.length){
    const opt = document.createElement("option");
    opt.value = -1;
    opt.textContent = "Aucune keyframe";
    keyframeSelect.appendChild(opt);
    return;
  }

  kfs.forEach((kf, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    const n = quadCount(kf.table_quad);
    opt.textContent = `frame ${kf.frame} · ${n}/4 points`;
    keyframeSelect.appendChild(opt);
  });

  currentKeyframeIndex = Math.max(0, Math.min(currentKeyframeIndex, kfs.length - 1));
  keyframeSelect.value = String(currentKeyframeIndex);
}

function updateSegmentList(){
  const c = currentClip();
  const segs = segmentsByClip[c?.clip_id] || [];
  segmentListEl.innerHTML = "";

  if(!segs.length){
    segmentListEl.textContent = "Aucun segment listé pour ce clip.";
    return;
  }

  segs.forEach(seg => {
    const b = document.createElement("button");
    b.className = "segment-btn";
    b.textContent = `${seg.review_id || ""} · f${seg.first_frame}-${seg.last_frame} · ${seg.final_decision_002B || ""} ${seg.human_decision_002A || ""}`;
    b.addEventListener("click", () => {
      video.currentTime = frameToTime(seg.mid_frame);
      ensureKeyframeAt(seg.mid_frame, true);
      updateAll();
      draw();
    });
    segmentListEl.appendChild(b);
  });
}

function completedKeyframes(){
  const out = {};
  clips.forEach(c => {
    const a = annotations.clips && annotations.clips[c.clip_id];
    const kfs = a && Array.isArray(a.table_keyframes) ? a.table_keyframes : [];
    out[c.clip_id] = {
      keyframes: kfs.length,
      complete: kfs.filter(k => quadCount(k.table_quad) === 4).length
    };
  });
  return out;
}

function exportPayload(){
  const out = {
    version: "003A3",
    exported_at: nowIso(),
    source: "TTFlux table_keyframes_003A3.html",
    clips: {}
  };

  clips.forEach(c => {
    const a = annotations.clips && annotations.clips[c.clip_id];

    out.clips[c.clip_id] = a || {
      clip_id: c.clip_id,
      filename: c.filename,
      video_path: c.video_path,
      width: c.width,
      height: c.height,
      fps: c.fps,
      frame_count: c.frame_count,
      table_keyframes: []
    };
  });

  return out;
}

function updateStatus(){
  const c = currentClip();
  const kf = currentKeyframe();
  const kfs = sortedKeyframes();
  const states = completedKeyframes();

  const totalComplete = Object.values(states).filter(s => s.complete > 0).length;

  const here = kf ? quadCount(kf.table_quad) : 0;

  statusEl.innerHTML = `
    <p>Clip : <b>${c ? c.filename : ""}</b></p>
    <p>Frame vidéo : <b>${currentFrame()}</b></p>
    <p>Keyframes clip : <b>${kfs.length}</b></p>
    <p>Keyframe active : <b class="${here === 4 ? "ok" : "warnText"}">${here}/4 points</b></p>
    <p>Clips avec au moins une keyframe complète : <b class="${totalComplete === clips.length ? "ok" : "warnText"}">${totalComplete}/${clips.length}</b></p>
  `;

  jsonPreviewEl.textContent = JSON.stringify(exportPayload(), null, 2);
}

function updateAll(){
  updateKeyframeSelect();
  updateSegmentList();
  updateStatus();
  draw();
}

function downloadJson(){
  const data = JSON.stringify(exportPayload(), null, 2);
  const blob = new Blob([data], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "table_keyframes_003A3.json";
  document.body.appendChild(a);
  a.click();
  URL.revokeObjectURL(a.href);
  a.remove();
}

function importJsonFile(file){
  const reader = new FileReader();
  reader.onload = () => {
    try{
      const data = JSON.parse(String(reader.result || "{}"));
      if(!data.clips) throw new Error("clips missing");

      annotations = {
        version: "003A3",
        updated_at: nowIso(),
        clips: {}
      };

      clips.forEach(c => {
        const old = data.clips[c.clip_id];
        if(!old) return;

        let kfs = [];

        if(Array.isArray(old.table_keyframes)){
          kfs = old.table_keyframes;
        }else if(Array.isArray(old.table_quad)){
          kfs = [{
            frame: Number(old.frame_ref || 0),
            table_quad: old.table_quad,
            updated_at: old.updated_at || nowIso(),
            source: "imported_003A"
          }];
        }

        annotations.clips[c.clip_id] = {
          clip_id: c.clip_id,
          filename: c.filename,
          video_path: c.video_path,
          width: c.width,
          height: c.height,
          fps: c.fps,
          frame_count: c.frame_count,
          table_keyframes: kfs
        };
      });

      saveAnnotations();
      updateAll();
    }catch(e){
      alert("JSON invalide ou incompatible.");
    }
  };
  reader.readAsText(file);
}

document.querySelectorAll(".cornerBtn").forEach(btn => {
  btn.addEventListener("click", () => setCorner(Number(btn.dataset.corner)));
});

keyframeSelect.addEventListener("change", () => {
  currentKeyframeIndex = Number(keyframeSelect.value || 0);
  const kf = currentKeyframe();
  if(kf) video.currentTime = frameToTime(Number(kf.frame || 0));
  updateAll();
});

document.getElementById("prevClipBtn").addEventListener("click", () => loadClip(currentClipIndex - 1));
document.getElementById("nextClipBtn").addEventListener("click", () => loadClip(currentClipIndex + 1));
document.getElementById("playBtn").addEventListener("click", () => video.paused ? video.play() : video.pause());
document.getElementById("nearestKfBtn").addEventListener("click", chooseNearestKeyframe);

document.getElementById("addKfBtn").addEventListener("click", addKeyframeHere);
document.getElementById("copyNearestBtn").addEventListener("click", copyNearestHere);
document.getElementById("deleteKfBtn").addEventListener("click", deleteKeyframe);

document.getElementById("startBtn").addEventListener("click", () => { video.currentTime = 0; });
document.getElementById("midBtn").addEventListener("click", () => { if(video.duration) video.currentTime = video.duration / 2; });
document.getElementById("endBtn").addEventListener("click", () => { if(video.duration) video.currentTime = Math.max(0, video.duration - 0.05); });

document.getElementById("undoPointBtn").addEventListener("click", undoPoint);
document.getElementById("clearKfBtn").addEventListener("click", clearKeyframe);

document.getElementById("exportBtn").addEventListener("click", downloadJson);
document.getElementById("importBtn").addEventListener("click", () => document.getElementById("importInput").click());
document.getElementById("importInput").addEventListener("change", evt => {
  const file = evt.target.files && evt.target.files[0];
  if(file) importJsonFile(file);
});

canvas.addEventListener("click", clickCanvas);
window.addEventListener("resize", resizeCanvas);
video.addEventListener("loadedmetadata", () => { resizeCanvas(); updateAll(); });
video.addEventListener("timeupdate", () => { updateStatus(); draw(); });

document.addEventListener("keydown", evt => {
  if(evt.target && ["INPUT","TEXTAREA","SELECT"].includes(evt.target.tagName)) return;

  if(evt.key >= "1" && evt.key <= "4"){
    setCorner(Number(evt.key) - 1);
    evt.preventDefault();
  }else if(evt.key.toLowerCase() === "a"){
    addKeyframeHere();
    evt.preventDefault();
  }else if(evt.key.toLowerCase() === "k"){
    chooseNearestKeyframe();
    evt.preventDefault();
  }else if(evt.key.toLowerCase() === "z"){
    undoPoint();
    evt.preventDefault();
  }else if(evt.key.toLowerCase() === "n"){
    loadClip(currentClipIndex + 1);
    evt.preventDefault();
  }else if(evt.key.toLowerCase() === "p"){
    loadClip(currentClipIndex - 1);
    evt.preventDefault();
  }else if(evt.code === "Space"){
    video.paused ? video.play() : video.pause();
    evt.preventDefault();
  }
});

initSelect();
loadClip(0);
setCorner(0);
</script>
</body>
</html>'''.replace("__PAYLOAD__", payload_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", default="runs/batch_001E/table_scene_003A/table_annotation_payload_003A.json")
    parser.add_argument("--seed-json", default="runs/batch_001E/table_scene_003A/table_annotations_003A_merged.json")
    parser.add_argument("--final-csv", default="runs/batch_001E/final_decisions_002B/final_decisions_002B_all.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/table_scene_003A3")
    args = parser.parse_args()

    payload_path = Path(args.payload)
    seed_path = Path(args.seed_json)
    final_csv = Path(args.final_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = read_json(payload_path)
    if not payload or not isinstance(payload.get("clips"), list):
        raise SystemExit("[003A3] payload 003A invalide ou introuvable.")

    clips = payload["clips"]
    seed = read_json(seed_path)
    seed_annotations = convert_seed(seed, clips)
    segments_by_clip = collect_segments(final_csv, clips)

    out_payload = {
        "version": "003A3",
        "clips": clips,
        "seed_annotations": seed_annotations,
        "segments_by_clip": segments_by_clip,
    }

    html_path = out_dir / "table_keyframes_003A3.html"
    payload_out = out_dir / "table_keyframes_payload_003A3.json"

    html_path.write_text(build_html(out_payload), encoding="utf-8")
    payload_out.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[003A3] clips     : {len(clips)}")
    print(f"[003A3] seed json : {seed_path if seed_path.exists() else 'none'}")
    print(f"[003A3] final csv : {final_csv if final_csv.exists() else 'none'}")
    print(f"[003A3] out dir   : {out_dir}")
    print(f"[003A3] html      : {html_path}")


if __name__ == "__main__":
    main()
