from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def walk_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v)


def resolve_video_path(value: str, roots: list[Path]) -> Path | None:
    value = str(value or "").strip().strip('"')

    if not value:
        return None

    p = Path(value)

    if p.suffix.lower() not in VIDEO_EXTS:
        return None

    if p.is_absolute() and p.exists():
        return p

    for root in roots:
        q = root / p
        if q.exists():
            return q

    return None


def video_meta(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return {
            "width": 0,
            "height": 0,
            "fps": 0,
            "frame_count": 0,
            "duration_sec": 0,
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    duration = frame_count / fps if fps > 0 else 0

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "duration_sec": round(duration, 3),
    }


def collect_videos(config_dir: Path, root: Path) -> list[dict[str, Any]]:
    videos: list[Path] = []
    seen = set()

    json_files = sorted(config_dir.rglob("*.json"))

    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            try:
                data = json.loads(jp.read_text(encoding="utf-8-sig"))
            except Exception:
                continue

        roots = [root, config_dir, jp.parent]

        for s in walk_strings(data):
            p = resolve_video_path(s, roots)

            if p is None:
                continue

            key = str(p.resolve()).lower()

            if key in seen:
                continue

            seen.add(key)
            videos.append(p.resolve())

    out = []

    for i, p in enumerate(videos, start=1):
        meta = video_meta(p)
        clip_id = f"C{i:03d}_{p.stem}"

        out.append({
            "clip_id": clip_id,
            "video_path": str(p),
            "video_uri": p.as_uri(),
            "filename": p.name,
            **meta,
        })

    return out


def build_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)

    return r'''<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux table annotation 003A</title>
<style>
:root{
  --bg:#101218;
  --panel:#181b22;
  --line:#2b303b;
  --text:#edf0f7;
  --muted:#9ea7b8;
  --accent:#74d99f;
  --warn:#ffd37a;
}
*{box-sizing:border-box}
body{
  margin:0;
  background:var(--bg);
  color:var(--text);
  font-family:system-ui,Segoe UI,sans-serif;
}
header{
  padding:14px 18px;
  border-bottom:1px solid var(--line);
  display:flex;
  gap:12px;
  align-items:center;
  flex-wrap:wrap;
}
h1{
  margin:0;
  font-size:18px;
}
main{
  display:grid;
  grid-template-columns:minmax(0,1fr) 340px;
  gap:14px;
  padding:14px;
}
.panel{
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:14px;
  padding:12px;
}
.viewer{
  position:relative;
  width:100%;
  min-height:300px;
  background:#000;
  border-radius:14px;
  overflow:hidden;
}
video{
  display:block;
  width:100%;
  max-height:78vh;
  background:#000;
}
canvas{
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  cursor:crosshair;
}
button,select,input{
  background:#20242e;
  color:var(--text);
  border:1px solid var(--line);
  border-radius:10px;
  padding:8px 10px;
}
button{
  cursor:pointer;
}
button.active{
  border-color:var(--accent);
  box-shadow:0 0 0 1px var(--accent) inset;
}
button.warn{
  border-color:#7b6040;
  color:var(--warn);
}
.row{
  display:flex;
  gap:8px;
  flex-wrap:wrap;
  align-items:center;
  margin:8px 0;
}
.corner-grid{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:8px;
}
.small{
  font-size:12px;
  color:var(--muted);
}
.kbd{
  font-family:ui-monospace,Consolas,monospace;
  background:#101218;
  border:1px solid var(--line);
  border-radius:6px;
  padding:1px 5px;
}
pre{
  white-space:pre-wrap;
  word-break:break-word;
  background:#101218;
  border:1px solid var(--line);
  border-radius:10px;
  padding:10px;
  max-height:220px;
  overflow:auto;
  font-size:12px;
}
.status-ok{color:var(--accent)}
.status-warn{color:var(--warn)}
@media(max-width:1000px){
  main{grid-template-columns:1fr}
}
</style>
</head>
<body>
<header>
  <h1>TTFlux · annotation table 003A</h1>
  <select id="clipSelect"></select>
  <button id="prevBtn">← clip</button>
  <button id="nextBtn">clip →</button>
  <button id="startBtn">début</button>
  <button id="midBtn">milieu</button>
  <button id="playBtn">play/pause</button>
</header>

<main>
  <section class="panel">
    <div class="viewer" id="viewer">
      <video id="video" controls preload="metadata"></video>
      <canvas id="canvas"></canvas>
    </div>
    <p class="small">
      Clique les 4 coins de la table dans l’ordre :
      <b>avant-gauche</b>, <b>avant-droite</b>, <b>arrière-droite</b>, <b>arrière-gauche</b>.
      L’ordre exact sert surtout à obtenir un quadrilatère cohérent.
    </p>
  </section>

  <aside class="panel">
    <h2>Contrôle</h2>

    <div class="corner-grid">
      <button data-corner="0" class="cornerBtn active">1 · avant-gauche</button>
      <button data-corner="1" class="cornerBtn">2 · avant-droite</button>
      <button data-corner="2" class="cornerBtn">3 · arrière-droite</button>
      <button data-corner="3" class="cornerBtn">4 · arrière-gauche</button>
    </div>

    <div class="row">
      <button id="undoBtn">Annuler point</button>
      <button id="clearBtn" class="warn">Effacer clip</button>
    </div>

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
      <span class="kbd">1–4</span> choisir coin ·
      <span class="kbd">Z</span> annuler ·
      <span class="kbd">C</span> effacer clip ·
      <span class="kbd">N/P</span> clip suivant/précédent ·
      <span class="kbd">Espace</span> play/pause
    </p>

    <h3>Clip</h3>
    <pre id="clipInfo"></pre>

    <h3>Export actuel</h3>
    <pre id="jsonPreview"></pre>
  </aside>
</main>

<script>
const payload = __PAYLOAD__;
const clips = payload.clips || [];
const storageKey = "ttflux_003A_table_annotations_v1";

const cornerLabels = [
  "front_left",
  "front_right",
  "back_right",
  "back_left",
];

const cornerNames = [
  "avant-gauche",
  "avant-droite",
  "arrière-droite",
  "arrière-gauche",
];

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const viewer = document.getElementById("viewer");
const clipSelect = document.getElementById("clipSelect");
const statusEl = document.getElementById("status");
const clipInfoEl = document.getElementById("clipInfo");
const jsonPreviewEl = document.getElementById("jsonPreview");

let currentClipIndex = 0;
let currentCorner = 0;

let annotations = loadAnnotations();

function nowIso(){
  return new Date().toISOString();
}

function loadAnnotations(){
  try{
    const raw = localStorage.getItem(storageKey);
    if(raw) return JSON.parse(raw);
  }catch(e){}
  return {
    version: "003A",
    updated_at: nowIso(),
    clips: {}
  };
}

function saveAnnotations(){
  annotations.updated_at = nowIso();
  localStorage.setItem(storageKey, JSON.stringify(annotations));
  updateStatus();
}

function currentClip(){
  return clips[currentClipIndex];
}

function currentAnn(){
  const c = currentClip();
  if(!c) return null;

  if(!annotations.clips[c.clip_id]){
    annotations.clips[c.clip_id] = {
      clip_id: c.clip_id,
      filename: c.filename,
      video_path: c.video_path,
      width: c.width,
      height: c.height,
      fps: c.fps,
      frame_count: c.frame_count,
      table_quad: [null,null,null,null],
      frame_ref: 0,
      updated_at: nowIso()
    };
  }

  return annotations.clips[c.clip_id];
}

function initSelect(){
  clipSelect.innerHTML = "";

  clips.forEach((clip, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `${idx + 1}/${clips.length} · ${clip.filename}`;
    clipSelect.appendChild(opt);
  });

  clipSelect.addEventListener("change", () => {
    loadClip(Number(clipSelect.value || 0));
  });
}

function loadClip(index){
  if(!clips.length) return;

  currentClipIndex = Math.max(0, Math.min(index, clips.length - 1));
  const clip = currentClip();

  clipSelect.value = String(currentClipIndex);
  video.src = clip.video_uri;
  video.load();

  video.onloadedmetadata = () => {
    resizeCanvas();
    draw();
    updateStatus();
  };

  updateStatus();
}

function resizeCanvas(){
  const rect = viewer.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
  draw();
}

function getVideoGeom(){
  const vw = video.videoWidth || currentClip()?.width || 1;
  const vh = video.videoHeight || currentClip()?.height || 1;
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
  return {
    x: g.ox + pt.x * g.scale,
    y: g.oy + pt.y * g.scale
  };
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

function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);

  const ann = currentAnn();
  if(!ann) return;

  const pts = ann.table_quad || [];

  ctx.save();

  // Cadre vidéo réel dans canvas.
  const g = getVideoGeom();
  ctx.strokeStyle = "rgba(255,255,255,.28)";
  ctx.lineWidth = 1;
  ctx.strokeRect(g.ox, g.oy, g.drawW, g.drawH);

  const valid = pts
    .map((p, i) => p ? {...p, idx:i} : null)
    .filter(Boolean);

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

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#101218";
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "13px system-ui";
    ctx.fillText(`${i + 1} · ${cornerNames[i]}`, q.x + 10, q.y - 10);
  });

  ctx.restore();
}

function clickCanvas(evt){
  const ann = currentAnn();
  if(!ann) return;

  const p = eventToVideo(evt);

  ann.table_quad[currentCorner] = {
    label: cornerLabels[currentCorner],
    x: Math.round(p.x * 100) / 100,
    y: Math.round(p.y * 100) / 100
  };

  const clip = currentClip();
  const fps = Number(clip.fps || 0);
  ann.frame_ref = fps > 0 ? Math.round(video.currentTime * fps) : 0;
  ann.updated_at = nowIso();

  if(currentCorner < 3) setCorner(currentCorner + 1);

  saveAnnotations();
  draw();
}

function undoPoint(){
  const ann = currentAnn();
  if(!ann) return;

  for(let i = 3; i >= 0; i--){
    if(ann.table_quad[i]){
      ann.table_quad[i] = null;
      setCorner(i);
      break;
    }
  }

  ann.updated_at = nowIso();
  saveAnnotations();
  draw();
}

function clearClip(){
  const ann = currentAnn();
  if(!ann) return;

  ann.table_quad = [null,null,null,null];
  ann.updated_at = nowIso();
  setCorner(0);
  saveAnnotations();
  draw();
}

function completedCount(){
  let n = 0;

  clips.forEach(c => {
    const a = annotations.clips[c.clip_id];
    if(a && a.table_quad && a.table_quad.filter(Boolean).length === 4) n++;
  });

  return n;
}

function exportPayload(){
  const out = {
    version: "003A",
    exported_at: nowIso(),
    source: "TTFlux table_annotation_003A.html",
    clips: {}
  };

  clips.forEach(c => {
    const a = annotations.clips[c.clip_id];

    if(a){
      out.clips[c.clip_id] = a;
    }else{
      out.clips[c.clip_id] = {
        clip_id: c.clip_id,
        filename: c.filename,
        video_path: c.video_path,
        width: c.width,
        height: c.height,
        fps: c.fps,
        frame_count: c.frame_count,
        table_quad: [null,null,null,null],
        frame_ref: 0,
        updated_at: null
      };
    }
  });

  return out;
}

function updateStatus(){
  const c = currentClip();
  const ann = currentAnn();

  const done = completedCount();
  const here = ann ? ann.table_quad.filter(Boolean).length : 0;

  statusEl.innerHTML = `
    <p>Clip annoté : <b class="${here === 4 ? "status-ok" : "status-warn"}">${here}/4 coins</b></p>
    <p>Total : <b class="${done === clips.length ? "status-ok" : "status-warn"}">${done}/${clips.length} clips complets</b></p>
    <p>Coin courant : <b>${currentCorner + 1} · ${cornerNames[currentCorner]}</b></p>
  `;

  clipInfoEl.textContent = JSON.stringify(c || {}, null, 2);
  jsonPreviewEl.textContent = JSON.stringify(exportPayload(), null, 2);
}

function downloadJson(){
  const data = JSON.stringify(exportPayload(), null, 2);
  const blob = new Blob([data], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "table_annotations_003A.json";
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

      if(data.clips){
        annotations = {
          version: "003A",
          updated_at: nowIso(),
          clips: data.clips
        };
        saveAnnotations();
        draw();
        updateStatus();
      }else{
        alert("JSON invalide : champ clips manquant.");
      }
    }catch(e){
      alert("Impossible de lire le JSON.");
    }
  };

  reader.readAsText(file);
}

document.querySelectorAll(".cornerBtn").forEach(btn => {
  btn.addEventListener("click", () => setCorner(Number(btn.dataset.corner)));
});

document.getElementById("prevBtn").addEventListener("click", () => loadClip(currentClipIndex - 1));
document.getElementById("nextBtn").addEventListener("click", () => loadClip(currentClipIndex + 1));

document.getElementById("startBtn").addEventListener("click", () => {
  video.currentTime = 0;
});

document.getElementById("midBtn").addEventListener("click", () => {
  if(video.duration) video.currentTime = video.duration / 2;
});

document.getElementById("playBtn").addEventListener("click", () => {
  if(video.paused) video.play();
  else video.pause();
});

document.getElementById("undoBtn").addEventListener("click", undoPoint);
document.getElementById("clearBtn").addEventListener("click", clearClip);
document.getElementById("exportBtn").addEventListener("click", downloadJson);

document.getElementById("importBtn").addEventListener("click", () => {
  document.getElementById("importInput").click();
});

document.getElementById("importInput").addEventListener("change", evt => {
  const file = evt.target.files && evt.target.files[0];
  if(file) importJsonFile(file);
});

canvas.addEventListener("click", clickCanvas);

video.addEventListener("loadedmetadata", () => {
  resizeCanvas();
  updateStatus();
});

video.addEventListener("timeupdate", updateStatus);

window.addEventListener("resize", resizeCanvas);

document.addEventListener("keydown", evt => {
  if(evt.target && ["INPUT", "TEXTAREA", "SELECT"].includes(evt.target.tagName)) return;

  if(evt.key >= "1" && evt.key <= "4"){
    setCorner(Number(evt.key) - 1);
    evt.preventDefault();
  }else if(evt.key.toLowerCase() === "z"){
    undoPoint();
    evt.preventDefault();
  }else if(evt.key.toLowerCase() === "c"){
    clearClip();
    evt.preventDefault();
  }else if(evt.key.toLowerCase() === "n"){
    loadClip(currentClipIndex + 1);
    evt.preventDefault();
  }else if(evt.key.toLowerCase() === "p"){
    loadClip(currentClipIndex - 1);
    evt.preventDefault();
  }else if(evt.code === "Space"){
    if(video.paused) video.play();
    else video.pause();
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
    parser.add_argument("--config-dir", default="configs/batch_001E")
    parser.add_argument("--out-dir", default="runs/batch_001E/table_scene_003A")
    args = parser.parse_args()

    root = Path.cwd()
    config_dir = Path(args.config_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = collect_videos(config_dir=config_dir, root=root)

    if not clips:
        raise SystemExit(
            "[003A] Aucun fichier vidéo trouvé dans les JSON de config. "
            "Vérifie --config-dir."
        )

    payload = {
        "version": "003A",
        "project_root": str(root),
        "config_dir": str(config_dir),
        "clip_count": len(clips),
        "clips": clips,
    }

    html_path = out_dir / "table_annotation_003A.html"
    payload_path = out_dir / "table_annotation_payload_003A.json"

    html_path.write_text(build_html(payload), encoding="utf-8")
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[003A] clips : {len(clips)}")
    for c in clips:
        print(f"  - {c['clip_id']} | {c['width']}x{c['height']} | {c['filename']}")
    print(f"[003A] out   : {out_dir}")
    print(f"[003A] html  : {html_path}")


if __name__ == "__main__":
    main()
