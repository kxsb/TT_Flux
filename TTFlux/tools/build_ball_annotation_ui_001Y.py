from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import cv2


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_json_values(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from iter_json_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_json_values(v)


def find_video_in_config(config_path: Path) -> Path | None:
    data = read_json(config_path)
    base = config_path.parent
    candidates: list[Path] = []

    for key, value in iter_json_values(data):
        if not isinstance(value, str):
            continue

        low_key = str(key).lower()
        low_val = value.lower()

        if (
            "video" not in low_key
            and "clip" not in low_key
            and not low_val.endswith((".mp4", ".avi", ".mov", ".mkv"))
        ):
            continue

        if not low_val.endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        p = Path(value)

        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((base / p).resolve())
            candidates.append((Path.cwd() / p).resolve())

    for p in candidates:
        if p.exists():
            return p

    return None


def build_clip_video_map(config_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}

    for cfg in sorted(config_dir.glob("*.json")):
        video = find_video_in_config(cfg)
        if video is None:
            continue

        stem = cfg.stem

        m = re.search(r"(?:batch_001E_)?\d{2}_(.+)$", stem)
        if m:
            out[m.group(1)] = video

        out[stem] = video

    return out


def resolve_path(run_dir: Path, value: str) -> Path | None:
    value = str(value or "").strip()
    if not value:
        return None

    p = Path(value)

    if p.is_absolute():
        return p if p.exists() else None

    p1 = run_dir / p
    if p1.exists():
        return p1

    p2 = Path.cwd() / p
    if p2.exists():
        return p2

    return None


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def load_track_points(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.exists():
        return []

    rows = read_csv(path)
    points: list[dict[str, float]] = []

    for row in rows:
        frame = row.get("frame") or row.get("frame_idx") or row.get("f")
        x = row.get("x") or row.get("cx") or row.get("ball_x")
        y = row.get("y") or row.get("cy") or row.get("ball_y")

        if frame is None or x is None or y is None:
            continue

        points.append(
            {
                "frame": fnum(frame),
                "x": fnum(x),
                "y": fnum(y),
            }
        )

    return sorted(points, key=lambda p: p["frame"])


def video_meta(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return {
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "frame_count": 0,
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    cap.release()

    if fps <= 1:
        fps = 30.0

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def make_segment_payload(
    rows: list[dict[str, str]],
    run_dir: Path,
    config_dir: Path,
) -> list[dict[str, Any]]:
    video_map = build_clip_video_map(config_dir)
    payload: list[dict[str, Any]] = []

    for row in rows:
        clip_id = row.get("clip_id", "")
        video_path = video_map.get(clip_id)

        if video_path is None:
            continue

        csv_path = resolve_path(run_dir, row.get("csv", ""))
        track_points = load_track_points(csv_path)

        meta = video_meta(video_path)

        first_frame = int(fnum(row.get("first_frame"), 0))
        last_frame = int(fnum(row.get("last_frame"), 0))

        if track_points:
            first_frame = int(min(p["frame"] for p in track_points))
            last_frame = int(max(p["frame"] for p in track_points))

        item = {
            "review_id": row.get("review_id", ""),
            "clip_id": clip_id,
            "segment_name": row.get("segment_name", ""),
            "decision_001U": row.get("decision_001U", ""),
            "decision_reason_001U": row.get("decision_reason_001U", ""),
            "risk_score_001G": row.get("risk_score_001G", ""),
            "center_blob_false_score_001N": row.get("center_blob_false_score_001N", ""),
            "micro_keep_score_001O": row.get("micro_keep_score_001O", ""),
            "micro_false_score_001O": row.get("micro_false_score_001O", ""),
            "micro_distance_med_001O": row.get("micro_distance_med_001O", ""),
            "video_uri": video_path.resolve().as_uri(),
            "video_path": str(video_path),
            "csv_path": str(csv_path) if csv_path else "",
            "fps": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "frame_count": meta["frame_count"],
            "first_frame": first_frame,
            "last_frame": last_frame,
            "track_points": track_points,
        }

        payload.append(item)

    return payload


HTML_TEMPLATE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux ball trace annotation 001Y</title>
<style>
body {
  margin:0;
  background:#101218;
  color:#edf0f7;
  font-family:system-ui, Segoe UI, sans-serif;
}
header {
  position:sticky;
  top:0;
  z-index:20;
  background:#181b22;
  border-bottom:1px solid #2b303b;
  padding:12px 18px;
}
main {
  display:grid;
  grid-template-columns:minmax(420px, 1fr) 360px;
  gap:16px;
  padding:16px;
}
section, aside {
  background:#181b22;
  border:1px solid #2b303b;
  border-radius:16px;
  padding:14px;
}
h1, h2, h3 { margin-top:0; }
select, button, input {
  background:#101218;
  color:#edf0f7;
  border:1px solid #3a4150;
  border-radius:10px;
  padding:8px 10px;
}
button {
  cursor:pointer;
  font-weight:700;
}
button:hover { filter:brightness(1.18); }
.toolbar {
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  align-items:center;
}
.stage {
  position:relative;
  background:#000;
  border-radius:14px;
  overflow:hidden;
  width:100%;
}
#video {
  display:block;
  width:100%;
  max-height:78vh;
  background:#000;
}
#canvas {
  position:absolute;
  left:0;
  top:0;
  width:100%;
  height:100%;
  cursor:crosshair;
}
.kbd {
  background:#0b0d12;
  border:1px solid #3a4150;
  padding:2px 6px;
  border-radius:6px;
}
.grid2 {
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:8px;
}
.meta {
  color:#9ea7b8;
  font-size:13px;
  line-height:1.45;
}
.good { color:#9dffbc; }
.warn { color:#ffd37a; }
.bad { color:#ff9aa4; }
textarea {
  width:100%;
  min-height:180px;
  box-sizing:border-box;
  background:#0b0d12;
  color:#edf0f7;
  border:1px solid #3a4150;
  border-radius:12px;
  padding:10px;
  font-family:Consolas, monospace;
}
.point-list {
  max-height:260px;
  overflow:auto;
  border:1px solid #2b303b;
  border-radius:12px;
  padding:8px;
  font-family:Consolas, monospace;
  font-size:12px;
  background:#0b0d12;
}
.small { font-size:12px; color:#9ea7b8; }
</style>
</head>
<body>
<header>
  <h1>TTFlux ball trace annotation 001Y</h1>
  <div class="toolbar">
    <label>Segment :
      <select id="segmentSelect"></select>
    </label>
    <button onclick="prevSegment()">Segment précédent</button>
    <button onclick="nextSegment()">Segment suivant</button>
    <button onclick="exportCsv()">Exporter CSV</button>
    <button onclick="downloadCsv()">Télécharger CSV</button>
    <button onclick="clearCurrent()">Effacer ce segment</button>
  </div>
</header>

<main>
  <section>
    <div class="toolbar">
      <button onclick="stepFrame(-10)">-10f</button>
      <button onclick="stepFrame(-5)">-5f</button>
      <button onclick="stepFrame(-1)">-1f</button>
      <button onclick="togglePlay()">Play/Pause</button>
      <button onclick="stepFrame(1)">+1f</button>
      <button onclick="stepFrame(5)">+5f</button>
      <button onclick="stepFrame(10)">+10f</button>
      <button onclick="jumpStart()">Début segment</button>
      <button onclick="jumpEnd()">Fin segment</button>
      <label>
        <input type="checkbox" id="showTrack" onchange="draw()" checked>
        afficher tracker auto
      </label>
      <label>
        <input type="checkbox" id="showHuman" onchange="draw()" checked>
        afficher humain
      </label>
    </div>

    <p class="meta" id="frameInfo"></p>

    <div class="stage" id="stage">
      <video id="video" muted preload="auto"></video>
      <canvas id="canvas"></canvas>
    </div>

    <p class="small">
      Clic sur la balle = point humain pour la frame courante.
      Le point est stocké aux coordonnées vidéo originales.
    </p>
  </section>

  <aside>
    <h2>Mode annotation</h2>

    <div class="grid2">
      <button onclick="markInvisible()">Balle invisible</button>
      <button onclick="markUncertain()">Incertain</button>
      <button onclick="deleteCurrentFrame()">Supprimer frame</button>
      <button onclick="copyPreviousPoint()">Copier point précédent</button>
    </div>

    <h3>Raccourcis</h3>
    <p class="meta">
      <span class="kbd">Espace</span> play/pause<br>
      <span class="kbd">A</span>/<span class="kbd">D</span> frame -1 / +1<br>
      <span class="kbd">Q</span>/<span class="kbd">E</span> frame -5 / +5<br>
      <span class="kbd">W</span> invisible<br>
      <span class="kbd">S</span> incertain<br>
      <span class="kbd">X</span> supprimer frame<br>
      <span class="kbd">T</span> toggle tracker auto
    </p>

    <h3>Conseil</h3>
    <p class="meta">
      Ne trace pas forcément toutes les frames. Mets les points-clés :
      début, rebond, contact raquette, changement de direction, fin.
      On interpolera ensuite.
    </p>

    <h3>Segment courant</h3>
    <p class="meta" id="segmentMeta"></p>

    <h3>Points humains</h3>
    <div class="point-list" id="pointList"></div>

    <h3>CSV export</h3>
    <textarea id="csvBox" placeholder="Clique Exporter CSV ou Télécharger CSV."></textarea>
  </aside>
</main>

<script>
const SEGMENTS = __SEGMENTS_JSON__;
const STORAGE_KEY = "ttflux_001Y_ball_trace_v1";

let currentIndex = 0;
let currentSegment = null;
let mousePos = null;

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const segmentSelect = document.getElementById("segmentSelect");

function loadState() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch (e) {
    return {};
  }
}

function saveState(state) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function getSegState(rid) {
  const state = loadState();
  if (!state[rid]) state[rid] = { points: {}, note: "" };
  return { state, segState: state[rid] };
}

function initSelect() {
  segmentSelect.innerHTML = "";
  SEGMENTS.forEach((seg, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = `${idx + 1}. ${seg.review_id} · ${seg.segment_name}`;
    segmentSelect.appendChild(opt);
  });

  segmentSelect.addEventListener("change", () => {
    loadSegment(Number(segmentSelect.value));
  });
}

function loadSegment(idx) {
  currentIndex = Math.max(0, Math.min(idx, SEGMENTS.length - 1));
  currentSegment = SEGMENTS[currentIndex];
  segmentSelect.value = String(currentIndex);

  video.src = currentSegment.video_uri;
  video.onloadedmetadata = () => {
    resizeCanvas();
    setFrame(currentSegment.first_frame || 0);
    refreshInfo();
    draw();
  };

  refreshInfo();
  refreshPointList();
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
  draw();
}

function currentFrame() {
  if (!currentSegment) return 0;
  return Math.round(video.currentTime * currentSegment.fps);
}

function setFrame(frame) {
  if (!currentSegment) return;

  const minFrame = 0;
  const maxFrame = currentSegment.frame_count > 0 ? currentSegment.frame_count - 1 : frame;
  const clamped = Math.max(minFrame, Math.min(frame, maxFrame));

  video.pause();
  video.currentTime = clamped / currentSegment.fps;
  setTimeout(() => {
    refreshInfo();
    draw();
  }, 30);
}

function stepFrame(delta) {
  setFrame(currentFrame() + delta);
}

function togglePlay() {
  if (video.paused) video.play();
  else video.pause();
}

function jumpStart() {
  setFrame(currentSegment.first_frame || 0);
}

function jumpEnd() {
  setFrame(currentSegment.last_frame || currentFrame());
}

function prevSegment() {
  loadSegment(currentIndex - 1);
}

function nextSegment() {
  loadSegment(currentIndex + 1);
}

function videoToCanvas(x, y) {
  return {
    x: x / currentSegment.width * canvas.width,
    y: y / currentSegment.height * canvas.height,
  };
}

function canvasToVideo(evt) {
  const rect = canvas.getBoundingClientRect();
  const cx = evt.clientX - rect.left;
  const cy = evt.clientY - rect.top;

  return {
    x: cx / rect.width * currentSegment.width,
    y: cy / rect.height * currentSegment.height,
  };
}

function addHumanPoint(x, y, quality) {
  if (!currentSegment) return;

  const frame = currentFrame();
  const { state, segState } = getSegState(currentSegment.review_id);

  segState.points[String(frame)] = {
    frame,
    x: Math.round(x * 100) / 100,
    y: Math.round(y * 100) / 100,
    visible: quality === "visible" ? 1 : 0,
    quality,
    updated_at: new Date().toISOString(),
  };

  saveState(state);
  refreshInfo();
  refreshPointList();
  draw();
}

canvas.addEventListener("click", (evt) => {
  const p = canvasToVideo(evt);
  addHumanPoint(p.x, p.y, "visible");
});

canvas.addEventListener("mousemove", (evt) => {
  mousePos = canvasToVideo(evt);
  draw();
});

canvas.addEventListener("mouseleave", () => {
  mousePos = null;
  draw();
});

function markInvisible() {
  if (!currentSegment) return;
  const frame = currentFrame();
  const { state, segState } = getSegState(currentSegment.review_id);

  segState.points[String(frame)] = {
    frame,
    x: "",
    y: "",
    visible: 0,
    quality: "invisible",
    updated_at: new Date().toISOString(),
  };

  saveState(state);
  refreshInfo();
  refreshPointList();
  draw();
}

function markUncertain() {
  if (!currentSegment) return;
  const frame = currentFrame();
  const { state, segState } = getSegState(currentSegment.review_id);

  segState.points[String(frame)] = {
    frame,
    x: "",
    y: "",
    visible: 0,
    quality: "uncertain",
    updated_at: new Date().toISOString(),
  };

  saveState(state);
  refreshInfo();
  refreshPointList();
  draw();
}

function deleteCurrentFrame() {
  if (!currentSegment) return;
  const frame = currentFrame();
  const { state, segState } = getSegState(currentSegment.review_id);

  delete segState.points[String(frame)];

  saveState(state);
  refreshInfo();
  refreshPointList();
  draw();
}

function copyPreviousPoint() {
  if (!currentSegment) return;

  const frame = currentFrame();
  const { state, segState } = getSegState(currentSegment.review_id);

  const frames = Object.keys(segState.points)
    .map(x => Number(x))
    .filter(x => x < frame)
    .sort((a, b) => b - a);

  if (!frames.length) return;

  const prev = segState.points[String(frames[0])];
  segState.points[String(frame)] = {
    ...prev,
    frame,
    updated_at: new Date().toISOString(),
  };

  saveState(state);
  refreshInfo();
  refreshPointList();
  draw();
}

function clearCurrent() {
  if (!currentSegment) return;
  if (!confirm("Effacer les annotations humaines de ce segment ?")) return;

  const state = loadState();
  delete state[currentSegment.review_id];
  saveState(state);

  refreshInfo();
  refreshPointList();
  draw();
}

function drawCross(x, y, color, size = 10) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x - size, y);
  ctx.lineTo(x + size, y);
  ctx.moveTo(x, y - size);
  ctx.lineTo(x, y + size);
  ctx.stroke();
}

function drawCircle(x, y, color, radius = 7) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.stroke();
}

function drawTrack() {
  if (!document.getElementById("showTrack").checked) return;
  if (!currentSegment || !currentSegment.track_points) return;

  const pts = currentSegment.track_points;

  ctx.strokeStyle = "rgba(255, 220, 0, 0.75)";
  ctx.lineWidth = 2;
  ctx.beginPath();

  let started = false;

  pts.forEach(p => {
    const q = videoToCanvas(p.x, p.y);
    if (!started) {
      ctx.moveTo(q.x, q.y);
      started = true;
    } else {
      ctx.lineTo(q.x, q.y);
    }
  });

  ctx.stroke();

  const frame = currentFrame();
  const nearest = pts.reduce((best, p) => {
    const d = Math.abs(p.frame - frame);
    if (!best || d < best.d) return { p, d };
    return best;
  }, null);

  if (nearest && nearest.d <= 2) {
    const q = videoToCanvas(nearest.p.x, nearest.p.y);
    drawCross(q.x, q.y, "rgba(255,60,60,0.95)", 11);
    drawCircle(q.x, q.y, "rgba(255,220,0,0.95)", 9);
  }
}

function drawHuman() {
  if (!document.getElementById("showHuman").checked) return;
  if (!currentSegment) return;

  const state = loadState();
  const segState = state[currentSegment.review_id] || { points: {} };

  const pts = Object.values(segState.points || {})
    .filter(p => p.quality === "visible" && p.x !== "" && p.y !== "")
    .sort((a, b) => a.frame - b.frame);

  ctx.strokeStyle = "rgba(80,255,140,0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();

  let started = false;

  pts.forEach(p => {
    const q = videoToCanvas(Number(p.x), Number(p.y));
    if (!started) {
      ctx.moveTo(q.x, q.y);
      started = true;
    } else {
      ctx.lineTo(q.x, q.y);
    }
  });

  ctx.stroke();

  pts.forEach(p => {
    const q = videoToCanvas(Number(p.x), Number(p.y));
    drawCircle(q.x, q.y, "rgba(80,255,140,0.95)", 5);
  });

  const frame = currentFrame();
  const cur = segState.points[String(frame)];

  if (cur && cur.quality === "visible") {
    const q = videoToCanvas(Number(cur.x), Number(cur.y));
    drawCross(q.x, q.y, "rgba(80,255,140,1)", 13);
  }
}

function drawMouse() {
  if (!mousePos || !currentSegment) return;
  const q = videoToCanvas(mousePos.x, mousePos.y);
  drawCross(q.x, q.y, "rgba(120,180,255,0.85)", 8);
}

function draw() {
  resizeCanvasNoDraw();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawTrack();
  drawHuman();
  drawMouse();
}

function resizeCanvasNoDraw() {
  const rect = video.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));

  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function refreshInfo() {
  if (!currentSegment) return;

  const frame = currentFrame();
  const state = loadState();
  const segState = state[currentSegment.review_id] || { points: {} };
  const cur = segState.points[String(frame)];
  const nPoints = Object.keys(segState.points || {}).length;

  document.getElementById("frameInfo").innerHTML =
    `Frame courante : <b>${frame}</b> · segment ${currentSegment.first_frame} → ${currentSegment.last_frame} · ` +
    `annotations humaines : <span class="good">${nPoints}</span>` +
    (cur ? ` · frame annotée : <span class="warn">${cur.quality}</span>` : "");

  document.getElementById("segmentMeta").innerHTML =
    `<b>${currentSegment.review_id}</b><br>` +
    `${currentSegment.segment_name}<br>` +
    `clip : <code>${currentSegment.clip_id}</code><br>` +
    `raison 001U : <code>${currentSegment.decision_reason_001U}</code><br>` +
    `risk=${currentSegment.risk_score_001G} · center=${currentSegment.center_blob_false_score_001N} · ` +
    `micro_keep=${currentSegment.micro_keep_score_001O} · micro_false=${currentSegment.micro_false_score_001O}<br>` +
    `vidéo : ${currentSegment.width}x${currentSegment.height} · fps=${currentSegment.fps.toFixed(3)}`;
}

function refreshPointList() {
  if (!currentSegment) return;

  const state = loadState();
  const segState = state[currentSegment.review_id] || { points: {} };

  const pts = Object.values(segState.points || {})
    .sort((a, b) => a.frame - b.frame);

  if (!pts.length) {
    document.getElementById("pointList").textContent = "aucun point humain";
    return;
  }

  document.getElementById("pointList").innerHTML = pts.map(p => {
    if (p.quality === "visible") {
      return `f${p.frame} · x=${p.x} y=${p.y} · visible`;
    }
    return `f${p.frame} · ${p.quality}`;
  }).join("<br>");
}

function csvEscape(value) {
  value = String(value ?? "");
  if (value.includes('"') || value.includes(",") || value.includes("\n") || value.includes("\r")) {
    return '"' + value.replaceAll('"', '""') + '"';
  }
  return value;
}

function buildCsv() {
  const state = loadState();

  const cols = [
    "review_id",
    "clip_id",
    "segment_name",
    "frame",
    "x",
    "y",
    "visible",
    "quality",
    "source",
    "video_path",
    "fps",
    "width",
    "height",
    "updated_at"
  ];

  const lines = [cols.join(",")];

  SEGMENTS.forEach(seg => {
    const segState = state[seg.review_id] || { points: {} };

    Object.values(segState.points || {})
      .sort((a, b) => a.frame - b.frame)
      .forEach(p => {
        const row = {
          review_id: seg.review_id,
          clip_id: seg.clip_id,
          segment_name: seg.segment_name,
          frame: p.frame,
          x: p.x ?? "",
          y: p.y ?? "",
          visible: p.visible ?? "",
          quality: p.quality ?? "",
          source: "human_001Y",
          video_path: seg.video_path,
          fps: seg.fps,
          width: seg.width,
          height: seg.height,
          updated_at: p.updated_at || "",
        };

        lines.push(cols.map(c => csvEscape(row[c] || "")).join(","));
      });
  });

  return lines.join("\n");
}

function exportCsv() {
  const csv = buildCsv();
  document.getElementById("csvBox").value = csv;
  return csv;
}

function downloadCsv() {
  const csv = exportCsv();
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "ball_trace_annotations_001Y.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();

  URL.revokeObjectURL(url);
}

document.addEventListener("keydown", ev => {
  if (ev.target && ["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName)) return;

  if (ev.key === " ") {
    ev.preventDefault();
    togglePlay();
  } else if (ev.key.toLowerCase() === "a") {
    stepFrame(-1);
  } else if (ev.key.toLowerCase() === "d") {
    stepFrame(1);
  } else if (ev.key.toLowerCase() === "q") {
    stepFrame(-5);
  } else if (ev.key.toLowerCase() === "e") {
    stepFrame(5);
  } else if (ev.key.toLowerCase() === "w") {
    markInvisible();
  } else if (ev.key.toLowerCase() === "s") {
    markUncertain();
  } else if (ev.key.toLowerCase() === "x") {
    deleteCurrentFrame();
  } else if (ev.key.toLowerCase() === "t") {
    const cb = document.getElementById("showTrack");
    cb.checked = !cb.checked;
    draw();
  }
});

video.addEventListener("timeupdate", () => {
  refreshInfo();
  draw();
});

video.addEventListener("seeked", () => {
  refreshInfo();
  draw();
});

window.addEventListener("resize", () => {
  resizeCanvas();
});

initSelect();
loadSegment(0);
</script>
</body>
</html>
"""


def write_html(out_html: Path, segments: list[dict[str, Any]]) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    doc = HTML_TEMPLATE.replace(
        "__SEGMENTS_JSON__",
        json.dumps(segments, ensure_ascii=False),
    )

    out_html.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--config-dir", default="configs/batch_001E")
    parser.add_argument("--review-csv", default="runs/batch_001E/arbiter_001V/arbiter_001U_review.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/ball_annotation_001Y.html")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_dir = Path(args.config_dir).resolve()
    review_csv = Path(args.review_csv)
    out_html = Path(args.out_html)

    rows = read_csv(review_csv)
    segments = make_segment_payload(rows, run_dir=run_dir, config_dir=config_dir)

    write_html(out_html, segments)

    print(f"[001Y] review rows : {len(rows)}")
    print(f"[001Y] segments    : {len(segments)}")
    print(f"[001Y] wrote HTML  : {out_html}")


if __name__ == "__main__":
    main()
