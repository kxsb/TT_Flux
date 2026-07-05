from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import shutil
import subprocess
import time
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

import pandas as pd


VERSION = "004J"


STATE = {
    "root": None,
    "run_dir": None,
    "packet_csv": None,
    "out_dir": None,
    "assets_dir": None,
    "clicks_csv": None,
    "records": [],
}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def safe_name(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("_")
    v = "".join(out).strip("_")
    return v or "unknown"


def relpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def transcode_h264(ffmpeg: str, src: Path, dst: Path) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_file() and dst.stat().st_size > 0:
        return True, ""

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(src),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        return False, (p.stderr or p.stdout or f"ffmpeg_failed_{p.returncode}").strip()

    if not dst.is_file() or dst.stat().st_size <= 0:
        return False, "empty_output"

    return True, ""


def prepare_records(packet_csv: Path, run_dir: Path, out_dir: Path, limit: int = 0) -> list[dict]:
    if not packet_csv.is_file():
        raise SystemExit(f"Packet absent: {packet_csv}")

    df = pd.read_csv(packet_csv)

    if limit and limit > 0:
        df = df.head(limit).copy()

    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg introuvable dans le PATH")

    records = []

    print("004J prepare records =", len(df))
    print("004J ffmpeg =", ffmpeg)

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        review_id = str(r.get("review_id", f"R{i:04d}"))
        rank = int(r.get("gold_rank_004I", i))
        video_id = str(r.get("video_id", ""))
        clip_id = str(r.get("clip_id", ""))
        segment_idx = str(r.get("segment_idx", ""))

        mp4_rel = str(r.get("mp4", ""))
        src = run_dir / mp4_rel

        asset_name = f"{rank:03d}_{safe_name(review_id)}_{safe_name(video_id)}_seg{safe_name(segment_idx)}.mp4"
        asset = assets_dir / asset_name

        ok = False
        err = ""

        if not src.is_file():
            err = f"source_missing: {src}"
        else:
            ok, err = transcode_h264(ffmpeg, src, asset)

        first_frame = int(float(r.get("first_frame", 0) or 0))
        last_frame = int(float(r.get("last_frame", 0) or 0))

        rec = {
            "idx": i - 1,
            "gold_rank_004I": rank,
            "review_id": review_id,
            "video_id": video_id,
            "clip_id": clip_id,
            "segment_idx": segment_idx,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "n_points": str(r.get("n_points", "")),
            "density": str(r.get("density", "")),
            "travel": str(r.get("travel", "")),
            "guess": str(r.get("guess", "")),
            "flags": "" if pd.isna(r.get("flags", "")) else str(r.get("flags", "")),
            "source_mp4": str(src),
            "asset": "/assets/" + asset.name if ok else "",
            "asset_path": str(asset) if ok else "",
            "asset_ok": bool(ok),
            "asset_error": err,
            "fps": 50.0,
        }

        records.append(rec)

        if i % 10 == 0 or i == len(df):
            print(f"  prepared {i}/{len(df)}")

    return records


def ensure_clicks_csv(path: Path) -> None:
    if path.is_file():
        return

    fieldnames = [
        "created_at",
        "review_id",
        "gold_rank_004I",
        "video_id",
        "clip_id",
        "segment_idx",
        "visibility",
        "local_time_sec",
        "local_frame",
        "source_frame",
        "x",
        "y",
        "video_w",
        "video_h",
        "comment",
        "user_action",
        "version",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_click(row: dict) -> None:
    path = STATE["clicks_csv"]
    ensure_clicks_csv(path)

    fieldnames = [
        "created_at",
        "review_id",
        "gold_rank_004I",
        "video_id",
        "clip_id",
        "segment_idx",
        "visibility",
        "local_time_sec",
        "local_frame",
        "source_frame",
        "x",
        "y",
        "video_w",
        "video_h",
        "comment",
        "user_action",
        "version",
    ]

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_clicks() -> list[dict]:
    path = STATE["clicks_csv"]
    if not path.is_file():
        return []

    try:
        return pd.read_csv(path).fillna("").to_dict("records")
    except Exception:
        return []


def delete_last_click(review_id: str | None = None) -> dict:
    path = STATE["clicks_csv"]
    if not path.is_file():
        return {"ok": False, "error": "clicks_csv_missing"}

    df = pd.read_csv(path).fillna("")
    if df.empty:
        return {"ok": False, "error": "no_clicks"}

    if review_id:
        mask = df["review_id"].astype(str).eq(str(review_id))
        idxs = df.index[mask].tolist()
        if not idxs:
            return {"ok": False, "error": f"no_click_for_review_id={review_id}"}
        drop_idx = idxs[-1]
    else:
        drop_idx = df.index[-1]

    removed = df.loc[drop_idx].to_dict()
    df = df.drop(index=drop_idx)
    df.to_csv(path, index=False, encoding="utf-8")

    return {"ok": True, "removed": removed}


def make_index_html() -> str:
    records_json = json.dumps(STATE["records"], ensure_ascii=False)
    clicks_json = json.dumps(read_clicks(), ensure_ascii=False)

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004J Ball Goldset Clicker</title>
<style>
:root {{
  --bg:#101218;
  --panel:#181b22;
  --line:#2b303b;
  --text:#edf0f7;
  --muted:#aeb7c8;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0;
  background:var(--bg);
  color:var(--text);
  font-family:system-ui,Segoe UI,sans-serif;
}}
header {{
  padding:12px 16px;
  border-bottom:1px solid var(--line);
  background:#151820;
  display:flex;
  gap:16px;
  align-items:center;
  position:sticky;
  top:0;
  z-index:5;
}}
button, select, input {{
  background:#101218;
  color:var(--text);
  border:1px solid #3b4150;
  border-radius:8px;
  padding:8px 10px;
}}
button:hover {{ background:#242936; cursor:pointer; }}
main {{
  display:grid;
  grid-template-columns: 330px 1fr;
  min-height:calc(100vh - 56px);
}}
aside {{
  border-right:1px solid var(--line);
  overflow:auto;
  max-height:calc(100vh - 56px);
  padding:10px;
}}
.item {{
  border:1px solid var(--line);
  border-radius:10px;
  padding:8px;
  margin-bottom:8px;
  background:var(--panel);
  cursor:pointer;
}}
.item.active {{ outline:2px solid #8bb8ff; }}
.item .small {{ color:var(--muted); font-size:12px; }}
.stage {{
  padding:16px;
}}
.panel {{
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:14px;
  padding:12px;
  margin-bottom:12px;
}}
.videoWrap {{
  position:relative;
  display:inline-block;
  background:#000;
  border-radius:12px;
  overflow:hidden;
  max-width:100%;
}}
video {{
  display:block;
  max-width: min(100%, 1100px);
  max-height: 70vh;
  background:#000;
}}
canvas {{
  position:absolute;
  left:0;
  top:0;
  pointer-events:auto;
}}
.row {{
  display:flex;
  gap:8px;
  flex-wrap:wrap;
  align-items:center;
}}
.kbd {{
  font-family:ui-monospace,Consolas,monospace;
  color:#cbd3e6;
  font-size:12px;
}}
.status {{
  color:#cbd3e6;
  font-size:13px;
}}
table {{
  width:100%;
  border-collapse:collapse;
}}
td, th {{
  border-bottom:1px solid var(--line);
  padding:6px;
  text-align:left;
  font-size:12px;
}}
th {{ color:#cbd3e6; }}
.good {{ color:#8df0b0; }}
.bad {{ color:#ff9d9d; }}
</style>
</head>
<body>
<header>
  <strong>TTFlux 004J · clic balle réelle</strong>
  <span id="topStatus" class="status"></span>
  <button id="prevBtn">← segment</button>
  <button id="nextBtn">segment →</button>
  <button id="undoBtn">annuler dernier clic</button>
</header>

<main>
  <aside>
    <div class="panel">
      <div><strong>Segments</strong></div>
      <div class="small">Clique un segment, puis clique la balle dans la vidéo.</div>
    </div>
    <div id="list"></div>
  </aside>

  <section class="stage">
    <div class="panel">
      <div class="row">
        <select id="visibility">
          <option value="ball">ball</option>
          <option value="not_visible">not_visible</option>
          <option value="unsure">unsure</option>
        </select>
        <input id="comment" placeholder="commentaire court" style="min-width:320px">
        <button id="markBtn">marquer frame sans clic</button>
        <button id="backFrameBtn">frame -1</button>
        <button id="forwardFrameBtn">frame +1</button>
        <button id="playBtn">play/pause</button>
      </div>
      <p class="kbd">
        Raccourcis : espace play/pause · A/D frame -/+ · 1 ball · 2 not_visible · 3 unsure · N marque not_visible · U marque unsure · Z undo.
      </p>
    </div>

    <div class="panel">
      <div id="meta"></div>
    </div>

    <div class="videoWrap" id="videoWrap">
      <video id="video" controls preload="metadata"></video>
      <canvas id="canvas"></canvas>
    </div>

    <div class="panel">
      <h3>Clics du segment courant</h3>
      <table>
        <thead>
          <tr><th>visibility</th><th>local_frame</th><th>source_frame</th><th>x</th><th>y</th><th>comment</th></tr>
        </thead>
        <tbody id="clickRows"></tbody>
      </table>
    </div>
  </section>
</main>

<script>
const RECORDS = {records_json};
let clicks = {clicks_json};
let current = 0;

const listEl = document.getElementById("list");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const meta = document.getElementById("meta");
const topStatus = document.getElementById("topStatus");
const visibility = document.getElementById("visibility");
const comment = document.getElementById("comment");
const clickRows = document.getElementById("clickRows");

function rec() {{ return RECORDS[current]; }}

function clickCount(reviewId) {{
  return clicks.filter(c => String(c.review_id) === String(reviewId)).length;
}}

function renderList() {{
  listEl.innerHTML = "";
  RECORDS.forEach((r, i) => {{
    const div = document.createElement("div");
    div.className = "item" + (i === current ? " active" : "");
    const c = clickCount(r.review_id);
    div.innerHTML = `
      <div><strong>${{r.gold_rank_004I}} · ${{r.review_id}}</strong> <span class="${{c ? 'good' : 'bad'}}">${{c}} clics</span></div>
      <div class="small">${{r.video_id}} · seg ${{r.segment_idx}} · travel ${{r.travel}}</div>
      <div class="small">${{r.flags || ""}}</div>
    `;
    div.onclick = () => loadSegment(i);
    listEl.appendChild(div);
  }});
}}

function loadSegment(i) {{
  current = Math.max(0, Math.min(RECORDS.length - 1, i));
  const r = rec();

  video.src = r.asset || "";
  video.load();

  meta.innerHTML = `
    <strong>${{r.review_id}}</strong> · ${{r.video_id}} · ${{r.clip_id}} · seg ${{r.segment_idx}}<br>
    frames source : ${{r.first_frame}} → ${{r.last_frame}} · n=${{r.n_points}} · density=${{r.density}} · travel=${{r.travel}}<br>
    flags : ${{r.flags || ""}}<br>
    asset : ${{r.asset_ok ? "OK" : "ERROR " + r.asset_error}}
  `;

  topStatus.textContent = `${{current + 1}} / ${{RECORDS.length}} · ${{r.review_id}} · clics=${{clickCount(r.review_id)}}`;

  renderList();
  renderClicks();
  resizeCanvasSoon();
}}

function resizeCanvasSoon() {{
  setTimeout(resizeCanvas, 250);
}}

function resizeCanvas() {{
  const rect = video.getBoundingClientRect();
  canvas.width = Math.round(rect.width);
  canvas.height = Math.round(rect.height);
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";
  drawOverlay();
}}

function currentFrame() {{
  const fps = Number(rec().fps || 50);
  return Math.round(video.currentTime * fps);
}}

function sourceFrame() {{
  return Number(rec().first_frame || 0) + currentFrame();
}}

function drawOverlay() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const r = rec();
  const segClicks = clicks.filter(c => String(c.review_id) === String(r.review_id));

  for (const c of segClicks) {{
    if (c.visibility !== "ball") continue;
    if (c.x === "" || c.y === "") continue;

    const vw = Number(c.video_w || video.videoWidth || 1);
    const vh = Number(c.video_h || video.videoHeight || 1);
    const x = Number(c.x) * canvas.width / vw;
    const y = Number(c.y) * canvas.height / vh;

    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#00ff88";
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x - 8, y);
    ctx.lineTo(x + 8, y);
    ctx.moveTo(x, y - 8);
    ctx.lineTo(x, y + 8);
    ctx.stroke();
  }}
}}

async function saveAnnotation(payload) {{
  const res = await fetch("/api/annotate", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify(payload)
  }});
  const data = await res.json();
  if (!data.ok) {{
    alert("save error: " + data.error);
    return;
  }}
  clicks = data.clicks;
  renderList();
  renderClicks();
  drawOverlay();
  topStatus.textContent = `${{current + 1}} / ${{RECORDS.length}} · ${{rec().review_id}} · clics=${{clickCount(rec().review_id)}}`;
}}

function makePayload(x, y, action) {{
  const r = rec();
  const vf = currentFrame();
  const sf = sourceFrame();
  return {{
    review_id: r.review_id,
    gold_rank_004I: r.gold_rank_004I,
    video_id: r.video_id,
    clip_id: r.clip_id,
    segment_idx: r.segment_idx,
    visibility: visibility.value,
    local_time_sec: Number(video.currentTime || 0).toFixed(4),
    local_frame: vf,
    source_frame: sf,
    x: x,
    y: y,
    video_w: video.videoWidth || "",
    video_h: video.videoHeight || "",
    comment: comment.value || "",
    user_action: action
  }};
}}

canvas.addEventListener("click", async (ev) => {{
  const r = rec();
  if (!r.asset_ok) return;

  const rect = canvas.getBoundingClientRect();
  const rx = ev.clientX - rect.left;
  const ry = ev.clientY - rect.top;

  const x = Math.round(rx * (video.videoWidth || canvas.width) / canvas.width);
  const y = Math.round(ry * (video.videoHeight || canvas.height) / canvas.height);

  visibility.value = "ball";
  await saveAnnotation(makePayload(x, y, "click_ball"));
}});

document.getElementById("markBtn").onclick = async () => {{
  await saveAnnotation(makePayload("", "", "mark_frame"));
}};

document.getElementById("prevBtn").onclick = () => loadSegment(current - 1);
document.getElementById("nextBtn").onclick = () => loadSegment(current + 1);
document.getElementById("playBtn").onclick = () => {{
  if (video.paused) video.play(); else video.pause();
}};

function stepFrame(delta) {{
  const fps = Number(rec().fps || 50);
  video.pause();
  video.currentTime = Math.max(0, video.currentTime + delta / fps);
}}

document.getElementById("backFrameBtn").onclick = () => stepFrame(-1);
document.getElementById("forwardFrameBtn").onclick = () => stepFrame(1);

document.getElementById("undoBtn").onclick = async () => {{
  const res = await fetch("/api/delete_last", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{review_id: rec().review_id}})
  }});
  const data = await res.json();
  if (!data.ok) {{
    alert(data.error);
    return;
  }}
  clicks = data.clicks;
  renderList();
  renderClicks();
  drawOverlay();
}};

function renderClicks() {{
  const r = rec();
  const segClicks = clicks.filter(c => String(c.review_id) === String(r.review_id));
  clickRows.innerHTML = "";
  for (const c of segClicks) {{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${{c.visibility}}</td>
      <td>${{c.local_frame}}</td>
      <td>${{c.source_frame}}</td>
      <td>${{c.x}}</td>
      <td>${{c.y}}</td>
      <td>${{c.comment || ""}}</td>
    `;
    clickRows.appendChild(tr);
  }}
}}

window.addEventListener("resize", resizeCanvas);
video.addEventListener("loadedmetadata", resizeCanvasSoon);
video.addEventListener("timeupdate", drawOverlay);

document.addEventListener("keydown", async (ev) => {{
  if (ev.target && ["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName)) return;

  if (ev.code === "Space") {{
    ev.preventDefault();
    if (video.paused) video.play(); else video.pause();
  }}
  if (ev.key.toLowerCase() === "a") stepFrame(-1);
  if (ev.key.toLowerCase() === "d") stepFrame(1);
  if (ev.key === "1") visibility.value = "ball";
  if (ev.key === "2") visibility.value = "not_visible";
  if (ev.key === "3") visibility.value = "unsure";
  if (ev.key.toLowerCase() === "n") {{
    visibility.value = "not_visible";
    await saveAnnotation(makePayload("", "", "mark_not_visible"));
  }}
  if (ev.key.toLowerCase() === "u") {{
    visibility.value = "unsure";
    await saveAnnotation(makePayload("", "", "mark_unsure"));
  }}
  if (ev.key.toLowerCase() === "z") {{
    document.getElementById("undoBtn").click();
  }}
  if (ev.key === "ArrowRight") loadSegment(current + 1);
  if (ev.key === "ArrowLeft") loadSegment(current - 1);
}});

renderList();
loadSegment(0);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            data = make_index_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/api/state":
            self.send_json({
                "ok": True,
                "version": VERSION,
                "records": STATE["records"],
                "clicks": read_clicks(),
            })
            return

        if path.startswith("/assets/"):
            name = unquote(path.split("/assets/", 1)[1])
            file_path = STATE["assets_dir"] / name
            self.serve_file(file_path)
            return

        self.send_error(404, "not found")

    def serve_file(self, file_path: Path):
        if not file_path.is_file():
            self.send_error(404, "file missing")
            return

        ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        size = file_path.stat().st_size

        # Support Range minimal pour lecture vidéo.
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            try:
                start_s, end_s = range_header.replace("bytes=", "").split("-", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()

                with file_path.open("rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return
            except Exception:
                pass

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.end_headers()

        with file_path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
        except Exception as exc:
            self.send_json({"ok": False, "error": repr(exc)}, status=400)
            return

        if path == "/api/annotate":
            try:
                row = {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "review_id": str(payload.get("review_id", "")),
                    "gold_rank_004I": payload.get("gold_rank_004I", ""),
                    "video_id": str(payload.get("video_id", "")),
                    "clip_id": str(payload.get("clip_id", "")),
                    "segment_idx": payload.get("segment_idx", ""),
                    "visibility": str(payload.get("visibility", "ball")),
                    "local_time_sec": payload.get("local_time_sec", ""),
                    "local_frame": payload.get("local_frame", ""),
                    "source_frame": payload.get("source_frame", ""),
                    "x": payload.get("x", ""),
                    "y": payload.get("y", ""),
                    "video_w": payload.get("video_w", ""),
                    "video_h": payload.get("video_h", ""),
                    "comment": str(payload.get("comment", "")),
                    "user_action": str(payload.get("user_action", "")),
                    "version": VERSION,
                }

                append_click(row)
                self.send_json({"ok": True, "clicks": read_clicks()})
                return

            except Exception as exc:
                self.send_json({"ok": False, "error": repr(exc)}, status=500)
                return

        if path == "/api/delete_last":
            review_id = payload.get("review_id")
            res = delete_last_click(str(review_id) if review_id else None)
            res["clicks"] = read_clicks()
            self.send_json(res)
            return

        self.send_json({"ok": False, "error": "unknown endpoint"}, status=404)

    def log_message(self, fmt, *args):
        # Plus silencieux.
        return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet-csv", default="runs/ball_goldset_004I/ball_goldset_packet_004I.csv")
    ap.add_argument("--run-dir", default="runs/batch_004F_full240")
    ap.add_argument("--out-dir", default="runs/ball_goldset_004J")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path.cwd()

    packet_csv = Path(args.packet_csv)
    if not packet_csv.is_absolute():
        packet_csv = root / packet_csv

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    clicks_csv = out_dir / "ball_clicks_004J.csv"
    ensure_clicks_csv(clicks_csv)

    records = prepare_records(packet_csv, run_dir, out_dir, limit=args.limit)

    STATE.update({
        "root": root,
        "run_dir": run_dir,
        "packet_csv": packet_csv,
        "out_dir": out_dir,
        "assets_dir": assets_dir,
        "clicks_csv": clicks_csv,
        "records": records,
    })

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "packet_csv": str(packet_csv),
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "assets_dir": str(assets_dir),
        "clicks_csv": str(clicks_csv),
        "records": len(records),
        "asset_ok": sum(1 for r in records if r.get("asset_ok")),
        "asset_failed": sum(1 for r in records if not r.get("asset_ok")),
        "url": f"http://127.0.0.1:{args.port}/",
    }

    (out_dir / "ball_goldset_server_summary_004J.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("004J status=SERVER_READY")
    print("records=", summary["records"])
    print("asset_ok=", summary["asset_ok"])
    print("asset_failed=", summary["asset_failed"])
    print("clicks_csv=", clicks_csv)
    print("url=", summary["url"])
    print("Stop serveur: Ctrl+C")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
