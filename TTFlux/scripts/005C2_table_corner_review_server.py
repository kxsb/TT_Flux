from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd


VERSION = "005C2"

TABLE_LENGTH_M = 2.740
TABLE_WIDTH_M = 1.525
TABLE_HALF_L = TABLE_LENGTH_M / 2.0
TABLE_HALF_W = TABLE_WIDTH_M / 2.0


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def order_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.asarray([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def encode_jpg_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def draw_quad(img, quad, color, label):
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    qi = np.round(q).astype(int)

    for a, b in zip(qi, np.vstack([qi[1:], qi[:1]])):
        cv2.line(img, tuple(a), tuple(b), color, 3, cv2.LINE_AA)

    names = ["TL", "TR", "BR", "BL"]
    for i, p in enumerate(qi):
        cv2.circle(img, tuple(p), 7, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            names[i],
            (int(p[0]) + 8, int(p[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        img,
        label,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        color,
        2,
        cv2.LINE_AA,
    )


def table_homographies_from_quad_src(quad_src):
    table_pts = np.asarray([
        [-TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W,  TABLE_HALF_L],
        [-TABLE_HALF_W,  TABLE_HALF_L],
    ], dtype=np.float32)

    quad_src = np.asarray(quad_src, dtype=np.float32).reshape(4, 2)
    h_img_to_table = cv2.getPerspectiveTransform(quad_src, table_pts)
    h_table_to_img = cv2.getPerspectiveTransform(table_pts, quad_src)
    return h_img_to_table, h_table_to_img


def load_frame(clip_path: Path, frame_idx: int, max_w: int):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None, 1.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None, 1.0

    h, w = frame.shape[:2]
    scale = 1.0

    if max_w > 0 and w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)

    return frame, scale


def build_items(table_objects: pd.DataFrame, max_w: int):
    items = []

    for idx, r in table_objects.iterrows():
        clip_path = Path(str(r["clip_path"]))
        frame_idx = int(to_num(pd.Series([r.get("best_frame_005C1", r.get("first_frame", 0))])).fillna(0).iloc[0])

        img, scale = load_frame(clip_path, frame_idx, max_w=max_w)
        if img is None:
            continue

        quad_small = np.asarray([
            [float(r["quad_tl_x_005C1"]), float(r["quad_tl_y_005C1"])],
            [float(r["quad_tr_x_005C1"]), float(r["quad_tr_y_005C1"])],
            [float(r["quad_br_x_005C1"]), float(r["quad_br_y_005C1"])],
            [float(r["quad_bl_x_005C1"]), float(r["quad_bl_y_005C1"])],
        ], dtype=np.float32)

        # 005C1 quad est en resize_w/resize_h.
        src_w = float(r["src_w"])
        src_h = float(r["src_h"])
        resize_w = float(r["resize_w"])
        resize_h = float(r["resize_h"])

        sx = src_w / max(1.0, resize_w)
        sy = src_h / max(1.0, resize_h)

        quad_src = quad_small.copy()
        quad_src[:, 0] *= sx
        quad_src[:, 1] *= sy

        quad_disp = quad_src.copy() * scale

        draw = img.copy()
        draw_quad(
            draw,
            quad_disp,
            (0, 255, 255),
            f"{r['review_id']} seg={r['camera_segment_id_005B2']} proposal",
        )

        items.append({
            "idx": len(items),
            "review_id": str(r["review_id"]),
            "camera_segment_id": int(r["camera_segment_id_005B2"]),
            "video_id": str(r.get("video_id", "")),
            "rally_id": str(r.get("rally_id", "")),
            "frame": frame_idx,
            "src_w": int(src_w),
            "src_h": int(src_h),
            "display_w": int(draw.shape[1]),
            "display_h": int(draw.shape[0]),
            "display_scale": float(scale),
            "clip_path": str(clip_path),
            "proposal_quad_src": quad_src.round(3).tolist(),
            "proposal_quad_display": quad_disp.round(3).tolist(),
            "table_confidence_005C1": float(r.get("table_confidence_005C1", 0.0)),
            "img_b64": encode_jpg_b64(draw),
        })

    return items


class ServerState:
    def __init__(self, out_dir: Path, items: list[dict]):
        self.out_dir = out_dir
        self.items = items
        self.lock = threading.Lock()
        self.corrections_csv = out_dir / "table_corner_corrections_005C2.csv"
        self.votes = self.load_votes()

    def key(self, review_id, camera_segment_id):
        return f"{review_id}__seg{camera_segment_id}"

    def load_votes(self):
        if not self.corrections_csv.is_file():
            return {}

        try:
            df = pd.read_csv(self.corrections_csv).fillna("")
        except Exception:
            return {}

        out = {}
        for _, r in df.iterrows():
            out[self.key(r["review_id"], r["camera_segment_id"])] = r.to_dict()
        return out

    def save(self, payload):
        review_id = str(payload["review_id"])
        seg = int(payload["camera_segment_id"])
        scale = float(payload["display_scale"])
        points = payload.get("points", [])

        item = next(
            (x for x in self.items if x["review_id"] == review_id and int(x["camera_segment_id"]) == seg),
            None,
        )

        if item is None:
            return {"ok": False, "error": "item_not_found"}

        if len(points) != 4:
            return {"ok": False, "error": "need_4_points"}

        pts_display = np.asarray([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float32)
        pts_src = pts_display / max(1e-9, scale)
        pts_src = order_quad(pts_src)

        H_img_to_table, H_table_to_img = table_homographies_from_quad_src(pts_src)

        row = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "version": VERSION,
            "review_id": review_id,
            "camera_segment_id": seg,
            "video_id": item.get("video_id", ""),
            "rally_id": item.get("rally_id", ""),
            "frame": int(item["frame"]),
            "src_w": int(item["src_w"]),
            "src_h": int(item["src_h"]),
            "display_scale": scale,
            "status": str(payload.get("status", "corrected")),
            "comment": str(payload.get("comment", "")),
            "tl_x": round(float(pts_src[0, 0]), 3),
            "tl_y": round(float(pts_src[0, 1]), 3),
            "tr_x": round(float(pts_src[1, 0]), 3),
            "tr_y": round(float(pts_src[1, 1]), 3),
            "br_x": round(float(pts_src[2, 0]), 3),
            "br_y": round(float(pts_src[2, 1]), 3),
            "bl_x": round(float(pts_src[3, 0]), 3),
            "bl_y": round(float(pts_src[3, 1]), 3),
            "H_img_to_table_005C2": json.dumps(H_img_to_table.reshape(-1).round(10).tolist()),
            "H_table_to_img_005C2": json.dumps(H_table_to_img.reshape(-1).round(10).tolist()),
            "clip_path": item["clip_path"],
        }

        with self.lock:
            self.votes[self.key(review_id, seg)] = row
            pd.DataFrame(list(self.votes.values())).to_csv(self.corrections_csv, index=False, encoding="utf-8")

        return {"ok": True, "saved": row}


def index_html():
    return r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 005C2 · correction table</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#101219;color:#eef1f8;font-family:system-ui,Segoe UI,sans-serif}
header{padding:12px 18px;background:#171b25;border-bottom:1px solid #303746;display:flex;justify-content:space-between;gap:12px}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:14px;padding:14px}
.canvasBox{background:#05070b;border:1px solid #303746;border-radius:12px;padding:10px;overflow:auto}
canvas{max-width:100%;background:#000;border-radius:8px;cursor:crosshair}
.panel{background:#181d27;border:1px solid #303746;border-radius:12px;padding:12px}
button{border:1px solid #3b4659;background:#252d3d;color:#eef1f8;border-radius:9px;padding:9px 11px;cursor:pointer;margin:3px}
button:hover{background:#303a50}
button.good{background:#245b37}
button.bad{background:#653030}
textarea{width:100%;min-height:70px;background:#101219;color:#eef1f8;border:1px solid #3b4659;border-radius:9px;padding:8px;box-sizing:border-box}
.meta{font-size:13px;color:#c1cada;line-height:1.55}
.small{font-size:12px;color:#9faabd}
.pt{font-family:ui-monospace,Consolas,monospace;font-size:12px}
</style>
</head>
<body>
<header>
  <b>TTFlux 005C2 · correction table 4 coins</b>
  <div id="progress">chargement…</div>
</header>

<main>
  <section class="canvasBox">
    <canvas id="canvas"></canvas>
  </section>

  <aside class="panel">
    <div class="meta" id="meta"></div>
    <hr>
    <p class="small">
      Clique 4 coins dans l’ordre : haut-gauche, haut-droite, bas-droite, bas-gauche.
      <br>Raccourcis : N suivant, P précédent, S sauver, R reset, A accepter proposition.
    </p>

    <div id="points" class="pt"></div>

    <textarea id="comment" placeholder="commentaire optionnel"></textarea>

    <div>
      <button class="good" onclick="acceptProposal()">A · accepter proposition</button>
      <button onclick="saveCorrection('corrected', false)">S · sauver</button>
      <button onclick="saveCorrection('corrected', true)">sauver + suivant</button>
      <button class="bad" onclick="saveCorrection('bad_frame', true)">bad frame</button>
      <button onclick="resetPoints()">R · reset</button>
    </div>

    <div>
      <button onclick="prevItem()">P · précédent</button>
      <button onclick="nextItem()">N · suivant</button>
    </div>
  </aside>
</main>

<script>
let items = [];
let votes = {};
let idx = Number(localStorage.getItem("ttflux_005C2_idx") || 0);
let img = new Image();
let points = [];
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

async function load() {
  const res = await fetch("/api/items");
  const data = await res.json();
  items = data.items || [];
  votes = data.votes || {};
  if (idx >= items.length) idx = 0;
  render();
}

function current(){ return items[idx]; }

function key(item){ return item.review_id + "__seg" + item.camera_segment_id; }

function render() {
  const item = current();
  if (!item) return;

  localStorage.setItem("ttflux_005C2_idx", String(idx));
  document.getElementById("progress").textContent = `${idx+1} / ${items.length}`;

  const saved = votes[key(item)] || null;

  document.getElementById("meta").innerHTML =
    `<b>${item.review_id}</b> · seg ${item.camera_segment_id}<br>` +
    `frame ${item.frame} · conf auto ${item.table_confidence_005C1.toFixed(3)}<br>` +
    `video ${item.video_id}<br>` +
    `status: <b>${saved ? saved.status : "non corrigé"}</b>`;

  document.getElementById("comment").value = saved ? (saved.comment || "") : "";

  points = [];

  img = new Image();
  img.onload = () => {
    canvas.width = item.display_w;
    canvas.height = item.display_h;
    redraw();
  };
  img.src = "data:image/jpeg;base64," + item.img_b64;
}

function redraw() {
  const item = current();
  ctx.drawImage(img, 0, 0);

  drawQuad(item.proposal_quad_display, "rgba(255,255,0,0.95)", "auto");

  if (points.length) {
    drawQuad(points.map(p => [p.x, p.y]), "rgba(0,255,0,0.95)", "manual");

    points.forEach((p, i) => {
      ctx.fillStyle = "lime";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI*2);
      ctx.fill();
      ctx.font = "18px sans-serif";
      ctx.fillText(String(i+1), p.x+8, p.y-8);
    });
  }

  document.getElementById("points").innerHTML = points.map((p,i) =>
    `${i+1}: ${p.x.toFixed(1)}, ${p.y.toFixed(1)}`
  ).join("<br>");
}

function drawQuad(q, color, label) {
  if (!q || q.length !== 4) return;

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(q[0][0], q[0][1]);
  for (let i=1;i<4;i++) ctx.lineTo(q[i][0], q[i][1]);
  ctx.closePath();
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.font = "18px sans-serif";
  ctx.fillText(label, q[0][0]+8, q[0][1]+22);
}

canvas.addEventListener("click", ev => {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  const x = (ev.clientX - rect.left) * sx;
  const y = (ev.clientY - rect.top) * sy;

  if (points.length >= 4) points = [];
  points.push({x, y});
  redraw();
});

function resetPoints(){
  points = [];
  redraw();
}

function acceptProposal(){
  const item = current();
  points = item.proposal_quad_display.map(p => ({x:p[0], y:p[1]}));
  redraw();
}

async function saveCorrection(status, goNext){
  const item = current();

  let usePoints = points;
  if (status === "bad_frame" && usePoints.length !== 4) {
    usePoints = item.proposal_quad_display.map(p => ({x:p[0], y:p[1]}));
  }

  const payload = {
    review_id: item.review_id,
    camera_segment_id: item.camera_segment_id,
    display_scale: item.display_scale,
    points: usePoints,
    status,
    comment: document.getElementById("comment").value || "",
  };

  const res = await fetch("/api/save", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(payload)
  });

  const data = await res.json();
  if (!data.ok) {
    alert(data.error || "erreur");
    return;
  }

  votes[key(item)] = data.saved;
  if (goNext) nextItem();
  else render();
}

function nextItem(){ idx = Math.min(items.length-1, idx+1); render(); }
function prevItem(){ idx = Math.max(0, idx-1); render(); }

document.addEventListener("keydown", ev => {
  if (ev.target && ev.target.tagName === "TEXTAREA") return;
  const k = ev.key.toLowerCase();
  if (k === "n") nextItem();
  if (k === "p") prevItem();
  if (k === "s") saveCorrection("corrected", false);
  if (k === "r") resetPoints();
  if (k === "a") acceptProposal();
});

load();
</script>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-objects", default="runs/rally_table_object_005C1_pass33/table_objects_005C1.csv")
    ap.add_argument("--out-dir", default="runs/rally_table_object_005C2_corner_review")
    ap.add_argument("--port", type=int, default=8774)
    ap.add_argument("--max-w", type=int, default=1280)
    ap.add_argument("--only-low-confidence", action="store_true")
    ap.add_argument("--confidence-max", type=float, default=0.70)
    args = ap.parse_args()

    root = Path.cwd()

    table_path = Path(args.table_objects)
    out_dir = Path(args.out_dir)

    if not table_path.is_absolute():
        table_path = root / table_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(table_path).fillna("")

    if args.only_low_confidence:
        table = table[to_num(table["table_confidence_005C1"]).fillna(0) <= args.confidence_max].copy()

    table = table.sort_values(
        ["table_ok_005C1", "table_confidence_005C1", "review_id", "camera_segment_id_005B2"],
        ascending=[True, True, True, True],
    ).copy()

    items = build_items(table, max_w=args.max_w)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "table_objects": str(table_path),
        "out_dir": str(out_dir),
        "items": len(items),
        "only_low_confidence": bool(args.only_low_confidence),
        "confidence_max": args.confidence_max,
        "corrections_csv": str(out_dir / "table_corner_corrections_005C2.csv"),
    }

    (out_dir / "table_corner_review_summary_005C2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("005C2 server READY")
    print("items=", len(items))
    print("url=", f"http://127.0.0.1:{args.port}/")
    print("corrections_csv=", out_dir / "table_corner_corrections_005C2.csv")

    if not items:
        print("Aucun item à corriger.")
        return 1

    state = ServerState(out_dir, items)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def send_json(self, obj, status=200):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_html(self, text):
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_html(index_html())
            elif path == "/api/items":
                self.send_json({"items": state.items, "votes": state.votes})
            else:
                self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/api/save":
                self.send_error(404)
                return

            n = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(n).decode("utf-8")

            try:
                payload = json.loads(raw)
            except Exception:
                self.send_json({"ok": False, "error": "invalid_json"}, status=400)
                return

            res = state.save(payload)
            self.send_json(res, 200 if res.get("ok") else 400)

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
