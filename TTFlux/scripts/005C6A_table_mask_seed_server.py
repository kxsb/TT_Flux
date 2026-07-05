from __future__ import annotations

import argparse
import base64
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd


VERSION = "005C6A"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def encode_jpg_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


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


def draw_current_quad(img, r):
    try:
        quad = np.asarray([
            [float(r["quad_tl_x_005C3"]), float(r["quad_tl_y_005C3"])],
            [float(r["quad_tr_x_005C3"]), float(r["quad_tr_y_005C3"])],
            [float(r["quad_br_x_005C3"]), float(r["quad_br_y_005C3"])],
            [float(r["quad_bl_x_005C3"]), float(r["quad_bl_y_005C3"])],
        ], dtype=np.float32)
    except Exception:
        return

    q = np.round(quad).astype(int)

    for a, b in zip(q, np.vstack([q[1:], q[:1]])):
        cv2.line(img, tuple(a), tuple(b), (0, 255, 255), 2, cv2.LINE_AA)

    for p in q:
        cv2.circle(img, tuple(p), 5, (0, 255, 255), -1, cv2.LINE_AA)


def build_items(table_objects: pd.DataFrame, audit: pd.DataFrame | None, max_w: int):
    items = []

    table = table_objects.copy()

    if audit is not None and not audit.empty:
        audit = audit.copy()
        audit["camera_segment_id_005B2"] = to_num(audit["camera_segment_id"]).fillna(1).astype(int)

        keep_cols = [
            "review_id",
            "camera_segment_id_005B2",
            "points_projected",
            "points_inside",
            "inside_ratio",
            "image",
            "status_005C4",
        ]

        keep_cols = [c for c in keep_cols if c in audit.columns]

        table = table.merge(
            audit[keep_cols],
            on=["review_id", "camera_segment_id_005B2"],
            how="left",
        )

    # Priorité : cas douteux / partiels / inside faible, puis quelques bons pour contraste.
    table["_inside"] = to_num(table.get("inside_ratio", 0)).fillna(-1)
    table["_points"] = to_num(table.get("points_projected", 0)).fillna(0)
    table["_source_priority"] = table.get("table_source_005C3", "").astype(str).map({
        "005C5B_semantic": 0,
        "005C2_human": 1,
        "005C1_auto": 2,
    }).fillna(3)

    table = table.sort_values(
        ["_inside", "_source_priority", "_points"],
        ascending=[True, True, False],
    ).copy()

    for _, r in table.iterrows():
        clip_path = Path(str(r["clip_path"]))
        frame_idx = int(to_num(pd.Series([r.get("best_frame_005C1", r.get("first_frame", 0))])).fillna(0).iloc[0])

        img, scale = load_frame(clip_path, frame_idx, max_w=max_w)
        if img is None:
            continue

        # Quad 005C3 est en coords source. On scale en coords display.
        draw_img = img.copy()

        rr = r.copy()
        for c in [
            "quad_tl_x_005C3", "quad_tr_x_005C3", "quad_br_x_005C3", "quad_bl_x_005C3",
        ]:
            if c in rr:
                rr[c] = float(rr[c]) * scale

        for c in [
            "quad_tl_y_005C3", "quad_tr_y_005C3", "quad_br_y_005C3", "quad_bl_y_005C3",
        ]:
            if c in rr:
                rr[c] = float(rr[c]) * scale

        draw_current_quad(draw_img, rr)

        cv2.putText(
            draw_img,
            f"{r['review_id']} seg={int(r['camera_segment_id_005B2'])} source={r.get('table_source_005C3','')}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        inside_ratio = float(to_num(pd.Series([r.get("inside_ratio", -1)])).fillna(-1).iloc[0])
        points_projected = int(to_num(pd.Series([r.get("points_projected", 0)])).fillna(0).iloc[0])

        cv2.putText(
            draw_img,
            f"inside_ratio={inside_ratio:.3f} points={points_projected}",
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        items.append({
            "idx": len(items),
            "review_id": str(r["review_id"]),
            "camera_segment_id": int(r["camera_segment_id_005B2"]),
            "video_id": str(r.get("video_id", "")),
            "rally_id": str(r.get("rally_id", "")),
            "frame": frame_idx,
            "display_w": int(draw_img.shape[1]),
            "display_h": int(draw_img.shape[0]),
            "display_scale": float(scale),
            "clip_path": str(clip_path),
            "table_source": str(r.get("table_source_005C3", "")),
            "table_status": str(r.get("table_status_005C3", "")),
            "inside_ratio": inside_ratio,
            "points_projected": points_projected,
            "img_b64": encode_jpg_b64(draw_img),
        })

    return items


class ServerState:
    def __init__(self, out_dir: Path, items: list[dict]):
        self.out_dir = out_dir
        self.items = items
        self.lock = threading.Lock()
        self.csv_path = out_dir / "table_mask_seed_points_005C6A.csv"
        self.saved = self.load_saved()

    def key(self, review_id, seg):
        return f"{review_id}__seg{seg}"

    def load_saved(self):
        if not self.csv_path.is_file():
            return {}

        try:
            df = pd.read_csv(self.csv_path).fillna("")
        except Exception:
            return {}

        saved = {}

        for _, r in df.iterrows():
            k = self.key(r["review_id"], r["camera_segment_id"])
            saved.setdefault(k, []).append(r.to_dict())

        return saved

    def save_points(self, payload):
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

        rows = []

        for p in points:
            label = str(p.get("label", "table"))
            x_disp = float(p["x"])
            y_disp = float(p["y"])
            radius_disp = float(p.get("radius", 8))
            source = str(p.get("source", "click"))

            rows.append({
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "version": VERSION,
                "review_id": review_id,
                "camera_segment_id": seg,
                "video_id": item.get("video_id", ""),
                "rally_id": item.get("rally_id", ""),
                "frame": int(item["frame"]),
                "label": label,
                "x_display": round(x_disp, 3),
                "y_display": round(y_disp, 3),
                "radius_display": round(radius_disp, 3),
                "x_src": round(x_disp / max(1e-9, scale), 3),
                "y_src": round(y_disp / max(1e-9, scale), 3),
                "radius_src": round(radius_disp / max(1e-9, scale), 3),
                "display_scale": scale,
                "source": source,
                "clip_path": item["clip_path"],
            })

        with self.lock:
            k = self.key(review_id, seg)
            self.saved[k] = rows

            all_rows = []
            for arr in self.saved.values():
                all_rows.extend(arr)

            pd.DataFrame(all_rows).to_csv(self.csv_path, index=False, encoding="utf-8")

        return {"ok": True, "saved_count": len(rows), "total_keys": len(self.saved)}


def index_html():
    return r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 005C6A · table mask seeds</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#101219;color:#eef1f8;font-family:system-ui,Segoe UI,sans-serif}
header{padding:12px 18px;background:#171b25;border-bottom:1px solid #303746;display:flex;justify-content:space-between;gap:12px}
main{display:grid;grid-template-columns:minmax(0,1fr) 390px;gap:14px;padding:14px}
.canvasBox{background:#05070b;border:1px solid #303746;border-radius:12px;padding:10px;overflow:auto}
canvas{max-width:100%;background:#000;border-radius:8px;cursor:crosshair}
.panel{background:#181d27;border:1px solid #303746;border-radius:12px;padding:12px}
button{border:1px solid #3b4659;background:#252d3d;color:#eef1f8;border-radius:9px;padding:9px 11px;cursor:pointer;margin:3px}
button:hover{background:#303a50}
button.active{outline:2px solid #fff}
button.table{background:#245b37}
button.notable{background:#633030}
button.edge{background:#59522a}
.meta{font-size:13px;color:#c1cada;line-height:1.55}
.warn{background:#182516;border:1px solid #345c2b;border-radius:8px;padding:8px;color:#cfeec8;margin:8px 0}
.pt{font-family:ui-monospace,Consolas,monospace;font-size:12px;max-height:180px;overflow:auto}
input{width:80px}
</style>
</head>
<body>
<header>
  <b>TTFlux 005C6A · graines masque table / non-table</b>
  <div id="progress">chargement…</div>
</header>

<main>
  <section class="canvasBox">
    <canvas id="canvas"></canvas>
  </section>

  <aside class="panel">
    <div class="meta" id="meta"></div>

    <div class="warn">
      But : donner des exemples pixels.
      <br><b>T</b> = table, <b>N</b> = non-table, <b>E</b> = bord/ligne table.
      <br>Clic ou drag. Pas besoin d’être ultra précis.
    </div>

    <div>
      <button id="btnTable" class="table active" onclick="setLabel('table')">T table</button>
      <button id="btnNon" class="notable" onclick="setLabel('non_table')">N non-table</button>
      <button id="btnEdge" class="edge" onclick="setLabel('edge')">E bord/ligne</button>
    </div>

    <div>
      Rayon <input id="radius" type="number" value="9" min="2" max="50">
    </div>

    <div>
      <button onclick="save(false)">S sauver</button>
      <button onclick="save(true)">sauver + suivant</button>
      <button onclick="clearCurrent()">R reset image</button>
    </div>

    <div>
      <button onclick="prevItem()">P précédent</button>
      <button onclick="nextItem()">N suivant</button>
    </div>

    <hr>
    <div class="pt" id="points"></div>
  </aside>
</main>

<script>
let items = [];
let saved = {};
let idx = Number(localStorage.getItem("ttflux_005C6A_idx") || 0);
let img = new Image();
let points = [];
let label = "table";
let drawing = false;
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

async function load() {
  const res = await fetch("/api/items");
  const data = await res.json();
  items = data.items || [];
  saved = data.saved || {};
  if (idx >= items.length) idx = 0;
  render();
}

function key(item){ return item.review_id + "__seg" + item.camera_segment_id; }
function current(){ return items[idx]; }

function setLabel(v) {
  label = v;
  document.getElementById("btnTable").classList.toggle("active", v==="table");
  document.getElementById("btnNon").classList.toggle("active", v==="non_table");
  document.getElementById("btnEdge").classList.toggle("active", v==="edge");
}

function render() {
  const item = current();
  if (!item) return;

  localStorage.setItem("ttflux_005C6A_idx", String(idx));
  document.getElementById("progress").textContent = `${idx+1} / ${items.length}`;

  document.getElementById("meta").innerHTML =
    `<b>${item.review_id}</b> · seg ${item.camera_segment_id}<br>` +
    `frame ${item.frame}<br>` +
    `source table : ${item.table_source}<br>` +
    `inside_ratio : ${item.inside_ratio}<br>` +
    `points projetés : ${item.points_projected}`;

  points = (saved[key(item)] || []).map(p => ({
    x: Number(p.x_display),
    y: Number(p.y_display),
    radius: Number(p.radius_display || 9),
    label: String(p.label || "table"),
    source: String(p.source || "saved")
  }));

  img = new Image();
  img.onload = () => {
    canvas.width = item.display_w;
    canvas.height = item.display_h;
    redraw();
  };
  img.src = "data:image/jpeg;base64," + item.img_b64;
}

function colorFor(l) {
  if (l === "table") return "rgba(0,255,0,0.85)";
  if (l === "non_table") return "rgba(255,60,60,0.85)";
  return "rgba(255,210,0,0.90)";
}

function redraw() {
  ctx.drawImage(img, 0, 0);

  for (const p of points) {
    ctx.fillStyle = colorFor(p.label);
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.radius, 0, Math.PI*2);
    ctx.fill();
  }

  const counts = {};
  for (const p of points) counts[p.label] = (counts[p.label] || 0) + 1;

  document.getElementById("points").innerHTML =
    `table=${counts.table||0} non_table=${counts.non_table||0} edge=${counts.edge||0}<br>` +
    points.slice(-40).map((p,i) => `${p.label}: ${p.x.toFixed(1)}, ${p.y.toFixed(1)}, r=${p.radius}`).join("<br>");
}

function addPoint(ev, source) {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  const x = (ev.clientX - rect.left) * sx;
  const y = (ev.clientY - rect.top) * sy;
  const radius = Number(document.getElementById("radius").value || 9);

  const last = points.length ? points[points.length - 1] : null;
  if (last && Math.hypot(last.x - x, last.y - y) < radius * 0.6 && last.label === label) {
    return;
  }

  points.push({x, y, radius, label, source});
  redraw();
}

canvas.addEventListener("mousedown", ev => {
  drawing = true;
  addPoint(ev, "mousedown");
});
canvas.addEventListener("mousemove", ev => {
  if (drawing) addPoint(ev, "drag");
});
window.addEventListener("mouseup", () => drawing = false);

function clearCurrent() {
  points = [];
  redraw();
}

async function save(goNext) {
  const item = current();

  const payload = {
    review_id: item.review_id,
    camera_segment_id: item.camera_segment_id,
    display_scale: item.display_scale,
    points
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

  saved[key(item)] = points.map(p => ({
    review_id: item.review_id,
    camera_segment_id: item.camera_segment_id,
    x_display: p.x,
    y_display: p.y,
    radius_display: p.radius,
    label: p.label,
    source: p.source
  }));

  if (goNext) nextItem();
  else render();
}

function nextItem(){ idx = Math.min(items.length-1, idx+1); render(); }
function prevItem(){ idx = Math.max(0, idx-1); render(); }

document.addEventListener("keydown", ev => {
  if (ev.target && ev.target.tagName === "INPUT") return;
  const k = ev.key.toLowerCase();
  if (k === "t") setLabel("table");
  if (k === "n") setLabel("non_table");
  if (k === "e") setLabel("edge");
  if (k === "s") save(false);
  if (k === "r") clearCurrent();
  if (k === "arrowright") nextItem();
  if (k === "arrowleft") prevItem();
  if (k === "p") prevItem();
});

load();
</script>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-objects", default="runs/rally_table_object_005C5C2_merged/table_objects_merged_005C5C2.csv")
    ap.add_argument("--audit-csv", default="runs/rally_table_object_005C5C2_merged_audit/table_object_canonical_audit_005C4.csv")
    ap.add_argument("--out-dir", default="runs/rally_table_mask_seed_005C6A")
    ap.add_argument("--port", type=int, default=8776)
    ap.add_argument("--max-w", type=int, default=1280)
    ap.add_argument("--max-items", type=int, default=0)
    args = ap.parse_args()

    root = Path.cwd()

    table_path = Path(args.table_objects)
    audit_path = Path(args.audit_csv)
    out_dir = Path(args.out_dir)

    if not table_path.is_absolute():
        table_path = root / table_path
    if not audit_path.is_absolute():
        audit_path = root / audit_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(table_path).fillna("")
    audit = pd.read_csv(audit_path).fillna("") if audit_path.is_file() else pd.DataFrame()

    if args.max_items > 0:
        table = table.head(args.max_items).copy()

    items = build_items(table, audit, max_w=args.max_w)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "human_seed_points_for_table_vs_non_table_segmentation",
        "table_objects": str(table_path),
        "audit_csv": str(audit_path),
        "out_dir": str(out_dir),
        "items": len(items),
        "seeds_csv": str(out_dir / "table_mask_seed_points_005C6A.csv"),
    }

    (out_dir / "table_mask_seed_server_summary_005C6A.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("005C6A server READY")
    print("items=", len(items))
    print("url=", f"http://127.0.0.1:{args.port}/")
    print("seeds_csv=", out_dir / "table_mask_seed_points_005C6A.csv")

    if not items:
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
                self.send_json({"items": state.items, "saved": state.saved})
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

            res = state.save_points(payload)
            self.send_json(res, 200 if res.get("ok") else 400)

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.serve_forever()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
