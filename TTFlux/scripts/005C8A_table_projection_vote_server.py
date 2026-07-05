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


VERSION = "005C8A"

TABLE_LENGTH_M = 2.740
TABLE_WIDTH_M = 1.525
TABLE_HALF_L = TABLE_LENGTH_M / 2.0
TABLE_HALF_W = TABLE_WIDTH_M / 2.0


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def encode_jpg_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def read_frame(clip_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    return frame


def resize_max_w(img, max_w: int):
    h, w = img.shape[:2]
    if max_w <= 0 or w <= max_w:
        return img.copy(), 1.0

    scale = max_w / max(1, w)
    out = cv2.resize(img, (max_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def snap_quad_src(row):
    return np.asarray([
        [float(row["snap_tl_x_005C7A"]), float(row["snap_tl_y_005C7A"])],
        [float(row["snap_tr_x_005C7A"]), float(row["snap_tr_y_005C7A"])],
        [float(row["snap_br_x_005C7A"]), float(row["snap_br_y_005C7A"])],
        [float(row["snap_bl_x_005C7A"]), float(row["snap_bl_y_005C7A"])],
    ], dtype=np.float32)


def old_quad_src(row):
    return np.asarray([
        [float(row["quad_tl_x_005C3"]), float(row["quad_tl_y_005C3"])],
        [float(row["quad_tr_x_005C3"]), float(row["quad_tr_y_005C3"])],
        [float(row["quad_br_x_005C3"]), float(row["quad_br_y_005C3"])],
        [float(row["quad_bl_x_005C3"]), float(row["quad_bl_y_005C3"])],
    ], dtype=np.float32)


def table_reference_points():
    # Coordonnées métriques officielles de la surface de jeu.
    perimeter = [
        (-TABLE_HALF_W, -TABLE_HALF_L),
        ( TABLE_HALF_W, -TABLE_HALF_L),
        ( TABLE_HALF_W,  TABLE_HALF_L),
        (-TABLE_HALF_W,  TABLE_HALF_L),
        (-TABLE_HALF_W, -TABLE_HALF_L),
    ]

    net = [
        (-TABLE_HALF_W, 0.0),
        ( TABLE_HALF_W, 0.0),
    ]

    center = [
        (0.0, -TABLE_HALF_L),
        (0.0,  TABLE_HALF_L),
    ]

    grid_lines = []

    for x in np.linspace(-TABLE_HALF_W, TABLE_HALF_W, 5):
        grid_lines.append([(float(x), -TABLE_HALF_L), (float(x), TABLE_HALF_L)])

    for y in np.linspace(-TABLE_HALF_L, TABLE_HALF_L, 7):
        grid_lines.append([(-TABLE_HALF_W, float(y)), (TABLE_HALF_W, float(y))])

    return perimeter, net, center, grid_lines


def homography_table_to_img_from_quad(quad):
    table_corners = np.asarray([
        [-TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W,  TABLE_HALF_L],
        [-TABLE_HALF_W,  TABLE_HALF_L],
    ], dtype=np.float32)

    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(table_corners, quad)


def project_table_points(H, pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H.astype(np.float64))
    return out.reshape(-1, 2)


def draw_reference_table(frame, quad, label, color_main=(255, 0, 255)):
    H = homography_table_to_img_from_quad(quad)

    perimeter, net, center, grid_lines = table_reference_points()

    def pt(p):
        return (int(round(float(p[0]))), int(round(float(p[1]))))

    # Grille fine.
    for line in grid_lines:
        pp = project_table_points(H, line)
        cv2.line(frame, pt(pp[0]), pt(pp[1]), (90, 180, 180), 1, cv2.LINE_AA)

    # Contour officiel.
    pp = project_table_points(H, perimeter)
    for a, b in zip(pp[:-1], pp[1:]):
        cv2.line(frame, pt(a), pt(b), color_main, 3, cv2.LINE_AA)

    # Filet.
    pp = project_table_points(H, net)
    cv2.line(frame, pt(pp[0]), pt(pp[1]), (0, 180, 255), 3, cv2.LINE_AA)

    # Ligne centrale.
    pp = project_table_points(H, center)
    cv2.line(frame, pt(pp[0]), pt(pp[1]), (255, 255, 0), 2, cv2.LINE_AA)

    # Coins.
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    names = ["FG", "FD", "PD", "PG"]
    for i, p in enumerate(q):
        qpt = pt(p)
        cv2.circle(frame, qpt, 6, color_main, -1, cv2.LINE_AA)
        cv2.putText(frame, names[i], (qpt[0] + 7, qpt[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_main, 2, cv2.LINE_AA)

    cv2.putText(frame, label, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color_main, 2, cv2.LINE_AA)


def build_items(metric_csv: pd.DataFrame, max_w: int, include_quality: str):
    root = Path.cwd()

    df = metric_csv.copy()

    if include_quality != "all":
        allowed = set(x.strip() for x in include_quality.split(",") if x.strip())
        df = df[df["metric_quality_005C7B"].astype(str).isin(allowed)].copy()

    # Priorité : les STRICT puis REVIEW, mais dans l’ordre le plus utile à juger.
    quality_order = {
        "METRIC_TABLE_STRICT": 0,
        "METRIC_TABLE_REVIEW": 1,
        "PARTIAL_TABLE_OBJECT": 2,
        "BAD_TABLE_OBJECT": 3,
    }

    df["_qorder"] = df["metric_quality_005C7B"].astype(str).map(quality_order).fillna(9)
    df = df.sort_values(
        ["_qorder", "review_id", "camera_segment_id_005B2"],
        ascending=[True, True, True],
    ).copy()

    items = []

    for _, r in df.iterrows():
        clip_path = Path(str(r["clip_path"]))
        if not clip_path.is_absolute():
            clip_path = root / clip_path

        frame_idx = int(to_num(pd.Series([
            r.get("metric_frame_005C7B", r.get("mask_frame_005C6B", r.get("best_frame_005C1", r.get("first_frame", 0))))
        ])).fillna(0).iloc[0])

        frame = read_frame(clip_path, frame_idx)
        if frame is None:
            continue

        disp, scale = resize_max_w(frame, max_w=max_w)

        try:
            snap_q = snap_quad_src(r.to_dict()) * scale
        except Exception:
            continue

        try:
            old_q = old_quad_src(r.to_dict()) * scale
        except Exception:
            old_q = None

        # Ancien quad jaune discret.
        if old_q is not None:
            qi = np.round(old_q).astype(int)
            for a, b in zip(qi, np.vstack([qi[1:], qi[:1]])):
                cv2.line(disp, tuple(a), tuple(b), (0, 255, 255), 1, cv2.LINE_AA)

        draw_reference_table(
            disp,
            snap_q,
            label=f"{r['review_id']} seg={int(r['camera_segment_id_005B2'])} reference-table projection",
            color_main=(255, 0, 255),
        )

        cv2.putText(
            disp,
            f"metric={r.get('metric_quality_005C7B','')} snap={r.get('snap_status_005C7A','')} conf={float(to_num(pd.Series([r.get('snap_confidence_005C7A', 0)])).fillna(0).iloc[0]):.2f}",
            (24, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
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
            "display_w": int(disp.shape[1]),
            "display_h": int(disp.shape[0]),
            "display_scale": float(scale),
            "clip_path": str(clip_path),
            "metric_quality": str(r.get("metric_quality_005C7B", "")),
            "metric_reason": str(r.get("metric_reason_005C7B", "")),
            "snap_status": str(r.get("snap_status_005C7A", "")),
            "snap_confidence": float(to_num(pd.Series([r.get("snap_confidence_005C7A", 0)])).fillna(0).iloc[0]),
            "img_b64": encode_jpg_b64(disp),
        })

    return items


class ServerState:
    def __init__(self, out_dir: Path, items: list[dict]):
        self.out_dir = out_dir
        self.items = items
        self.lock = threading.Lock()
        self.csv_path = out_dir / "table_projection_votes_005C8A.csv"
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

        out = {}
        for _, r in df.iterrows():
            out[self.key(r["review_id"], r["camera_segment_id"])] = r.to_dict()
        return out

    def save(self, payload):
        review_id = str(payload["review_id"])
        seg = int(payload["camera_segment_id"])

        item = next(
            (x for x in self.items if x["review_id"] == review_id and int(x["camera_segment_id"]) == seg),
            None,
        )

        if item is None:
            return {"ok": False, "error": "item_not_found"}

        rating = int(payload.get("rating", 0))
        if rating < 1 or rating > 10:
            return {"ok": False, "error": "rating_must_be_1_to_10"}

        row = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "version": VERSION,
            "review_id": review_id,
            "camera_segment_id": seg,
            "video_id": item.get("video_id", ""),
            "rally_id": item.get("rally_id", ""),
            "frame": int(item["frame"]),
            "rating_1_10": rating,
            "usable_metric": int(rating >= 8),
            "review_metric": int(5 <= rating <= 7),
            "reject_metric": int(rating <= 4),
            "tag": str(payload.get("tag", "")),
            "comment": str(payload.get("comment", "")),
            "metric_quality_005C7B": item.get("metric_quality", ""),
            "metric_reason_005C7B": item.get("metric_reason", ""),
            "snap_status_005C7A": item.get("snap_status", ""),
            "snap_confidence_005C7A": item.get("snap_confidence", ""),
            "clip_path": item["clip_path"],
        }

        with self.lock:
            self.saved[self.key(review_id, seg)] = row
            pd.DataFrame(list(self.saved.values())).to_csv(self.csv_path, index=False, encoding="utf-8")

        return {"ok": True, "saved": row}


def index_html():
    return r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 005C8A · vote projection table</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#101219;color:#eef1f8;font-family:system-ui,Segoe UI,sans-serif}
header{padding:12px 18px;background:#171b25;border-bottom:1px solid #303746;display:flex;justify-content:space-between;gap:12px}
main{display:grid;grid-template-columns:minmax(0,1fr) 390px;gap:14px;padding:14px}
.canvasBox{background:#05070b;border:1px solid #303746;border-radius:12px;padding:10px;overflow:auto}
img{max-width:100%;border-radius:8px}
.panel{background:#181d27;border:1px solid #303746;border-radius:12px;padding:12px}
button{border:1px solid #3b4659;background:#252d3d;color:#eef1f8;border-radius:9px;padding:9px 11px;cursor:pointer;margin:3px}
button:hover{background:#303a50}
.score button{font-weight:700;width:42px}
.score button.active{outline:3px solid white}
textarea{width:100%;min-height:70px;background:#101219;color:#eef1f8;border:1px solid #3b4659;border-radius:9px;padding:8px;box-sizing:border-box}
select{width:100%;background:#101219;color:#eef1f8;border:1px solid #3b4659;border-radius:9px;padding:8px}
.meta{font-size:13px;color:#c1cada;line-height:1.55}
.help{background:#182516;border:1px solid #345c2b;border-radius:8px;padding:8px;color:#cfeec8;margin:8px 0}
</style>
</head>
<body>
<header>
  <b>TTFlux 005C8A · note projection table référence</b>
  <div id="progress">chargement…</div>
</header>

<main>
  <section class="canvasBox">
    <img id="img">
  </section>

  <aside class="panel">
    <div class="meta" id="meta"></div>

    <div class="help">
      Note la transposition de la table référence rose.
      <br>1 = faux objet / inutilisable
      <br>5 = partiel ou approximatif
      <br>8 = exploitable métriquement
      <br>10 = très propre
    </div>

    <div class="score" id="score"></div>

    <select id="tag">
      <option value="">tag optionnel</option>
      <option value="good_metric">bonne projection métrique</option>
      <option value="partial_table">table partielle</option>
      <option value="wrong_perspective">mauvaise perspective</option>
      <option value="wrong_object">mauvais objet</option>
      <option value="too_zoomed">plan trop serré</option>
      <option value="occluded">table trop masquée</option>
      <option value="line_mismatch">lignes/bords décalés</option>
    </select>

    <textarea id="comment" placeholder="commentaire optionnel"></textarea>

    <div>
      <button onclick="save(false)">S sauver</button>
      <button onclick="save(true)">sauver + suivant</button>
      <button onclick="prevItem()">P précédent</button>
      <button onclick="nextItem()">N suivant</button>
    </div>
  </aside>
</main>

<script>
let items = [];
let saved = {};
let idx = Number(localStorage.getItem("ttflux_005C8A_idx") || 0);
let rating = 0;

async function load() {
  const res = await fetch("/api/items");
  const data = await res.json();
  items = data.items || [];
  saved = data.saved || {};
  if (idx >= items.length) idx = 0;
  buildScoreButtons();
  render();
}

function key(item){ return item.review_id + "__seg" + item.camera_segment_id; }
function current(){ return items[idx]; }

function buildScoreButtons() {
  const box = document.getElementById("score");
  box.innerHTML = "";
  for (let i=1; i<=10; i++) {
    const b = document.createElement("button");
    b.textContent = String(i);
    b.onclick = () => setRating(i);
    b.id = "score_" + i;
    box.appendChild(b);
  }
}

function setRating(v) {
  rating = v;
  for (let i=1; i<=10; i++) {
    document.getElementById("score_" + i).classList.toggle("active", i === v);
  }
}

function render() {
  const item = current();
  if (!item) return;

  localStorage.setItem("ttflux_005C8A_idx", String(idx));
  document.getElementById("progress").textContent = `${idx+1} / ${items.length}`;

  const s = saved[key(item)] || null;

  document.getElementById("meta").innerHTML =
    `<b>${item.review_id}</b> · seg ${item.camera_segment_id}<br>` +
    `frame ${item.frame}<br>` +
    `metric : ${item.metric_quality}<br>` +
    `reason : ${item.metric_reason}<br>` +
    `snap : ${item.snap_status} conf=${item.snap_confidence.toFixed(3)}<br>` +
    `saved : <b>${s ? s.rating_1_10 + "/10" : "non noté"}</b>`;

  document.getElementById("img").src = "data:image/jpeg;base64," + item.img_b64;

  setRating(s ? Number(s.rating_1_10) : 0);
  document.getElementById("tag").value = s ? (s.tag || "") : "";
  document.getElementById("comment").value = s ? (s.comment || "") : "";
}

async function save(goNext) {
  const item = current();

  if (rating < 1 || rating > 10) {
    alert("Choisis une note de 1 à 10.");
    return;
  }

  const payload = {
    review_id: item.review_id,
    camera_segment_id: item.camera_segment_id,
    rating,
    tag: document.getElementById("tag").value || "",
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

  saved[key(item)] = data.saved;
  if (goNext) nextItem();
  else render();
}

function nextItem(){ idx = Math.min(items.length-1, idx+1); render(); }
function prevItem(){ idx = Math.max(0, idx-1); render(); }

document.addEventListener("keydown", ev => {
  if (ev.target && (ev.target.tagName === "TEXTAREA" || ev.target.tagName === "SELECT")) return;

  const k = ev.key.toLowerCase();
  if (k >= "1" && k <= "9") setRating(Number(k));
  if (k === "0") setRating(10);
  if (k === "s") save(false);
  if (k === "n" || k === "arrowright") nextItem();
  if (k === "p" || k === "arrowleft") prevItem();
});

load();
</script>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-csv", default="runs/rally_table_metric_qa_005C7B/table_metric_quality_005C7B.csv")
    ap.add_argument("--out-dir", default="runs/rally_table_projection_vote_005C8A")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--max-w", type=int, default=1280)
    ap.add_argument("--include-quality", default="METRIC_TABLE_STRICT,METRIC_TABLE_REVIEW")
    args = ap.parse_args()

    root = Path.cwd()

    metric_path = Path(args.metric_csv)
    out_dir = Path(args.out_dir)

    if not metric_path.is_absolute():
        metric_path = root / metric_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    metric = pd.read_csv(metric_path).fillna("")
    items = build_items(metric, max_w=args.max_w, include_quality=args.include_quality)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "human_rating_1_to_10_for_reference_table_projection_quality",
        "metric_csv": str(metric_path),
        "out_dir": str(out_dir),
        "include_quality": args.include_quality,
        "items": len(items),
        "votes_csv": str(out_dir / "table_projection_votes_005C8A.csv"),
    }

    (out_dir / "table_projection_vote_server_summary_005C8A.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("005C8A server READY")
    print("items=", len(items))
    print("url=", f"http://127.0.0.1:{args.port}/")
    print("votes_csv=", out_dir / "table_projection_votes_005C8A.csv")

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

            res = state.save(payload)
            self.send_json(res, 200 if res.get("ok") else 400)

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.serve_forever()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
