from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import pandas as pd


VERSION = "004Z"


def safe_name(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_") or "item"


def find_ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p

    patterns = [
        r"C:\Program Files\ffmpeg-*\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg*\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]

    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]

    return ""


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "ok": False,
            "duration_sec": 0,
            "fps": 0,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "error": "open_failed",
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration = frame_count / fps if fps else 0

    return {
        "ok": True,
        "duration_sec": round(duration, 3),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "error": "",
    }


def transcode_for_browser(ffmpeg: str, src: Path, dst: Path, max_sec: float, force: bool = False) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_file() and not force:
        return True

    if not ffmpeg:
        shutil.copyfile(src, dst)
        return False

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-t",
        str(max_sec),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if p.returncode != 0:
        print("ffmpeg failed:", src)
        print(p.stderr[-2000:])
        shutil.copyfile(src, dst)
        return False

    return True


def build_packet(
    review_summary: Path,
    out_dir: Path,
    min_sec: float,
    max_sec: float,
    include_short: bool,
    max_items: int,
    force_transcode: bool,
) -> tuple[list[dict], dict]:
    df = pd.read_csv(review_summary).fillna("")

    if "overlay" not in df.columns:
        raise SystemExit("Colonne overlay absente du review summary.")

    root = Path.cwd()
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    rows = []
    skipped = []

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        overlay = str(r.get("overlay", "")).strip()
        if not overlay:
            skipped.append({"review_id": str(r.get("review_id", "")), "reason": "no_overlay"})
            continue

        src = Path(overlay)
        if not src.is_absolute():
            src = root / src

        if not src.is_file():
            skipped.append({"review_id": str(r.get("review_id", "")), "reason": "overlay_missing", "overlay": str(src)})
            continue

        probe = probe_video(src)
        if not probe["ok"]:
            skipped.append({"review_id": str(r.get("review_id", "")), "reason": probe.get("error", "probe_failed"), "overlay": str(src)})
            continue

        duration = float(probe["duration_sec"])

        if duration < min_sec and not include_short:
            skipped.append({
                "review_id": str(r.get("review_id", "")),
                "reason": "too_short",
                "duration_sec": duration,
                "overlay": str(src),
            })
            continue

        review_id = str(r.get("review_id", f"R{i:04d}"))
        video_id = str(r.get("video_id", ""))
        rally_id = str(r.get("rally_id", ""))

        asset_name = f"{i:03d}_{safe_name(review_id)}_{safe_name(video_id)}_004Z.mp4"
        dst = assets_dir / asset_name

        transcoded = transcode_for_browser(
            ffmpeg=ffmpeg,
            src=src,
            dst=dst,
            max_sec=max_sec,
            force=force_transcode,
        )

        asset_probe = probe_video(dst)

        row = {
            "idx": len(rows),
            "review_id": review_id,
            "rally_id": rally_id,
            "video_id": video_id,
            "duration_src_sec": duration,
            "duration_vote_sec": asset_probe.get("duration_sec", ""),
            "fps": asset_probe.get("fps", ""),
            "width": asset_probe.get("width", ""),
            "height": asset_probe.get("height", ""),
            "path_points": str(r.get("path_points", "")),
            "emitted_points": str(r.get("emitted_points", "")),
            "coverage_points": str(r.get("coverage_points", "")),
            "tracklets": str(r.get("tracklets", "")),
            "longest_tracklet_points": str(r.get("longest_tracklet_points", "")),
            "score_med_emitted": str(r.get("score_med_emitted", "")),
            "review_score_med_004Y": str(r.get("review_score_med_004Y", "")),
            "overlay_src": str(src),
            "asset_name": asset_name,
            "asset_path": str(dst),
            "transcoded_h264": bool(transcoded),
        }

        rows.append(row)

        if max_items > 0 and len(rows) >= max_items:
            break

    packet_csv = out_dir / "vote_packet_004Z.csv"
    skipped_csv = out_dir / "vote_packet_skipped_004Z.csv"
    summary_json = out_dir / "vote_packet_summary_004Z.json"

    if rows:
        pd.DataFrame(rows).to_csv(packet_csv, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=["idx", "review_id", "asset_name"]).to_csv(packet_csv, index=False, encoding="utf-8")

    pd.DataFrame(skipped).to_csv(skipped_csv, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "review_summary": str(review_summary),
        "out_dir": str(out_dir),
        "ffmpeg": ffmpeg,
        "min_sec": min_sec,
        "max_sec": max_sec,
        "include_short": include_short,
        "max_items": max_items,
        "items": len(rows),
        "skipped": len(skipped),
        "packet_csv": str(packet_csv),
        "skipped_csv": str(skipped_csv),
        "votes_csv": str(out_dir / "votes_004Z.csv"),
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return rows, summary


class VoteServer:
    def __init__(self, out_dir: Path, items: list[dict], summary: dict):
        self.out_dir = out_dir
        self.assets_dir = out_dir / "assets"
        self.votes_csv = out_dir / "votes_004Z.csv"
        self.items = items
        self.summary = summary
        self.lock = threading.Lock()
        self.votes = self.load_votes()

    def load_votes(self) -> dict:
        if not self.votes_csv.is_file():
            return {}

        try:
            df = pd.read_csv(self.votes_csv).fillna("")
        except Exception:
            return {}

        votes = {}
        for _, r in df.iterrows():
            votes[str(r.get("review_id", ""))] = r.to_dict()

        return votes

    def save_vote(self, payload: dict):
        review_id = str(payload.get("review_id", ""))
        if not review_id:
            return

        item = next((x for x in self.items if str(x.get("review_id")) == review_id), {})

        row = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "review_id": review_id,
            "video_id": str(item.get("video_id", "")),
            "rally_id": str(item.get("rally_id", "")),
            "duration_vote_sec": str(item.get("duration_vote_sec", "")),
            "rating": str(payload.get("rating", "")),
            "tags": "|".join(payload.get("tags", [])) if isinstance(payload.get("tags"), list) else str(payload.get("tags", "")),
            "comment": str(payload.get("comment", "")),
            "asset_name": str(item.get("asset_name", "")),
            "overlay_src": str(item.get("overlay_src", "")),
        }

        with self.lock:
            self.votes[review_id] = row
            rows = list(self.votes.values())
            pd.DataFrame(rows).to_csv(self.votes_csv, index=False, encoding="utf-8")

    def make_handler(self):
        server_ref = self

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

            def send_text(self, text, status=200, content_type="text/html; charset=utf-8"):
                data = text.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path

                if path == "/":
                    self.send_text(index_html())
                    return

                if path == "/api/items":
                    self.send_json({
                        "summary": server_ref.summary,
                        "items": server_ref.items,
                        "votes": server_ref.votes,
                    })
                    return

                if path.startswith("/assets/"):
                    name = unquote(path[len("/assets/"):])
                    name = os.path.basename(name)
                    f = server_ref.assets_dir / name

                    if not f.is_file():
                        self.send_error(404)
                        return

                    ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
                    data = f.read_bytes()

                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

                self.send_error(404)

            def do_POST(self):
                parsed = urlparse(self.path)

                if parsed.path != "/api/vote":
                    self.send_error(404)
                    return

                n = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(n).decode("utf-8")

                try:
                    payload = json.loads(raw)
                except Exception:
                    self.send_json({"ok": False, "error": "invalid_json"}, status=400)
                    return

                server_ref.save_vote(payload)
                self.send_json({"ok": True, "votes": server_ref.votes})

        return Handler


def index_html() -> str:
    return r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004Z · vote vidéo tracking balle</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0f1117;color:#eef1f8;font-family:system-ui,Segoe UI,sans-serif}
header{padding:14px 20px;border-bottom:1px solid #2b303b;background:#161922;display:flex;gap:12px;align-items:center;justify-content:space-between}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:16px;padding:16px}
.videoBox{background:#05060a;border:1px solid #2b303b;border-radius:14px;padding:12px}
video{width:100%;max-height:76vh;background:#000;border-radius:10px}
.panel{background:#181b22;border:1px solid #2b303b;border-radius:14px;padding:14px}
h1{font-size:18px;margin:0}
h2{font-size:15px;margin:0 0 10px}
.meta{font-size:12px;color:#bac3d6;line-height:1.5}
button{border:1px solid #3a4252;background:#232937;color:#eef1f8;border-radius:10px;padding:10px 12px;cursor:pointer;font-size:14px}
button:hover{background:#2e3749}
button.primary{background:#245c35}
button.mid{background:#5f4a1b}
button.bad{background:#6a2626}
button.zero{background:#3b3f48}
.row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
textarea{width:100%;min-height:90px;border-radius:10px;background:#0f1117;color:#eef1f8;border:1px solid #3a4252;padding:10px;box-sizing:border-box}
label{display:block;margin:6px 0;font-size:13px}
.progress{font-size:13px;color:#cbd3e7}
.small{font-size:12px;color:#9da8bd}
.voteState{padding:8px;border-radius:10px;background:#10131b;border:1px solid #2b303b;margin:8px 0}
kbd{border:1px solid #566070;background:#10131b;border-radius:4px;padding:1px 5px}
@media(max-width:1000px){main{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>TTFlux 004Z · vote vidéo tracking balle</h1>
  <div class="progress" id="progress">chargement…</div>
</header>

<main>
  <section class="videoBox">
    <video id="video" controls preload="metadata"></video>
  </section>

  <aside class="panel">
    <h2 id="title">—</h2>
    <div class="meta" id="meta"></div>

    <div class="voteState" id="voteState">aucun vote</div>

    <div class="row">
      <button class="primary" onclick="setRating('bon')">3 · bon</button>
      <button class="mid" onclick="setRating('moyen')">2 · moyen</button>
      <button class="bad" onclick="setRating('mauvais')">1 · mauvais</button>
      <button class="zero" onclick="setRating('inutilisable')">0 · inutilisable</button>
    </div>

    <h2>Tags d’erreur</h2>
    <label><input type="checkbox" value="points_sur_balle"> points verts sur balle</label>
    <label><input type="checkbox" value="points_trop_rares"> points trop rares</label>
    <label><input type="checkbox" value="faux_joueur"> accroche joueur / corps</label>
    <label><input type="checkbox" value="faux_raquette"> accroche raquette / bras</label>
    <label><input type="checkbox" value="faux_table"> accroche table / filet / bord</label>
    <label><input type="checkbox" value="faux_logo"> accroche logo / texte / score</label>
    <label><input type="checkbox" value="no_ball_ok"> NO_BALL pertinent</label>
    <label><input type="checkbox" value="no_ball_trop_agressif"> NO_BALL trop agressif</label>
    <label><input type="checkbox" value="segment_pas_rally"> segment pas assez rally</label>

    <h2>Commentaire</h2>
    <textarea id="comment" placeholder="Ex : bons points au début, puis accroche maillot blanc…"></textarea>

    <div class="row">
      <button onclick="saveVote()">S · enregistrer</button>
      <button onclick="prevItem()">P · précédent</button>
      <button onclick="nextItem()">N · suivant</button>
    </div>

    <p class="small">
      Raccourcis : <kbd>3</kbd> bon, <kbd>2</kbd> moyen, <kbd>1</kbd> mauvais,
      <kbd>0</kbd> inutilisable, <kbd>S</kbd> sauvegarder, <kbd>N</kbd>/<kbd>P</kbd>.
    </p>
  </aside>
</main>

<script>
let items = [];
let votes = {};
let idx = Number(localStorage.getItem("ttflux_004Z_idx") || 0);
let currentRating = "";

async function load() {
  const res = await fetch("/api/items");
  const data = await res.json();
  items = data.items || [];
  votes = data.votes || {};
  if (idx >= items.length) idx = 0;
  render();
}

function current() {
  return items[idx];
}

function render() {
  const item = current();
  if (!item) {
    document.getElementById("title").textContent = "Aucune vidéo";
    document.getElementById("progress").textContent = "0 / 0";
    return;
  }

  localStorage.setItem("ttflux_004Z_idx", String(idx));

  document.getElementById("progress").textContent = `${idx + 1} / ${items.length}`;
  document.getElementById("title").textContent = `${item.review_id} · ${item.video_id}`;
  document.getElementById("video").src = `/assets/${encodeURIComponent(item.asset_name)}`;

  document.getElementById("meta").innerHTML = `
    durée vote : <b>${item.duration_vote_sec}s</b><br>
    durée source : ${item.duration_src_sec}s<br>
    emitted : ${item.emitted_points || "?"} · tracklets : ${item.tracklets || "?"}<br>
    review score : ${item.review_score_med_004Y || "?"}<br>
    overlay : <code>${escapeHtml(item.overlay_src || "")}</code>
  `;

  const vote = votes[item.review_id] || {};
  currentRating = vote.rating || "";

  document.getElementById("voteState").textContent = currentRating
    ? `vote actuel : ${currentRating} · ${vote.tags || ""}`
    : "aucun vote";

  document.getElementById("comment").value = vote.comment || "";

  const tags = String(vote.tags || "").split("|").filter(Boolean);
  document.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.checked = tags.includes(cb.value);
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, m => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
  }[m]));
}

function setRating(rating) {
  currentRating = rating;
  document.getElementById("voteState").textContent = `vote non sauvegardé : ${rating}`;
}

async function saveVote(goNext=false) {
  const item = current();
  if (!item) return;

  const tags = Array.from(document.querySelectorAll("input[type=checkbox]:checked")).map(x => x.value);
  const comment = document.getElementById("comment").value || "";

  const payload = {
    review_id: item.review_id,
    rating: currentRating,
    tags,
    comment,
  };

  const res = await fetch("/api/vote", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (data.ok) {
    votes = data.votes || votes;
    if (goNext) nextItem();
    else render();
  }
}

function nextItem() {
  idx = Math.min(items.length - 1, idx + 1);
  render();
}

function prevItem() {
  idx = Math.max(0, idx - 1);
  render();
}

document.addEventListener("keydown", ev => {
  if (ev.target && ["TEXTAREA", "INPUT"].includes(ev.target.tagName)) return;

  if (ev.key === "3") setRating("bon");
  if (ev.key === "2") setRating("moyen");
  if (ev.key === "1") setRating("mauvais");
  if (ev.key === "0") setRating("inutilisable");
  if (ev.key.toLowerCase() === "s") saveVote(false);
  if (ev.key.toLowerCase() === "n") saveVote(true);
  if (ev.key.toLowerCase() === "p") prevItem();
});

load();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-summary", default="runs/rally_gated_paths_004Y_annotated17_p080_r030_vote/gated_review_summary_004Y.csv")
    ap.add_argument("--out-dir", default="runs/rally_video_vote_004Z_p080_r030")
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--min-sec", type=float, default=20.0)
    ap.add_argument("--max-sec", type=float, default=60.0)
    ap.add_argument("--include-short", action="store_true")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--force-transcode", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()

    review_summary = Path(args.review_summary)
    if not review_summary.is_absolute():
        review_summary = root / review_summary

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    items, summary = build_packet(
        review_summary=review_summary,
        out_dir=out_dir,
        min_sec=args.min_sec,
        max_sec=args.max_sec,
        include_short=args.include_short,
        max_items=args.max_items,
        force_transcode=args.force_transcode,
    )

    print("004Z status=READY")
    print("items=", len(items))
    print("out_dir=", out_dir)
    print("votes_csv=", out_dir / "votes_004Z.csv")
    print("url=", f"http://127.0.0.1:{args.port}/")

    if not items:
        print("Aucune vidéo dans le packet. Relance avec --include-short ou baisse --min-sec.")
        return 1

    server_state = VoteServer(out_dir=out_dir, items=items, summary=summary)
    handler = server_state.make_handler()

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    httpd.serve_forever()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
