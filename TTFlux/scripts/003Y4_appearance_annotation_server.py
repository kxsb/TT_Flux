from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import os
import shutil
import subprocess
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd


LABELS = ["reject", "keep", "partial", "unsure"]


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def load_existing_labels(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}

    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("review_id", "")).strip()
            if rid:
                out[rid] = row
    return out


def write_labels(path: Path, labels: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["review_id", "human_label_003Y4", "comment_003Y4", "updated_at"]
    tmp = path.with_suffix(".tmp.csv")

    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for rid in sorted(labels.keys()):
            row = labels[rid]
            writer.writerow({
                "review_id": rid,
                "human_label_003Y4": row.get("human_label_003Y4", ""),
                "comment_003Y4": row.get("comment_003Y4", ""),
                "updated_at": row.get("updated_at", ""),
            })

    tmp.replace(path)


def resolve_path(raw, root: Path, run_dir: Path) -> Path | None:
    if raw is None:
        return None

    s = str(raw).strip().strip('"').strip("'")
    if not s or s.lower() == "nan":
        return None

    p = Path(s.replace("\\", os.sep))
    candidates = [p] if p.is_absolute() else [root / p, run_dir / p]

    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            pass

    return None


def locate_source_video(row: pd.Series, root: Path, run_dir: Path) -> Path | None:
    rid = str(row.get("review_id", "")).strip()

    for col in ["manifest__mp4", "manifest__video", "manifest__video_path", "true003E__mp4", "mp4"]:
        if col in row.index:
            p = resolve_path(row.get(col), root, run_dir)
            if p:
                return p

    asset_dir = run_dir / "appearance_shadow_rule_003Y3_assets" / rid
    if asset_dir.is_dir():
        mp4s = sorted(asset_dir.glob("*.mp4"))
        if mp4s:
            return mp4s[0].resolve()

    return None


def convert_to_h264(src: Path, dst: Path, ffmpeg: str | None) -> tuple[bool, str]:
    if dst.is_file() and dst.stat().st_size > 0:
        return True, ""

    if ffmpeg is None:
        return False, "ffmpeg introuvable"

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
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

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if proc.returncode != 0:
        return False, proc.stderr[-1000:]

    if not dst.is_file() or dst.stat().st_size <= 0:
        return False, "conversion produced no file"

    return True, ""


def prepare_rows(run_dir: Path, source_csv: Path, root: Path) -> list[dict]:
    df = pd.read_csv(source_csv)

    if "003Y3_shadow_hit" not in df.columns:
        raise SystemExit("Colonne 003Y3_shadow_hit absente.")

    hits = df[df["003Y3_shadow_hit"].map(boolish)].copy()

    if "review_id" not in hits.columns:
        raise SystemExit("review_id absent.")

    ffmpeg = find_ffmpeg()
    h264_dir = run_dir / "appearance_review_003Y4_h264"
    h264_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, row in hits.iterrows():
        rid = str(row.get("review_id", "")).strip()
        src = locate_source_video(row, root, run_dir)

        h264_rel = ""
        convert_ok = False
        convert_error = ""

        if src is None:
            convert_error = "source video not found"
        else:
            dst = h264_dir / f"{rid}_003Y4_h264.mp4"
            convert_ok, convert_error = convert_to_h264(src, dst, ffmpeg)
            if convert_ok:
                h264_rel = dst.relative_to(run_dir).as_posix()

        rows.append({
            "review_id": rid,
            "video_rel": h264_rel,
            "source_video": str(src) if src else "",
            "convert_ok": convert_ok,
            "convert_error": convert_error,
            "feature_value": row.get("003Y3_feature_value", ""),
            "feature_col": row.get("003Y3_feature_col_used", ""),
            "target_class_003G": row.get("target_class_003G", row.get("true003E__target_class_003G", "")),
            "comment_previous": row.get("comment_003S", ""),
            "label_previous": row.get("human_label_003S", ""),
        })

    return rows


def make_app(run_dir: Path, source_csv: Path, labels_csv: Path):
    root = Path.cwd()
    rows = prepare_rows(run_dir, source_csv, root)

    state = {
        "run_dir": run_dir,
        "source_csv": source_csv,
        "labels_csv": labels_csv,
        "rows": rows,
        "labels": load_existing_labels(labels_csv),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def send_json(self, obj, status=200):
            data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)

            if parsed.path in {"/", "/index.html"}:
                self.serve_index()
                return

            if parsed.path == "/api/state":
                self.send_json({
                    "rows": state["rows"],
                    "labels": state["labels"],
                    "labels_csv": str(state["labels_csv"]),
                })
                return

            if parsed.path.startswith("/media/"):
                rel = urllib.parse.unquote(parsed.path[len("/media/"):])
                target = (state["run_dir"] / rel).resolve()

                try:
                    target.relative_to(state["run_dir"].resolve())
                except Exception:
                    self.send_error(403)
                    return

                if not target.is_file():
                    self.send_error(404)
                    return

                mime, _ = mimetypes.guess_type(str(target))
                mime = mime or "application/octet-stream"

                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_error(404)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)

            if parsed.path != "/api/save":
                self.send_error(404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as exc:
                self.send_json({"ok": False, "error": repr(exc)}, status=400)
                return

            rid = str(payload.get("review_id", "")).strip()
            label = str(payload.get("human_label_003Y4", "")).strip()
            comment = str(payload.get("comment_003Y4", "")).strip()

            if not rid:
                self.send_json({"ok": False, "error": "review_id missing"}, status=400)
                return

            if label and label not in LABELS:
                self.send_json({"ok": False, "error": f"invalid label: {label}"}, status=400)
                return

            state["labels"][rid] = {
                "review_id": rid,
                "human_label_003Y4": label,
                "comment_003Y4": comment,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }

            write_labels(state["labels_csv"], state["labels"])

            done = sum(
                1 for r in state["rows"]
                if state["labels"].get(r["review_id"], {}).get("human_label_003Y4")
            )

            self.send_json({
                "ok": True,
                "done": done,
                "total": len(state["rows"]),
                "labels_csv": str(state["labels_csv"]),
            })

        def serve_index(self):
            rows_json = json.dumps(state["rows"], ensure_ascii=False)
            labels_json = json.dumps(state["labels"], ensure_ascii=False)

            doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003Y4 annotation UI</title>
<style>
:root {{
  color-scheme: dark;
  --bg:#101218;
  --card:#181b22;
  --line:#2b303b;
  --text:#edf0f7;
  --muted:#aab2c5;
  --yellow:#ffc850;
  --green:#74d99f;
  --red:#ff7070;
}}
body {{
  margin:0;
  padding:24px;
  background:var(--bg);
  color:var(--text);
  font-family:system-ui, Segoe UI, sans-serif;
}}
header,.card {{
  background:var(--card);
  border:1px solid var(--line);
  border-radius:16px;
  padding:16px;
  margin-bottom:16px;
}}
header {{
  position:sticky;
  top:0;
  z-index:10;
  box-shadow:0 10px 30px rgba(0,0,0,.25);
}}
.grid {{
  display:grid;
  grid-template-columns:420px 1fr;
  gap:16px;
}}
@media(max-width:1000px) {{
  .grid {{ grid-template-columns:1fr; }}
}}
table {{
  width:100%;
  border-collapse:collapse;
}}
td,th {{
  border-bottom:1px solid var(--line);
  padding:7px;
  text-align:left;
  vertical-align:top;
  font-size:12px;
}}
th {{
  width:210px;
  background:#20242e;
}}
video {{
  display:block;
  width:900px;
  max-width:100%;
  max-height:560px;
  object-fit:contain;
  border:1px solid var(--line);
  border-radius:10px;
  background:#05060a;
  margin-bottom:12px;
}}
button {{
  border:1px solid #3a4050;
  background:#20242e;
  color:var(--text);
  border-radius:10px;
  padding:10px 12px;
  cursor:pointer;
  font-weight:700;
  margin-right:8px;
  margin-bottom:8px;
}}
button.active.reject {{ border-color:var(--red); background:rgba(255,112,112,.18); }}
button.active.keep {{ border-color:var(--green); background:rgba(116,217,159,.18); }}
button.active.partial {{ border-color:var(--yellow); background:rgba(255,200,80,.18); }}
button.active.unsure {{ border-color:#9db4ff; background:rgba(157,180,255,.18); }}
textarea {{
  width:100%;
  min-height:80px;
  background:#101218;
  color:var(--text);
  border:1px solid var(--line);
  border-radius:10px;
  padding:10px;
  box-sizing:border-box;
}}
.done {{
  border-color:rgba(116,217,159,.65);
  box-shadow:inset 4px 0 0 rgba(116,217,159,.8);
}}
.todo {{
  border-color:rgba(255,200,80,.65);
  box-shadow:inset 4px 0 0 rgba(255,200,80,.8);
}}
.muted {{ color:var(--muted); }}
.warn {{ color:var(--yellow); }}
.badge {{
  display:inline-block;
  padding:3px 8px;
  border:1px solid #3a4050;
  border-radius:999px;
  font-size:12px;
  background:#20242e;
  margin-right:6px;
}}
a {{ color:#b9cdfa; }}
</style>
</head>
<body>
<header>
  <h1>TTFlux · 003Y4 annotation UI</h1>
  <p>
    <span class="badge">run: {esc(str(state["run_dir"]))}</span>
    <span class="badge">source: {esc(str(state["source_csv"]))}</span>
    <span class="badge">labels: {esc(str(state["labels_csv"]))}</span>
  </p>
  <h2 id="progress">Chargement…</h2>
  <p class="muted">But : vérifier les hits cross-batch de la règle 003Y appearance/logo/shoes.</p>
</header>

<div id="app"></div>

<script>
const ROWS = {rows_json};
let LABELS = {labels_json};
const VALID = ["reject", "keep", "partial", "unsure"];

function esc(s) {{
  return String(s ?? "").replace(/[&<>"']/g, c => ({{
    "&":"&amp;",
    "<":"&lt;",
    ">":"&gt;",
    "\\"":"&quot;",
    "'":"&#039;"
  }}[c]));
}}

function labelOf(rid) {{
  return (LABELS[rid] && LABELS[rid].human_label_003Y4) || "";
}}

function commentOf(rid) {{
  return (LABELS[rid] && LABELS[rid].comment_003Y4) || "";
}}

function renderProgress() {{
  const done = ROWS.filter(r => labelOf(r.review_id)).length;
  document.getElementById("progress").innerHTML = `Progression : ${{done}} / ${{ROWS.length}}`;
}}

async function saveLabel(rid, label, comment) {{
  const res = await fetch("/api/save", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{
      review_id: rid,
      human_label_003Y4: label,
      comment_003Y4: comment
    }})
  }});

  const data = await res.json();

  if (!data.ok) {{
    alert("Erreur sauvegarde: " + data.error);
    return;
  }}

  LABELS[rid] = {{
    review_id: rid,
    human_label_003Y4: label,
    comment_003Y4: comment,
    updated_at: new Date().toISOString()
  }};

  const card = document.getElementById("card-" + rid);
  card.classList.remove("todo", "done");
  card.classList.add(label ? "done" : "todo");

  renderButtons(rid);
  renderProgress();
}}

function renderButtons(rid) {{
  const box = document.getElementById("buttons-" + rid);
  const current = labelOf(rid);
  box.innerHTML = "";

  for (const lab of VALID) {{
    const b = document.createElement("button");
    b.className = lab + (current === lab ? " active" : "");
    b.textContent = lab;
    b.onclick = () => {{
      const comment = document.getElementById("comment-" + rid).value;
      saveLabel(rid, lab, comment);
    }};
    box.appendChild(b);
  }}

  const clear = document.createElement("button");
  clear.textContent = "clear";
  clear.onclick = () => {{
    const comment = document.getElementById("comment-" + rid).value;
    saveLabel(rid, "", comment);
  }};
  box.appendChild(clear);
}}

function render() {{
  const app = document.getElementById("app");

  app.innerHTML = ROWS.map((r, idx) => {{
    const rid = r.review_id;
    const label = labelOf(rid);
    const comment = commentOf(rid);
    const cardCls = label ? "card done" : "card todo";

    const video = r.video_rel
      ? `<video controls preload="metadata" src="/media/${{esc(r.video_rel)}}"></video>
         <p><a href="/media/${{esc(r.video_rel)}}" target="_blank">ouvrir vidéo</a></p>`
      : `<p class="warn">Vidéo H264 absente : ${{esc(r.convert_error)}}</p>`;

    return `
<div class="${{cardCls}}" id="card-${{esc(rid)}}">
  <h2>${{idx + 1}} / ${{ROWS.length}} · ${{esc(rid)}}</h2>
  <div class="grid">
    <div>
      <h3>Données</h3>
      <table>
        <tbody>
          <tr><th>review_id</th><td>${{esc(rid)}}</td></tr>
          <tr><th>feature</th><td>${{esc(r.feature_col)}}</td></tr>
          <tr><th>value</th><td>${{esc(r.feature_value)}}</td></tr>
          <tr><th>target_class</th><td>${{esc(r.target_class_003G)}}</td></tr>
          <tr><th>previous label</th><td>${{esc(r.label_previous)}}</td></tr>
          <tr><th>previous comment</th><td>${{esc(r.comment_previous)}}</td></tr>
          <tr><th>source video</th><td>${{esc(r.source_video)}}</td></tr>
        </tbody>
      </table>

      <h3>Annotation 003Y4</h3>
      <div id="buttons-${{esc(rid)}}"></div>
      <textarea id="comment-${{esc(rid)}}" placeholder="Commentaire optionnel">${{esc(comment)}}</textarea>
      <p class="muted">Le commentaire est sauvegardé quand tu cliques sur un label.</p>
    </div>
    <div>
      <h3>Vidéo</h3>
      ${{video}}
    </div>
  </div>
</div>`;
  }}).join("");

  for (const r of ROWS) {{
    renderButtons(r.review_id);
  }}

  renderProgress();
}}

render();
</script>
</body>
</html>"""

            data = doc.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_001E")
    ap.add_argument("--source", default="appearance_shadow_rule_run_003Y3.csv")
    ap.add_argument("--labels", default="appearance_review_003Y4_labels.csv")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    root = Path.cwd()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    source = Path(args.source)
    if not source.is_absolute():
        source = run_dir / source

    labels = Path(args.labels)
    if not labels.is_absolute():
        labels = run_dir / labels

    if not source.is_file():
        raise SystemExit(f"Source introuvable: {source}")

    handler = make_app(run_dir, source, labels)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print("003Y4 annotation server")
    print("run_dir =", run_dir)
    print("source  =", source)
    print("labels  =", labels)
    print(f"url     = http://{args.host}:{args.port}/")
    print("")
    print("Garde cette fenêtre ouverte pendant l'annotation.")
    print("Ctrl+C pour arrêter.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("")
        print("server stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
