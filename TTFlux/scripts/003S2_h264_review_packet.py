from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003S2"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def find_ffmpeg() -> str | None:
    p = shutil.which("ffmpeg")
    return p


def convert_mp4(src: Path, dst: Path, ffmpeg: str) -> tuple[bool, str]:
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

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return False, repr(exc)

    if proc.returncode != 0:
        return False, proc.stderr[-2000:]

    if not dst.is_file() or dst.stat().st_size <= 0:
        return False, "conversion produced no file"

    return True, ""


def locate_source_video(row: pd.Series, run_dir: Path) -> Path | None:
    rid = str(row.get("review_id", "")).strip()

    # 1) Chemin mp4 original.
    raw = str(row.get("mp4", "")).strip()
    if raw and raw.lower() != "nan":
        p = Path(raw)
        if p.is_file():
            return p

    # 2) Asset déjà copié par 003S.
    d = run_dir / "human_review_packet_003S_assets" / rid
    if d.is_dir():
        mp4s = sorted(d.glob("*.mp4"))
        if mp4s:
            return mp4s[0]

    # 3) Asset 003N.
    d = run_dir / "frozen_shadow_rule_003N_assets" / rid
    if d.is_dir():
        mp4s = sorted(d.glob("*.mp4"))
        if mp4s:
            return mp4s[0]

    return None


def write_html(path: Path, rows: list[dict], summary: dict) -> None:
    cards = []

    for r in rows:
        rid = r["review_id"]
        video_rel = r.get("h264_rel", "")
        original = r.get("source_video", "")

        trs = []
        for c in [
            "review_id",
            "target_class_003G",
            "003N_player_motion_inside_value",
            "003N_old_table_distance_value",
            "003N_max_accel_value",
            "conversion_ok",
            "conversion_error",
            "source_video",
        ]:
            trs.append(f"<tr><th>{esc(c)}</th><td>{esc(r.get(c, ''))}</td></tr>")

        if video_rel:
            media = f"""
<p class="muted">H264 web · {esc(video_rel)}</p>
<video controls preload="metadata" src="{esc(video_rel)}" type="video/mp4"></video>
<p><a href="{esc(video_rel)}">ouvrir H264</a></p>
"""
        else:
            media = f"""
<p class="warn">Conversion absente ou échouée.</p>
<p>Original : {esc(original)}</p>
"""

        cards.append(f"""
<div class="card">
<h2>{esc(rid)}</h2>
<div class="grid">
  <div>
    <h3>Données</h3>
    <table><tbody>{''.join(trs)}</tbody></table>
  </div>
  <div>
    <h3>Vidéo lisible navigateur</h3>
    {media}
  </div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003S2 H264 review packet</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section,.card{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
.card{{border-color:rgba(255,200,80,.75);box-shadow:inset 4px 0 0 rgba(255,200,80,.85)}}
h1,h2,h3{{margin-top:0}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{width:280px;background:#20242e}}
.grid{{display:grid;grid-template-columns:430px 1fr;gap:16px}}
@media(max-width:950px){{.grid{{grid-template-columns:1fr}}}}
video{{display:block;width:900px;max-width:100%;max-height:560px;object-fit:contain;border:1px solid #2b303b;border-radius:10px;background:#05060a;margin-bottom:10px}}
a{{color:#b9cdfa}}
.muted{{color:#aab2c5}}
.warn{{color:#ffcc66}}
</style>
</head>
<body>
<h1>TTFlux · 003S2 H264 human review packet</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

<section>
<h2>Annotation</h2>
<pre>Remplir ensuite human_review_003S_template.csv :
reject / keep / partial / unsure</pre>
</section>

{''.join(cards)}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_002A")
    ap.add_argument("--source", default="frozen_shadow_rule_run_003N.csv")
    args = ap.parse_args()

    root = Path.cwd()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    source = Path(args.source)
    if not source.is_absolute():
        source = run_dir / source

    if not source.is_file():
        raise SystemExit(f"Source introuvable: {source}")

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit("ffmpeg introuvable dans PATH. Installe/active ffmpeg puis relance.")

    df = pd.read_csv(source)

    if "003N_shadow_hit" not in df.columns:
        raise SystemExit("Colonne 003N_shadow_hit absente.")

    hits = df[df["003N_shadow_hit"].map(boolish)].copy()

    out_dir = run_dir / "human_review_packet_003S2_h264"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, row in hits.iterrows():
        rid = str(row.get("review_id", "")).strip()
        src = locate_source_video(row, run_dir)

        dst = out_dir / f"{rid}_review_h264.mp4"

        ok = False
        err = ""

        if src is None:
            err = "source video not found"
        else:
            ok, err = convert_mp4(src, dst, ffmpeg)

        rel = ""
        if ok:
            rel = dst.relative_to(run_dir).as_posix()

        out = {
            "review_id": rid,
            "target_class_003G": row.get("target_class_003G", ""),
            "003N_player_motion_inside_value": row.get("003N_player_motion_inside_value", ""),
            "003N_old_table_distance_value": row.get("003N_old_table_distance_value", ""),
            "003N_max_accel_value": row.get("003N_max_accel_value", ""),
            "source_video": str(src) if src else "",
            "h264_path": str(dst) if ok else "",
            "h264_rel": rel,
            "conversion_ok": ok,
            "conversion_error": err,
        }

        rows.append(out)

        print(f"[003S2] {rid} ok={ok} src={src} dst={dst if ok else '-'}")
        if err and not ok:
            print("  error:", err[:500])

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "browser_playable_review_packet_only_no_live_filter",
        "run_dir": str(run_dir),
        "source": str(source),
        "ffmpeg": ffmpeg,
        "hit_total": int(len(hits)),
        "converted_ok": int(sum(1 for r in rows if r["conversion_ok"])),
        "converted_failed": int(sum(1 for r in rows if not r["conversion_ok"])),
        "html": str(run_dir / "human_review_packet_003S2_h264.html"),
        "template_to_fill": str(run_dir / "human_review_003S_template.csv"),
    }

    out_json = run_dir / "human_review_packet_summary_003S2.json"
    out_html = run_dir / "human_review_packet_003S2_h264.html"

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, rows, summary)

    print("")
    print("003S2 status=OK")
    print("converted_ok=", summary["converted_ok"])
    print("converted_failed=", summary["converted_failed"])
    print("html=", out_html)
    print("template=", run_dir / "human_review_003S_template.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
