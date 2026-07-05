from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "004H2"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def safe_name(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def transcode_h264(ffmpeg: str, src: Path, dst: Path) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)

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


def write_html(path: Path, rows: pd.DataFrame, summary: dict):
    trs = []

    for _, r in rows.iterrows():
        src = str(r.get("asset_relpath_004H2", ""))
        video = ""
        if src:
            video = f"<video controls preload='metadata' width='360' src='{esc(src)}' type='video/mp4'></video>"

        label_cell = """
<select>
  <option></option>
  <option>keep</option>
  <option>partial</option>
  <option>reject</option>
  <option>unsure</option>
</select>
"""

        trs.append(
            "<tr>"
            f"<td>{esc(r.get('review_rank_004H'))}</td>"
            f"<td>{esc(r.get('review_id'))}</td>"
            f"<td>{esc(r.get('video_id'))}</td>"
            f"<td>{esc(r.get('clip_id'))}</td>"
            f"<td>{esc(r.get('segment_idx'))}</td>"
            f"<td>{esc(r.get('n_points'))}</td>"
            f"<td>{esc(r.get('density'))}</td>"
            f"<td>{esc(r.get('travel'))}</td>"
            f"<td>{esc(r.get('guess'))}</td>"
            f"<td>{video}</td>"
            f"<td>{label_cell}</td>"
            f"<td>{esc(r.get('flags'))}</td>"
            f"<td>{esc(r.get('transcode_error_004H2'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004H2 review packet H264</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:80vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
video{{border-radius:8px;background:#000}}
select{{background:#101218;color:#edf0f7;border:1px solid #3b4150;border-radius:6px;padding:6px}}
</style>
</head>
<body>
<h1>TTFlux · 004H2 review packet H.264</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
<section>
<h2>Revue visuelle</h2>
<div class="wrap">
<table>
<thead>
<tr>
<th>rank</th><th>review</th><th>video</th><th>clip</th><th>seg</th><th>n</th><th>density</th><th>travel</th><th>guess</th><th>preview</th><th>label visuel</th><th>flags</th><th>error</th>
</tr>
</thead>
<tbody>{''.join(trs)}</tbody>
</table>
</div>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet-csv", default="runs/review_packet_004H/review_packet_004H.csv")
    ap.add_argument("--run-dir", default="runs/batch_004F_full240")
    ap.add_argument("--out-dir", default="runs/review_packet_004H2_h264")
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

    if not packet_csv.is_file():
        raise SystemExit(f"packet absent: {packet_csv}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg introuvable dans le PATH")

    df = pd.read_csv(packet_csv)

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    rows = []
    ok_count = 0
    fail_count = 0

    print("004H2 rows=", len(df))
    print("004H2 ffmpeg=", ffmpeg)

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        review_id = str(r.get("review_id", f"R{i:04d}"))
        video_id = safe_name(str(r.get("video_id", "video")))
        seg = str(r.get("segment_idx", "seg"))

        mp4_rel = str(r.get("mp4", ""))
        src = run_dir / mp4_rel

        dst_name = f"{int(r.get('review_rank_004H', i)):03d}_{safe_name(review_id)}_{video_id}_seg{safe_name(seg)}.mp4"
        dst = assets_dir / dst_name

        ok = False
        err = ""

        if not src.is_file():
            err = f"source_missing: {src}"
        else:
            ok, err = transcode_h264(ffmpeg, src, dst)

        row = r.to_dict()
        row["asset_path_004H2"] = str(dst) if ok else ""
        row["asset_relpath_004H2"] = "assets/" + dst.name if ok else ""
        row["transcode_ok_004H2"] = bool(ok)
        row["transcode_error_004H2"] = err

        if ok:
            ok_count += 1
        else:
            fail_count += 1

        rows.append(row)

        if i % 20 == 0 or i == len(df):
            print(f"  {i}/{len(df)} ok={ok_count} fail={fail_count}")

    out = pd.DataFrame(rows)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "browser_review_packet_h264_only_no_tracking_no_delete",
        "packet_csv": str(packet_csv),
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "rows": int(len(out)),
        "transcode_ok": int(ok_count),
        "transcode_failed": int(fail_count),
        "instruction": "Annoter visuellement: keep / partial / reject / unsure. Les menus ne sauvegardent pas encore automatiquement.",
    }

    out_csv = out_dir / "review_packet_004H2_h264.csv"
    out_json = out_dir / "review_packet_004H2_summary.json"
    out_html = out_dir / "review_packet_004H2.html"

    out.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, out, summary)

    print("004H2 status=OK")
    print("rows=", len(out))
    print("transcode_ok=", ok_count)
    print("transcode_failed=", fail_count)
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
