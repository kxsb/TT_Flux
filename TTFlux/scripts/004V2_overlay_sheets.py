from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "004V2"


def esc(x):
    return html.escape("" if x is None else str(x))


def imwrite_unicode(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def make_sheet(frames, cols=4, thumb_w=520):
    if not frames:
        return None

    thumbs = []
    for img in frames:
        h, w = img.shape[:2]
        scale = thumb_w / max(1, w)
        thumb_h = int(round(h * scale))
        t = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumbs.append(t)

    max_h = max(t.shape[0] for t in thumbs)

    padded = []
    for t in thumbs:
        if t.shape[0] < max_h:
            pad = np.zeros((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8)
            t = np.vstack([t, pad])
        padded.append(t)

    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row))

    return np.vstack(rows)


def sample_overlay(video_path: Path, n=12):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"ok": False, "error": "open_failed"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

    if frame_count <= 0:
        cap.release()
        return [], {"ok": False, "error": "empty_video"}

    idxs = np.linspace(0, frame_count - 1, n).round().astype(int).tolist()

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        cv2.putText(
            frame,
            f"sample frame={idx} t={idx / fps:.2f}s" if fps else f"sample frame={idx}",
            (24, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(frame)

    cap.release()

    return frames, {
        "ok": True,
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "duration_sec": round(frame_count / fps, 3) if fps else "",
        "sample_count": len(frames),
    }


def write_html(path: Path, rows, summary):
    cards = []

    for r in rows:
        sheet = r.get("sheet_rel", "")
        img = f"<img src='{esc(sheet)}'>" if sheet else ""

        cards.append(f"""
<section>
<h2>{esc(r.get('review_id'))} · {esc(r.get('video_id'))}</h2>
<p>
frames={esc(r.get('frames_processed'))} · path_score_med={esc(r.get('path_score_med'))} ·
speed_med={esc(r.get('speed_px_sec_med'))} · speed_p90={esc(r.get('speed_px_sec_p90'))}
</p>
{img}
<p><code>{esc(r.get('overlay'))}</code></p>
</section>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004V2 overlay sheets</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
img{{max-width:100%;border-radius:10px;border:1px solid #2b303b;background:#000}}
code{{font-size:12px;color:#c8d3ff}}
</style>
</head>
<body>
<h1>TTFlux · 004V2 overlay sheets</h1>
<section><h2>Résumé</h2><pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre></section>
{''.join(cards)}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", default="runs/rally_apply_ranker_004V_probe12/rally_ranker_review_summary_004V.csv")
    ap.add_argument("--out-dir", default="runs/rally_apply_ranker_004V_probe12_sheets")
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()

    root = Path.cwd()

    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = root / summary_csv

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    sheets_dir = out_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv).fillna("")

    rows = []

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        overlay = str(r.get("overlay", ""))
        if not overlay:
            continue

        overlay_path = Path(overlay)
        if not overlay_path.is_absolute():
            overlay_path = root / overlay_path

        frames, probe = sample_overlay(overlay_path, n=args.samples)

        sheet_path = ""
        sheet_rel = ""

        if frames:
            sheet = make_sheet(frames, cols=4, thumb_w=520)
            if sheet is not None:
                sheet_path = sheets_dir / f"{i:02d}_{r.get('review_id')}_{r.get('video_id')}_sheet.jpg"
                imwrite_unicode(sheet_path, sheet)
                sheet_rel = str(sheet_path.relative_to(out_dir)).replace("\\", "/")

        row = {
            **r.to_dict(),
            **probe,
            "sheet": str(sheet_path),
            "sheet_rel": sheet_rel,
        }
        rows.append(row)

        print(f"  {i}/{len(df)} {r.get('review_id')} samples={len(frames)} sheet={sheet_path}")

    out_csv = out_dir / "overlay_sheets_004V2.csv"
    out_json = out_dir / "overlay_sheets_summary_004V2.json"
    out_html = out_dir / "overlay_sheets_004V2.html"

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(summary_csv),
        "rows": len(rows),
        "samples_per_overlay": args.samples,
        "instruction": {
            "orange": "top1 frame-by-frame",
            "blue": "path Viterbi/ranker",
            "good": "blue follows plausible ball trajectory over several seconds",
            "bad": "blue sticks to players, table edges, logos, shirts, rackets"
        }
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, rows, summary)

    print("004V2 status=OK")
    print("rows=", len(rows))
    print("sheets_dir=", sheets_dir)
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
