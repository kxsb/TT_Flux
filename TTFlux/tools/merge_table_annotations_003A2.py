from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_time(value: Any) -> float:
    if not value:
        return 0.0

    text = str(value).strip()

    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return 0.0


def valid_point(p: Any) -> bool:
    return isinstance(p, dict) and p.get("x") is not None and p.get("y") is not None


def quad_count(clip: dict[str, Any]) -> int:
    q = clip.get("table_quad") or []
    return sum(1 for p in q if valid_point(p))


def normalize_quad(clip: dict[str, Any]) -> list[Any]:
    q = clip.get("table_quad") or []

    out = []
    for i in range(4):
        if i < len(q) and valid_point(q[i]):
            p = q[i]
            out.append({
                "label": p.get("label", ["front_left", "front_right", "back_right", "back_left"][i]),
                "x": float(p["x"]),
                "y": float(p["y"]),
            })
        else:
            out.append(None)

    return out


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 92) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if frame_count > 0:
        frame_idx = max(0, min(frame_idx, frame_count - 1))
    else:
        frame_idx = max(0, frame_idx)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()

    if not ok or img is None:
        return None

    return img


def draw_overlay(img: np.ndarray, clip: dict[str, Any]) -> np.ndarray:
    out = img.copy()
    quad = normalize_quad(clip)

    pts = []
    for p in quad:
        if valid_point(p):
            pts.append([int(round(float(p["x"]))), int(round(float(p["y"])))])

    if len(pts) >= 2:
        arr = np.array(pts, dtype=np.int32)

        if len(pts) == 4:
            cv2.polylines(out, [arr], isClosed=True, color=(80, 230, 120), thickness=3, lineType=cv2.LINE_AA)
            fill = out.copy()
            cv2.fillPoly(fill, [arr], color=(80, 230, 120))
            out = cv2.addWeighted(fill, 0.18, out, 0.82, 0)
        else:
            cv2.polylines(out, [arr], isClosed=False, color=(80, 230, 120), thickness=3, lineType=cv2.LINE_AA)

    labels = ["1 front_left", "2 front_right", "3 back_right", "4 back_left"]

    for i, p in enumerate(quad):
        if not valid_point(p):
            continue

        x = int(round(float(p["x"])))
        y = int(round(float(p["y"])))

        cv2.circle(out, (x, y), 9, (80, 230, 120), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 9, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(out, labels[i], (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, labels[i], (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

    return out


def merge_exports(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    files = sorted(input_dir.glob("table_annotations_003A*.json"))

    # Exclut les fichiers de sortie éventuels.
    files = [
        p for p in files
        if "merged" not in p.name
        and "summary" not in p.name
        and "payload" not in p.name
    ]

    best: dict[str, dict[str, Any]] = {}
    seen_by_clip: dict[str, list[dict[str, Any]]] = {}

    for path in files:
        data = load_json(path)

        if not data or not isinstance(data.get("clips"), dict):
            continue

        exported_ts = parse_time(data.get("exported_at")) or path.stat().st_mtime

        for clip_id, clip in data["clips"].items():
            if not isinstance(clip, dict):
                continue

            c = dict(clip)
            c["clip_id"] = c.get("clip_id") or clip_id
            c["table_quad"] = normalize_quad(c)
            c["_source_file_003A2"] = path.name
            c["_source_mtime_003A2"] = path.stat().st_mtime
            c["_completeness_003A2"] = quad_count(c)

            updated_ts = parse_time(c.get("updated_at"))
            c["_rank_time_003A2"] = max(updated_ts, exported_ts, path.stat().st_mtime)

            seen_by_clip.setdefault(clip_id, []).append({
                "source_file": path.name,
                "complete_points": c["_completeness_003A2"],
                "updated_at": c.get("updated_at", ""),
            })

            prev = best.get(clip_id)

            if prev is None:
                best[clip_id] = c
                continue

            prev_key = (int(prev.get("_completeness_003A2", 0)), float(prev.get("_rank_time_003A2", 0)))
            new_key = (int(c.get("_completeness_003A2", 0)), float(c.get("_rank_time_003A2", 0)))

            if new_key > prev_key:
                best[clip_id] = c

    merged = {
        "version": "003A2_merged",
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(input_dir),
        "source_files": [p.name for p in files],
        "clips": {},
    }

    for clip_id in sorted(best):
        c = dict(best[clip_id])
        # On garde source/completeness dans le JSON, c'est utile pour audit.
        merged["clips"][clip_id] = c

    completed = [
        cid for cid, c in merged["clips"].items()
        if quad_count(c) == 4
    ]

    incomplete = [
        cid for cid, c in merged["clips"].items()
        if quad_count(c) < 4
    ]

    summary = {
        "source_files_count": len(files),
        "source_files": [p.name for p in files],
        "clips_total": len(merged["clips"]),
        "clips_complete": len(completed),
        "clips_incomplete": len(incomplete),
        "complete_clip_ids": completed,
        "incomplete_clip_ids": incomplete,
        "selected_by_clip": {
            cid: {
                "source_file": c.get("_source_file_003A2", ""),
                "complete_points": quad_count(c),
                "updated_at": c.get("updated_at", ""),
                "filename": c.get("filename", ""),
            }
            for cid, c in merged["clips"].items()
        },
        "all_exports_by_clip": seen_by_clip,
    }

    return merged, summary


def build_overlays(merged: dict[str, Any], out_dir: Path) -> dict[str, str]:
    overlay_dir = out_dir / "overlays"
    overlay_map: dict[str, str] = {}

    for clip_id, clip in merged.get("clips", {}).items():
        if quad_count(clip) < 4:
            continue

        video_path = Path(str(clip.get("video_path", "")))

        if not video_path.exists():
            continue

        frame_ref = int(float(clip.get("frame_ref") or 0))
        if frame_ref <= 0:
            frame_ref = int(float(clip.get("frame_count") or 0) // 2)

        img = read_video_frame(video_path, frame_ref)

        if img is None:
            continue

        overlay = draw_overlay(img, clip)
        out_path = overlay_dir / f"table_overlay_{clip_id}.jpg"

        if imwrite_unicode(out_path, overlay, quality=92):
            overlay_map[clip_id] = str(out_path.relative_to(out_dir))

    return overlay_map


def write_review_html(path: Path, merged: dict[str, Any], summary: dict[str, Any], overlay_map: dict[str, str]) -> None:
    rows = []

    for clip_id, clip in merged.get("clips", {}).items():
        count = quad_count(clip)
        status = "ok" if count == 4 else "warn"

        quad = normalize_quad(clip)
        quad_txt = html.escape(json.dumps(quad, ensure_ascii=False))

        img_html = ""
        if clip_id in overlay_map:
            img_html = f'<img src="{html.escape(overlay_map[clip_id])}" loading="lazy">'

        rows.append(f"""
<section>
<h2>{html.escape(clip_id)} · <span class="{status}">{count}/4 points</span></h2>
<p><b>Fichier :</b> {html.escape(str(clip.get("filename", "")))}</p>
<p><b>Source export :</b> <code>{html.escape(str(clip.get("_source_file_003A2", "")))}</code></p>
<p><b>Frame ref :</b> {html.escape(str(clip.get("frame_ref", "")))}</p>
<pre>{quad_txt}</pre>
{img_html}
</section>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux table annotations 003A2 review</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left}}
code,pre{{color:#dce4ff}}
pre{{white-space:pre-wrap;word-break:break-word;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
img{{max-width:100%;border-radius:12px;border:1px solid #2b303b;background:#000}}
.ok{{color:#74d99f}}
.warn{{color:#ffd37a}}
</style>
</head>
<body>
<h1>TTFlux table annotations 003A2 review</h1>

<section>
<h2>Résumé</h2>
<table>
<tr><th>source_files_count</th><td>{summary["source_files_count"]}</td></tr>
<tr><th>clips_total</th><td>{summary["clips_total"]}</td></tr>
<tr><th>clips_complete</th><td>{summary["clips_complete"]}</td></tr>
<tr><th>clips_incomplete</th><td>{summary["clips_incomplete"]}</td></tr>
</table>
</section>

{''.join(rows)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="runs/batch_001E/table_scene_003A")
    parser.add_argument("--out-dir", default="runs/batch_001E/table_scene_003A")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, summary = merge_exports(input_dir)
    overlay_map = build_overlays(merged, out_dir)

    merged_path = out_dir / "table_annotations_003A_merged.json"
    summary_path = out_dir / "table_annotations_003A_summary.json"
    html_path = out_dir / "table_annotations_003A_review.html"

    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_review_html(html_path, merged, summary, overlay_map)

    print(f"[003A2] source files     : {summary['source_files_count']}")
    print(f"[003A2] clips total      : {summary['clips_total']}")
    print(f"[003A2] clips complete   : {summary['clips_complete']}")
    print(f"[003A2] clips incomplete : {summary['clips_incomplete']}")
    print(f"[003A2] overlays         : {len(overlay_map)}")
    print(f"[003A2] merged           : {merged_path}")
    print(f"[003A2] summary          : {summary_path}")
    print(f"[003A2] html             : {html_path}")


if __name__ == "__main__":
    main()
