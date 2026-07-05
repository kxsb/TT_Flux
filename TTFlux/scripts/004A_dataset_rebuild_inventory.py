from __future__ import annotations

import argparse
import csv
import html
import json
import os
from datetime import datetime
from pathlib import Path

import cv2


VERSION = "004A"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def norm_path(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))

    out = {
        "video_path": str(path),
        "exists": path.is_file(),
        "probe_ok": False,
        "width": "",
        "height": "",
        "fps": "",
        "frame_count": "",
        "duration_sec": "",
        "duration_min": "",
        "error": "",
    }

    if not path.is_file():
        out["error"] = "missing_file"
        return out

    if not cap.isOpened():
        out["error"] = "opencv_open_failed"
        return out

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        duration = frames / fps if fps > 0 else 0.0

        out.update({
            "probe_ok": True,
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "frame_count": frames,
            "duration_sec": round(duration, 3),
            "duration_min": round(duration / 60.0, 3),
        })

    except Exception as exc:
        out["error"] = repr(exc)

    finally:
        cap.release()

    return out


def classify_video(path: Path) -> str:
    s = str(path).lower().replace("\\", "/")
    name = path.name.lower()

    if "normalized" in name:
        return "normalized_source"
    if name == "source.mp4":
        return "source_original"
    if "/clips/" in s or "dataset2_clips" in s:
        return "existing_clip"
    if "overlay" in name or "review" in name or "h264" in name:
        return "generated_review_media"
    if "/runs/batch_" in s:
        return "generated_batch_media"

    return "unknown_mp4"


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "items" in data:
        return data["items"]

    return []


def manifest_rows(manifest_path: Path) -> list[dict]:
    rows = []

    for item in read_manifest(manifest_path):
        if not isinstance(item, dict):
            continue

        for kind in ["normalized_path", "source_path"]:
            raw = str(item.get(kind, "")).strip()
            if not raw:
                continue

            p = Path(raw)

            rows.append({
                "source": "manifest",
                "manifest_path": str(manifest_path),
                "manifest_idx": item.get("idx", ""),
                "video_id": item.get("video_id", ""),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "manifest_status": item.get("status", ""),
                "manifest_error": item.get("error", ""),
                "manifest_duration": item.get("duration", ""),
                "manifest_resolution": item.get("resolution", ""),
                "manifest_fps": item.get("fps", ""),
                "path_kind": kind,
                "video_path": str(p),
            })

    return rows


def scan_mp4_roots(roots: list[Path], max_files: int) -> list[dict]:
    rows = []
    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for p in root.rglob("*.mp4"):
            key = norm_path(p).lower()

            if key in seen:
                continue

            seen.add(key)

            rows.append({
                "source": "recursive_scan",
                "manifest_path": "",
                "manifest_idx": "",
                "video_id": "",
                "title": "",
                "url": "",
                "manifest_status": "",
                "manifest_error": "",
                "manifest_duration": "",
                "manifest_resolution": "",
                "manifest_fps": "",
                "path_kind": classify_video(p),
                "video_path": str(p),
            })

            if len(rows) >= max_files:
                return rows

    return rows


def write_html(path: Path, summary: dict, rows: list[dict]) -> None:
    trs = []

    for r in rows:
        cls = "ok" if r.get("probe_ok") else "bad"
        trs.append(
            f"<tr class='{cls}'>"
            f"<td>{esc(r.get('source'))}</td>"
            f"<td>{esc(r.get('path_kind'))}</td>"
            f"<td>{esc(r.get('video_id'))}</td>"
            f"<td>{esc(r.get('manifest_status'))}</td>"
            f"<td>{esc(r.get('probe_ok'))}</td>"
            f"<td>{esc(r.get('duration_min'))}</td>"
            f"<td>{esc(r.get('fps'))}</td>"
            f"<td>{esc(r.get('width'))}x{esc(r.get('height'))}</td>"
            f"<td>{esc(r.get('title'))}</td>"
            f"<td>{esc(r.get('video_path'))}</td>"
            f"<td>{esc(r.get('error') or r.get('manifest_error'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004A dataset rebuild inventory</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:78vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
tr.ok td{{background:rgba(116,217,159,.06)}}
tr.bad td{{background:rgba(255,80,80,.10)}}
</style>
</head>
<body>
<h1>TTFlux · 004A dataset rebuild inventory</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
<section>
<h2>Vidéos</h2>
<div class="wrap">
<table>
<thead>
<tr>
<th>source</th><th>kind</th><th>video_id</th><th>manifest status</th><th>probe</th><th>min</th><th>fps</th><th>res</th><th>title</th><th>path</th><th>error</th>
</tr>
</thead>
<tbody>
{''.join(trs)}
</tbody>
</table>
</div>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=r"C:\Users\micka\pingcoach_probe\videos_dataset2_v60\dataset2_manifest_v60.json",
    )
    ap.add_argument(
        "--scan-roots",
        nargs="*",
        default=[
            r"C:\Users\micka\pingcoach_probe\videos_dataset2_v60",
            r"C:\Users\micka\pingcoach_probe\runs\dataset2_clips_v62",
        ],
    )
    ap.add_argument("--out-dir", default="runs/dataset_rebuild_004A")
    ap.add_argument("--max-scan-files", type=int, default=500)
    args = ap.parse_args()

    root = Path.cwd()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest)

    raw_rows = []
    raw_rows.extend(manifest_rows(manifest))

    scan_roots = [Path(x) for x in args.scan_roots]
    raw_rows.extend(scan_mp4_roots(scan_roots, args.max_scan_files))

    # Déduplication par chemin résolu.
    dedup = {}
    for r in raw_rows:
        p = Path(str(r.get("video_path", "")))
        key = norm_path(p).lower()

        if key not in dedup:
            dedup[key] = r
        else:
            # Priorité aux infos du manifest.
            if dedup[key].get("source") != "manifest" and r.get("source") == "manifest":
                dedup[key] = r

    rows = []

    for r in dedup.values():
        p = Path(str(r["video_path"]))
        pr = probe_video(p)

        row = {
            **r,
            **pr,
            "path_kind": r.get("path_kind") or classify_video(p),
        }
        rows.append(row)

    rows = sorted(
        rows,
        key=lambda r: (
            str(r.get("path_kind", "")),
            str(r.get("video_id", "")),
            str(r.get("video_path", "")),
        )
    )

    ok_rows = [r for r in rows if r.get("probe_ok")]
    normalized = [r for r in ok_rows if r.get("path_kind") == "normalized_source"]
    originals = [r for r in ok_rows if r.get("path_kind") == "source_original"]
    existing_clips = [r for r in ok_rows if r.get("path_kind") == "existing_clip"]

    total_sec_all = sum(float(r.get("duration_sec") or 0) for r in ok_rows)
    total_sec_norm = sum(float(r.get("duration_sec") or 0) for r in normalized)

    by_kind = {}
    for r in rows:
        k = r.get("path_kind", "unknown")
        by_kind.setdefault(k, {"count": 0, "ok": 0, "duration_min": 0.0})
        by_kind[k]["count"] += 1
        if r.get("probe_ok"):
            by_kind[k]["ok"] += 1
            by_kind[k]["duration_min"] += float(r.get("duration_min") or 0)

    for k in by_kind:
        by_kind[k]["duration_min"] = round(by_kind[k]["duration_min"], 3)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "dataset_inventory_only_no_download_no_delete",
        "manifest": str(manifest),
        "scan_roots": [str(x) for x in scan_roots],
        "raw_rows": len(raw_rows),
        "unique_videos": len(rows),
        "probe_ok": len(ok_rows),
        "probe_failed": len(rows) - len(ok_rows),
        "normalized_source_ok": len(normalized),
        "source_original_ok": len(originals),
        "existing_clip_ok": len(existing_clips),
        "total_duration_all_ok_min": round(total_sec_all / 60.0, 3),
        "total_duration_normalized_ok_min": round(total_sec_norm / 60.0, 3),
        "by_kind": by_kind,
        "recommendation": "Use normalized_source videos as primary source for rebuilding clips. Existing clips are secondary/debug only.",
    }

    out_csv = out_dir / "dataset_rebuild_inventory_004A.csv"
    out_json = out_dir / "dataset_rebuild_inventory_summary_004A.json"
    out_html = out_dir / "dataset_rebuild_inventory_004A.html"

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, summary, rows)

    print("004A status=OK")
    print("unique_videos=", len(rows))
    print("probe_ok=", len(ok_rows))
    print("normalized_source_ok=", len(normalized))
    print("existing_clip_ok=", len(existing_clips))
    print("total_duration_all_ok_min=", summary["total_duration_all_ok_min"])
    print("total_duration_normalized_ok_min=", summary["total_duration_normalized_ok_min"])
    print("by_kind=", json.dumps(by_kind, ensure_ascii=False))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
