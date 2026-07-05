from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "006A3_click_overlay_review"


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def imwrite_unicode(path: Path, img, jpg_quality=92):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    buf.tofile(str(path))


def as_int(v):
    try:
        return int(float(str(v).strip()))
    except Exception:
        return None


def as_float(v):
    try:
        return float(str(v).strip().replace(",", "."))
    except Exception:
        return None


def make_sheet(images, cell_w=320, cell_h=180, cols=4):
    if not images:
        return None

    rows = math.ceil(len(images) / cols)
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        thumb = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = thumb

    return sheet


def crop_around(img, x, y, size=220):
    h, w = img.shape[:2]
    half = size // 2

    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)

    crop = img[y1:y2, x1:x2].copy()

    # Pad si bord image.
    out = np.zeros((size, size, 3), dtype=np.uint8)
    ch, cw = crop.shape[:2]
    out[:ch, :cw] = crop
    return out, x1, y1


def main():
    root = Path.cwd()
    label_dir = Path("runs/006A_human_frame_labels")
    in_csv = label_dir / "006A2_position_training_rows.csv"

    out_dir = label_dir / "006A3_click_overlay_review"
    frames_dir = out_dir / "frames"
    crops_dir = out_dir / "crops"

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(in_csv)
    rows = [r for r in rows if r.get("usable_for_position") == "1"]

    if not rows:
        raise SystemExit(f"No usable position rows found: {in_csv}")

    video_paths = sorted({r.get("video_path", "").strip() for r in rows if r.get("video_path", "").strip()})
    if len(video_paths) != 1:
        raise SystemExit(f"Expected exactly 1 video_path, got {len(video_paths)}: {video_paths}")

    video_path = root / video_paths[0]
    if not video_path.exists():
        raise SystemExit(f"Video missing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    manifest = []
    full_thumbs = []
    crop_thumbs = []

    for idx, r in enumerate(rows, 1):
        frame = as_int(r.get("video_frame"))
        x_f = as_float(r.get("x"))
        y_f = as_float(r.get("y"))

        if frame is None or x_f is None or y_f is None:
            manifest.append({
                "label_uid": r.get("label_uid", ""),
                "frame": r.get("video_frame", ""),
                "read_ok": False,
                "status": "bad_row",
                "x": r.get("x", ""),
                "y": r.get("y", ""),
            })
            continue

        x = int(round(x_f))
        y = int(round(y_f))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()

        if not ok or img is None:
            manifest.append({
                "label_uid": r.get("label_uid", ""),
                "frame": frame,
                "read_ok": False,
                "status": "frame_read_failed",
                "x": x,
                "y": y,
            })
            continue

        overlay = img.copy()

        # Point clique humain.
        cv2.circle(overlay, (x, y), 14, (0, 0, 255), 2)
        cv2.drawMarker(
            overlay,
            (x, y),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=28,
            thickness=2,
        )
        cv2.putText(
            overlay,
            f"{idx:02d} f={frame} x={x} y={y}",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        out_frame = frames_dir / f"{idx:02d}_frame_{frame:06d}_overlay.jpg"
        imwrite_unicode(out_frame, overlay)

        crop, x1, y1 = crop_around(img, x, y, size=260)
        cx = x - x1
        cy = y - y1
        cv2.circle(crop, (cx, cy), 16, (0, 0, 255), 2)
        cv2.drawMarker(
            crop,
            (cx, cy),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=30,
            thickness=2,
        )
        cv2.putText(
            crop,
            f"{idx:02d} f={frame}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        out_crop = crops_dir / f"{idx:02d}_frame_{frame:06d}_crop.jpg"
        imwrite_unicode(out_crop, crop)

        full_thumbs.append(overlay)
        crop_thumbs.append(crop)

        manifest.append({
            "label_uid": r.get("label_uid", ""),
            "sample_id": r.get("sample_id", ""),
            "frame": frame,
            "x": x,
            "y": y,
            "read_ok": True,
            "status": "ok",
            "overlay_path": str(out_frame),
            "crop_path": str(out_crop),
        })

    cap.release()

    sheet_full = make_sheet(full_thumbs, cell_w=320, cell_h=180, cols=4)
    sheet_crop = make_sheet(crop_thumbs, cell_w=260, cell_h=260, cols=4)

    full_sheet_path = out_dir / "006A3_click_overlay_contact_sheet.jpg"
    crop_sheet_path = out_dir / "006A3_click_crop_contact_sheet.jpg"

    if sheet_full is not None:
        imwrite_unicode(full_sheet_path, sheet_full)

    if sheet_crop is not None:
        imwrite_unicode(crop_sheet_path, sheet_crop)

    manifest_path = out_dir / "006A3_click_overlay_manifest.csv"
    summary_path = out_dir / "006A3_click_overlay_summary.json"
    md_path = out_dir / "006A3_click_overlay_summary.md"

    write_csv(manifest_path, manifest)

    ok_count = sum(1 for r in manifest if r.get("status") == "ok")

    summary = {
        "patch": PATCH_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(in_csv),
        "video_path": str(video_path),
        "video_width": width,
        "video_height": height,
        "fps": fps,
        "frame_count": frame_count,
        "rows": len(rows),
        "ok_count": ok_count,
        "error_count": len(rows) - ok_count,
        "full_contact_sheet": str(full_sheet_path),
        "crop_contact_sheet": str(crop_sheet_path),
        "manifest": str(manifest_path),
        "ok_for_next": ok_count == len(rows) and ok_count >= 10,
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# 006A3 click overlay review")
    md.append("")
    md.append(f"- Video: `{video_path}`")
    md.append(f"- Resolution: **{width}x{height}**")
    md.append(f"- FPS: **{fps}**")
    md.append(f"- Rows: **{len(rows)}**")
    md.append(f"- OK: **{ok_count}**")
    md.append(f"- Errors: **{len(rows) - ok_count}**")
    md.append(f"- Full sheet: `{full_sheet_path}`")
    md.append(f"- Crop sheet: `{crop_sheet_path}`")
    md.append(f"- Manifest: `{manifest_path}`")
    md.append(f"- ok_for_next: **{summary['ok_for_next']}**")
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print("=" * 72)
    print("")
    print("OK 006A3")
    print(f"summary = {summary_path}")
    print(f"full_sheet = {full_sheet_path}")
    print(f"crop_sheet = {crop_sheet_path}")
    print(f"manifest = {manifest_path}")
    print("")
    print(f"rows={len(rows)} ok={ok_count} errors={len(rows) - ok_count}")
    print(f"ok_for_next={summary['ok_for_next']}")


if __name__ == "__main__":
    main()
