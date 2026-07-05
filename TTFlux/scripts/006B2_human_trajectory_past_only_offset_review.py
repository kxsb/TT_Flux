from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "006B2_human_trajectory_past_only_offset_review"


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v, default=-1):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return int(float(v))
    except Exception:
        return default


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def draw_text_bg(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1):
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 5, y - th - 8), (x + tw + 5, y + base + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def shifted_label_map(rows, shift):
    """
    shift > 0 : les labels sont affichés plus tard.
    shift < 0 : les labels sont affichés plus tôt.
    """
    labels = {}
    for r in rows:
        fr = safe_int(r.get("frame"))
        if fr < 0:
            continue
        rr = dict(r)
        rr["_original_frame"] = fr
        rr["_display_frame"] = fr + shift
        labels[fr + shift] = rr
    return labels


def draw_review(frame, frame_idx, labels, trail=45, shift=0):
    out = frame.copy()
    h, w = out.shape[:2]

    prev = None

    # Past-only : jamais de futur.
    for fr in range(frame_idx - trail, frame_idx + 1):
        r = labels.get(fr)

        if not r or r.get("visible") != "1" or not r.get("x") or not r.get("y"):
            prev = None
            continue

        x = int(round(safe_float(r.get("x"))))
        y = int(round(safe_float(r.get("y"))))

        if not (0 <= x < w and 0 <= y < h):
            prev = None
            continue

        ptype = r.get("point_type", "")

        if ptype == "manual":
            color = (0, 255, 255)
            radius = 6
        else:
            color = (0, 160, 255)
            radius = 3

        if prev is not None:
            cv2.line(out, prev, (x, y), (0, 190, 255), 2)

        cv2.circle(out, (x, y), radius, color, -1)
        prev = (x, y)

    cur = labels.get(frame_idx)
    status = "unlabeled"

    if cur:
        status = f"{cur.get('label','')} / {cur.get('point_type','')} / original_f={cur.get('_original_frame')}"
        if cur.get("visible") == "1" and cur.get("x") and cur.get("y"):
            x = int(round(safe_float(cur.get("x"))))
            y = int(round(safe_float(cur.get("y"))))
            cv2.circle(out, (x, y), 17, (255, 255, 255), 2)
            cv2.drawMarker(out, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)

    draw_text_bg(out, f"{PATCH_ID} | f={frame_idx} | shift={shift} | past-only | {status}", (12, 28), scale=0.55)

    return out


def make_video(video_path: Path, out_video: Path, labels, shift, pad=20):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if labels:
        min_frame = max(0, min(labels) - pad)
        max_frame = min(total - 1, max(labels) + pad)
    else:
        min_frame = 0
        max_frame = min(total - 1, int(15 * fps))

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire vidéo: {out_video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)

    written = 0
    frame_idx = min_frame

    while frame_idx <= max_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        out = draw_review(frame, frame_idx, labels, shift=shift)
        writer.write(out)

        written += 1
        frame_idx += 1

    cap.release()
    writer.release()

    return {
        "shift": shift,
        "out_video": str(out_video),
        "start_frame": min_frame,
        "end_frame": max_frame,
        "written_frames": written,
        "duration_sec": written / fps if fps else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="runs/006B_human_trajectory_labels/006B1_human_trajectory_merged.csv")
    ap.add_argument("--video", default="runs/005F3_align_clean_video/005F3_clean_aligned_segment.mp4")
    ap.add_argument("--out-dir", default="runs/006B_human_trajectory_labels/006B2_offset_review")
    ap.add_argument("--shifts", default="0,-3,-2,-1,1,2,3")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(labels_path)
    if not rows:
        raise RuntimeError(f"Aucun label trouvé: {labels_path}")
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    shifts = []
    for s in args.shifts.split(","):
        s = s.strip()
        if s:
            shifts.append(int(s))

    results = []

    for shift in shifts:
        labels = shifted_label_map(rows, shift)
        sign = "p" if shift >= 0 else "m"
        out_video = out_dir / f"006B2_past_only_shift_{sign}{abs(shift):02d}.mp4"
        meta = make_video(video_path, out_video, labels, shift)
        results.append(meta)

    summary = {
        "patch": PATCH_ID,
        "labels": str(labels_path),
        "video": str(video_path),
        "out_dir": str(out_dir),
        "shift_definition": "shift > 0 = labels affichés plus tard ; shift < 0 = labels affichés plus tôt",
        "row_count": len(rows),
        "results": results,
        "note": "Commencer par shift 0. Si le marqueur courant reste en avance sur la balle, essayer shift +1/+2/+3.",
    }

    summary_path = out_dir / "006B2_offset_review_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"labels = {labels_path}")
    print(f"video  = {video_path}")
    print("=" * 72)
    print("")
    print("OK 006B2")
    print(f"summary = {summary_path}")
    print("")
    print("Vidéos générées :")
    for r in results:
        print(f"  shift={r['shift']}: {r['out_video']}")
    print("")
    print("Lecture :")
    print("  shift 0  = correction du bug past-only seulement")
    print("  shift +1 = labels affichés 1 frame plus tard")
    print("  shift +2 = labels affichés 2 frames plus tard")
    print("  shift -1 = labels affichés 1 frame plus tôt")


if __name__ == "__main__":
    main()
