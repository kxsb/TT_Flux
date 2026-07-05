from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "006B1_human_trajectory_audit_review"


FIELDS = [
    "source_file",
    "session_id",
    "video_path",
    "frame",
    "timestamp_sec",
    "visible",
    "x",
    "y",
    "label",
    "point_type",
    "saved_at",
    "patch",
]


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


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in sorted(rows, key=lambda x: safe_int(x.get("frame"))):
            wr.writerow({k: r.get(k, "") for k in fields})


def draw_text_bg(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1):
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 5, y - th - 8), (x + tw + 5, y + base + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def merge_labels(label_dir: Path):
    files = sorted(label_dir.glob("006B_trajectory_labels_*.csv"))
    rows = []

    for p in files:
        for r in read_csv(p):
            rr = dict(r)
            rr["source_file"] = str(p)
            rows.append(rr)

    # Dédupliquer par frame : priorité manual > interpolated > invisible/unsure.
    priority = {
        "manual": 3,
        "interpolated": 2,
        "": 1,
    }

    by_frame = {}
    for r in rows:
        fr = safe_int(r.get("frame"))
        if fr < 0:
            continue

        old = by_frame.get(fr)
        if old is None:
            by_frame[fr] = r
            continue

        old_prio = priority.get(old.get("point_type", ""), 1)
        new_prio = priority.get(r.get("point_type", ""), 1)

        if new_prio >= old_prio:
            by_frame[fr] = r

    merged = [by_frame[k] for k in sorted(by_frame)]

    return files, merged


def label_map(rows):
    out = {}
    for r in rows:
        fr = safe_int(r.get("frame"))
        if fr >= 0:
            out[fr] = r
    return out


def draw_review(frame, frame_idx, labels, trail=45):
    out = frame.copy()
    h, w = out.shape[:2]

    prev = None

    for fr in range(frame_idx - trail, frame_idx + trail + 1):
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
        status = f"{cur.get('label','')} / {cur.get('point_type','')}"
        if cur.get("visible") == "1" and cur.get("x") and cur.get("y"):
            x = int(round(safe_float(cur.get("x"))))
            y = int(round(safe_float(cur.get("y"))))
            cv2.circle(out, (x, y), 16, (255, 255, 255), 2)
            cv2.drawMarker(out, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
        elif cur.get("visible") == "0":
            cv2.rectangle(out, (w - 190, 14), (w - 14, 54), (0, 0, 120), -1)
            cv2.putText(out, "INVISIBLE", (w - 176, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2, cv2.LINE_AA)

    draw_text_bg(out, f"{PATCH_ID} | f={frame_idx} | {status}", (12, 28), scale=0.58)

    return out


def make_review_video(video_path: Path, out_video: Path, labels: dict[int, dict]):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire vidéo: {out_video}")

    if labels:
        min_frame = max(0, min(labels) - 20)
        max_frame = min(total - 1, max(labels) + 20)
    else:
        min_frame = 0
        max_frame = min(total - 1, int(15 * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)

    written = 0
    frame_idx = min_frame

    while frame_idx <= max_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        out = draw_review(frame, frame_idx, labels)
        writer.write(out)

        written += 1
        frame_idx += 1

    cap.release()
    writer.release()

    return {
        "fps": fps,
        "width": w,
        "height": h,
        "source_total_frames": total,
        "review_start_frame": min_frame,
        "review_end_frame": max_frame,
        "written_frames": written,
        "duration_sec": written / fps if fps else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="runs/006B_human_trajectory_labels")
    ap.add_argument("--video", default="")
    args = ap.parse_args()

    label_dir = Path(args.label_dir)
    files, rows = merge_labels(label_dir)

    if not files:
        raise RuntimeError(f"Aucun fichier 006B_trajectory_labels_*.csv trouvé dans {label_dir}")

    video_path = Path(args.video) if args.video else None

    if video_path is None:
        for r in rows:
            v = str(r.get("video_path", "")).strip()
            if v:
                video_path = Path(v)
                break

    if video_path is None or not video_path.exists():
        raise RuntimeError("Vidéo introuvable. Passe --video chemin_video.mp4")

    labels = label_map(rows)

    merged_csv = label_dir / "006B1_human_trajectory_merged.csv"
    review_video = label_dir / "006B1_human_trajectory_review.mp4"
    summary_path = label_dir / "006B1_human_trajectory_audit_summary.json"

    write_csv(merged_csv, rows, FIELDS)

    video_meta = make_review_video(video_path, review_video, labels)

    visible = [r for r in rows if r.get("visible") == "1"]
    manual = [r for r in rows if r.get("point_type") == "manual"]
    interpolated = [r for r in rows if r.get("point_type") == "interpolated"]
    invisible = [r for r in rows if r.get("label") == "invisible"]
    unsure = [r for r in rows if r.get("label") == "unsure"]

    frames = sorted(labels)

    summary = {
        "patch": PATCH_ID,
        "label_dir": str(label_dir),
        "source_files": [str(p) for p in files],
        "video": str(video_path),
        "merged_csv": str(merged_csv),
        "review_video": str(review_video),
        "row_count": len(rows),
        "frame_count": len(frames),
        "visible_count": len(visible),
        "manual_count": len(manual),
        "interpolated_count": len(interpolated),
        "invisible_count": len(invisible),
        "unsure_count": len(unsure),
        "min_frame": min(frames) if frames else None,
        "max_frame": max(frames) if frames else None,
        "video_meta": video_meta,
        "ok_for_dataset_join": len(manual) >= 2 and len(visible) >= 5,
        "note": "Audit labels humains trajectoire. Ne modifie aucun tracker.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"label_dir = {label_dir}")
    print(f"video     = {video_path}")
    print("=" * 72)
    print("")
    print("OK 006B1")
    print(f"merged_csv  = {merged_csv}")
    print(f"review_video = {review_video}")
    print(f"summary     = {summary_path}")
    print("")
    print(f"row_count={len(rows)}")
    print(f"frame_count={len(frames)}")
    print(f"visible_count={len(visible)}")
    print(f"manual_count={len(manual)}")
    print(f"interpolated_count={len(interpolated)}")
    print(f"invisible_count={len(invisible)}")
    print(f"unsure_count={len(unsure)}")
    print(f"ok_for_dataset_join={summary['ok_for_dataset_join']}")


if __name__ == "__main__":
    main()
