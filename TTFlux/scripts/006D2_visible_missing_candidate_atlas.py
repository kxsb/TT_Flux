from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "006D2_visible_ball_missing_candidate_atlas"


def read_csv(path: Path):
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def safe_float(v, default=math.nan):
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


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def crop_around(img, x, y, radius=72):
    h, w = img.shape[:2]
    x = int(round(x))
    y = int(round(y))

    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius)
    y2 = min(h, y + radius)

    crop = img[y1:y2, x1:x2].copy()

    if crop.size == 0:
        crop = np.zeros((radius * 2, radius * 2, 3), dtype=np.uint8)

    return crop, x1, y1


def draw_text_bg(img, text, org, scale=0.5, color=(255, 255, 255), thickness=1):
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 4, y - th - 7), (x + tw + 4, y + base + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def read_frame(cap, frame_no):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ok, img = cap.read()
    if not ok or img is None:
        return None
    return img


def load_missing_rows(coverage_csv: Path):
    rows, _ = read_csv(coverage_csv)

    missing = []
    for r in rows:
        human_visible = str(r.get("human_visible", "")).strip()
        has_xy = str(r.get("human_has_xy", "")).strip()
        candidate_count = safe_int(r.get("candidate_count"), 0)

        if human_visible == "1" and has_xy == "1" and candidate_count == 0:
            fr = safe_int(r.get("frame"))
            x = safe_float(r.get("human_x"))
            y = safe_float(r.get("human_y"))

            if fr >= 0 and math.isfinite(x) and math.isfinite(y):
                rr = dict(r)
                rr["_frame"] = fr
                rr["_x"] = x
                rr["_y"] = y
                missing.append(rr)

    return missing


def make_contact_sheet(video_path: Path, rows: list[dict], out_path: Path, max_items=80):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    panels = []

    for idx, r in enumerate(rows[:max_items]):
        fr = r["_frame"]
        x = r["_x"]
        y = r["_y"]

        img = read_frame(cap, fr)
        if img is None:
            continue

        crop, ox, oy = crop_around(img, x, y, radius=84)

        # agrandissement crop
        crop_big = cv2.resize(crop, (252, 252), interpolation=cv2.INTER_CUBIC)

        cx = int(round((x - ox) * 252 / max(1, crop.shape[1])))
        cy = int(round((y - oy) * 252 / max(1, crop.shape[0])))

        cv2.circle(crop_big, (cx, cy), 15, (0, 255, 255), 2)
        cv2.drawMarker(crop_big, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)

        draw_text_bg(crop_big, f"#{idx} f={fr}", (8, 22), scale=0.5)

        panels.append(crop_big)

    cap.release()

    if not panels:
        raise RuntimeError("Aucun panel généré.")

    cols = 5
    rows_n = int(math.ceil(len(panels) / cols))
    cell = 252
    sheet = np.zeros((rows_n * cell, cols * cell, 3), dtype=np.uint8)

    for i, p in enumerate(panels):
        rr = i // cols
        cc = i % cols
        y = rr * cell
        x = cc * cell
        sheet[y:y+cell, x:x+cell] = p

    imwrite_unicode(out_path, sheet)


def make_review_video(video_path: Path, rows: list[dict], out_path: Path, pad=8):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # vidéo compacte : uniquement autour des frames manquantes
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire vidéo: {out_path}")

    written = 0

    # On limite la review aux 100 premières pour rester lisible.
    for idx, r in enumerate(rows[:100]):
        fr0 = max(0, r["_frame"] - pad)
        fr1 = min(total - 1, r["_frame"] + pad)

        for fr in range(fr0, fr1 + 1):
            img = read_frame(cap, fr)
            if img is None:
                continue

            out = img.copy()

            x = int(round(r["_x"]))
            y = int(round(r["_y"]))

            if fr == r["_frame"]:
                cv2.circle(out, (x, y), 20, (0, 255, 255), 3)
                cv2.drawMarker(out, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
                status = "TARGET HUMAN BALL / NO CANDIDATE"
                color = (0, 255, 255)
            else:
                cv2.circle(out, (x, y), 12, (0, 160, 255), 2)
                status = "neighbor frame"
                color = (230, 230, 230)

            draw_text_bg(out, f"{PATCH_ID} #{idx} f={fr} target={r['_frame']} {status}", (12, 28), scale=0.55, color=color)
            writer.write(out)
            written += 1

    cap.release()
    writer.release()

    return {
        "fps": fps,
        "width": w,
        "height": h,
        "source_total_frames": total,
        "written_frames": written,
        "duration_sec": written / fps if fps else None,
    }


def frame_run_lengths(frames):
    if not frames:
        return []

    runs = []
    start = frames[0]
    prev = frames[0]

    for fr in frames[1:]:
        if fr == prev + 1:
            prev = fr
            continue

        runs.append((start, prev, prev - start + 1))
        start = fr
        prev = fr

    runs.append((start, prev, prev - start + 1))
    return runs


def main():
    root = Path(".").resolve()
    out_dir = root / "runs" / "006D2_visible_missing_candidate_atlas"
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage_csv = root / "runs/006D1_candidate_human_training_rows/006D1_human_frame_candidate_coverage.csv"
    video_path = root / "runs/005F3_align_clean_video/005F3_clean_aligned_segment.mp4"

    if not coverage_csv.exists():
        raise FileNotFoundError(coverage_csv)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    missing = load_missing_rows(coverage_csv)
    frames = sorted(r["_frame"] for r in missing)
    runs = frame_run_lengths(frames)

    missing_csv = out_dir / "006D2_visible_xy_frames_without_candidate.csv"
    contact = out_dir / "006D2_visible_missing_candidate_contact.jpg"
    review_video = out_dir / "006D2_visible_missing_candidate_review.mp4"
    runs_csv = out_dir / "006D2_missing_frame_runs.csv"
    summary_json = out_dir / "006D2_visible_missing_candidate_summary.json"

    write_csv(
        missing_csv,
        missing,
        ["frame", "human_visible", "human_has_xy", "human_x", "human_y", "label_source", "candidate_count"]
    )

    run_rows = [
        {
            "run_start": a,
            "run_end": b,
            "run_len": n,
        }
        for a, b, n in runs
    ]
    write_csv(runs_csv, run_rows, ["run_start", "run_end", "run_len"])

    make_contact_sheet(video_path, missing, contact, max_items=80)
    video_meta = make_review_video(video_path, missing, review_video, pad=8)

    summary = {
        "patch": PATCH_ID,
        "coverage_csv": str(coverage_csv),
        "video": str(video_path),
        "missing_visible_xy_count": len(missing),
        "missing_run_count": len(runs),
        "missing_runs_top": sorted(
            [
                {"run_start": a, "run_end": b, "run_len": n}
                for a, b, n in runs
            ],
            key=lambda x: x["run_len"],
            reverse=True,
        )[:20],
        "outputs": {
            "missing_csv": str(missing_csv),
            "runs_csv": str(runs_csv),
            "contact": str(contact),
            "review_video": str(review_video),
        },
        "video_meta": video_meta,
        "next": "Construire 006D3 nouveau détecteur candidat ciblé sur ces frames manquantes.",
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print("=" * 72)
    print("")
    print("OK 006D2")
    print(f"missing_csv  = {missing_csv}")
    print(f"runs_csv     = {runs_csv}")
    print(f"contact      = {contact}")
    print(f"review_video = {review_video}")
    print(f"summary_json = {summary_json}")
    print("")
    print(f"missing_visible_xy_count={len(missing)}")
    print(f"missing_run_count={len(runs)}")
    print("")
    print("TOP MISSING RUNS")
    for a, b, n in sorted(runs, key=lambda x: x[2], reverse=True)[:12]:
        print(f"  f={a}->{b} len={n}")


if __name__ == "__main__":
    main()
