from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005F3_align_gap_focus_to_clean_video"


def frame_signature(img):
    # On masque grossièrement les zones score/overlay et le bas écran.
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Zone centrale robuste : table + joueurs, peu d'overlay.
    y1 = int(h * 0.12)
    y2 = int(h * 0.78)
    x1 = int(w * 0.05)
    x2 = int(w * 0.95)
    roi = gray[y1:y2, x1:x2]

    small = cv2.resize(roi, (160, 90), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0)

    # Normalisation pour réduire l'effet compression/overlay.
    small = small.astype(np.float32)
    small = (small - small.mean()) / (small.std() + 1e-6)
    return small


def read_frame(cap, fr):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
    ok, img = cap.read()
    if not ok or img is None:
        return None
    return img


def sig_dist(a, b):
    d = a - b
    return float(np.mean(d * d))


def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-video", default="runs/rally_ball_gap_audit_005D5/gap_focus/RLY0074_005D5_gap_focus.mp4")
    ap.add_argument("--clean-video", default="videos_dataset2_v60/05_-0bM0t0qS8Q/normalized_720p50.mp4")
    ap.add_argument("--out-dir", default="runs/005F3_align_clean_video")
    ap.add_argument("--coarse-step", type=int, default=25)
    ap.add_argument("--fine-radius", type=int, default=80)
    args = ap.parse_args()

    gap_video = Path(args.gap_video)
    clean_video = Path(args.clean_video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gap_video.exists():
        raise FileNotFoundError(gap_video)
    if not clean_video.exists():
        raise FileNotFoundError(clean_video)

    gap_info = video_info(gap_video)
    clean_info = video_info(clean_video)

    gap_cap = cv2.VideoCapture(str(gap_video))
    clean_cap = cv2.VideoCapture(str(clean_video))

    if not gap_cap.isOpened() or not clean_cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir une des vidéos.")

    # Frames échantillons dans gap_focus. On évite tout début/fin.
    gap_samples = [
        int(gap_info["frames"] * p)
        for p in [0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88]
    ]
    gap_samples = sorted(set(max(0, min(gap_info["frames"] - 1, x)) for x in gap_samples))

    gap_sigs = []
    for gf in gap_samples:
        img = read_frame(gap_cap, gf)
        if img is None:
            continue
        gap_sigs.append((gf, frame_signature(img)))

    if len(gap_sigs) < 3:
        raise RuntimeError("Pas assez de signatures gap_focus.")

    max_offset = clean_info["frames"] - gap_info["frames"] - 1
    if max_offset <= 0:
        raise RuntimeError("Vidéo clean plus courte que gap_focus.")

    coarse_scores = []

    for off in range(0, max_offset + 1, args.coarse_step):
        vals = []
        for gf, gs in gap_sigs:
            cf = off + gf
            img = read_frame(clean_cap, cf)
            if img is None:
                vals.append(9999.0)
                continue
            cs = frame_signature(img)
            vals.append(sig_dist(gs, cs))

        score = float(np.median(vals))
        coarse_scores.append((score, off))

    coarse_scores.sort(key=lambda x: x[0])
    coarse_best_score, coarse_best_offset = coarse_scores[0]

    fine_start = max(0, coarse_best_offset - args.fine_radius)
    fine_end = min(max_offset, coarse_best_offset + args.fine_radius)

    fine_scores = []
    for off in range(fine_start, fine_end + 1):
        vals = []
        for gf, gs in gap_sigs:
            cf = off + gf
            img = read_frame(clean_cap, cf)
            if img is None:
                vals.append(9999.0)
                continue
            cs = frame_signature(img)
            vals.append(sig_dist(gs, cs))

        score = float(np.median(vals))
        fine_scores.append((score, off))

    fine_scores.sort(key=lambda x: x[0])
    best_score, best_offset = fine_scores[0]

    # Générer une petite vidéo clean aligned du même span.
    aligned_video = out_dir / "005F3_clean_aligned_segment.mp4"

    clean_cap.set(cv2.CAP_PROP_POS_FRAMES, best_offset)
    writer = cv2.VideoWriter(
        str(aligned_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        clean_info["fps"] or 50.0,
        (clean_info["width"], clean_info["height"]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Impossible d'écrire {aligned_video}")

    written = 0
    for i in range(gap_info["frames"]):
        ok, img = clean_cap.read()
        if not ok or img is None:
            break
        writer.write(img)
        written += 1

    writer.release()
    gap_cap.release()
    clean_cap.release()

    summary_path = out_dir / "005F3_align_clean_video_summary.json"
    next_video = out_dir / "NEXT_CLEAN_ALIGNED_VIDEO.txt"

    summary = {
        "patch": PATCH_ID,
        "gap_video": str(gap_video),
        "clean_video": str(clean_video),
        "gap_info": gap_info,
        "clean_info": clean_info,
        "gap_samples": gap_samples,
        "coarse_step": args.coarse_step,
        "fine_radius": args.fine_radius,
        "coarse_best_offset": coarse_best_offset,
        "coarse_best_score": coarse_best_score,
        "best_offset": best_offset,
        "best_score": best_score,
        "top_coarse": [{"offset": o, "score": s} for s, o in coarse_scores[:12]],
        "top_fine": [{"offset": o, "score": s} for s, o in fine_scores[:20]],
        "aligned_video": str(aligned_video),
        "aligned_written_frames": written,
        "note": "Utiliser aligned_video pour les rendus de tracking local 0..N.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    next_video.write_text(str(aligned_video), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"gap_video   = {gap_video}")
    print(f"clean_video = {clean_video}")
    print("=" * 72)
    print("")
    print("OK 005F3")
    print(f"summary       = {summary_path}")
    print(f"aligned_video = {aligned_video}")
    print(f"next_video    = {next_video}")
    print("")
    print(f"gap_frames={gap_info['frames']} clean_frames={clean_info['frames']}")
    print(f"best_offset={best_offset}")
    print(f"best_score={best_score:.6f}")
    print(f"aligned_written_frames={written}")
    print("")
    print("top offsets:")
    for s, o in fine_scores[:8]:
        print(f"  offset={o} score={s:.6f}")


if __name__ == "__main__":
    main()
