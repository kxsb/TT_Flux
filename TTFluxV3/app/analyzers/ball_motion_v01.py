from pathlib import Path
import csv
import json
import math
import time
import cv2


def run_analysis(input_video: Path, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)

    preview_path = run_dir / "analysis_preview.mp4"
    csv_path = run_dir / "ball_tracking.csv"
    summary_path = run_dir / "summary.json"

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    rows = []
    prev_gray = None
    frame_idx = 0
    found_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        best = None

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)

            # Nettoyage léger.
            mask = cv2.medianBlur(mask, 3)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidates = []
            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                area = bw * bh

                # Balle = petit objet en mouvement. Très brut, volontairement.
                if 2 <= bw <= 18 and 2 <= bh <= 18 and 4 <= area <= 220:
                    cx = x + bw / 2
                    cy = y + bh / 2

                    # Score simple : mouvement compact + plutôt clair.
                    patch = gray[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
                    brightness = float(patch.mean()) if patch.size else 0.0
                    compact = 1.0 / (1.0 + abs(bw - bh))
                    score = (brightness / 255.0) * 0.7 + compact * 0.3

                    candidates.append((score, cx, cy, bw, bh, area))

            if candidates:
                candidates.sort(reverse=True, key=lambda t: t[0])
                best = candidates[0]

        if best is not None:
            score, cx, cy, bw, bh, area = best
            found_count += 1

            rows.append({
                "frame_num": frame_idx,
                "time_sec": round(frame_idx / fps, 4),
                "x_num": round(cx, 3),
                "y_num": round(cy, 3),
                "score_num": round(score, 6),
                "state": "BALL_CANDIDATE",
                "display_source": "motion_v01",
                "is_interp": 0,
                "display_segment_id": 1,
                "bbox_w": bw,
                "bbox_h": bh,
            })

            cv2.circle(frame, (int(cx), int(cy)), 9, (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"ball? {score:.2f}",
                (int(cx) + 12, int(cy) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(frame, (16, 16), (455, 74), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"TTFlux V3 analysis | frame {frame_idx}",
            (28, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        prev_gray = gray
        frame_idx += 1

    cap.release()
    writer.release()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame_num", "time_sec", "x_num", "y_num", "score_num",
            "state", "display_source", "is_interp", "display_segment_id",
            "bbox_w", "bbox_h",
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    summary = {
        "version": "V3-001_motion_v01",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_video": str(input_video),
        "preview_video": str(preview_path),
        "tracking_csv": str(csv_path),
        "frames_read": frame_idx,
        "fps": fps,
        "width": w,
        "height": h,
        "candidate_frames": found_count,
        "candidate_ratio": round(found_count / max(frame_idx, 1), 4),
        "note": "Détection mouvement brute. Contrat V3 stable, tracking à améliorer ensuite.",
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "preview_video": str(preview_path),
        "tracking_csv": str(csv_path),
        "summary_json": str(summary_path),
    }
