from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np

from ttflux.tracking.candidates.scorers import (
    BallCandidateScorer,
    BallCandidateScoringInput,
    HeuristicV1BallCandidateScorer,
)


@dataclass(frozen=True)
class CandidateConfig:
    motion_threshold: int = 18
    min_area: int = 2
    max_area: int = 160
    max_dimension: int = 26
    min_fill_ratio: float = 0.10
    max_candidates_per_frame: int = 24
    blur_kernel: int = 3

    def validate(self) -> None:
        if self.motion_threshold < 1:
            raise ValueError("motion_threshold doit être positif.")
        if self.min_area < 1 or self.max_area < self.min_area:
            raise ValueError("Plage d'aire candidate invalide.")
        if self.max_dimension < 2:
            raise ValueError("max_dimension doit être supérieur à 1.")
        if not 0 <= self.min_fill_ratio <= 1:
            raise ValueError("min_fill_ratio doit être compris entre 0 et 1.")
        if self.max_candidates_per_frame < 1:
            raise ValueError("max_candidates_per_frame doit être positif.")
        if self.blur_kernel not in {1, 3, 5, 7}:
            raise ValueError("blur_kernel doit valoir 1, 3, 5 ou 7.")


CSV_FIELDS = (
    "candidate_id",
    "frame",
    "time_s",
    "rank",
    "x",
    "y",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "area",
    "mean_brightness",
    "motion_strength",
    "fill_ratio",
    "circularity",
    "score",
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_gray(frame: np.ndarray, blur_kernel: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if blur_kernel > 1:
        gray = cv2.GaussianBlur(
            gray,
            (blur_kernel, blur_kernel),
            0,
        )

    return gray


def _component_circularity(component_mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return 0.0

    area = cv2.contourArea(contour)

    return float(
        max(
            0.0,
            min(1.0, 4.0 * math.pi * area / (perimeter * perimeter)),
        )
    )


def detect_frame_candidates(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    next_gray: np.ndarray,
    config: CandidateConfig | None = None,
    scorer: BallCandidateScorer | None = None,
) -> list[dict[str, float | int]]:
    """Détecte de petits composants mobiles sur la frame centrale."""

    resolved = config or CandidateConfig()
    resolved.validate()
    resolved_scorer = scorer or HeuristicV1BallCandidateScorer()

    if (
        previous_gray.shape != current_gray.shape
        or current_gray.shape != next_gray.shape
    ):
        raise ValueError("Les trois images doivent avoir la même dimension.")

    diff_previous = cv2.absdiff(current_gray, previous_gray)
    diff_next = cv2.absdiff(current_gray, next_gray)
    symmetric_motion = cv2.min(diff_previous, diff_next)

    _, binary = cv2.threshold(
        symmetric_motion,
        resolved.motion_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    component_count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    candidates: list[dict[str, float | int]] = []

    for label_id in range(1, component_count):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < resolved.min_area or area > resolved.max_area:
            continue

        if width > resolved.max_dimension or height > resolved.max_dimension:
            continue

        if width < 1 or height < 1:
            continue

        fill_ratio = area / float(width * height)

        if fill_ratio < resolved.min_fill_ratio:
            continue

        local_labels = labels[y : y + height, x : x + width]
        local_component = (local_labels == label_id).astype(np.uint8)
        local_mask = local_component.astype(bool)

        local_gray = current_gray[y : y + height, x : x + width]
        local_motion = symmetric_motion[
            y : y + height,
            x : x + width,
        ]

        mean_brightness = float(local_gray[local_mask].mean())
        motion_strength = float(local_motion[local_mask].mean())
        circularity = _component_circularity(local_component * 255)

        centroid_x = float(centroids[label_id][0])
        centroid_y = float(centroids[label_id][1])

        scoring_input = BallCandidateScoringInput(
            previous_gray=previous_gray,
            current_gray=current_gray,
            next_gray=next_gray,
            x=centroid_x,
            y=centroid_y,
            bbox_x=x,
            bbox_y=y,
            bbox_w=width,
            bbox_h=height,
            area=area,
            mean_brightness=mean_brightness,
            motion_strength=motion_strength,
            fill_ratio=fill_ratio,
            circularity=circularity,
        )
        score = resolved_scorer.score(scoring_input)

        candidates.append(
            {
                "x": round(centroid_x, 3),
                "y": round(centroid_y, 3),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": width,
                "bbox_h": height,
                "area": area,
                "mean_brightness": round(mean_brightness, 3),
                "motion_strength": round(motion_strength, 3),
                "fill_ratio": round(fill_ratio, 4),
                "circularity": round(circularity, 4),
                "score": round(score, 6),
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item["score"]),
            float(item["motion_strength"]),
        ),
        reverse=True,
    )

    return candidates[: resolved.max_candidates_per_frame]


def summarize_candidate_counts(
    counts: list[int],
    scores: list[float],
) -> dict[str, Any]:
    analyzed_frames = len(counts)
    frames_with_candidates = sum(1 for count in counts if count > 0)
    total_candidates = sum(counts)

    return {
        "analyzed_frames": analyzed_frames,
        "frames_with_candidates": frames_with_candidates,
        "coverage_ratio": round(
            frames_with_candidates / analyzed_frames,
            6,
        )
        if analyzed_frames
        else 0.0,
        "total_candidates": total_candidates,
        "mean_candidates_per_frame": round(mean(counts), 3)
        if counts
        else 0.0,
        "median_candidates_per_frame": round(float(median(counts)), 3)
        if counts
        else 0.0,
        "max_candidates_per_frame": max(counts) if counts else 0,
        "mean_candidate_score": round(mean(scores), 6)
        if scores
        else 0.0,
        "max_candidate_score": round(max(scores), 6)
        if scores
        else 0.0,
    }


def _draw_overlay(
    frame: np.ndarray,
    frame_index: int,
    candidates: list[dict[str, Any]],
) -> np.ndarray:
    output = frame.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (420, 34),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        output,
        f"frame {frame_index} | candidats {len(candidates)}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for rank, candidate in enumerate(candidates, start=1):
        x = int(round(float(candidate["x"])))
        y = int(round(float(candidate["y"])))
        width = int(candidate["bbox_w"])
        height = int(candidate["bbox_h"])
        radius = max(5, min(12, max(width, height) // 2 + 3))
        color = (70, 230, 90) if rank == 1 else (0, 180, 255)

        cv2.circle(
            output,
            (x, y),
            radius,
            color,
            1,
            cv2.LINE_AA,
        )

        if rank <= 8:
            cv2.putText(
                output,
                str(rank),
                (x + radius + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    return output


def _transcode_overlay(
    intermediate_path: Path,
    destination_path: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError("ffmpeg est introuvable dans le PATH.")

    temporary_path = destination_path.with_suffix(".partial.mp4")
    temporary_path.unlink(missing_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(intermediate_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        temporary_path.replace(destination_path)
    except subprocess.CalledProcessError as exc:
        temporary_path.unlink(missing_ok=True)
        message = exc.stderr.strip() or "Échec inconnu de ffmpeg"
        raise RuntimeError(
            f"Encodage de l'overlay impossible : {message}"
        ) from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_candidates_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def analyze_candidates(
    video_path: Path,
    csv_path: Path,
    metrics_path: Path,
    overlay_path: Path,
    config: CandidateConfig | None = None,
    scorer: BallCandidateScorer | None = None,
) -> dict[str, Any]:
    """Produit le réservoir brut de candidats et son overlay diagnostic."""

    resolved = config or CandidateConfig()
    resolved.validate()
    resolved_scorer = scorer or HeuristicV1BallCandidateScorer()

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir le segment : {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Dimensions vidéo invalides.")

    intermediate_path = overlay_path.with_name(
        overlay_path.stem + ".intermediate.mp4"
    )
    intermediate_path.unlink(missing_ok=True)
    overlay_path.unlink(missing_ok=True)

    writer = cv2.VideoWriter(
        str(intermediate_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Impossible de créer la vidéo overlay intermédiaire.")

    rows: list[dict[str, Any]] = []
    counts: list[int] = []
    scores: list[float] = []
    processed_frames = 0

    try:
        ok_previous, previous_frame = capture.read()
        ok_current, current_frame = capture.read()

        if not ok_previous:
            raise RuntimeError("Le segment ne contient aucune image.")

        writer.write(
            _draw_overlay(
                previous_frame,
                0,
                [],
            )
        )
        processed_frames = 1

        if not ok_current:
            writer.release()
            _transcode_overlay(intermediate_path, overlay_path)
            summary = summarize_candidate_counts([], [])
            metrics = {
                "schema_version": 1,
                "algorithm": {
                    "name": "triple_frame_motion_components",
                    "version": 1,
                },
                "parameters": asdict(resolved),
                "video": {
                    "fps": round(fps, 6),
                    "width": width,
                    "height": height,
                    "processed_frames": processed_frames,
                },
                "summary": summary,
                "artifacts": {
                    "candidates": csv_path.name,
                    "overlay": overlay_path.name,
                },
            }
            _write_candidates_csv(csv_path, rows)
            _atomic_write_json(metrics_path, metrics)
            return metrics

        previous_gray = _prepare_gray(
            previous_frame,
            resolved.blur_kernel,
        )
        current_gray = _prepare_gray(
            current_frame,
            resolved.blur_kernel,
        )
        frame_index = 1

        while True:
            ok_next, next_frame = capture.read()

            if not ok_next:
                writer.write(
                    _draw_overlay(
                        current_frame,
                        frame_index,
                        [],
                    )
                )
                processed_frames += 1
                break

            next_gray = _prepare_gray(
                next_frame,
                resolved.blur_kernel,
            )
            candidates = detect_frame_candidates(
                previous_gray,
                current_gray,
                next_gray,
                resolved,
                resolved_scorer,
            )

            counts.append(len(candidates))

            for rank, candidate in enumerate(candidates, start=1):
                score = float(candidate["score"])
                scores.append(score)
                rows.append(
                    {
                        "candidate_id": (
                            f"f{frame_index:06d}_c{rank:02d}"
                        ),
                        "frame": frame_index,
                        "time_s": round(frame_index / fps, 6),
                        "rank": rank,
                        **candidate,
                    }
                )

            writer.write(
                _draw_overlay(
                    current_frame,
                    frame_index,
                    candidates,
                )
            )
            processed_frames += 1

            previous_gray = current_gray
            current_gray = next_gray
            previous_frame = current_frame
            current_frame = next_frame
            frame_index += 1

        writer.release()
        capture.release()

        _write_candidates_csv(csv_path, rows)
        _transcode_overlay(intermediate_path, overlay_path)

        summary = summarize_candidate_counts(counts, scores)
        metrics = {
            "schema_version": 1,
            "algorithm": {
                "name": "triple_frame_motion_components",
                "version": 1,
            },
            "parameters": asdict(resolved),
            "video": {
                "fps": round(fps, 6),
                "width": width,
                "height": height,
                "processed_frames": processed_frames,
            },
            "summary": summary,
            "artifacts": {
                "candidates": csv_path.name,
                "overlay": overlay_path.name,
            },
        }
        _atomic_write_json(metrics_path, metrics)
        return metrics
    except Exception:
        writer.release()
        capture.release()
        intermediate_path.unlink(missing_ok=True)
        overlay_path.unlink(missing_ok=True)
        csv_path.with_suffix(csv_path.suffix + ".tmp").unlink(
            missing_ok=True
        )
        raise
    finally:
        intermediate_path.unlink(missing_ok=True)
