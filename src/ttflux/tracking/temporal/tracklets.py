from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackConfig:
    max_gap_frames: int = 2
    max_jump_per_frame: float = 58.0
    prediction_error_limit_px: float = 26.0
    max_acceleration_px_per_frame2: float = 24.0
    max_turn_degrees: float = 115.0
    max_track_span_frames: int = 45
    beam_width_per_seed: int = 4
    branch_factor: int = 2
    seed_candidates_per_frame: int = 6
    seed_stride_frames: int = 3
    min_points: int = 5
    max_output_tracks: int = 24
    trail_frames: int = 12
    duplicate_overlap_ratio: float = 0.65
    cut_mad_threshold: float = 30.0

    def validate(self) -> None:
        if self.max_gap_frames < 0 or self.max_gap_frames > 6:
            raise ValueError("max_gap_frames doit être compris entre 0 et 6.")
        if self.max_jump_per_frame <= 0:
            raise ValueError("max_jump_per_frame doit être positif.")
        if self.prediction_error_limit_px <= 0:
            raise ValueError("prediction_error_limit_px doit être positif.")
        if self.max_acceleration_px_per_frame2 <= 0:
            raise ValueError("max_acceleration_px_per_frame2 doit être positif.")
        if not 0 < self.max_turn_degrees <= 180:
            raise ValueError("max_turn_degrees doit être dans ]0, 180].")
        if self.max_track_span_frames < 5:
            raise ValueError("max_track_span_frames doit être au moins égal à 5.")
        if self.beam_width_per_seed < 1:
            raise ValueError("beam_width_per_seed doit être positif.")
        if self.branch_factor < 1:
            raise ValueError("branch_factor doit être positif.")
        if self.seed_candidates_per_frame < 1:
            raise ValueError("seed_candidates_per_frame doit être positif.")
        if self.seed_stride_frames < 1:
            raise ValueError("seed_stride_frames doit être positif.")
        if self.min_points < 2:
            raise ValueError("min_points doit être au moins égal à 2.")
        if self.max_output_tracks < 1:
            raise ValueError("max_output_tracks doit être positif.")
        if self.trail_frames < 1:
            raise ValueError("trail_frames doit être positif.")
        if not 0 < self.duplicate_overlap_ratio <= 1:
            raise ValueError(
                "duplicate_overlap_ratio doit être dans ]0, 1]."
            )
        if self.cut_mad_threshold <= 0:
            raise ValueError("cut_mad_threshold doit être positif.")


@dataclass(frozen=True)
class CandidatePoint:
    candidate_id: str
    frame: int
    time_s: float
    x: float
    y: float
    score: float
    area: float = 0.0
    mean_brightness: float = 0.0
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    rank: int = 0


@dataclass
class Hypothesis:
    points: list[CandidatePoint]
    link_score: float = 0.0
    prediction_errors: list[float] = field(default_factory=list)
    accelerations: list[float] = field(default_factory=list)
    gaps: int = 0

    @property
    def first_frame(self) -> int:
        return self.points[0].frame

    @property
    def last_frame(self) -> int:
        return self.points[-1].frame

    @property
    def span_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    def copy_with(
        self,
        point: CandidatePoint,
        score_delta: float,
        prediction_error: float,
        acceleration: float,
    ) -> "Hypothesis":
        gap = max(0, point.frame - self.last_frame - 1)
        return Hypothesis(
            points=[*self.points, point],
            link_score=self.link_score + score_delta,
            prediction_errors=[*self.prediction_errors, prediction_error],
            accelerations=[*self.accelerations, acceleration],
            gaps=self.gaps + gap,
        )


TRACK_CSV_FIELDS = (
    "track_id",
    "track_rank",
    "point_index",
    "candidate_id",
    "frame",
    "time_s",
    "x",
    "y",
    "candidate_score",
    "prediction_error_px",
)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in {None, ""}:
        return default
    return float(value)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value in {None, ""}:
        return default
    return int(float(value))


def read_candidates(path: Path) -> list[CandidatePoint]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "candidate_id",
            "frame",
            "time_s",
            "x",
            "y",
            "score",
        }

        if reader.fieldnames is None or not required.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError("Colonnes candidates.csv incomplètes.")

        points: list[CandidatePoint] = []
        for row in reader:
            points.append(
                CandidatePoint(
                    candidate_id=str(row["candidate_id"]),
                    frame=_int(row, "frame"),
                    time_s=_float(row, "time_s"),
                    x=_float(row, "x"),
                    y=_float(row, "y"),
                    score=_float(row, "score"),
                    area=_float(row, "area"),
                    mean_brightness=_float(row, "mean_brightness"),
                    bbox_w=_float(row, "bbox_w"),
                    bbox_h=_float(row, "bbox_h"),
                    rank=_int(row, "rank"),
                )
            )

    points.sort(key=lambda point: (point.frame, point.rank, -point.score))
    return points


def group_candidates(
    candidates: Iterable[CandidatePoint],
) -> dict[int, list[CandidatePoint]]:
    grouped: dict[int, list[CandidatePoint]] = defaultdict(list)
    for point in candidates:
        grouped[point.frame].append(point)
    for points in grouped.values():
        points.sort(key=lambda point: (point.rank, -point.score))
    return dict(grouped)


def _vector(
    first: CandidatePoint,
    second: CandidatePoint,
) -> tuple[float, float]:
    delta_frames = max(1, second.frame - first.frame)
    return (
        (second.x - first.x) / delta_frames,
        (second.y - first.y) / delta_frames,
    )


def _norm(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _turn_degrees(
    previous_velocity: tuple[float, float],
    next_velocity: tuple[float, float],
) -> float:
    previous_norm = _norm(previous_velocity)
    next_norm = _norm(next_velocity)
    if previous_norm < 1e-6 or next_norm < 1e-6:
        return 0.0
    cosine = (
        previous_velocity[0] * next_velocity[0]
        + previous_velocity[1] * next_velocity[1]
    ) / (previous_norm * next_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _appearance_penalty(
    previous: CandidatePoint,
    candidate: CandidatePoint,
) -> float:
    penalty = 0.0
    if previous.area > 0 and candidate.area > 0:
        penalty += min(2.0, abs(math.log(candidate.area / previous.area)))
    if previous.mean_brightness > 0 and candidate.mean_brightness > 0:
        penalty += abs(
            candidate.mean_brightness - previous.mean_brightness
        ) / 255.0
    previous_size = max(previous.bbox_w, previous.bbox_h)
    candidate_size = max(candidate.bbox_w, candidate.bbox_h)
    if previous_size > 0 and candidate_size > 0:
        penalty += 0.5 * min(
            2.0,
            abs(math.log(candidate_size / previous_size)),
        )
    return penalty


def evaluate_link(
    hypothesis: Hypothesis,
    candidate: CandidatePoint,
    config: TrackConfig,
) -> tuple[float, float, float] | None:
    last = hypothesis.points[-1]
    frame_delta = candidate.frame - last.frame
    if frame_delta < 1 or frame_delta > config.max_gap_frames + 1:
        return None

    jump = math.hypot(candidate.x - last.x, candidate.y - last.y)
    if jump / frame_delta > config.max_jump_per_frame:
        return None

    if len(hypothesis.points) >= 2:
        previous_velocity = _vector(
            hypothesis.points[-2],
            hypothesis.points[-1],
        )
        predicted_x = last.x + previous_velocity[0] * frame_delta
        predicted_y = last.y + previous_velocity[1] * frame_delta
        prediction_error = math.hypot(
            candidate.x - predicted_x,
            candidate.y - predicted_y,
        )
        prediction_limit = (
            config.prediction_error_limit_px + 4.0 * (frame_delta - 1)
        )
        if prediction_error > prediction_limit:
            return None
    else:
        previous_velocity = (0.0, 0.0)
        prediction_error = 0.0

    next_velocity = (
        (candidate.x - last.x) / frame_delta,
        (candidate.y - last.y) / frame_delta,
    )
    acceleration = _norm(
        (
            next_velocity[0] - previous_velocity[0],
            next_velocity[1] - previous_velocity[1],
        )
    )

    if len(hypothesis.points) >= 2:
        if acceleration > config.max_acceleration_px_per_frame2:
            return None
        if (
            _turn_degrees(previous_velocity, next_velocity)
            > config.max_turn_degrees
        ):
            return None
    else:
        acceleration = 0.0

    appearance = _appearance_penalty(last, candidate)
    score_delta = (
        1.65 * candidate.score
        - 0.035 * prediction_error
        - 0.022 * acceleration
        - 0.010 * jump
        - 0.24 * appearance
        - 0.10 * (frame_delta - 1)
    )
    return score_delta, prediction_error, acceleration


def detect_scene_cuts(
    video_path: Path,
    threshold: float,
) -> set[int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir le segment : {video_path}")

    cuts: set[int] = set()
    previous_small: np.ndarray | None = None
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
            if previous_small is not None:
                mad = float(
                    np.mean(
                        cv2.absdiff(small, previous_small).astype(np.float32)
                    )
                )
                if mad >= threshold:
                    cuts.add(frame_index)
            previous_small = small
            frame_index += 1
    finally:
        capture.release()

    return cuts


def _crosses_cut(start_frame: int, end_frame: int, cuts: set[int]) -> bool:
    return any(start_frame < cut <= end_frame for cut in cuts)


def _hypothesis_sort_key(hypothesis: Hypothesis) -> tuple[float, int]:
    density = len(hypothesis.points) / max(1, hypothesis.span_frames)
    return (
        hypothesis.link_score + 0.16 * len(hypothesis.points) + 0.25 * density,
        len(hypothesis.points),
    )


def build_tracklets(
    candidates: list[CandidatePoint],
    config: TrackConfig | None = None,
    scene_cuts: set[int] | None = None,
) -> tuple[list[Hypothesis], dict[str, int]]:
    resolved = config or TrackConfig()
    resolved.validate()
    cuts = scene_cuts or set()
    grouped = group_candidates(candidates)
    frames = sorted(grouped)
    if not frames:
        return [], {
            "generated_hypotheses": 0,
            "completed_hypotheses": 0,
            "eligible_hypotheses": 0,
        }

    generated = 0
    completed: list[Hypothesis] = []

    for seed_frame in frames:
        if seed_frame % resolved.seed_stride_frames != 0:
            continue
        seeds = grouped[seed_frame][: resolved.seed_candidates_per_frame]
        for seed in seeds:
            beam = [
                Hypothesis(
                    points=[seed],
                    link_score=1.2 * seed.score,
                )
            ]
            generated += 1

            while beam:
                next_beam: list[Hypothesis] = []
                for hypothesis in beam:
                    if hypothesis.span_frames >= resolved.max_track_span_frames:
                        completed.append(hypothesis)
                        continue

                    extensions: list[
                        tuple[float, CandidatePoint, float, float, float]
                    ] = []
                    last_frame = hypothesis.last_frame
                    for frame_delta in range(
                        1,
                        resolved.max_gap_frames + 2,
                    ):
                        target_frame = last_frame + frame_delta
                        if (
                            target_frame - hypothesis.first_frame + 1
                            > resolved.max_track_span_frames
                        ):
                            break
                        if _crosses_cut(last_frame, target_frame, cuts):
                            break
                        for candidate in grouped.get(target_frame, []):
                            evaluated = evaluate_link(
                                hypothesis,
                                candidate,
                                resolved,
                            )
                            if evaluated is None:
                                continue
                            delta, error, acceleration = evaluated
                            extensions.append(
                                (
                                    delta,
                                    candidate,
                                    error,
                                    acceleration,
                                    candidate.score,
                                )
                            )

                    if not extensions:
                        completed.append(hypothesis)
                        continue

                    extensions.sort(
                        key=lambda item: (item[0], item[4]),
                        reverse=True,
                    )
                    for delta, point, error, acceleration, _ in extensions[
                        : resolved.branch_factor
                    ]:
                        next_beam.append(
                            hypothesis.copy_with(
                                point,
                                delta,
                                error,
                                acceleration,
                            )
                        )
                        generated += 1

                if not next_beam:
                    break
                next_beam.sort(key=_hypothesis_sort_key, reverse=True)
                beam = next_beam[: resolved.beam_width_per_seed]

    eligible = [
        hypothesis
        for hypothesis in completed
        if len(hypothesis.points) >= resolved.min_points
    ]
    eligible.sort(key=_final_track_score, reverse=True)

    selected: list[Hypothesis] = []
    for hypothesis in eligible:
        if any(
            _duplicate_ratio(hypothesis, previous)
            >= resolved.duplicate_overlap_ratio
            for previous in selected
        ):
            continue
        selected.append(hypothesis)
        if len(selected) >= resolved.max_output_tracks:
            break

    return selected, {
        "generated_hypotheses": generated,
        "completed_hypotheses": len(completed),
        "eligible_hypotheses": len(eligible),
    }


def _duplicate_ratio(first: Hypothesis, second: Hypothesis) -> float:
    first_ids = {point.candidate_id for point in first.points}
    second_ids = {point.candidate_id for point in second.points}
    denominator = min(len(first_ids), len(second_ids))
    if denominator == 0:
        return 0.0
    return len(first_ids & second_ids) / denominator


def _final_track_score(hypothesis: Hypothesis) -> float:
    points = hypothesis.points
    coverage = len(points) / max(1, hypothesis.span_frames)
    candidate_mean = mean(point.score for point in points)
    error_mean = mean(hypothesis.prediction_errors) if hypothesis.prediction_errors else 0.0
    acceleration_mean = mean(hypothesis.accelerations) if hypothesis.accelerations else 0.0
    distances = [
        math.hypot(second.x - first.x, second.y - first.y)
        for first, second in zip(points, points[1:])
    ]
    speed_mean = mean(distances) if distances else 0.0
    useful_motion = min(1.0, speed_mean / 18.0)
    return (
        1.8 * candidate_mean
        + 0.55 * coverage
        + 0.32 * math.log1p(len(points))
        + 0.25 * useful_motion
        - 0.030 * error_mean
        - 0.018 * acceleration_mean
        - 0.025 * hypothesis.gaps
    )


def summarize_track(
    hypothesis: Hypothesis,
    track_id: str,
    rank: int,
) -> dict[str, Any]:
    points = hypothesis.points
    distances = [
        math.hypot(second.x - first.x, second.y - first.y)
        for first, second in zip(points, points[1:])
    ]
    total_distance = sum(distances)
    mean_speed = mean(distances) if distances else 0.0
    return {
        "track_id": track_id,
        "rank": rank,
        "score": round(_final_track_score(hypothesis), 6),
        "point_count": len(points),
        "first_frame": hypothesis.first_frame,
        "last_frame": hypothesis.last_frame,
        "span_frames": hypothesis.span_frames,
        "coverage_ratio": round(
            len(points) / max(1, hypothesis.span_frames),
            6,
        ),
        "gap_frames": hypothesis.gaps,
        "total_distance_px": round(total_distance, 3),
        "mean_speed_px_per_frame": round(mean_speed, 3),
        "mean_candidate_score": round(
            mean(point.score for point in points),
            6,
        ),
        "mean_prediction_error_px": round(
            mean(hypothesis.prediction_errors),
            3,
        )
        if hypothesis.prediction_errors
        else 0.0,
        "mean_acceleration_px_per_frame2": round(
            mean(hypothesis.accelerations),
            3,
        )
        if hypothesis.accelerations
        else 0.0,
    }


def _write_tracks_csv(
    path: Path,
    tracks: list[Hypothesis],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_CSV_FIELDS)
        writer.writeheader()
        for rank, hypothesis in enumerate(tracks, start=1):
            track_id = f"T{rank:03d}"
            errors = [0.0, *hypothesis.prediction_errors]
            for point_index, (point, error) in enumerate(
                zip(hypothesis.points, errors),
                start=1,
            ):
                writer.writerow(
                    {
                        "track_id": track_id,
                        "track_rank": rank,
                        "point_index": point_index,
                        "candidate_id": point.candidate_id,
                        "frame": point.frame,
                        "time_s": point.time_s,
                        "x": point.x,
                        "y": point.y,
                        "candidate_score": point.score,
                        "prediction_error_px": round(error, 3),
                    }
                )
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _transcode_overlay(intermediate: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg est introuvable dans le PATH.")
    temporary = destination.with_suffix(".partial.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(intermediate),
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
        str(temporary),
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
        temporary.replace(destination)
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        message = exc.stderr.strip() or "Échec inconnu de ffmpeg"
        raise RuntimeError(
            f"Encodage de l'overlay pistes impossible : {message}"
        ) from exc


def _draw_tracks_overlay(
    video_path: Path,
    overlay_path: Path,
    tracks: list[Hypothesis],
    config: TrackConfig,
) -> None:
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

    intermediate = overlay_path.with_name(
        overlay_path.stem + ".intermediate.mp4"
    )
    intermediate.unlink(missing_ok=True)
    overlay_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Impossible de créer l'overlay pistes.")

    colors = [
        (70, 230, 90),
        (0, 180, 255),
        (255, 170, 40),
        (220, 80, 220),
        (255, 255, 70),
        (80, 220, 220),
    ]
    indexed: list[tuple[str, tuple[int, int, int], Hypothesis]] = []
    for rank, hypothesis in enumerate(tracks, start=1):
        indexed.append(
            (
                f"T{rank:03d}",
                colors[(rank - 1) % len(colors)],
                hypothesis,
            )
        )

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output = frame.copy()
            visible = 0
            for track_id, color, hypothesis in indexed:
                active_points = [
                    point
                    for point in hypothesis.points
                    if frame_index - config.trail_frames < point.frame <= frame_index
                ]
                current = next(
                    (
                        point
                        for point in hypothesis.points
                        if point.frame == frame_index
                    ),
                    None,
                )
                if current is None and not active_points:
                    continue
                visible += 1
                if len(active_points) >= 2:
                    polyline = np.array(
                        [
                            [int(round(point.x)), int(round(point.y))]
                            for point in active_points
                        ],
                        dtype=np.int32,
                    ).reshape((-1, 1, 2))
                    cv2.polylines(
                        output,
                        [polyline],
                        False,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                if current is not None:
                    center = (
                        int(round(current.x)),
                        int(round(current.y)),
                    )
                    cv2.circle(output, center, 7, color, 2, cv2.LINE_AA)
                    cv2.putText(
                        output,
                        track_id,
                        (center[0] + 9, center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            cv2.rectangle(output, (0, 0), (510, 36), (0, 0, 0), -1)
            cv2.putText(
                output,
                f"frame {frame_index} | tracklets visibles {visible}/{len(tracks)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(output)
            frame_index += 1
    finally:
        writer.release()
        capture.release()

    try:
        _transcode_overlay(intermediate, overlay_path)
    finally:
        intermediate.unlink(missing_ok=True)


def _resolve_analyze_paths(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Path, Path, Path, Path, Path, TrackConfig | None]:
    config = kwargs.get("config")
    if config is None:
        config = next(
            (value for value in args if isinstance(value, TrackConfig)),
            None,
        )

    named = {
        key: Path(value)
        for key, value in kwargs.items()
        if key != "config" and isinstance(value, (str, Path))
    }
    paths = [
        Path(value)
        for value in args
        if isinstance(value, (str, Path))
    ]

    def named_or(predicate, *names: str) -> Path:
        for name in names:
            if name in named:
                return named[name]
        for path in paths:
            if predicate(path):
                return path
        raise TypeError(f"Chemin introuvable pour {names[0]}.")

    candidates_path = named_or(
        lambda path: path.suffix.lower() == ".csv"
        and "candidate" in path.stem.lower(),
        "candidates_path",
        "candidate_path",
    )
    video_path = named_or(
        lambda path: path.suffix.lower() == ".mp4"
        and "track" not in path.stem.lower()
        and "overlay" not in path.stem.lower(),
        "video_path",
        "clip_path",
    )
    tracks_path = named_or(
        lambda path: path.suffix.lower() == ".csv"
        and "track" in path.stem.lower(),
        "tracks_path",
        "csv_path",
    )
    metrics_path = named_or(
        lambda path: path.suffix.lower() == ".json"
        and "track" in path.stem.lower(),
        "metrics_path",
        "track_metrics_path",
    )
    overlay_path = named_or(
        lambda path: path.suffix.lower() == ".mp4"
        and "track" in path.stem.lower(),
        "overlay_path",
        "track_overlay_path",
    )
    return (
        candidates_path,
        video_path,
        tracks_path,
        metrics_path,
        overlay_path,
        config,
    )


def analyze_tracks(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Produit des tracklets locaux sans forcer une piste globale."""
    (
        candidates_path,
        video_path,
        tracks_path,
        metrics_path,
        overlay_path,
        config,
    ) = _resolve_analyze_paths(args, kwargs)
    resolved = config or TrackConfig()
    resolved.validate()

    candidates = read_candidates(candidates_path)
    grouped = group_candidates(candidates)
    scene_cuts = detect_scene_cuts(
        video_path,
        resolved.cut_mad_threshold,
    )
    tracks, counters = build_tracklets(
        candidates,
        resolved,
        scene_cuts,
    )

    track_rows = [
        summarize_track(hypothesis, f"T{rank:03d}", rank)
        for rank, hypothesis in enumerate(tracks, start=1)
    ]
    point_counts = [row["point_count"] for row in track_rows]
    coverage_values = [row["coverage_ratio"] for row in track_rows]
    prediction_errors = [
        row["mean_prediction_error_px"] for row in track_rows
    ]

    summary = {
        "selected_tracks": len(track_rows),
        "total_track_points": sum(point_counts),
        "longest_track_points": max(point_counts) if point_counts else 0,
        "longest_track_span_frames": max(
            (row["span_frames"] for row in track_rows),
            default=0,
        ),
        "best_track_score": track_rows[0]["score"] if track_rows else 0.0,
        "mean_selected_points": round(mean(point_counts), 3)
        if point_counts
        else 0.0,
        "mean_selected_coverage": round(mean(coverage_values), 6)
        if coverage_values
        else 0.0,
        "median_prediction_error_px": round(
            median(prediction_errors),
            3,
        )
        if prediction_errors
        else 0.0,
        "scene_cut_count": len(scene_cuts),
        "max_track_span_frames": resolved.max_track_span_frames,
    }

    metrics = {
        "schema_version": 2,
        "algorithm": {
            "name": "local_motion_tracklets_probe",
            "version": 2,
        },
        "parameters": asdict(resolved),
        "input": {
            "input_candidates": len(candidates),
            "input_frames": len(grouped),
            "scene_cuts": sorted(scene_cuts),
            **counters,
        },
        "summary": summary,
        "tracks": track_rows,
        "artifacts": {
            "tracks": tracks_path.name,
            "overlay": overlay_path.name,
        },
    }

    try:
        _write_tracks_csv(tracks_path, tracks)
        _atomic_write_json(metrics_path, metrics)
        _draw_tracks_overlay(
            video_path,
            overlay_path,
            tracks,
            resolved,
        )
    except Exception:
        tracks_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
        overlay_path.unlink(missing_ok=True)
        raise

    return metrics
