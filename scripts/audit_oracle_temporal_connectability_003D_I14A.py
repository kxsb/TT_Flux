from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ttflux.analysis.candidates import (
    CandidateConfig,
    detect_frame_candidates,
)
from ttflux.analysis.tracks import (
    CandidatePoint,
    Hypothesis,
    TrackConfig,
    detect_scene_cuts,
    evaluate_link,
)


CAPS = (6, 24, 4096)
RADIUS_PX = 20.0
MODES = (
    "current_seed_pruned",
    "unrestricted_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "runs/_ball_gt_seed_003D_I12A/"
            "i12a_frozen_manifest.csv"
        ),
    )

    parser.add_argument(
        "--gt",
        type=Path,
        default=Path(
            "runs/_ball_gt_benchmark_003D_I12D/"
            "i12d_frozen_gt.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/_ball_oracle_connectability_003D_I14A"
        ),
    )

    return parser.parse_args()


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            f"Aucune ligne à écrire : {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_float(
    value: Any,
) -> float | None:
    text = str(
        value or ""
    ).strip().replace(",", ".")

    if not text:
        return None

    try:
        result = float(text)
    except ValueError:
        return None

    if not math.isfinite(result):
        return None

    return result


def parse_int(
    value: Any,
) -> int:
    result = parse_float(value)

    if result is None:
        raise ValueError(
            f"Entier invalide : {value!r}"
        )

    return int(round(result))


def ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return round(
        numerator / denominator,
        6,
    )


def prepare_gray(
    frame: np.ndarray,
    blur_kernel: int,
) -> np.ndarray:
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    if blur_kernel > 1:
        gray = cv2.GaussianBlur(
            gray,
            (
                blur_kernel,
                blur_kernel,
            ),
            0,
        )

    return gray


def decode_context(
    video_path: Path,
    start_frame: int,
    end_frame_exclusive: int,
    blur_kernel: int,
) -> tuple[
    dict[int, np.ndarray],
    float,
]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Vidéo inaccessible : {video_path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    first_needed = max(
        0,
        start_frame - 1,
    )

    last_needed = (
        end_frame_exclusive
    )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        float(first_needed),
    )

    decoded: dict[
        int,
        np.ndarray,
    ] = {}

    frame_index = first_needed

    try:
        while (
            frame_index
            <= last_needed
        ):
            ok, frame = capture.read()

            if not ok:
                break

            decoded[
                frame_index
            ] = prepare_gray(
                frame,
                blur_kernel,
            )

            frame_index += 1
    finally:
        capture.release()

    missing = [
        frame
        for frame in range(
            start_frame,
            end_frame_exclusive,
        )
        if frame not in decoded
    ]

    if missing:
        raise RuntimeError(
            f"{video_path.name} : "
            f"frames manquantes "
            f"{missing[:10]}"
        )

    return decoded, fps


def vector(
    first: CandidatePoint,
    second: CandidatePoint,
) -> tuple[float, float]:
    frame_delta = max(
        1,
        second.frame - first.frame,
    )

    return (
        (second.x - first.x)
        / frame_delta,
        (second.y - first.y)
        / frame_delta,
    )


def norm(
    value: tuple[float, float],
) -> float:
    return math.hypot(
        value[0],
        value[1],
    )


def turn_degrees(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    first_norm = norm(first)
    second_norm = norm(second)

    if (
        first_norm < 1e-6
        or second_norm < 1e-6
    ):
        return 0.0

    cosine = (
        first[0] * second[0]
        + first[1] * second[1]
    ) / (
        first_norm
        * second_norm
    )

    cosine = max(
        -1.0,
        min(1.0, cosine),
    )

    return math.degrees(
        math.acos(cosine)
    )


def diagnose_link(
    hypothesis: Hypothesis,
    candidate: CandidatePoint,
    config: TrackConfig,
) -> str | None:
    last = hypothesis.points[-1]

    frame_delta = (
        candidate.frame
        - last.frame
    )

    if (
        frame_delta < 1
        or frame_delta
        > config.max_gap_frames + 1
    ):
        return "frame_delta"

    jump = math.hypot(
        candidate.x - last.x,
        candidate.y - last.y,
    )

    if (
        jump / frame_delta
        > config.max_jump_per_frame
    ):
        return "jump"

    if len(hypothesis.points) < 2:
        return None

    previous_velocity = vector(
        hypothesis.points[-2],
        hypothesis.points[-1],
    )

    predicted_x = (
        last.x
        + previous_velocity[0]
        * frame_delta
    )

    predicted_y = (
        last.y
        + previous_velocity[1]
        * frame_delta
    )

    prediction_error = math.hypot(
        candidate.x - predicted_x,
        candidate.y - predicted_y,
    )

    prediction_limit = (
        config.prediction_error_limit_px
        + 4.0
        * (frame_delta - 1)
    )

    if prediction_error > prediction_limit:
        return "prediction_error"

    next_velocity = (
        (
            candidate.x
            - last.x
        )
        / frame_delta,
        (
            candidate.y
            - last.y
        )
        / frame_delta,
    )

    acceleration = norm(
        (
            next_velocity[0]
            - previous_velocity[0],
            next_velocity[1]
            - previous_velocity[1],
        )
    )

    if (
        acceleration
        > config.max_acceleration_px_per_frame2
    ):
        return "acceleration"

    if (
        turn_degrees(
            previous_velocity,
            next_velocity,
        )
        > config.max_turn_degrees
    ):
        return "turn"

    return None


def crosses_cut(
    start_frame: int,
    end_frame: int,
    cuts: set[int],
) -> bool:
    return any(
        start_frame < cut <= end_frame
        for cut in cuts
    )


def deduplicate_hypotheses(
    hypotheses: list[Hypothesis],
) -> list[Hypothesis]:
    best: dict[
        tuple[str, str],
        Hypothesis,
    ] = {}

    for hypothesis in hypotheses:
        if len(hypothesis.points) >= 2:
            key = (
                hypothesis.points[-2].candidate_id,
                hypothesis.points[-1].candidate_id,
            )
        else:
            key = (
                "",
                hypothesis.points[-1].candidate_id,
            )

        previous = best.get(key)

        if previous is None:
            best[key] = hypothesis
            continue

        previous_key = (
            len(previous.points),
            previous.link_score,
        )

        candidate_key = (
            len(hypothesis.points),
            hypothesis.link_score,
        )

        if candidate_key > previous_key:
            best[key] = hypothesis

    return list(best.values())


def build_oracle_paths(
    hits_by_frame: dict[
        int,
        list[CandidatePoint],
    ],
    config: TrackConfig,
    cuts: set[int],
    mode: str,
) -> tuple[
    list[Hypothesis],
    dict[str, Any],
]:
    if mode not in MODES:
        raise ValueError(
            f"Mode inconnu : {mode}"
        )

    seeds: list[
        CandidatePoint
    ] = []

    for frame in sorted(
        hits_by_frame
    ):
        for candidate in hits_by_frame[
            frame
        ]:
            if (
                mode
                == "current_seed_pruned"
            ):
                if (
                    frame
                    % config.seed_stride_frames
                    != 0
                ):
                    continue

                if (
                    candidate.rank
                    > config.seed_candidates_per_frame
                ):
                    continue

            seeds.append(candidate)

    completed: list[
        Hypothesis
    ] = []

    generated = 0

    rejection_counts: Counter[
        str
    ] = Counter()

    for seed in seeds:
        beam = [
            Hypothesis(
                points=[seed],
                link_score=(
                    1.2 * seed.score
                ),
            )
        ]

        generated += 1

        while beam:
            next_beam: list[
                Hypothesis
            ] = []

            for hypothesis in beam:
                if (
                    hypothesis.span_frames
                    >= config.max_track_span_frames
                ):
                    rejection_counts[
                        "span_limit"
                    ] += 1

                    completed.append(
                        hypothesis
                    )

                    continue

                extensions: list[
                    tuple[
                        float,
                        CandidatePoint,
                        float,
                        float,
                        float,
                    ]
                ] = []

                target_candidate_seen = False

                last_frame = (
                    hypothesis.last_frame
                )

                for frame_delta in range(
                    1,
                    config.max_gap_frames + 2,
                ):
                    target_frame = (
                        last_frame
                        + frame_delta
                    )

                    if (
                        target_frame
                        - hypothesis.first_frame
                        + 1
                        > config.max_track_span_frames
                    ):
                        break

                    if crosses_cut(
                        last_frame,
                        target_frame,
                        cuts,
                    ):
                        rejection_counts[
                            "scene_cut"
                        ] += 1

                        break

                    target_candidates = (
                        hits_by_frame.get(
                            target_frame,
                            [],
                        )
                    )

                    if target_candidates:
                        target_candidate_seen = True

                    for candidate in target_candidates:
                        reason = diagnose_link(
                            hypothesis,
                            candidate,
                            config,
                        )

                        if reason is not None:
                            rejection_counts[
                                reason
                            ] += 1

                            continue

                        evaluated = evaluate_link(
                            hypothesis,
                            candidate,
                            config,
                        )

                        if evaluated is None:
                            rejection_counts[
                                "unknown_rejection"
                            ] += 1

                            continue

                        (
                            score_delta,
                            prediction_error,
                            acceleration,
                        ) = evaluated

                        extensions.append(
                            (
                                score_delta,
                                candidate,
                                prediction_error,
                                acceleration,
                                candidate.score,
                            )
                        )

                if not extensions:
                    if not target_candidate_seen:
                        rejection_counts[
                            "no_target_hit"
                        ] += 1
                    else:
                        rejection_counts[
                            "all_links_rejected"
                        ] += 1

                    completed.append(
                        hypothesis
                    )

                    continue

                extensions.sort(
                    key=lambda item: (
                        item[0],
                        item[4],
                    ),
                    reverse=True,
                )

                if (
                    mode
                    == "current_seed_pruned"
                ):
                    extensions = extensions[
                        : config.branch_factor
                    ]

                for (
                    score_delta,
                    candidate,
                    prediction_error,
                    acceleration,
                    _,
                ) in extensions:
                    next_beam.append(
                        hypothesis.copy_with(
                            candidate,
                            score_delta,
                            prediction_error,
                            acceleration,
                        )
                    )

                    generated += 1

            if not next_beam:
                break

            if (
                mode
                == "current_seed_pruned"
            ):
                next_beam.sort(
                    key=lambda hypothesis: (
                        hypothesis.link_score
                        + 0.16
                        * len(
                            hypothesis.points
                        ),
                        len(
                            hypothesis.points
                        ),
                    ),
                    reverse=True,
                )

                beam = next_beam[
                    : config.beam_width_per_seed
                ]
            else:
                beam = deduplicate_hypotheses(
                    next_beam
                )

    unique: dict[
        tuple[str, ...],
        Hypothesis,
    ] = {}

    for hypothesis in completed:
        if (
            len(hypothesis.points)
            < config.min_points
        ):
            continue

        key = tuple(
            point.candidate_id
            for point
            in hypothesis.points
        )

        unique[key] = hypothesis

    eligible = list(
        unique.values()
    )

    return eligible, {
        "seed_count":
            len(seeds),
        "generated_hypotheses":
            generated,
        "completed_hypotheses":
            len(completed),
        "eligible_paths":
            len(eligible),
        "rejection_attempts":
            dict(rejection_counts),
    }


def summarize_paths(
    paths: list[Hypothesis],
    hit_frames: set[int],
    visible_frames: set[int],
) -> dict[str, Any]:
    covered_frames = {
        point.frame
        for path in paths
        for point in path.points
    }

    eligible_hit_frames = (
        covered_frames
        & hit_frames
    )

    eligible_visible_frames = (
        covered_frames
        & visible_frames
    )

    longest_points = max(
        (
            len(path.points)
            for path in paths
        ),
        default=0,
    )

    longest_span = max(
        (
            path.span_frames
            for path in paths
        ),
        default=0,
    )

    return {
        "eligible_paths":
            len(paths),
        "covered_frames":
            len(covered_frames),
        "covered_hit_frames":
            len(eligible_hit_frames),
        "covered_hit_ratio":
            ratio(
                len(eligible_hit_frames),
                len(hit_frames),
            ),
        "covered_visible_frames":
            len(
                eligible_visible_frames
            ),
        "covered_visible_ratio":
            ratio(
                len(
                    eligible_visible_frames
                ),
                len(visible_frames),
            ),
        "longest_path_points":
            longest_points,
        "longest_path_span_frames":
            longest_span,
        "_covered_frames":
            covered_frames,
    }


def aggregate_mode_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    rejection_counts: Counter[
        str
    ] = Counter()

    for summary in summaries:
        rejection_counts.update(
            summary.get(
                "rejection_attempts",
                {},
            )
        )

    visible_frames = sum(
        summary["visible_frames"]
        for summary in summaries
    )

    hit_frames = sum(
        summary["hit_frames"]
        for summary in summaries
    )

    covered_visible = sum(
        summary[
            "covered_visible_frames"
        ]
        for summary in summaries
    )

    covered_hits = sum(
        summary[
            "covered_hit_frames"
        ]
        for summary in summaries
    )

    return {
        "visible_frames":
            visible_frames,
        "hit_frames":
            hit_frames,
        "oracle_recall":
            ratio(
                hit_frames,
                visible_frames,
            ),
        "seed_count":
            sum(
                summary["seed_count"]
                for summary in summaries
            ),
        "eligible_paths":
            sum(
                summary["eligible_paths"]
                for summary in summaries
            ),
        "covered_hit_frames":
            covered_hits,
        "covered_hit_ratio":
            ratio(
                covered_hits,
                hit_frames,
            ),
        "covered_visible_frames":
            covered_visible,
        "covered_visible_ratio":
            ratio(
                covered_visible,
                visible_frames,
            ),
        "longest_path_points":
            max(
                (
                    summary[
                        "longest_path_points"
                    ]
                    for summary
                    in summaries
                ),
                default=0,
            ),
        "longest_path_span_frames":
            max(
                (
                    summary[
                        "longest_path_span_frames"
                    ]
                    for summary
                    in summaries
                ),
                default=0,
            ),
        "generated_hypotheses":
            sum(
                summary[
                    "generated_hypotheses"
                ]
                for summary in summaries
            ),
        "completed_hypotheses":
            sum(
                summary[
                    "completed_hypotheses"
                ]
                for summary in summaries
            ),
        "rejection_attempts":
            dict(
                rejection_counts
            ),
    }


def main() -> None:
    args = parse_args()

    manifest_path = (
        args.manifest.resolve()
    )

    gt_path = (
        args.gt.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    manifest_rows = [
        row
        for row in read_csv(
            manifest_path
        )
        if str(
            row.get(
                "review_status"
            )
            or ""
        ).strip().lower() == "accepted"
    ]

    gt_rows = read_csv(
        gt_path
    )

    if len(manifest_rows) != 3:
        raise RuntimeError(
            "Trois clips acceptés attendus."
        )

    if len(gt_rows) != 450:
        raise RuntimeError(
            "450 lignes GT attendues."
        )

    gt_by_clip: dict[
        str,
        dict[int, dict[str, str]],
    ] = defaultdict(dict)

    for row in gt_rows:
        clip_id = str(
            row["clip_id"]
        )

        local_frame = parse_int(
            row["local_frame"]
        )

        gt_by_clip[
            clip_id
        ][local_frame] = row

    candidate_config = replace(
        CandidateConfig(),
        max_candidates_per_frame=4096,
    )

    track_config = TrackConfig()

    frame_rows: list[
        dict[str, Any]
    ] = []

    per_clip: dict[
        str,
        dict[str, Any],
    ] = {}

    overall_mode_inputs: dict[
        tuple[int, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for manifest in manifest_rows:
        clip_id = str(
            manifest["clip_id"]
        )

        source_key = str(
            manifest["source_key"]
        )

        video_path = Path(
            manifest["source_video"]
        ).resolve()

        start_frame = parse_int(
            manifest["start_frame"]
        )

        end_frame = parse_int(
            manifest[
                "end_frame_exclusive"
            ]
        )

        if (
            end_frame - start_frame
        ) != 150:
            raise RuntimeError(
                f"{clip_id} : "
                "150 frames attendues."
            )

        clip_gt = gt_by_clip[
            clip_id
        ]

        if len(clip_gt) != 150:
            raise RuntimeError(
                f"{clip_id} : "
                "150 lignes GT attendues."
            )

        decoded, fps = decode_context(
            video_path,
            start_frame,
            end_frame,
            candidate_config.blur_kernel,
        )

        source_cuts = detect_scene_cuts(
            video_path,
            track_config.cut_mad_threshold,
        )

        local_cuts = {
            cut - start_frame
            for cut in source_cuts
            if (
                start_frame
                < cut
                < end_frame
            )
        }

        candidates_by_frame: dict[
            int,
            list[dict[str, Any]],
        ] = {}

        visible_frames: set[
            int
        ] = set()

        clip_frame_rows: dict[
            int,
            dict[str, Any],
        ] = {}

        for local_frame in range(150):
            source_frame = (
                start_frame
                + local_frame
            )

            gt = clip_gt[
                local_frame
            ]

            gt_x = parse_float(
                gt.get("x")
            )

            gt_y = parse_float(
                gt.get("y")
            )

            visible = (
                gt_x is not None
                and gt_y is not None
            )

            if visible:
                visible_frames.add(
                    local_frame
                )

            current_gray = decoded[
                source_frame
            ]

            previous_gray = decoded.get(
                source_frame - 1,
                current_gray,
            )

            next_gray = decoded.get(
                source_frame + 1,
                current_gray,
            )

            detected = detect_frame_candidates(
                previous_gray,
                current_gray,
                next_gray,
                candidate_config,
            )

            candidates: list[
                dict[str, Any]
            ] = []

            for rank, candidate in enumerate(
                detected,
                start=1,
            ):
                if visible:
                    distance = math.hypot(
                        float(
                            candidate["x"]
                        )
                        - gt_x,
                        float(
                            candidate["y"]
                        )
                        - gt_y,
                    )
                else:
                    distance = None

                point = CandidatePoint(
                    candidate_id=(
                        f"{clip_id}_"
                        f"f{local_frame:03d}_"
                        f"c{rank:04d}"
                    ),
                    frame=local_frame,
                    time_s=(
                        local_frame
                        / fps
                    ),
                    x=float(
                        candidate["x"]
                    ),
                    y=float(
                        candidate["y"]
                    ),
                    score=float(
                        candidate["score"]
                    ),
                    area=float(
                        candidate["area"]
                    ),
                    mean_brightness=float(
                        candidate[
                            "mean_brightness"
                        ]
                    ),
                    bbox_w=float(
                        candidate["bbox_w"]
                    ),
                    bbox_h=float(
                        candidate["bbox_h"]
                    ),
                    rank=rank,
                )

                candidates.append({
                    "point":
                        point,
                    "rank":
                        rank,
                    "distance":
                        distance,
                })

            candidates_by_frame[
                local_frame
            ] = candidates

            hit_ranks = [
                candidate["rank"]
                for candidate
                in candidates
                if (
                    candidate["distance"]
                    is not None
                    and candidate[
                        "distance"
                    ] <= RADIUS_PX
                )
            ]

            distances = [
                float(
                    candidate["distance"]
                )
                for candidate
                in candidates
                if candidate[
                    "distance"
                ] is not None
            ]

            row: dict[
                str,
                Any,
            ] = {
                "clip_id":
                    clip_id,
                "source_key":
                    source_key,
                "local_frame":
                    local_frame,
                "source_frame":
                    source_frame,
                "status":
                    gt.get(
                        "status",
                        "",
                    ),
                "gt_visible":
                    int(visible),
                "gt_x":
                    gt_x
                    if visible
                    else "",
                "gt_y":
                    gt_y
                    if visible
                    else "",
                "candidate_count_full":
                    len(candidates),
                "nearest_distance_px":
                    (
                        round(
                            min(distances),
                            6,
                        )
                        if distances
                        else ""
                    ),
                "first_hit_rank_20px":
                    min(hit_ranks)
                    if hit_ranks
                    else "",
            }

            clip_frame_rows[
                local_frame
            ] = row

        clip_report: dict[
            str,
            Any,
        ] = {
            "visible_frames":
                len(visible_frames),
            "scene_cuts":
                sorted(local_cuts),
            "caps":
                {},
        }

        for cap in CAPS:
            hits_by_frame: dict[
                int,
                list[CandidatePoint],
            ] = {}

            for (
                local_frame,
                candidates,
            ) in candidates_by_frame.items():
                hits = [
                    candidate["point"]
                    for candidate
                    in candidates
                    if (
                        candidate["rank"]
                        <= cap
                        and candidate[
                            "distance"
                        ] is not None
                        and candidate[
                            "distance"
                        ] <= RADIUS_PX
                    )
                ]

                if hits:
                    hits_by_frame[
                        local_frame
                    ] = hits

                clip_frame_rows[
                    local_frame
                ][
                    f"hit_count_cap{cap}"
                ] = len(hits)

            hit_frames = set(
                hits_by_frame
            )

            cap_report: dict[
                str,
                Any,
            ] = {
                "hit_frames":
                    len(hit_frames),
                "oracle_recall":
                    ratio(
                        len(hit_frames),
                        len(
                            visible_frames
                        ),
                    ),
                "modes":
                    {},
            }

            for mode in MODES:
                (
                    paths,
                    counters,
                ) = build_oracle_paths(
                    hits_by_frame,
                    track_config,
                    local_cuts,
                    mode,
                )

                summary = summarize_paths(
                    paths,
                    hit_frames,
                    visible_frames,
                )

                covered_frames = (
                    summary.pop(
                        "_covered_frames"
                    )
                )

                combined = {
                    "visible_frames":
                        len(
                            visible_frames
                        ),
                    "hit_frames":
                        len(hit_frames),
                    **counters,
                    **summary,
                }

                cap_report[
                    "modes"
                ][mode] = combined

                overall_mode_inputs[
                    (cap, mode)
                ].append(combined)

                for local_frame in range(150):
                    clip_frame_rows[
                        local_frame
                    ][
                        (
                            f"covered_"
                            f"{mode}_"
                            f"cap{cap}"
                        )
                    ] = int(
                        local_frame
                        in covered_frames
                    )

            clip_report[
                "caps"
            ][str(cap)] = (
                cap_report
            )

        frame_rows.extend(
            clip_frame_rows[
                local_frame
            ]
            for local_frame
            in range(150)
        )

        per_clip[
            clip_id
        ] = clip_report

    visible_count = sum(
        int(
            row["gt_visible"]
        )
        for row in frame_rows
    )

    if (
        len(frame_rows) != 450
        or visible_count != 416
    ):
        raise RuntimeError(
            "Contrat GT brisé : "
            f"frames={len(frame_rows)}, "
            f"visibles={visible_count}"
        )

    overall: dict[
        str,
        Any,
    ] = {
        "frames":
            len(frame_rows),
        "visible_frames":
            visible_count,
        "caps":
            {},
    }

    for cap in CAPS:
        cap_modes: dict[
            str,
            Any,
        ] = {}

        for mode in MODES:
            cap_modes[
                mode
            ] = aggregate_mode_summaries(
                overall_mode_inputs[
                    (cap, mode)
                ]
            )

        overall[
            "caps"
        ][str(cap)] = {
            "modes":
                cap_modes,
        }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_csv = (
        output_dir
        / "i14a_frame_connectability.csv"
    )

    report_json = (
        output_dir
        / "i14a_connectability_report.json"
    )

    write_csv(
        frame_csv,
        frame_rows,
    )

    report = {
        "experiment":
            (
                "003D_I14A_"
                "oracle_temporal_connectability"
            ),
        "manifest":
            str(manifest_path),
        "ground_truth":
            str(gt_path),
        "radius_px":
            RADIUS_PX,
        "caps":
            list(CAPS),
        "modes": {
            "current_seed_pruned":
                (
                    "Seeding rang <= 6 "
                    "une frame sur trois, "
                    "branch_factor et beam_width "
                    "actuels."
                ),
            "unrestricted_oracle":
                (
                    "Tout candidat oracle peut "
                    "démarrer une piste, sans "
                    "pruning du faisceau."
                ),
        },
        "candidate_parameters":
            asdict(candidate_config),
        "track_parameters":
            asdict(track_config),
        "overall":
            overall,
        "per_clip":
            per_clip,
        "artifacts": {
            "frame_csv":
                frame_csv.name,
            "report_json":
                report_json.name,
        },
    }

    report_json.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "I14A_ORACLE_TEMPORAL_CONNECTABILITY_OK"
    )
    print(
        "frames =",
        len(frame_rows),
    )
    print(
        "visible_frames =",
        visible_count,
    )

    for cap in CAPS:
        print()
        print(
            f"=== CAP {cap} ==="
        )

        for mode in MODES:
            summary = overall[
                "caps"
            ][str(cap)][
                "modes"
            ][mode]

            print(
                mode,
                (
                    f"hits="
                    f"{summary['hit_frames']}/"
                    f"{summary['visible_frames']}"
                ),
                (
                    f"oracle="
                    f"{summary['oracle_recall']:.4f}"
                ),
                (
                    f"seeds="
                    f"{summary['seed_count']}"
                ),
                (
                    f"paths="
                    f"{summary['eligible_paths']}"
                ),
                (
                    f"covered_visible="
                    f"{summary['covered_visible_frames']}"
                ),
                (
                    f"covered_ratio="
                    f"{summary['covered_visible_ratio']:.4f}"
                ),
                (
                    f"covered_hits="
                    f"{summary['covered_hit_frames']}"
                    f"/{summary['hit_frames']}"
                ),
                (
                    f"longest_points="
                    f"{summary['longest_path_points']}"
                ),
                (
                    f"longest_span="
                    f"{summary['longest_path_span_frames']}"
                ),
            )

            print(
                "  rejections =",
                summary[
                    "rejection_attempts"
                ],
            )

    print()
    print(
        "=== PAR CLIP, CAP 24 ==="
    )

    for (
        clip_id,
        clip_report,
    ) in per_clip.items():
        cap_report = clip_report[
            "caps"
        ]["24"]

        current = cap_report[
            "modes"
        ][
            "current_seed_pruned"
        ]

        unrestricted = cap_report[
            "modes"
        ][
            "unrestricted_oracle"
        ]

        print(
            clip_id,
            (
                f"hits="
                f"{cap_report['hit_frames']}"
            ),
            (
                f"current_covered="
                f"{current['covered_visible_frames']}"
            ),
            (
                f"unrestricted_covered="
                f"{unrestricted['covered_visible_frames']}"
            ),
            (
                f"current_longest="
                f"{current['longest_path_points']}"
            ),
            (
                f"unrestricted_longest="
                f"{unrestricted['longest_path_points']}"
            ),
        )

    print()
    print(
        "frame_csv =",
        frame_csv,
    )
    print(
        "report_json =",
        report_json,
    )


if __name__ == "__main__":
    main()
