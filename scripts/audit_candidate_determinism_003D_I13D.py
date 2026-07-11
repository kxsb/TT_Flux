from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ttflux.analysis.candidates import (
    CandidateConfig,
    detect_frame_candidates,
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
        "--i13b",
        type=Path,
        default=Path(
            "runs/_ball_candidate_oracle_003D_I13B/"
            "i13b_frame_oracle.csv"
        ),
    )

    parser.add_argument(
        "--i13c",
        type=Path,
        default=Path(
            "runs/_ball_candidate_capacity_003D_I13C/"
            "i13c_capacity_by_frame.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/_ball_candidate_determinism_003D_I13D"
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


def parse_flag(
    value: Any,
) -> int:
    number = parse_float(value)

    if number is None:
        return 0

    return int(number != 0)


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
) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Vidéo inaccessible : {video_path}"
        )

    first_needed = max(
        0,
        start_frame - 1,
    )

    last_needed = end_frame_exclusive

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
        while frame_index <= last_needed:
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
            f"frames absentes {missing[:10]}"
        )

    return decoded


def candidate_signature(
    candidate: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        candidate["x"],
        candidate["y"],
        candidate["bbox_x"],
        candidate["bbox_y"],
        candidate["bbox_w"],
        candidate["bbox_h"],
        candidate["area"],
        candidate["mean_brightness"],
        candidate["motion_strength"],
        candidate["fill_ratio"],
        candidate["circularity"],
        candidate["score"],
    )


def ordering_key(
    candidate: dict[str, Any],
) -> tuple[float, float]:
    return (
        float(candidate["score"]),
        float(
            candidate[
                "motion_strength"
            ]
        ),
    )


def best_candidate(
    candidates: list[dict[str, Any]],
    gt_x: float,
    gt_y: float,
    cap: int | None = None,
) -> tuple[int | None, float | None]:
    selected = (
        candidates
        if cap is None
        else candidates[:cap]
    )

    if not selected:
        return None, None

    distances = [
        (
            rank,
            math.hypot(
                float(candidate["x"])
                - gt_x,
                float(candidate["y"])
                - gt_y,
            ),
        )
        for rank, candidate
        in enumerate(
            selected,
            start=1,
        )
    ]

    return min(
        distances,
        key=lambda item: item[1],
    )


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


def main() -> None:
    args = parse_args()

    manifest_path = (
        args.manifest.resolve()
    )
    gt_path = args.gt.resolve()
    i13b_path = args.i13b.resolve()
    i13c_path = args.i13c.resolve()
    output_dir = (
        args.output_dir.resolve()
    )

    cv2.setNumThreads(1)

    try:
        cv2.setRNGSeed(0)
    except AttributeError:
        pass

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

    i13b_rows = read_csv(
        i13b_path
    )

    i13c_rows = read_csv(
        i13c_path
    )

    if len(manifest_rows) != 3:
        raise RuntimeError(
            "Trois clips acceptés attendus."
        )

    if (
        len(gt_rows) != 450
        or len(i13b_rows) != 450
        or len(i13c_rows) != 450
    ):
        raise RuntimeError(
            "Les trois tables de frames "
            "doivent contenir 450 lignes."
        )

    gt_index: dict[
        tuple[str, int],
        dict[str, str],
    ] = {}

    for row in gt_rows:
        key = (
            str(row["clip_id"]),
            parse_int(
                row["local_frame"]
            ),
        )

        if key in gt_index:
            raise RuntimeError(
                f"GT dupliqué : {key}"
            )

        gt_index[key] = row

    b_index = {
        (
            str(row["clip_id"]),
            parse_int(
                row["local_frame"]
            ),
        ):
            row
        for row in i13b_rows
    }

    c_index = {
        (
            str(row["clip_id"]),
            parse_int(
                row["local_frame"]
            ),
        ):
            row
        for row in i13c_rows
    }

    config_24 = CandidateConfig()

    config_extended = replace(
        config_24,
        max_candidates_per_frame=4096,
    )

    audit_rows: list[
        dict[str, Any]
    ] = []

    fresh_counts = {
        "top1_hits":
            0,
        "cap24_hits":
            0,
        "extended_hits":
            0,
    }

    prefix_mismatch_frames = 0
    repeat_mismatch_frames = 0
    top1_tie_frames = 0
    boundary24_tie_frames = 0

    for manifest in manifest_rows:
        clip_id = str(
            manifest["clip_id"]
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

        decoded = decode_context(
            video_path,
            start_frame,
            end_frame,
            config_24.blur_kernel,
        )

        for local_frame in range(150):
            key = (
                clip_id,
                local_frame,
            )

            gt = gt_index[key]
            saved_b = b_index[key]
            saved_c = c_index[key]

            source_frame = (
                start_frame
                + local_frame
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

            candidates_24 = (
                detect_frame_candidates(
                    previous_gray,
                    current_gray,
                    next_gray,
                    config_24,
                )
            )

            candidates_extended = (
                detect_frame_candidates(
                    previous_gray,
                    current_gray,
                    next_gray,
                    config_extended,
                )
            )

            candidates_repeat = (
                detect_frame_candidates(
                    previous_gray,
                    current_gray,
                    next_gray,
                    config_extended,
                )
            )

            signatures_24 = [
                candidate_signature(
                    candidate
                )
                for candidate
                in candidates_24
            ]

            signatures_prefix = [
                candidate_signature(
                    candidate
                )
                for candidate
                in candidates_extended[:24]
            ]

            signatures_repeat = [
                candidate_signature(
                    candidate
                )
                for candidate
                in candidates_repeat
            ]

            prefix_equal = (
                signatures_24
                == signatures_prefix
            )

            repeat_equal = (
                [
                    candidate_signature(
                        candidate
                    )
                    for candidate
                    in candidates_extended
                ]
                == signatures_repeat
            )

            if not prefix_equal:
                prefix_mismatch_frames += 1

            if not repeat_equal:
                repeat_mismatch_frames += 1

            top1_tie = (
                len(
                    candidates_extended
                ) >= 2
                and ordering_key(
                    candidates_extended[0]
                )
                == ordering_key(
                    candidates_extended[1]
                )
            )

            boundary24_tie = (
                len(
                    candidates_extended
                ) >= 25
                and ordering_key(
                    candidates_extended[23]
                )
                == ordering_key(
                    candidates_extended[24]
                )
            )

            top1_tie_frames += int(
                top1_tie
            )

            boundary24_tie_frames += int(
                boundary24_tie
            )

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

            fresh_top1_hit = 0
            fresh_cap24_hit = 0
            fresh_extended_hit = 0

            top1_distance = None
            cap24_rank = None
            cap24_distance = None
            extended_rank = None
            extended_distance = None

            if visible:
                if candidates_extended:
                    top1_distance = (
                        math.hypot(
                            float(
                                candidates_extended[
                                    0
                                ]["x"]
                            )
                            - gt_x,
                            float(
                                candidates_extended[
                                    0
                                ]["y"]
                            )
                            - gt_y,
                        )
                    )

                (
                    cap24_rank,
                    cap24_distance,
                ) = best_candidate(
                    candidates_extended,
                    gt_x,
                    gt_y,
                    cap=24,
                )

                (
                    extended_rank,
                    extended_distance,
                ) = best_candidate(
                    candidates_extended,
                    gt_x,
                    gt_y,
                    cap=None,
                )

                fresh_top1_hit = int(
                    top1_distance
                    is not None
                    and top1_distance
                    <= 20
                )

                fresh_cap24_hit = int(
                    cap24_distance
                    is not None
                    and cap24_distance
                    <= 20
                )

                fresh_extended_hit = int(
                    extended_distance
                    is not None
                    and extended_distance
                    <= 20
                )

                fresh_counts[
                    "top1_hits"
                ] += fresh_top1_hit

                fresh_counts[
                    "cap24_hits"
                ] += fresh_cap24_hit

                fresh_counts[
                    "extended_hits"
                ] += fresh_extended_hit

            saved_b_top1 = int(
                (
                    parse_float(
                        saved_b.get(
                            "top1_distance_px"
                        )
                    )
                    or math.inf
                )
                <= 20
            )

            saved_b_cap24 = parse_flag(
                saved_b.get(
                    "oracle_hit_20px"
                )
            )

            saved_c_top1 = parse_flag(
                saved_c.get(
                    "hit_cap1_20px"
                )
            )

            saved_c_cap24 = parse_flag(
                saved_c.get(
                    "hit_cap24_20px"
                )
            )

            saved_c_extended = parse_flag(
                saved_c.get(
                    "hit_cap4096_20px"
                )
            )

            audit_rows.append({
                "clip_id":
                    clip_id,
                "local_frame":
                    local_frame,
                "source_frame":
                    source_frame,
                "gt_visible":
                    int(visible),
                "candidate_count_24":
                    len(candidates_24),
                "candidate_count_extended":
                    len(
                        candidates_extended
                    ),
                "prefix_equal":
                    int(prefix_equal),
                "repeat_equal":
                    int(repeat_equal),
                "top1_tie":
                    int(top1_tie),
                "boundary24_tie":
                    int(boundary24_tie),
                "fresh_top1_hit_20px":
                    fresh_top1_hit,
                "fresh_cap24_hit_20px":
                    fresh_cap24_hit,
                "fresh_extended_hit_20px":
                    fresh_extended_hit,
                "fresh_cap24_best_rank":
                    cap24_rank
                    if cap24_rank
                    is not None
                    else "",
                "fresh_cap24_distance_px":
                    (
                        round(
                            cap24_distance,
                            6,
                        )
                        if cap24_distance
                        is not None
                        else ""
                    ),
                "fresh_extended_best_rank":
                    extended_rank
                    if extended_rank
                    is not None
                    else "",
                "fresh_extended_distance_px":
                    (
                        round(
                            extended_distance,
                            6,
                        )
                        if extended_distance
                        is not None
                        else ""
                    ),
                "saved_i13b_top1_hit":
                    saved_b_top1,
                "saved_i13b_cap24_hit":
                    saved_b_cap24,
                "saved_i13c_top1_hit":
                    saved_c_top1,
                "saved_i13c_cap24_hit":
                    saved_c_cap24,
                "saved_i13c_extended_hit":
                    saved_c_extended,
                "fresh_vs_i13b_top1_mismatch":
                    int(
                        fresh_top1_hit
                        != saved_b_top1
                    ),
                "fresh_vs_i13b_cap24_mismatch":
                    int(
                        fresh_cap24_hit
                        != saved_b_cap24
                    ),
                "fresh_vs_i13c_top1_mismatch":
                    int(
                        fresh_top1_hit
                        != saved_c_top1
                    ),
                "fresh_vs_i13c_cap24_mismatch":
                    int(
                        fresh_cap24_hit
                        != saved_c_cap24
                    ),
                "fresh_vs_i13c_extended_mismatch":
                    int(
                        fresh_extended_hit
                        != saved_c_extended
                    ),
            })

    saved_counts = {
        "i13b_top1_hits":
            sum(
                int(
                    (
                        parse_float(
                            row.get(
                                "top1_distance_px"
                            )
                        )
                        or math.inf
                    )
                    <= 20
                )
                for row in i13b_rows
            ),
        "i13b_cap24_hits":
            sum(
                parse_flag(
                    row.get(
                        "oracle_hit_20px"
                    )
                )
                for row in i13b_rows
            ),
        "i13c_top1_hits":
            sum(
                parse_flag(
                    row.get(
                        "hit_cap1_20px"
                    )
                )
                for row in i13c_rows
            ),
        "i13c_cap24_hits":
            sum(
                parse_flag(
                    row.get(
                        "hit_cap24_20px"
                    )
                )
                for row in i13c_rows
            ),
        "i13c_extended_hits":
            sum(
                parse_flag(
                    row.get(
                        "hit_cap4096_20px"
                    )
                )
                for row in i13c_rows
            ),
    }

    mismatch_counts = {
        column:
            sum(
                int(row[column])
                for row in audit_rows
            )
        for column in (
            "fresh_vs_i13b_top1_mismatch",
            "fresh_vs_i13b_cap24_mismatch",
            "fresh_vs_i13c_top1_mismatch",
            "fresh_vs_i13c_cap24_mismatch",
            "fresh_vs_i13c_extended_mismatch",
        )
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "i13d_determinism_by_frame.csv"
    )

    json_path = (
        output_dir
        / "i13d_determinism_report.json"
    )

    write_csv(
        csv_path,
        audit_rows,
    )

    report = {
        "experiment":
            (
                "003D_I13D_"
                "candidate_determinism"
            ),
        "opencv_threads":
            1,
        "fresh_counts":
            fresh_counts,
        "saved_counts":
            saved_counts,
        "prefix_mismatch_frames":
            prefix_mismatch_frames,
        "repeat_mismatch_frames":
            repeat_mismatch_frames,
        "top1_tie_frames":
            top1_tie_frames,
        "boundary24_tie_frames":
            boundary24_tie_frames,
        "saved_output_mismatches":
            mismatch_counts,
        "artifacts": {
            "frame_csv":
                csv_path.name,
            "report_json":
                json_path.name,
        },
    }

    json_path.write_text(
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
        "I13D_CANDIDATE_DETERMINISM_OK"
    )

    print()
    print(
        "=== CALCUL FRAIS "
        "MONO-THREAD ==="
    )
    print(
        "top1_hits =",
        fresh_counts[
            "top1_hits"
        ],
    )
    print(
        "cap24_hits =",
        fresh_counts[
            "cap24_hits"
        ],
    )
    print(
        "extended_hits =",
        fresh_counts[
            "extended_hits"
        ],
    )

    print()
    print(
        "=== SORTIES SAUVEGARDÉES ==="
    )

    for key, value in (
        saved_counts.items()
    ):
        print(
            key,
            "=",
            value,
        )

    print()
    print(
        "=== DÉTERMINISME ==="
    )
    print(
        "prefix_mismatch_frames =",
        prefix_mismatch_frames,
    )
    print(
        "repeat_mismatch_frames =",
        repeat_mismatch_frames,
    )
    print(
        "top1_tie_frames =",
        top1_tie_frames,
    )
    print(
        "boundary24_tie_frames =",
        boundary24_tie_frames,
    )

    print()
    print(
        "=== ÉCARTS AUX SORTIES "
        "PRÉCÉDENTES ==="
    )

    for key, value in (
        mismatch_counts.items()
    ):
        print(
            key,
            "=",
            value,
        )

    print()
    print(
        "csv =",
        csv_path,
    )
    print(
        "json =",
        json_path,
    )


if __name__ == "__main__":
    main()
