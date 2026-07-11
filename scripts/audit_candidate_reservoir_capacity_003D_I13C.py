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


CAP_LEVELS = (
    1,
    3,
    5,
    10,
    24,
    48,
    96,
    192,
    512,
    4096,
)

RADII = (
    10,
    20,
    30,
)

EXTENDED_CAP = max(CAP_LEVELS)


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
            "runs/_ball_candidate_capacity_003D_I13C"
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
            f"frames absentes "
            f"{missing[:10]}"
        )

    return decoded


def ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator,
        6,
    )


def percentile(
    values: list[int | float],
    q: float,
) -> float | None:
    if not values:
        return None

    return round(
        float(
            np.quantile(
                np.asarray(
                    values,
                    dtype=np.float64,
                ),
                q,
            )
        ),
        3,
    )


def classify_frame_20px(
    visible: bool,
    candidate_count: int,
    first_hit_rank_20px: int | None,
) -> str:
    if not visible:
        return "gt_invisible"

    if candidate_count == 0:
        return "empty_extended_reservoir"

    if first_hit_rank_20px is None:
        return "no_candidate_within_20"

    if first_hit_rank_20px == 1:
        return "top1_hit"

    if first_hit_rank_20px <= 24:
        return "recoverable_within_24"

    return "pruned_beyond_24"


def summarize(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    visible_rows = [
        row
        for row in rows
        if row["_visible"]
    ]

    counts = [
        int(
            row[
                "extended_candidate_count"
            ]
        )
        for row in rows
    ]

    categories = Counter(
        str(
            row[
                "failure_class_20px"
            ]
        )
        for row in rows
    )

    capacity_curve: dict[
        str,
        dict[str, Any],
    ] = {}

    for cap in CAP_LEVELS:
        cap_metrics: dict[
            str,
            Any,
        ] = {
            "candidate_cap":
                cap,
            "radii":
                {},
        }

        for radius in RADII:
            hits = sum(
                any(
                    rank <= cap
                    and distance <= radius
                    for rank, distance
                    in row["_ranked_distances"]
                )
                for row in visible_rows
            )

            cap_metrics[
                "radii"
            ][str(radius)] = {
                "hits":
                    hits,
                "recall":
                    ratio(
                        hits,
                        len(visible_rows),
                    ),
            }

        capacity_curve[
            str(cap)
        ] = cap_metrics

    extended_hits_20 = (
        capacity_curve[
            str(EXTENDED_CAP)
        ]["radii"]["20"]["hits"]
    )

    cap24_hits_20 = (
        capacity_curve[
            "24"
        ]["radii"]["20"]["hits"]
    )

    top1_hits_20 = (
        capacity_curve[
            "1"
        ]["radii"]["20"]["hits"]
    )

    return {
        "frames":
            len(rows),
        "visible_frames":
            len(visible_rows),
        "invisible_frames":
            len(rows)
            - len(visible_rows),
        "extended_candidate_total":
            sum(counts),
        "extended_candidate_mean":
            round(
                float(
                    np.mean(counts)
                ),
                3,
            ),
        "extended_candidate_median":
            percentile(
                counts,
                0.50,
            ),
        "extended_candidate_p90":
            percentile(
                counts,
                0.90,
            ),
        "extended_candidate_p95":
            percentile(
                counts,
                0.95,
            ),
        "extended_candidate_max":
            max(
                counts,
                default=0,
            ),
        "frames_above_24_candidates":
            sum(
                count > 24
                for count in counts
            ),
        "frames_at_extended_cap":
            sum(
                count >= EXTENDED_CAP
                for count in counts
            ),
        "failure_classes_20px":
            dict(categories),
        "capacity_curve":
            capacity_curve,
        "diagnosis_20px": {
            "top1_hits":
                top1_hits_20,
            "recoverable_ranks_2_to_24":
                cap24_hits_20
                - top1_hits_20,
            "recoverable_beyond_24":
                extended_hits_20
                - cap24_hits_20,
            "extended_generator_hits":
                extended_hits_20,
            "extended_generator_misses":
                len(visible_rows)
                - extended_hits_20,
        },
    }


def main() -> None:
    args = parse_args()

    manifest_path = (
        args.manifest.resolve()
    )
    gt_path = args.gt.resolve()
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
            "Trois clips acceptés "
            f"attendus, "
            f"{len(manifest_rows)} "
            "trouvés."
        )

    if len(gt_rows) != 450:
        raise RuntimeError(
            "450 lignes GT attendues, "
            f"{len(gt_rows)} trouvées."
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

        if (
            local_frame
            in gt_by_clip[clip_id]
        ):
            raise RuntimeError(
                "GT dupliqué : "
                f"{clip_id}/"
                f"{local_frame}"
            )

        gt_by_clip[
            clip_id
        ][local_frame] = row

    base_config = CandidateConfig()

    extended_config = replace(
        base_config,
        max_candidates_per_frame=(
            EXTENDED_CAP
        ),
    )

    extended_config.validate()

    output_rows: list[
        dict[str, Any]
    ] = []

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
            end_frame
            - start_frame
        ) != 150:
            raise RuntimeError(
                f"{clip_id} : "
                "150 frames attendues."
            )

        clip_gt = gt_by_clip.get(
            clip_id,
            {},
        )

        if len(clip_gt) != 150:
            raise RuntimeError(
                f"{clip_id} : "
                f"{len(clip_gt)} "
                "lignes GT."
            )

        decoded = decode_context(
            video_path,
            start_frame,
            end_frame,
            extended_config.blur_kernel,
        )

        for local_frame in range(150):
            source_frame = (
                start_frame
                + local_frame
            )

            gt = clip_gt[
                local_frame
            ]

            if (
                parse_int(
                    gt["source_frame"]
                )
                != source_frame
            ):
                raise RuntimeError(
                    f"{clip_id}/"
                    f"{local_frame} : "
                    "source_frame "
                    "incohérent."
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

            candidates = (
                detect_frame_candidates(
                    previous_gray,
                    current_gray,
                    next_gray,
                    extended_config,
                )
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

            ranked_distances: list[
                tuple[int, float]
            ] = []

            if visible:
                for (
                    rank,
                    candidate,
                ) in enumerate(
                    candidates,
                    start=1,
                ):
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

                    ranked_distances.append(
                        (
                            rank,
                            distance,
                        )
                    )

            if ranked_distances:
                (
                    best_rank,
                    best_distance,
                ) = min(
                    ranked_distances,
                    key=lambda item:
                        item[1],
                )
            else:
                best_rank = None
                best_distance = None

            first_hit_rank_20px = next(
                (
                    rank
                    for rank, distance
                    in ranked_distances
                    if distance <= 20
                ),
                None,
            )

            failure_class = (
                classify_frame_20px(
                    visible,
                    len(candidates),
                    first_hit_rank_20px,
                )
            )

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
                "extended_candidate_count":
                    len(candidates),
                "best_candidate_rank":
                    best_rank
                    if best_rank
                    is not None
                    else "",
                "best_distance_px":
                    (
                        round(
                            best_distance,
                            6,
                        )
                        if best_distance
                        is not None
                        else ""
                    ),
                "first_hit_rank_20px":
                    (
                        first_hit_rank_20px
                        if first_hit_rank_20px
                        is not None
                        else ""
                    ),
                "failure_class_20px":
                    failure_class,
            }

            for cap in CAP_LEVELS:
                for radius in RADII:
                    row[
                        (
                            f"hit_cap"
                            f"{cap}_"
                            f"{radius}px"
                        )
                    ] = int(
                        best_rank
                        is not None
                        and best_rank
                        <= cap
                        and best_distance
                        is not None
                        and best_distance
                        <= radius
                    )

            output_rows.append({
                **row,
                "_visible":
                    visible,
                "_best_rank":
                    best_rank,
                "_best_distance":
                    best_distance,
                "_ranked_distances":
                    ranked_distances,
            })

    visible_count = sum(
        int(
            row["_visible"]
        )
        for row in output_rows
    )

    if (
        len(output_rows) != 450
        or visible_count != 416
    ):
        raise RuntimeError(
            "Contrat GT brisé : "
            f"frames="
            f"{len(output_rows)}, "
            f"visibles="
            f"{visible_count}"
        )

    public_rows = [
        {
            key: value
            for key, value
            in row.items()
            if not key.startswith("_")
        }
        for row in output_rows
    ]

    per_clip = {
        clip_id:
            summarize(
                [
                    row
                    for row
                    in output_rows
                    if row[
                        "clip_id"
                    ] == clip_id
                ]
            )
        for clip_id
        in sorted(gt_by_clip)
    }

    overall = summarize(
        output_rows
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_csv = (
        output_dir
        / "i13c_capacity_by_frame.csv"
    )

    report_json = (
        output_dir
        / "i13c_capacity_report.json"
    )

    write_csv(
        frame_csv,
        public_rows,
    )

    report = {
        "experiment":
            (
                "003D_I13C_"
                "candidate_reservoir_capacity"
            ),
        "manifest":
            str(manifest_path),
        "ground_truth":
            str(gt_path),
        "candidate_algorithm": {
            "name":
                (
                    "triple_frame_"
                    "motion_components"
                ),
            "base_parameters":
                asdict(base_config),
            "extended_parameters":
                asdict(
                    extended_config
                ),
        },
        "cap_levels":
            list(CAP_LEVELS),
        "radii_px":
            list(RADII),
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
        "I13C_RESERVOIR_CAPACITY_OK"
    )
    print(
        "frames =",
        overall["frames"],
    )
    print(
        "visible_frames =",
        overall[
            "visible_frames"
        ],
    )
    print(
        "extended_candidate_mean =",
        overall[
            "extended_candidate_mean"
        ],
    )
    print(
        "extended_candidate_p95 =",
        overall[
            "extended_candidate_p95"
        ],
    )
    print(
        "extended_candidate_max =",
        overall[
            "extended_candidate_max"
        ],
    )
    print(
        "frames_above_24_candidates =",
        overall[
            "frames_above_24_candidates"
        ],
    )
    print(
        "frames_at_extended_cap =",
        overall[
            "frames_at_extended_cap"
        ],
    )

    print()
    print(
        "=== COURBE DE CAPACITÉ "
        "À 20 PX ==="
    )

    for cap in CAP_LEVELS:
        metrics = overall[
            "capacity_curve"
        ][str(cap)][
            "radii"
        ]["20"]

        print(
            f"cap={cap:04d}",
            (
                f"hits="
                f"{metrics['hits']}"
            ),
            (
                f"recall="
                f"{metrics['recall']:.4f}"
            ),
        )

    print()
    print(
        "=== DÉCOMPOSITION "
        "20 PX ==="
    )

    diagnosis = overall[
        "diagnosis_20px"
    ]

    print(
        "top1_hits =",
        diagnosis[
            "top1_hits"
        ],
    )
    print(
        "recoverable_ranks_2_to_24 =",
        diagnosis[
            "recoverable_ranks_2_to_24"
        ],
    )
    print(
        "recoverable_beyond_24 =",
        diagnosis[
            "recoverable_beyond_24"
        ],
    )
    print(
        "extended_generator_hits =",
        diagnosis[
            "extended_generator_hits"
        ],
    )
    print(
        "extended_generator_misses =",
        diagnosis[
            "extended_generator_misses"
        ],
    )

    print()
    print(
        "failure_classes =",
        overall[
            "failure_classes_20px"
        ],
    )

    print()
    print(
        "=== PAR CLIP À 20 PX ==="
    )

    for (
        clip_id,
        summary,
    ) in per_clip.items():
        clip_diagnosis = summary[
            "diagnosis_20px"
        ]

        print(
            clip_id,
            (
                f"top1="
                f"{clip_diagnosis['top1_hits']}"
            ),
            (
                f"rank2_24="
                f"{clip_diagnosis['recoverable_ranks_2_to_24']}"
            ),
            (
                f"beyond24="
                f"{clip_diagnosis['recoverable_beyond_24']}"
            ),
            (
                f"generator_miss="
                f"{clip_diagnosis['extended_generator_misses']}"
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
