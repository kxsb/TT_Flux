from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ttflux.tracking.candidates.generator import (
    CandidateConfig,
    detect_frame_candidates,
)


RADII = (5, 10, 20, 30, 50)
TOP_K = (1, 3, 5, 10, 24)


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
            "runs/_ball_candidate_oracle_003D_I13B"
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


def parse_number(
    value: Any,
) -> float | None:
    text = str(value or "").strip().replace(",", ".")

    if not text:
        return None

    try:
        result = float(text)
    except ValueError:
        return None

    if not math.isfinite(result):
        return None

    return result


def parse_integer(
    value: Any,
) -> int:
    result = parse_number(value)

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
            (blur_kernel, blur_kernel),
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
    dict[str, Any],
]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Vidéo inaccessible : {video_path}"
        )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )
    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )
    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )
    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    first_needed = max(
        0,
        start_frame - 1,
    )

    last_needed = end_frame_exclusive

    if frame_count > 0:
        last_needed = min(
            last_needed,
            frame_count - 1,
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        float(first_needed),
    )

    decoded: dict[int, np.ndarray] = {}
    frame_index = first_needed

    try:
        while frame_index <= last_needed:
            ok, frame = capture.read()

            if not ok:
                break

            decoded[frame_index] = (
                prepare_gray(
                    frame,
                    blur_kernel,
                )
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
            f"frames manquantes {missing[:8]}"
        )

    return decoded, {
        "source_video":
            str(video_path),
        "width":
            width,
        "height":
            height,
        "fps":
            round(fps, 6)
            if math.isfinite(fps)
            else None,
        "frame_count":
            frame_count,
        "start_frame":
            start_frame,
        "end_frame_exclusive":
            end_frame_exclusive,
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            f"Aucune ligne à écrire dans {path}"
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


def rate(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator,
        6,
    )


def quantile_stats(
    values: list[float],
) -> dict[str, float | None]:
    if not values:
        return {
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "median":
            round(
                float(
                    np.quantile(
                        array,
                        0.50,
                    )
                ),
                3,
            ),
        "p90":
            round(
                float(
                    np.quantile(
                        array,
                        0.90,
                    )
                ),
                3,
            ),
        "p95":
            round(
                float(
                    np.quantile(
                        array,
                        0.95,
                    )
                ),
                3,
            ),
        "max":
            round(
                float(np.max(array)),
                3,
            ),
    }


def summarize(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    visible_rows = [
        row
        for row in rows
        if row["_visible"]
    ]

    invisible_rows = [
        row
        for row in rows
        if not row["_visible"]
    ]

    candidate_counts = [
        int(row["candidate_count"])
        for row in rows
    ]

    nonempty_visible = [
        row
        for row in visible_rows
        if int(
            row["candidate_count"]
        ) > 0
    ]

    visible_empty = sum(
        int(
            row["candidate_count"]
        ) == 0
        for row in visible_rows
    )

    result: dict[str, Any] = {
        "frames":
            len(rows),
        "visible_frames":
            len(visible_rows),
        "invisible_frames":
            len(invisible_rows),
        "total_candidates":
            sum(candidate_counts),
        "mean_candidates_per_frame":
            round(
                float(
                    np.mean(
                        candidate_counts
                    )
                ),
                3,
            ),
        "median_candidates_per_frame":
            round(
                float(
                    np.median(
                        candidate_counts
                    )
                ),
                3,
            ),
        "max_candidates_per_frame":
            max(
                candidate_counts,
                default=0,
            ),
        "visible_empty_reservoir":
            visible_empty,
        "invisible_frames_with_candidates":
            sum(
                int(
                    row["candidate_count"]
                ) > 0
                for row in invisible_rows
            ),
        "nearest_distance_nonempty_visible":
            quantile_stats(
                [
                    float(
                        row["_best_distance"]
                    )
                    for row
                    in nonempty_visible
                ]
            ),
        "thresholds":
            {},
        "top_k_at_20px":
            {},
    }

    for radius in RADII:
        oracle_hits = [
            row
            for row in visible_rows
            if (
                row["_best_distance"]
                is not None
                and float(
                    row["_best_distance"]
                ) <= radius
            )
        ]

        top1_hits = [
            row
            for row in visible_rows
            if (
                row["_top1_distance"]
                is not None
                and float(
                    row["_top1_distance"]
                ) <= radius
            )
        ]

        nonempty_generator_misses = (
            len(visible_rows)
            - visible_empty
            - len(oracle_hits)
        )

        rank_gt1 = sum(
            int(
                row["_best_rank"]
            ) > 1
            for row in oracle_hits
        )

        result["thresholds"][
            str(radius)
        ] = {
            "oracle_hits":
                len(oracle_hits),
            "oracle_recall":
                rate(
                    len(oracle_hits),
                    len(visible_rows),
                ),
            "top1_hits":
                len(top1_hits),
            "top1_recall":
                rate(
                    len(top1_hits),
                    len(visible_rows),
                ),
            "empty_reservoir_misses":
                visible_empty,
            "nonempty_generator_misses":
                nonempty_generator_misses,
            "oracle_hits_rank_gt1":
                rank_gt1,
            "rank1_gap_recoverable":
                (
                    len(oracle_hits)
                    - len(top1_hits)
                ),
        }

    for top_k in TOP_K:
        hits = sum(
            any(
                rank <= top_k
                and distance <= 20
                for rank, distance
                in row[
                    "_ranked_distances"
                ]
            )
            for row in visible_rows
        )

        result["top_k_at_20px"][
            str(top_k)
        ] = {
            "hits":
                hits,
            "recall":
                rate(
                    hits,
                    len(visible_rows),
                ),
        }

    metrics_20 = result[
        "thresholds"
    ]["20"]

    generator_misses = (
        metrics_20[
            "empty_reservoir_misses"
        ]
        + metrics_20[
            "nonempty_generator_misses"
        ]
    )

    rank1_gap = metrics_20[
        "rank1_gap_recoverable"
    ]

    if generator_misses > rank1_gap:
        dominant_axis = (
            "candidate_generation"
        )
    elif rank1_gap > generator_misses:
        dominant_axis = (
            "rank1_scoring"
        )
    else:
        dominant_axis = "mixed"

    result["diagnosis_at_20px"] = {
        "generator_misses":
            generator_misses,
        "rank1_gap_recoverable":
            rank1_gap,
        "dominant_axis":
            dominant_axis,
    }

    return result


def main() -> None:
    cli = parse_args()

    manifest_path = (
        cli.manifest.resolve()
    )
    gt_path = cli.gt.resolve()
    output_dir = (
        cli.output_dir.resolve()
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
            "Trois clips acceptés attendus, "
            f"{len(manifest_rows)} trouvés."
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
        local_frame = parse_integer(
            row["local_frame"]
        )

        if local_frame in gt_by_clip[
            clip_id
        ]:
            raise RuntimeError(
                "GT dupliqué : "
                f"{clip_id} frame "
                f"{local_frame}"
            )

        gt_by_clip[
            clip_id
        ][local_frame] = row

    config = CandidateConfig()
    config.validate()

    frame_rows: list[
        dict[str, Any]
    ] = []

    candidate_rows: list[
        dict[str, Any]
    ] = []

    source_summaries: dict[
        str,
        dict[str, Any],
    ] = {}

    for manifest in manifest_rows:
        clip_id = str(
            manifest["clip_id"]
        )
        source_key = str(
            manifest["source_key"]
        )
        source_video = Path(
            manifest["source_video"]
        ).resolve()

        start_frame = parse_integer(
            manifest["start_frame"]
        )
        end_frame = parse_integer(
            manifest[
                "end_frame_exclusive"
            ]
        )

        if (
            end_frame - start_frame
        ) != 150:
            raise RuntimeError(
                f"{clip_id} : "
                "fenêtre différente de "
                "150 frames."
            )

        clip_gt = gt_by_clip.get(
            clip_id,
            {},
        )

        if len(clip_gt) != 150:
            raise RuntimeError(
                f"{clip_id} : "
                f"{len(clip_gt)} GT "
                "au lieu de 150."
            )

        decoded, source_summary = (
            decode_context(
                source_video,
                start_frame,
                end_frame,
                config.blur_kernel,
            )
        )

        source_summaries[
            clip_id
        ] = {
            **source_summary,
            "source_key":
                source_key,
        }

        for local_frame in range(150):
            source_frame = (
                start_frame
                + local_frame
            )

            gt = clip_gt[
                local_frame
            ]

            gt_source_frame = (
                parse_integer(
                    gt[
                        "source_frame"
                    ]
                )
            )

            if (
                gt_source_frame
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
                    config,
                )
            )

            gt_x = parse_number(
                gt.get("x")
            )
            gt_y = parse_number(
                gt.get("y")
            )

            visible = (
                gt_x is not None
                and gt_y is not None
            )

            ranked_distances: list[
                tuple[int, float]
            ] = []

            for rank, candidate in enumerate(
                candidates,
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

                if distance is not None:
                    ranked_distances.append(
                        (
                            rank,
                            distance,
                        )
                    )

                candidate_rows.append({
                    "clip_id":
                        clip_id,
                    "source_key":
                        source_key,
                    "local_frame":
                        local_frame,
                    "source_frame":
                        source_frame,
                    "candidate_id":
                        (
                            f"{clip_id}_"
                            f"f{local_frame:03d}_"
                            f"c{rank:02d}"
                        ),
                    "rank":
                        rank,
                    "x":
                        candidate["x"],
                    "y":
                        candidate["y"],
                    "bbox_x":
                        candidate[
                            "bbox_x"
                        ],
                    "bbox_y":
                        candidate[
                            "bbox_y"
                        ],
                    "bbox_w":
                        candidate[
                            "bbox_w"
                        ],
                    "bbox_h":
                        candidate[
                            "bbox_h"
                        ],
                    "area":
                        candidate["area"],
                    "mean_brightness":
                        candidate[
                            "mean_brightness"
                        ],
                    "motion_strength":
                        candidate[
                            "motion_strength"
                        ],
                    "fill_ratio":
                        candidate[
                            "fill_ratio"
                        ],
                    "circularity":
                        candidate[
                            "circularity"
                        ],
                    "score":
                        candidate["score"],
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
                    "distance_to_gt_px":
                        (
                            round(
                                distance,
                                6,
                            )
                            if distance
                            is not None
                            else ""
                        ),
                })

            if ranked_distances:
                top1_distance = (
                    ranked_distances[
                        0
                    ][1]
                )

                (
                    best_rank,
                    best_distance,
                ) = min(
                    ranked_distances,
                    key=lambda item:
                        item[1],
                )

                best_candidate = (
                    candidates[
                        best_rank - 1
                    ]
                )
            else:
                top1_distance = None
                best_rank = None
                best_distance = None
                best_candidate = None

            public_row: dict[
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
                "candidate_count":
                    len(candidates),
                "top1_x":
                    (
                        candidates[
                            0
                        ]["x"]
                        if candidates
                        else ""
                    ),
                "top1_y":
                    (
                        candidates[
                            0
                        ]["y"]
                        if candidates
                        else ""
                    ),
                "top1_score":
                    (
                        candidates[
                            0
                        ]["score"]
                        if candidates
                        else ""
                    ),
                "top1_distance_px":
                    (
                        round(
                            top1_distance,
                            6,
                        )
                        if top1_distance
                        is not None
                        else ""
                    ),
                "best_candidate_rank":
                    (
                        best_rank
                        if best_rank
                        is not None
                        else ""
                    ),
                "best_candidate_x":
                    (
                        best_candidate[
                            "x"
                        ]
                        if best_candidate
                        is not None
                        else ""
                    ),
                "best_candidate_y":
                    (
                        best_candidate[
                            "y"
                        ]
                        if best_candidate
                        is not None
                        else ""
                    ),
                "best_candidate_score":
                    (
                        best_candidate[
                            "score"
                        ]
                        if best_candidate
                        is not None
                        else ""
                    ),
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
            }

            for radius in RADII:
                public_row[
                    f"oracle_hit_"
                    f"{radius}px"
                ] = int(
                    best_distance
                    is not None
                    and best_distance
                    <= radius
                )

            frame_rows.append({
                **public_row,
                "_visible":
                    visible,
                "_top1_distance":
                    top1_distance,
                "_best_distance":
                    best_distance,
                "_best_rank":
                    best_rank,
                "_ranked_distances":
                    ranked_distances,
            })

    visible_count = sum(
        int(row["_visible"])
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_csv = (
        output_dir
        / "i13b_frame_oracle.csv"
    )

    candidate_csv = (
        output_dir
        / (
            "i13b_candidates_"
            "with_gt_distance.csv"
        )
    )

    report_json = (
        output_dir
        / "i13b_oracle_report.json"
    )

    public_frame_rows = [
        {
            key: value
            for key, value
            in row.items()
            if not key.startswith("_")
        }
        for row in frame_rows
    ]

    write_csv(
        frame_csv,
        public_frame_rows,
    )

    write_csv(
        candidate_csv,
        candidate_rows,
    )

    per_clip = {
        clip_id:
            summarize(
                [
                    row
                    for row in frame_rows
                    if row["clip_id"]
                    == clip_id
                ]
            )
        for clip_id
        in sorted(gt_by_clip)
    }

    overall = summarize(
        frame_rows
    )

    report = {
        "experiment":
            (
                "003D_I13B_"
                "candidate_reservoir_oracle"
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
            "implementation":
                (
                    "ttflux.tracking."
                    "candidates.generator."
                    "detect_frame_candidates"
                ),
            "parameters":
                asdict(config),
        },
        "radii_px":
            list(RADII),
        "source_clips":
            source_summaries,
        "overall":
            overall,
        "per_clip":
            per_clip,
        "artifacts": {
            "frame_oracle_csv":
                frame_csv.name,
            "candidate_distance_csv":
                candidate_csv.name,
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
        "I13B_CANDIDATE_ORACLE_OK"
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
        "total_candidates =",
        overall[
            "total_candidates"
        ],
    )
    print(
        "mean_candidates_per_frame =",
        overall[
            "mean_candidates_per_frame"
        ],
    )
    print(
        "visible_empty_reservoir =",
        overall[
            "visible_empty_reservoir"
        ],
    )

    print()
    print("=== ORACLE GLOBAL ===")

    for radius in RADII:
        metrics = overall[
            "thresholds"
        ][str(radius)]

        print(
            f"@{radius:02d}px",
            (
                f"oracle="
                f"{metrics['oracle_hits']}/"
                f"{overall['visible_frames']}"
            ),
            (
                f"recall="
                f"{metrics['oracle_recall']:.4f}"
            ),
            (
                f"top1="
                f"{metrics['top1_hits']}"
            ),
            (
                f"top1_recall="
                f"{metrics['top1_recall']:.4f}"
            ),
            (
                f"empty="
                f"{metrics['empty_reservoir_misses']}"
            ),
            (
                f"nonempty_miss="
                f"{metrics['nonempty_generator_misses']}"
            ),
            (
                f"rank_gt1="
                f"{metrics['oracle_hits_rank_gt1']}"
            ),
        )

    print()
    print("=== TOP-K À 20 PX ===")

    for top_k in TOP_K:
        metrics = overall[
            "top_k_at_20px"
        ][str(top_k)]

        print(
            f"top{top_k:02d}",
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
        "=== PAR CLIP À 20 PX ==="
    )

    for (
        clip_id,
        summary,
    ) in per_clip.items():
        metrics = summary[
            "thresholds"
        ]["20"]

        print(
            clip_id,
            (
                f"visible="
                f"{summary['visible_frames']}"
            ),
            (
                f"candidates="
                f"{summary['total_candidates']}"
            ),
            (
                f"oracle="
                f"{metrics['oracle_hits']}"
            ),
            (
                f"recall="
                f"{metrics['oracle_recall']:.4f}"
            ),
            (
                f"top1="
                f"{metrics['top1_hits']}"
            ),
            (
                f"empty="
                f"{metrics['empty_reservoir_misses']}"
            ),
            (
                f"nonempty_miss="
                f"{metrics['nonempty_generator_misses']}"
            ),
            (
                f"rank_gt1="
                f"{metrics['oracle_hits_rank_gt1']}"
            ),
        )

    diagnosis = overall[
        "diagnosis_at_20px"
    ]

    print()
    print(
        "=== DIAGNOSTIC 20 PX ==="
    )
    print(
        "generator_misses =",
        diagnosis[
            "generator_misses"
        ],
    )
    print(
        "rank1_gap_recoverable =",
        diagnosis[
            "rank1_gap_recoverable"
        ],
    )
    print(
        "dominant_axis =",
        diagnosis[
            "dominant_axis"
        ],
    )

    print()
    print(
        "frame_csv =",
        frame_csv,
    )
    print(
        "candidate_csv =",
        candidate_csv,
    )
    print(
        "report_json =",
        report_json,
    )


if __name__ == "__main__":
    main()
