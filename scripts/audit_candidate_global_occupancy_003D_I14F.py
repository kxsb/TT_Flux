from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FRAME_PATH = Path(
    "runs/_ball_candidate_oracle_003D_I13B/"
    "i13b_frame_oracle.csv"
)

CANDIDATE_PATH = Path(
    "runs/_ball_candidate_oracle_003D_I13B/"
    "i13b_candidates_with_gt_distance.csv"
)

I14C_PREDICTION_PATH = Path(
    "runs/_ball_temporal_reranking_003D_I14C/"
    "i14c_temporal_predictions.csv"
)

OUTPUT_DIR = Path(
    "runs/_ball_candidate_global_occupancy_003D_I14F"
)

RADIUS_PX = 20.0
TOP_K = (1, 3, 5, 10, 24)

INTRINSIC_FEATURES = (
    "log_area",
    "brightness",
    "motion",
    "fill_ratio",
    "circularity",
    "log_dimension",
    "aspect_symmetry",
)

OCCUPANCY_FEATURES = (
    "global_support_16",
    "global_support_32",
    "global_support_64",
    "global_support_96",
    "global_density_32",
    "global_density_64",
    "global_density_96",
    "global_nearest_log",
)

FEATURE_SETS = {
    "intrinsic": INTRINSIC_FEATURES,
    "occupancy_only": OCCUPANCY_FEATURES,
    "intrinsic_plus_occupancy": (
        *INTRINSIC_FEATURES,
        *OCCUPANCY_FEATURES,
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
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


def parse_float(raw: Any) -> float | None:
    text = str(
        raw or ""
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


def parse_int(raw: Any) -> int:
    result = parse_float(raw)

    if result is None:
        raise ValueError(
            f"Entier invalide : {raw!r}"
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


def median_value(
    rows: list[dict[str, Any]],
    feature_name: str,
) -> float | None:
    if not rows:
        return None

    return round(
        float(
            np.median(
                np.asarray(
                    [
                        row["features"][
                            feature_name
                        ]
                        for row in rows
                    ],
                    dtype=np.float64,
                )
            )
        ),
        6,
    )


def intrinsic_features(
    row: dict[str, Any],
) -> dict[str, float]:
    area = max(
        0.0,
        float(row["area"]),
    )

    width = max(
        1.0,
        float(row["bbox_w"]),
    )

    height = max(
        1.0,
        float(row["bbox_h"]),
    )

    maximum = max(width, height)
    minimum = min(width, height)

    return {
        "log_area":
            math.log1p(area),
        "brightness":
            float(
                row["mean_brightness"]
            ) / 255.0,
        "motion":
            float(
                row["motion_strength"]
            ) / 255.0,
        "fill_ratio":
            float(row["fill_ratio"]),
        "circularity":
            float(row["circularity"]),
        "log_dimension":
            math.log1p(maximum),
        "aspect_symmetry":
            minimum / maximum,
    }


def build_clip_arrays(
    rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        "x":
            np.asarray(
                [
                    row["x"]
                    for row in rows
                ],
                dtype=np.float64,
            ),
        "y":
            np.asarray(
                [
                    row["y"]
                    for row in rows
                ],
                dtype=np.float64,
            ),
        "frame":
            np.asarray(
                [
                    row["local_frame"]
                    for row in rows
                ],
                dtype=np.int32,
            ),
    }


def occupancy_features(
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    clip_frame_count: int,
) -> dict[str, float]:
    frame = int(
        row["local_frame"]
    )

    other_frame_mask = (
        arrays["frame"] != frame
    )

    dx = (
        arrays["x"]
        - float(row["x"])
    )

    dy = (
        arrays["y"]
        - float(row["y"])
    )

    distances = np.sqrt(
        dx * dx + dy * dy
    )[other_frame_mask]

    other_frames = arrays[
        "frame"
    ][other_frame_mask]

    denominator = max(
        1,
        clip_frame_count - 1,
    )

    def support(radius: float) -> float:
        mask = distances <= radius

        if not np.any(mask):
            return 0.0

        unique_frames = np.unique(
            other_frames[mask]
        )

        return float(
            len(unique_frames)
            / denominator
        )

    def density(radius: float) -> float:
        return float(
            np.sum(
                distances <= radius
            )
            / denominator
        )

    nearest = (
        float(np.min(distances))
        if len(distances)
        else 500.0
    )

    return {
        "global_support_16":
            support(16.0),
        "global_support_32":
            support(32.0),
        "global_support_64":
            support(64.0),
        "global_support_96":
            support(96.0),
        "global_density_32":
            density(32.0),
        "global_density_64":
            density(64.0),
        "global_density_96":
            density(96.0),
        "global_nearest_log":
            math.log1p(
                min(nearest, 500.0)
            ),
    }


def build_matrix(
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        [
            [
                row["features"][name]
                for name in feature_names
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    positive_count = int(
        np.sum(labels == 1)
    )

    negative_count = int(
        np.sum(labels == 0)
    )

    if (
        positive_count == 0
        or negative_count == 0
    ):
        raise RuntimeError(
            "Les deux classes sont nécessaires."
        )

    means = np.mean(
        matrix,
        axis=0,
    )

    scales = np.std(
        matrix,
        axis=0,
    )

    scales[scales < 1e-8] = 1.0

    normalized = (
        matrix - means
    ) / scales

    design = np.column_stack(
        (
            np.ones(
                len(normalized),
                dtype=np.float64,
            ),
            normalized,
        )
    )

    weights = np.where(
        labels == 1,
        len(labels)
        / (2.0 * positive_count),
        len(labels)
        / (2.0 * negative_count),
    )

    coefficients = np.zeros(
        design.shape[1],
        dtype=np.float64,
    )

    regularization = np.eye(
        design.shape[1],
        dtype=np.float64,
    )

    regularization[0, 0] = 0.0

    for _ in range(60):
        logits = np.clip(
            design @ coefficients,
            -30.0,
            30.0,
        )

        probabilities = (
            1.0
            / (
                1.0
                + np.exp(-logits)
            )
        )

        gradient = (
            design.T
            @ (
                weights
                * (
                    probabilities - labels
                )
            )
            + regularization
            @ coefficients
        )

        curvature = (
            weights
            * probabilities
            * (
                1.0 - probabilities
            )
        )

        hessian = (
            design.T
            @ (
                design
                * curvature[:, None]
            )
            + regularization
            + 1e-8
            * np.eye(
                design.shape[1]
            )
        )

        step = np.linalg.solve(
            hessian,
            gradient,
        )

        coefficients -= step

        if float(
            np.linalg.norm(step)
        ) < 1e-7:
            break

    return (
        coefficients,
        means,
        scales,
    )


def predict_probabilities(
    matrix: np.ndarray,
    coefficients: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    normalized = (
        matrix - means
    ) / scales

    design = np.column_stack(
        (
            np.ones(
                len(normalized),
                dtype=np.float64,
            ),
            normalized,
        )
    )

    logits = np.clip(
        design @ coefficients,
        -30.0,
        30.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-logits)
        )
    )


def average_ranks(
    values: np.ndarray,
) -> np.ndarray:
    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )

    position = 0

    while position < len(order):
        end = position + 1

        while (
            end < len(order)
            and values[order[end]]
            == values[order[position]]
        ):
            end += 1

        average = (
            position + 1 + end
        ) / 2.0

        ranks[
            order[position:end]
        ] = average

        position = end

    return ranks


def roc_auc(
    values: np.ndarray,
    labels: np.ndarray,
) -> float | None:
    positives = int(
        np.sum(labels == 1)
    )

    negatives = int(
        np.sum(labels == 0)
    )

    if positives == 0 or negatives == 0:
        return None

    ranks = average_ranks(values)

    positive_rank_sum = float(
        np.sum(
            ranks[labels == 1]
        )
    )

    return (
        positive_rank_sum
        - positives
        * (positives + 1)
        / 2.0
    ) / (
        positives * negatives
    )


def evaluate(
    frame_keys: list[
        tuple[str, int]
    ],
    grouped: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ],
    score_lookup: dict[str, float],
) -> dict[str, Any]:
    hits = {
        top_k: 0
        for top_k in TOP_K
    }

    oracle_frames = 0

    top1_by_frame: dict[
        tuple[str, int],
        str | None,
    ] = {}

    for frame_key in frame_keys:
        rows = list(
            grouped.get(
                frame_key,
                [],
            )
        )

        if any(
            row["positive"]
            for row in rows
        ):
            oracle_frames += 1

        rows.sort(
            key=lambda row: (
                score_lookup.get(
                    row["candidate_id"],
                    -math.inf,
                ),
                -int(row["rank"]),
            ),
            reverse=True,
        )

        top1_by_frame[frame_key] = (
            rows[0]["candidate_id"]
            if rows
            else None
        )

        for top_k in TOP_K:
            if any(
                row["positive"]
                for row in rows[:top_k]
            ):
                hits[top_k] += 1

    return {
        "visible_frames":
            len(frame_keys),
        "oracle_frames":
            oracle_frames,
        "top_k": {
            str(top_k): {
                "hits":
                    hits[top_k],
                "recall":
                    ratio(
                        hits[top_k],
                        len(frame_keys),
                    ),
                "recoverable_ratio":
                    ratio(
                        hits[top_k],
                        oracle_frames,
                    ),
            }
            for top_k in TOP_K
        },
        "_top1_by_frame":
            top1_by_frame,
    }


def main() -> None:
    frame_rows = read_csv(
        FRAME_PATH
    )

    raw_candidate_rows = read_csv(
        CANDIDATE_PATH
    )

    if len(frame_rows) != 450:
        raise RuntimeError(
            "450 frames attendues."
        )

    frame_count_by_clip: dict[
        str,
        int,
    ] = defaultdict(int)

    visible_frames: list[
        tuple[str, int]
    ] = []

    visible_by_clip: dict[
        str,
        list[tuple[str, int]],
    ] = defaultdict(list)

    for row in frame_rows:
        clip_id = str(
            row["clip_id"]
        )

        local_frame = parse_int(
            row["local_frame"]
        )

        frame_count_by_clip[
            clip_id
        ] += 1

        if parse_int(
            row["gt_visible"]
        ) != 1:
            continue

        key = (
            clip_id,
            local_frame,
        )

        visible_frames.append(key)
        visible_by_clip[
            clip_id
        ].append(key)

    if len(visible_frames) != 416:
        raise RuntimeError(
            "416 frames visibles attendues."
        )

    candidates: list[
        dict[str, Any]
    ] = []

    candidates_by_clip: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for raw in raw_candidate_rows:
        distance = parse_float(
            raw.get(
                "distance_to_gt_px"
            )
        )

        gt_visible = parse_int(
            raw["gt_visible"]
        )

        row: dict[str, Any] = {
            "clip_id":
                str(raw["clip_id"]),
            "local_frame":
                parse_int(
                    raw["local_frame"]
                ),
            "candidate_id":
                str(raw["candidate_id"]),
            "rank":
                parse_int(raw["rank"]),
            "x":
                float(raw["x"]),
            "y":
                float(raw["y"]),
            "bbox_w":
                float(raw["bbox_w"]),
            "bbox_h":
                float(raw["bbox_h"]),
            "area":
                float(raw["area"]),
            "mean_brightness":
                float(
                    raw[
                        "mean_brightness"
                    ]
                ),
            "motion_strength":
                float(
                    raw[
                        "motion_strength"
                    ]
                ),
            "fill_ratio":
                float(raw["fill_ratio"]),
            "circularity":
                float(raw["circularity"]),
            "gt_visible":
                gt_visible,
            "distance_to_gt_px":
                distance,
            "positive":
                int(
                    gt_visible == 1
                    and distance is not None
                    and distance <= RADIUS_PX
                ),
        }

        candidates.append(row)
        candidates_by_clip[
            row["clip_id"]
        ].append(row)

    clip_arrays = {
        clip_id:
            build_clip_arrays(rows)
        for clip_id, rows
        in candidates_by_clip.items()
    }

    print(
        "Calcul des cartes d'occupation..."
    )

    for index, row in enumerate(
        candidates,
        start=1,
    ):
        clip_id = row["clip_id"]

        row["features"] = {
            **intrinsic_features(row),
            **occupancy_features(
                row,
                clip_arrays[clip_id],
                frame_count_by_clip[
                    clip_id
                ],
            ),
        }

        if index % 1000 == 0:
            print(
                f"  {index}/{len(candidates)}"
            )

    visible_frame_set = set(
        visible_frames
    )

    visible_candidates = [
        row
        for row in candidates
        if (
            row["clip_id"],
            row["local_frame"],
        ) in visible_frame_set
    ]

    grouped_visible: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in visible_candidates:
        grouped_visible[
            (
                row["clip_id"],
                row["local_frame"],
            )
        ].append(row)

    clips = sorted(
        visible_by_clip
    )

    labels = np.asarray(
        [
            row["positive"]
            for row in visible_candidates
        ],
        dtype=np.int64,
    )

    baseline_lookup = {
        row["candidate_id"]:
            -float(row["rank"])
        for row in visible_candidates
    }

    baseline = evaluate(
        visible_frames,
        grouped_visible,
        baseline_lookup,
    )

    if (
        baseline["top_k"]["1"]["hits"]
        != 258
    ):
        raise RuntimeError(
            "Baseline incohérente : "
            f"{baseline['top_k']['1']['hits']}"
        )

    feature_audit: dict[
        str,
        Any,
    ] = {}

    for feature_name in OCCUPANCY_FEATURES:
        values = np.asarray(
            [
                row["features"][
                    feature_name
                ]
                for row in visible_candidates
            ],
            dtype=np.float64,
        )

        raw_auc = roc_auc(
            values,
            labels,
        )

        separation = (
            max(
                raw_auc,
                1.0 - raw_auc,
            )
            if raw_auc is not None
            else None
        )

        feature_audit[
            feature_name
        ] = {
            "auc":
                round(raw_auc, 6)
                if raw_auc is not None
                else None,
            "separation_auc":
                round(separation, 6)
                if separation is not None
                else None,
            "direction":
                (
                    "high"
                    if raw_auc is not None
                    and raw_auc >= 0.5
                    else "low"
                ),
            "ball_median":
                round(
                    float(
                        np.median(
                            values[
                                labels == 1
                            ]
                        )
                    ),
                    6,
                ),
            "other_median":
                round(
                    float(
                        np.median(
                            values[
                                labels == 0
                            ]
                        )
                    ),
                    6,
                ),
        }

    model_reports: dict[
        str,
        Any,
    ] = {}

    prediction_rows: list[
        dict[str, Any]
    ] = []

    top1_by_model: dict[
        str,
        dict[
            tuple[str, int],
            str | None,
        ],
    ] = {}

    for model_name, feature_names in (
        FEATURE_SETS.items()
    ):
        per_clip: dict[
            str,
            Any,
        ] = {}

        aggregate_hits = {
            top_k: 0
            for top_k in TOP_K
        }

        aggregate_oracle = 0

        model_top1: dict[
            tuple[str, int],
            str | None,
        ] = {}

        coefficients_by_fold: dict[
            str,
            Any,
        ] = {}

        for held_out_clip in clips:
            training_rows = [
                row
                for row in visible_candidates
                if row["clip_id"]
                != held_out_clip
            ]

            testing_rows = [
                row
                for row in visible_candidates
                if row["clip_id"]
                == held_out_clip
            ]

            training_labels = np.asarray(
                [
                    row["positive"]
                    for row in training_rows
                ],
                dtype=np.float64,
            )

            coefficients, means, scales = (
                fit_logistic(
                    build_matrix(
                        training_rows,
                        feature_names,
                    ),
                    training_labels,
                )
            )

            probabilities = (
                predict_probabilities(
                    build_matrix(
                        testing_rows,
                        feature_names,
                    ),
                    coefficients,
                    means,
                    scales,
                )
            )

            score_lookup = {
                row["candidate_id"]:
                    float(probability)
                for row, probability
                in zip(
                    testing_rows,
                    probabilities,
                )
            }

            metrics = evaluate(
                visible_by_clip[
                    held_out_clip
                ],
                grouped_visible,
                score_lookup,
            )

            model_top1.update(
                metrics[
                    "_top1_by_frame"
                ]
            )

            per_clip[
                held_out_clip
            ] = {
                key: value
                for key, value
                in metrics.items()
                if not key.startswith("_")
            }

            aggregate_oracle += (
                metrics[
                    "oracle_frames"
                ]
            )

            for top_k in TOP_K:
                aggregate_hits[
                    top_k
                ] += metrics[
                    "top_k"
                ][str(top_k)][
                    "hits"
                ]

            coefficients_by_fold[
                held_out_clip
            ] = {
                name:
                    round(
                        float(
                            coefficients[
                                index + 1
                            ]
                        ),
                        6,
                    )
                for index, name
                in enumerate(
                    feature_names
                )
            }

            for row, probability in zip(
                testing_rows,
                probabilities,
            ):
                prediction_rows.append({
                    "model":
                        model_name,
                    "held_out_clip":
                        held_out_clip,
                    "clip_id":
                        row["clip_id"],
                    "local_frame":
                        row["local_frame"],
                    "candidate_id":
                        row["candidate_id"],
                    "original_rank":
                        row["rank"],
                    "probability":
                        round(
                            float(probability),
                            8,
                        ),
                    "positive_20px":
                        row["positive"],
                    **{
                        feature_name:
                            round(
                                float(
                                    row["features"][
                                        feature_name
                                    ]
                                ),
                                8,
                            )
                        for feature_name
                        in OCCUPANCY_FEATURES
                    },
                })

        top1_by_model[
            model_name
        ] = model_top1

        model_reports[
            model_name
        ] = {
            "features":
                list(feature_names),
            "aggregate": {
                "visible_frames":
                    len(visible_frames),
                "oracle_frames":
                    aggregate_oracle,
                "top_k": {
                    str(top_k): {
                        "hits":
                            aggregate_hits[
                                top_k
                            ],
                        "recall":
                            ratio(
                                aggregate_hits[
                                    top_k
                                ],
                                len(
                                    visible_frames
                                ),
                            ),
                        "recoverable_ratio":
                            ratio(
                                aggregate_hits[
                                    top_k
                                ],
                                aggregate_oracle,
                            ),
                    }
                    for top_k in TOP_K
                },
            },
            "per_clip":
                per_clip,
            "coefficients_by_fold":
                coefficients_by_fold,
        }

    intrinsic_hits = model_reports[
        "intrinsic"
    ]["aggregate"]["top_k"]["1"][
        "hits"
    ]

    if intrinsic_hits != 281:
        raise RuntimeError(
            "Reproduction I14B incohérente : "
            f"{intrinsic_hits}"
        )

    candidate_index = {
        row["candidate_id"]:
            row
        for row in visible_candidates
    }

    intrinsic_top1 = top1_by_model[
        "intrinsic"
    ]

    combined_top1 = top1_by_model[
        "intrinsic_plus_occupancy"
    ]

    comparisons: dict[
        str,
        Any,
    ] = {}

    for clip_id in clips:
        corrected = 0
        regressed = 0
        both_correct = 0
        both_wrong = 0

        for frame_key in visible_by_clip[
            clip_id
        ]:
            intrinsic_id = (
                intrinsic_top1.get(
                    frame_key
                )
            )

            combined_id = (
                combined_top1.get(
                    frame_key
                )
            )

            intrinsic_correct = bool(
                intrinsic_id
                and candidate_index[
                    intrinsic_id
                ]["positive"]
            )

            combined_correct = bool(
                combined_id
                and candidate_index[
                    combined_id
                ]["positive"]
            )

            if (
                not intrinsic_correct
                and combined_correct
            ):
                corrected += 1
            elif (
                intrinsic_correct
                and not combined_correct
            ):
                regressed += 1
            elif (
                intrinsic_correct
                and combined_correct
            ):
                both_correct += 1
            else:
                both_wrong += 1

        comparisons[clip_id] = {
            "corrected_by_occupancy":
                corrected,
            "regressed_by_occupancy":
                regressed,
            "net_gain":
                corrected - regressed,
            "both_correct":
                both_correct,
            "both_wrong":
                both_wrong,
        }

    wide_fp_summary: dict[
        str,
        Any,
    ] = {}

    if I14C_PREDICTION_PATH.exists():
        prediction_source_rows = read_csv(
            I14C_PREDICTION_PATH
        )

        wide_fp_ids = {
            str(row["candidate_id"])
            for row in prediction_source_rows
            if (
                str(row.get("strategy", ""))
                == "temporal_abstention"
                and str(row.get("clip_id", ""))
                == "i12a_wide_v61_4"
                and str(row.get("status", ""))
                in {
                    "wrong_visible",
                    "false_positive_invisible",
                }
                and str(
                    row.get(
                        "candidate_id",
                        "",
                    )
                ).strip()
            )
        }

        all_candidate_index = {
            row["candidate_id"]: row
            for row in candidates
        }

        wide_fp_rows = [
            all_candidate_index[
                candidate_id
            ]
            for candidate_id
            in wide_fp_ids
            if candidate_id
            in all_candidate_index
        ]

        wide_positive_rows = [
            row
            for row in visible_candidates
            if (
                row["clip_id"]
                == "i12a_wide_v61_4"
                and row["positive"]
            )
        ]

        wide_fp_summary = {
            "false_positive_candidate_count":
                len(wide_fp_rows),
            "wide_positive_candidate_count":
                len(wide_positive_rows),
            "features": {
                feature_name: {
                    "false_positive_median":
                        median_value(
                            wide_fp_rows,
                            feature_name,
                        ),
                    "ball_candidate_median":
                        median_value(
                            wide_positive_rows,
                            feature_name,
                        ),
                }
                for feature_name
                in OCCUPANCY_FEATURES
            },
        }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        OUTPUT_DIR
        / "i14f_occupancy_predictions.csv"
    )

    report_path = (
        OUTPUT_DIR
        / "i14f_occupancy_report.json"
    )

    write_csv(
        prediction_path,
        prediction_rows,
    )

    report = {
        "experiment":
            (
                "003D_I14F_"
                "candidate_global_occupancy"
            ),
        "protocol": {
            "evaluation":
                "leave-one-clip-out",
            "occupancy_map":
                (
                    "Construite sans GT sur tous les "
                    "candidats du clip, en excluant "
                    "la frame du candidat évalué."
                ),
            "causal":
                False,
        },
        "dataset": {
            "frames":
                len(frame_rows),
            "visible_frames":
                len(visible_frames),
            "all_candidate_rows":
                len(candidates),
            "visible_candidate_rows":
                len(visible_candidates),
            "positive_candidate_rows":
                int(np.sum(labels)),
            "clips":
                clips,
        },
        "feature_audit":
            feature_audit,
        "baseline_current_ranking": {
            key: value
            for key, value
            in baseline.items()
            if not key.startswith("_")
        },
        "models":
            model_reports,
        "intrinsic_vs_occupancy_comparison":
            comparisons,
        "wide_temporal_false_positive_summary":
            wide_fp_summary,
        "artifacts": {
            "predictions":
                prediction_path.name,
            "report":
                report_path.name,
        },
    }

    report_path.write_text(
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
        "I14F_GLOBAL_OCCUPANCY_OK"
    )
    print(
        "frames =",
        len(frame_rows),
    )
    print(
        "visible_frames =",
        len(visible_frames),
    )
    print(
        "all_candidate_rows =",
        len(candidates),
    )
    print(
        "positive_candidate_rows =",
        int(np.sum(labels)),
    )

    print()
    print(
        "=== FEATURES D'OCCUPATION ==="
    )

    for feature_name in OCCUPANCY_FEATURES:
        audit = feature_audit[
            feature_name
        ]

        print(
            feature_name,
            (
                f"auc_sep="
                f"{audit['separation_auc']}"
            ),
            (
                f"direction="
                f"{audit['direction']}"
            ),
            (
                f"ball="
                f"{audit['ball_median']}"
            ),
            (
                f"other="
                f"{audit['other_median']}"
            ),
        )

    print()
    print(
        "=== MODÈLES LOCO ==="
    )

    print(
        "baseline_current",
        (
            "top1="
            f"{baseline['top_k']['1']['hits']}"
        ),
    )

    for model_name, model_report in (
        model_reports.items()
    ):
        print()
        print(model_name)

        for top_k in TOP_K:
            metrics = model_report[
                "aggregate"
            ]["top_k"][str(top_k)]

            print(
                f"  top{top_k:02d}",
                (
                    f"hits="
                    f"{metrics['hits']}"
                ),
                (
                    f"recall="
                    f"{metrics['recall']:.4f}"
                ),
                (
                    f"recoverable="
                    f"{metrics['recoverable_ratio']:.4f}"
                ),
            )

        print(
            "  per_clip_top1 =",
            {
                clip_id:
                    model_report[
                        "per_clip"
                    ][clip_id][
                        "top_k"
                    ]["1"]["hits"]
                for clip_id in clips
            },
        )

    print()
    print(
        "=== INTRINSÈQUE VS OCCUPATION ==="
    )

    for clip_id in clips:
        comparison = comparisons[
            clip_id
        ]

        print(
            clip_id,
            (
                f"corrected="
                f"{comparison['corrected_by_occupancy']}"
            ),
            (
                f"regressed="
                f"{comparison['regressed_by_occupancy']}"
            ),
            (
                f"net_gain="
                f"{comparison['net_gain']}"
            ),
            (
                f"both_wrong="
                f"{comparison['both_wrong']}"
            ),
        )

    if wide_fp_summary:
        print()
        print(
            "=== WIDE : FP TEMPORELS VS BALLE ==="
        )

        print(
            "false_positive_candidates =",
            wide_fp_summary[
                "false_positive_candidate_count"
            ],
        )

        for feature_name in OCCUPANCY_FEATURES:
            values = wide_fp_summary[
                "features"
            ][feature_name]

            print(
                feature_name,
                (
                    f"fp="
                    f"{values['false_positive_median']}"
                ),
                (
                    f"ball="
                    f"{values['ball_candidate_median']}"
                ),
            )

    print()
    print(
        "predictions =",
        prediction_path.resolve(),
    )
    print(
        "report =",
        report_path.resolve(),
    )


if __name__ == "__main__":
    main()
