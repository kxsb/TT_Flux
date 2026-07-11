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

OUTPUT_DIR = Path(
    "runs/_ball_candidate_reranking_003D_I14B"
)

RADIUS_PX = 20.0
TOP_K = (1, 3, 5, 10, 24)

FEATURE_SETS = {
    "intrinsic": (
        "log_area",
        "brightness",
        "motion",
        "fill_ratio",
        "circularity",
        "log_dimension",
        "aspect_symmetry",
    ),
    "intrinsic_plus_current": (
        "log_area",
        "brightness",
        "motion",
        "fill_ratio",
        "circularity",
        "log_dimension",
        "aspect_symmetry",
        "current_score",
        "rank_normalized",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def value(raw: Any) -> float | None:
    text = str(raw or "").strip().replace(",", ".")

    if not text:
        return None

    try:
        result = float(text)
    except ValueError:
        return None

    return result if math.isfinite(result) else None


def integer(raw: Any) -> int:
    result = value(raw)

    if result is None:
        raise ValueError(
            f"Entier invalide : {raw!r}"
        )

    return int(round(result))


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


def features(
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

    rank = max(
        1,
        int(row["rank"]),
    )

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
        "current_score":
            float(row["score"]),
        "rank_normalized":
            (rank - 1) / 23.0,
    }


def matrix(
    rows: list[dict[str, Any]],
    names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        [
            [
                row["features"][name]
                for name in names
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0)

    scales[scales < 1e-8] = 1.0

    normalized = (
        x - means
    ) / scales

    design = np.column_stack(
        (
            np.ones(len(normalized)),
            normalized,
        )
    )

    positive_count = int(
        np.sum(y == 1)
    )

    negative_count = int(
        np.sum(y == 0)
    )

    if (
        positive_count == 0
        or negative_count == 0
    ):
        raise RuntimeError(
            "Deux classes sont nécessaires."
        )

    weights = np.where(
        y == 1,
        len(y) / (2.0 * positive_count),
        len(y) / (2.0 * negative_count),
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

    for _ in range(50):
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
                    probabilities - y
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

        if np.linalg.norm(step) < 1e-7:
            break

    return (
        coefficients,
        means,
        scales,
    )


def predict(
    x: np.ndarray,
    coefficients: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    normalized = (
        x - means
    ) / scales

    design = np.column_stack(
        (
            np.ones(len(normalized)),
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


def auc(
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

    rank_sum = float(
        np.sum(
            ranks[labels == 1]
        )
    )

    result = (
        rank_sum
        - positives
        * (positives + 1)
        / 2.0
    ) / (
        positives * negatives
    )

    return float(result)


def evaluate(
    frame_keys: list[
        tuple[str, int]
    ],
    grouped: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ],
    score_key: str,
) -> dict[str, Any]:
    hits = {
        top_k: 0
        for top_k in TOP_K
    }

    oracle_frames = 0

    for frame_key in frame_keys:
        candidates = grouped.get(
            frame_key,
            [],
        )

        if any(
            candidate["positive"]
            for candidate in candidates
        ):
            oracle_frames += 1

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                float(
                    candidate[score_key]
                ),
                float(
                    candidate["score"]
                ),
                -int(
                    candidate["rank"]
                ),
            ),
            reverse=True,
        )

        for top_k in TOP_K:
            if any(
                candidate["positive"]
                for candidate
                in ranked[:top_k]
            ):
                hits[top_k] += 1

    return {
        "visible_frames":
            len(frame_keys),
        "oracle_frames":
            oracle_frames,
        "oracle_recall":
            rate(
                oracle_frames,
                len(frame_keys),
            ),
        "top_k": {
            str(top_k): {
                "hits":
                    hits[top_k],
                "recall":
                    rate(
                        hits[top_k],
                        len(frame_keys),
                    ),
                "recoverable_ratio":
                    rate(
                        hits[top_k],
                        oracle_frames,
                    ),
            }
            for top_k in TOP_K
        },
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
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
    frame_rows = read_csv(
        FRAME_PATH
    )

    raw_candidates = read_csv(
        CANDIDATE_PATH
    )

    visible_frames: list[
        tuple[str, int]
    ] = []

    visible_by_clip: dict[
        str,
        list[tuple[str, int]],
    ] = defaultdict(list)

    for row in frame_rows:
        if integer(
            row["gt_visible"]
        ) != 1:
            continue

        frame_key = (
            str(row["clip_id"]),
            integer(
                row["local_frame"]
            ),
        )

        visible_frames.append(
            frame_key
        )

        visible_by_clip[
            frame_key[0]
        ].append(frame_key)

    if len(visible_frames) != 416:
        raise RuntimeError(
            "416 frames visibles attendues."
        )

    visible_set = set(
        visible_frames
    )

    candidates: list[
        dict[str, Any]
    ] = []

    grouped: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for raw in raw_candidates:
        frame_key = (
            str(raw["clip_id"]),
            integer(
                raw["local_frame"]
            ),
        )

        if frame_key not in visible_set:
            continue

        distance = value(
            raw.get(
                "distance_to_gt_px"
            )
        )

        row: dict[str, Any] = {
            "clip_id":
                frame_key[0],
            "local_frame":
                frame_key[1],
            "candidate_id":
                str(raw["candidate_id"]),
            "rank":
                integer(raw["rank"]),
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
                float(
                    raw["fill_ratio"]
                ),
            "circularity":
                float(
                    raw["circularity"]
                ),
            "score":
                float(raw["score"]),
            "distance_to_gt_px":
                distance,
            "positive":
                int(
                    distance is not None
                    and distance <= RADIUS_PX
                ),
        }

        row["features"] = features(row)

        # Baseline : le rang 1 reçoit le score le plus élevé.
        row["baseline_score"] = -float(
            row["rank"]
        )

        candidates.append(row)
        grouped[frame_key].append(row)

    clips = sorted(
        visible_by_clip
    )

    labels = np.asarray(
        [
            row["positive"]
            for row in candidates
        ],
        dtype=np.int64,
    )

    baseline = evaluate(
        visible_frames,
        grouped,
        "baseline_score",
    )

    feature_audit: dict[
        str,
        Any,
    ] = {}

    all_feature_names = tuple(
        dict.fromkeys(
            name
            for names
            in FEATURE_SETS.values()
            for name in names
        )
    )

    for name in all_feature_names:
        values = np.asarray(
            [
                row["features"][name]
                for row in candidates
            ],
            dtype=np.float64,
        )

        raw_auc = auc(
            values,
            labels,
        )

        feature_audit[name] = {
            "auc":
                round(raw_auc, 6)
                if raw_auc is not None
                else None,
            "separation_auc":
                round(
                    max(
                        raw_auc,
                        1.0 - raw_auc,
                    ),
                    6,
                )
                if raw_auc is not None
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

    for (
        model_name,
        feature_names,
    ) in FEATURE_SETS.items():
        per_clip: dict[
            str,
            Any,
        ] = {}

        for held_out_clip in clips:
            training = [
                row
                for row in candidates
                if row["clip_id"]
                != held_out_clip
            ]

            testing = [
                row
                for row in candidates
                if row["clip_id"]
                == held_out_clip
            ]

            train_x = matrix(
                training,
                feature_names,
            )

            train_y = np.asarray(
                [
                    row["positive"]
                    for row in training
                ],
                dtype=np.float64,
            )

            (
                coefficients,
                means,
                scales,
            ) = fit_logistic(
                train_x,
                train_y,
            )

            probabilities = predict(
                matrix(
                    testing,
                    feature_names,
                ),
                coefficients,
                means,
                scales,
            )

            for row, probability in zip(
                testing,
                probabilities,
            ):
                score_name = (
                    f"score_{model_name}"
                )

                row[score_name] = float(
                    probability
                )

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
                    "original_score":
                        row["score"],
                    "predicted_probability":
                        round(
                            float(
                                probability
                            ),
                            8,
                        ),
                    "positive_20px":
                        row["positive"],
                    "distance_to_gt_px":
                        (
                            row[
                                "distance_to_gt_px"
                            ]
                            if row[
                                "distance_to_gt_px"
                            ]
                            is not None
                            else ""
                        ),
                })

            per_clip[
                held_out_clip
            ] = evaluate(
                visible_by_clip[
                    held_out_clip
                ],
                grouped,
                f"score_{model_name}",
            )

        aggregate_hits = {
            top_k:
                sum(
                    per_clip[clip][
                        "top_k"
                    ][str(top_k)][
                        "hits"
                    ]
                    for clip in clips
                )
            for top_k in TOP_K
        }

        oracle_frames = sum(
            per_clip[clip][
                "oracle_frames"
            ]
            for clip in clips
        )

        model_reports[
            model_name
        ] = {
            "features":
                list(feature_names),
            "aggregate": {
                "visible_frames":
                    len(visible_frames),
                "oracle_frames":
                    oracle_frames,
                "top_k": {
                    str(top_k): {
                        "hits":
                            aggregate_hits[
                                top_k
                            ],
                        "recall":
                            rate(
                                aggregate_hits[
                                    top_k
                                ],
                                len(
                                    visible_frames
                                ),
                            ),
                        "recoverable_ratio":
                            rate(
                                aggregate_hits[
                                    top_k
                                ],
                                oracle_frames,
                            ),
                    }
                    for top_k in TOP_K
                },
            },
            "per_clip":
                per_clip,
        }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        OUTPUT_DIR
        / "i14b_loco_predictions.csv"
    )

    report_path = (
        OUTPUT_DIR
        / "i14b_static_reranking_report.json"
    )

    write_csv(
        prediction_path,
        prediction_rows,
    )

    report = {
        "experiment":
            "003D_I14B_static_reranking",
        "protocol":
            "leave-one-clip-out",
        "radius_px":
            RADIUS_PX,
        "dataset": {
            "visible_frames":
                len(visible_frames),
            "candidate_rows":
                len(candidates),
            "positive_rows":
                int(np.sum(labels)),
            "clips":
                clips,
        },
        "baseline":
            baseline,
        "feature_audit":
            feature_audit,
        "models":
            model_reports,
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
        "I14B_STATIC_RERANKING_OK"
    )
    print(
        "visible_frames =",
        len(visible_frames),
    )
    print(
        "candidate_rows =",
        len(candidates),
    )
    print(
        "positive_candidate_rows =",
        int(np.sum(labels)),
    )

    print()
    print(
        "=== BASELINE ACTUELLE ==="
    )

    for top_k in TOP_K:
        result = baseline[
            "top_k"
        ][str(top_k)]

        print(
            f"top{top_k:02d}",
            f"hits={result['hits']}",
            (
                f"recall="
                f"{result['recall']:.4f}"
            ),
        )

    print()
    print(
        "=== SÉPARATION FEATURES ==="
    )

    for name, audit in sorted(
        feature_audit.items(),
        key=lambda item:
            item[1][
                "separation_auc"
            ],
        reverse=True,
    ):
        print(
            name,
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

    for (
        model_name,
        model_report,
    ) in model_reports.items():
        print()
        print(model_name)

        for top_k in TOP_K:
            result = model_report[
                "aggregate"
            ]["top_k"][str(top_k)]

            print(
                f"  top{top_k:02d}",
                f"hits={result['hits']}",
                (
                    f"recall="
                    f"{result['recall']:.4f}"
                ),
                (
                    f"recoverable="
                    f"{result['recoverable_ratio']:.4f}"
                ),
            )

        print(
            "  per_clip_top1 =",
            {
                clip:
                    model_report[
                        "per_clip"
                    ][clip][
                        "top_k"
                    ]["1"]["hits"]
                for clip in clips
            },
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
