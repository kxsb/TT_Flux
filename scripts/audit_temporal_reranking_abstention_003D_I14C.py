from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
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
    "runs/_ball_temporal_reranking_003D_I14C"
)

FEATURE_NAMES = (
    "log_area",
    "brightness",
    "motion",
    "fill_ratio",
    "circularity",
    "log_dimension",
    "aspect_symmetry",
)

RADIUS_PX = 20.0
FRAME_COUNT_PER_CLIP = 150
MAX_GAP_FRAMES = 2
MAX_JUMP_PER_FRAME = 58.0
PREDICTION_ERROR_LIMIT = 26.0
MAX_ACCELERATION = 24.0
MAX_TURN_DEGREES = 115.0

STRATEGIES = (
    "current_rank1",
    "intrinsic_top1",
    "static_abstention",
    "temporal_abstention",
)


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


def feature_values(
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


def build_matrix(
    rows: list[dict[str, Any]],
) -> np.ndarray:
    return np.asarray(
        [
            [
                row["features"][name]
                for name in FEATURE_NAMES
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def fit_logistic(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(
            "Aucune ligne d'entraînement."
        )

    matrix = build_matrix(rows)

    labels = np.asarray(
        [
            row["positive"]
            for row in rows
        ],
        dtype=np.float64,
    )

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
            "Deux classes sont nécessaires."
        )

    means = np.mean(
        matrix,
        axis=0,
    )

    scales = np.std(
        matrix,
        axis=0,
    )

    scales[
        scales < 1e-8
    ] = 1.0

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

    sample_weights = np.where(
        labels == 1,
        len(labels)
        / (
            2.0
            * positive_count
        ),
        len(labels)
        / (
            2.0
            * negative_count
        ),
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
                sample_weights
                * (
                    probabilities
                    - labels
                )
            )
            + regularization
            @ coefficients
        )

        curvature = (
            sample_weights
            * probabilities
            * (
                1.0
                - probabilities
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

    return {
        "coefficients":
            coefficients,
        "means":
            means,
        "scales":
            scales,
    }


def predict_probabilities(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
) -> dict[str, float]:
    if not rows:
        return {}

    matrix = build_matrix(rows)

    normalized = (
        matrix
        - model["means"]
    ) / model["scales"]

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
        design
        @ model["coefficients"],
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

    return {
        row["candidate_id"]:
            float(probability)
        for row, probability
        in zip(
            rows,
            probabilities,
        )
    }


def probability_logit(
    probability: float,
) -> float:
    bounded = max(
        1e-6,
        min(
            1.0 - 1e-6,
            probability,
        ),
    )

    return math.log(
        bounded
        / (
            1.0 - bounded
        )
    )


def vector(
    first: dict[str, Any],
    second: dict[str, Any],
) -> tuple[float, float]:
    frame_delta = max(
        1,
        int(second["local_frame"])
        - int(first["local_frame"]),
    )

    return (
        (
            float(second["x"])
            - float(first["x"])
        )
        / frame_delta,
        (
            float(second["y"])
            - float(first["y"])
        )
        / frame_delta,
    )


def vector_norm(
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
    first_norm = vector_norm(first)
    second_norm = vector_norm(second)

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
        min(
            1.0,
            cosine,
        ),
    )

    return math.degrees(
        math.acos(cosine)
    )


def appearance_penalty(
    previous: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    penalty = 0.0

    previous_area = max(
        1.0,
        float(previous["area"]),
    )

    candidate_area = max(
        1.0,
        float(candidate["area"]),
    )

    penalty += min(
        2.0,
        abs(
            math.log(
                candidate_area
                / previous_area
            )
        ),
    )

    penalty += (
        abs(
            float(
                candidate[
                    "mean_brightness"
                ]
            )
            - float(
                previous[
                    "mean_brightness"
                ]
            )
        )
        / 255.0
    )

    previous_size = max(
        1.0,
        float(previous["bbox_w"]),
        float(previous["bbox_h"]),
    )

    candidate_size = max(
        1.0,
        float(candidate["bbox_w"]),
        float(candidate["bbox_h"]),
    )

    penalty += (
        0.5
        * min(
            2.0,
            abs(
                math.log(
                    candidate_size
                    / previous_size
                )
            ),
        )
    )

    return penalty


def transition_score(
    history: list[dict[str, Any]],
    candidate: dict[str, Any],
    probability: float,
) -> float | None:
    if not history:
        return probability_logit(
            probability
        )

    last = history[-1]

    frame_delta = (
        int(candidate["local_frame"])
        - int(last["local_frame"])
    )

    if (
        frame_delta < 1
        or frame_delta
        > MAX_GAP_FRAMES + 1
    ):
        return None

    jump = math.hypot(
        float(candidate["x"])
        - float(last["x"]),
        float(candidate["y"])
        - float(last["y"]),
    )

    jump_per_frame = (
        jump / frame_delta
    )

    if (
        jump_per_frame
        > MAX_JUMP_PER_FRAME
    ):
        return None

    prediction_error = 0.0
    acceleration = 0.0

    if len(history) >= 2:
        previous = history[-2]

        previous_velocity = vector(
            previous,
            last,
        )

        predicted_x = (
            float(last["x"])
            + previous_velocity[0]
            * frame_delta
        )

        predicted_y = (
            float(last["y"])
            + previous_velocity[1]
            * frame_delta
        )

        prediction_error = math.hypot(
            float(candidate["x"])
            - predicted_x,
            float(candidate["y"])
            - predicted_y,
        )

        prediction_limit = (
            PREDICTION_ERROR_LIMIT
            + 4.0
            * (
                frame_delta - 1
            )
        )

        if (
            prediction_error
            > prediction_limit
        ):
            return None

        next_velocity = (
            (
                float(candidate["x"])
                - float(last["x"])
            )
            / frame_delta,
            (
                float(candidate["y"])
                - float(last["y"])
            )
            / frame_delta,
        )

        acceleration = vector_norm(
            (
                next_velocity[0]
                - previous_velocity[0],
                next_velocity[1]
                - previous_velocity[1],
            )
        )

        if (
            acceleration
            > MAX_ACCELERATION
        ):
            return None

        if (
            turn_degrees(
                previous_velocity,
                next_velocity,
            )
            > MAX_TURN_DEGREES
        ):
            return None

    appearance = appearance_penalty(
        last,
        candidate,
    )

    temporal_penalty = (
        0.012 * jump
        + 0.025 * prediction_error
        + 0.012 * acceleration
        + 0.20 * appearance
    )

    return (
        probability_logit(
            probability
        )
        - temporal_penalty
    )


def group_candidates(
    clip_id: str,
    candidates: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for candidate in candidates:
        if (
            candidate["clip_id"]
            != clip_id
        ):
            continue

        grouped[
            int(
                candidate[
                    "local_frame"
                ]
            )
        ].append(candidate)

    for rows in grouped.values():
        rows.sort(
            key=lambda row:
                int(row["rank"])
        )

    return dict(grouped)


def current_rank1_predictions(
    grouped: dict[
        int,
        list[dict[str, Any]],
    ],
) -> list[str | None]:
    predictions: list[
        str | None
    ] = []

    for frame in range(
        FRAME_COUNT_PER_CLIP
    ):
        rows = grouped.get(
            frame,
            [],
        )

        if not rows:
            predictions.append(None)
            continue

        selected = min(
            rows,
            key=lambda row:
                int(row["rank"]),
        )

        predictions.append(
            selected["candidate_id"]
        )

    return predictions


def intrinsic_top1_predictions(
    grouped: dict[
        int,
        list[dict[str, Any]],
    ],
    probabilities: dict[str, float],
) -> list[str | None]:
    predictions: list[
        str | None
    ] = []

    for frame in range(
        FRAME_COUNT_PER_CLIP
    ):
        rows = grouped.get(
            frame,
            [],
        )

        if not rows:
            predictions.append(None)
            continue

        selected = max(
            rows,
            key=lambda row: (
                probabilities.get(
                    row["candidate_id"],
                    0.0,
                ),
                -int(row["rank"]),
            ),
        )

        predictions.append(
            selected["candidate_id"]
        )

    return predictions


def static_abstention_predictions(
    grouped: dict[
        int,
        list[dict[str, Any]],
    ],
    probabilities: dict[str, float],
    threshold: float,
    margin: float,
) -> list[str | None]:
    predictions: list[
        str | None
    ] = []

    for frame in range(
        FRAME_COUNT_PER_CLIP
    ):
        rows = sorted(
            grouped.get(
                frame,
                [],
            ),
            key=lambda row: (
                probabilities.get(
                    row["candidate_id"],
                    0.0,
                ),
                -int(row["rank"]),
            ),
            reverse=True,
        )

        if not rows:
            predictions.append(None)
            continue

        best = rows[0]

        best_probability = (
            probabilities.get(
                best["candidate_id"],
                0.0,
            )
        )

        second_probability = (
            probabilities.get(
                rows[1]["candidate_id"],
                0.0,
            )
            if len(rows) >= 2
            else 0.0
        )

        if (
            best_probability
            < threshold
            or (
                best_probability
                - second_probability
            )
            < margin
        ):
            predictions.append(None)
            continue

        predictions.append(
            best["candidate_id"]
        )

    return predictions


def choose_acquisition(
    rows: list[dict[str, Any]],
    probabilities: dict[str, float],
    threshold: float,
    margin: float,
) -> dict[str, Any] | None:
    ordered = sorted(
        rows,
        key=lambda row: (
            probabilities.get(
                row["candidate_id"],
                0.0,
            ),
            -int(row["rank"]),
        ),
        reverse=True,
    )

    if not ordered:
        return None

    best = ordered[0]

    best_probability = (
        probabilities.get(
            best["candidate_id"],
            0.0,
        )
    )

    second_probability = (
        probabilities.get(
            ordered[1]["candidate_id"],
            0.0,
        )
        if len(ordered) >= 2
        else 0.0
    )

    if (
        best_probability < threshold
        or (
            best_probability
            - second_probability
        ) < margin
    ):
        return None

    return best


def choose_continuation(
    rows: list[dict[str, Any]],
    probabilities: dict[str, float],
    history: list[dict[str, Any]],
    threshold: float,
    score_margin: float,
) -> dict[str, Any] | None:
    scored: list[
        tuple[
            float,
            float,
            dict[str, Any],
        ]
    ] = []

    for row in rows:
        probability = (
            probabilities.get(
                row["candidate_id"],
                0.0,
            )
        )

        if probability < threshold:
            continue

        score = transition_score(
            history,
            row,
            probability,
        )

        if score is None:
            continue

        scored.append(
            (
                score,
                probability,
                row,
            )
        )

    if not scored:
        return None

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
            -int(
                item[2]["rank"]
            ),
        ),
        reverse=True,
    )

    best = scored[0]

    second_score = (
        scored[1][0]
        if len(scored) >= 2
        else -math.inf
    )

    if (
        best[0] - second_score
        < score_margin
    ):
        return None

    return best[2]


def temporal_predictions(
    grouped: dict[
        int,
        list[dict[str, Any]],
    ],
    probabilities: dict[str, float],
    acquire_threshold: float,
    continue_threshold: float,
    acquire_margin: float,
    score_margin: float,
    minimum_acquire_run: int,
) -> list[str | None]:
    predictions: list[
        str | None
    ] = [
        None
        for _ in range(
            FRAME_COUNT_PER_CLIP
        )
    ]

    active = False

    history: list[
        dict[str, Any]
    ] = []

    pending: list[
        dict[str, Any]
    ] = []

    for frame in range(
        FRAME_COUNT_PER_CLIP
    ):
        rows = grouped.get(
            frame,
            [],
        )

        if active:
            continuation = (
                choose_continuation(
                    rows,
                    probabilities,
                    history,
                    continue_threshold,
                    score_margin,
                )
            )

            if continuation is not None:
                predictions[frame] = (
                    continuation[
                        "candidate_id"
                    ]
                )

                history.append(
                    continuation
                )

                history = history[-6:]
                continue

            if (
                history
                and (
                    frame
                    - int(
                        history[-1][
                            "local_frame"
                        ]
                    )
                )
                <= MAX_GAP_FRAMES
            ):
                continue

            active = False
            history = []
            pending = []

        if pending:
            continuation = (
                choose_continuation(
                    rows,
                    probabilities,
                    pending,
                    continue_threshold,
                    score_margin,
                )
            )

            if continuation is not None:
                pending.append(
                    continuation
                )

                if (
                    len(pending)
                    >= minimum_acquire_run
                ):
                    for point in pending:
                        predictions[
                            int(
                                point[
                                    "local_frame"
                                ]
                            )
                        ] = point[
                            "candidate_id"
                        ]

                    active = True
                    history = pending[-6:]
                    pending = []

                continue

            pending = []

        acquisition = (
            choose_acquisition(
                rows,
                probabilities,
                acquire_threshold,
                acquire_margin,
            )
        )

        if acquisition is not None:
            pending = [
                acquisition
            ]

            if minimum_acquire_run <= 1:
                predictions[frame] = (
                    acquisition[
                        "candidate_id"
                    ]
                )

                active = True
                history = [
                    acquisition
                ]
                pending = []

    return predictions


def run_lengths(
    values: list[bool],
) -> list[int]:
    result: list[int] = []
    current = 0

    for value in values:
        if value:
            current += 1
            continue

        if current > 0:
            result.append(current)
            current = 0

    if current > 0:
        result.append(current)

    return result


def f_beta(
    precision: float,
    recall: float,
    beta: float = 0.5,
) -> float:
    beta_squared = beta * beta

    denominator = (
        beta_squared
        * precision
        + recall
    )

    if denominator <= 0:
        return 0.0

    return (
        (
            1.0
            + beta_squared
        )
        * precision
        * recall
        / denominator
    )


def evaluate_one_clip(
    clip_id: str,
    predictions: list[str | None],
    frame_meta: dict[
        int,
        dict[str, Any],
    ],
    candidate_index: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    true_positive = 0
    false_positive = 0
    false_negative = 0

    predicted_frames = 0
    predicted_visible = 0
    invisible_predictions = 0

    visible_frames = 0
    invisible_frames = 0

    correct_flags: list[bool] = []
    wrong_flags: list[bool] = []
    predicted_flags: list[bool] = []

    statuses: list[str] = []

    for frame in range(
        FRAME_COUNT_PER_CLIP
    ):
        meta = frame_meta[frame]

        visible = bool(
            meta["gt_visible"]
        )

        prediction = (
            predictions[frame]
        )

        if visible:
            visible_frames += 1
        else:
            invisible_frames += 1

        if prediction is None:
            if visible:
                false_negative += 1
                status = (
                    "abstain_visible"
                )
            else:
                status = (
                    "abstain_invisible"
                )

            correct_flags.append(False)
            wrong_flags.append(False)
            predicted_flags.append(False)
            statuses.append(status)
            continue

        predicted_frames += 1
        predicted_flags.append(True)

        candidate = candidate_index[
            prediction
        ]

        correct = bool(
            candidate["positive"]
        )

        if visible:
            predicted_visible += 1

        if correct:
            true_positive += 1
            status = "correct"
        else:
            false_positive += 1

            if visible:
                false_negative += 1
                status = (
                    "wrong_visible"
                )
            else:
                invisible_predictions += 1
                status = (
                    "false_positive_invisible"
                )

        correct_flags.append(correct)
        wrong_flags.append(not correct)
        statuses.append(status)

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
        if (
            true_positive
            + false_positive
        ) > 0
        else 0.0
    )

    recall = (
        true_positive
        / visible_frames
        if visible_frames > 0
        else 0.0
    )

    correct_runs = run_lengths(
        correct_flags
    )

    wrong_runs = run_lengths(
        wrong_flags
    )

    predicted_runs = run_lengths(
        predicted_flags
    )

    return {
        "clip_id":
            clip_id,
        "frames":
            FRAME_COUNT_PER_CLIP,
        "visible_frames":
            visible_frames,
        "invisible_frames":
            invisible_frames,
        "predicted_frames":
            predicted_frames,
        "coverage":
            ratio(
                predicted_frames,
                FRAME_COUNT_PER_CLIP,
            ),
        "visible_coverage":
            ratio(
                predicted_visible,
                visible_frames,
            ),
        "true_positive":
            true_positive,
        "false_positive":
            false_positive,
        "false_negative":
            false_negative,
        "precision":
            round(
                precision,
                6,
            ),
        "recall":
            round(
                recall,
                6,
            ),
        "f0_5":
            round(
                f_beta(
                    precision,
                    recall,
                    beta=0.5,
                ),
                6,
            ),
        "invisible_predictions":
            invisible_predictions,
        "invisible_prediction_rate":
            ratio(
                invisible_predictions,
                invisible_frames,
            ),
        "correct_run_count":
            len(correct_runs),
        "correct_run_median":
            (
                round(
                    float(
                        median(
                            correct_runs
                        )
                    ),
                    3,
                )
                if correct_runs
                else 0.0
            ),
        "longest_correct_run":
            max(
                correct_runs,
                default=0,
            ),
        "correct_runs_ge_3":
            sum(
                length >= 3
                for length
                in correct_runs
            ),
        "correct_runs_ge_5":
            sum(
                length >= 5
                for length
                in correct_runs
            ),
        "correct_runs_ge_10":
            sum(
                length >= 10
                for length
                in correct_runs
            ),
        "longest_wrong_run":
            max(
                wrong_runs,
                default=0,
            ),
        "longest_predicted_run":
            max(
                predicted_runs,
                default=0,
            ),
        "_statuses":
            statuses,
        "_correct_runs":
            correct_runs,
        "_wrong_runs":
            wrong_runs,
        "_predicted_runs":
            predicted_runs,
    }


def aggregate_metrics(
    clip_metrics: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    true_positive = sum(
        item["true_positive"]
        for item in clip_metrics
    )

    false_positive = sum(
        item["false_positive"]
        for item in clip_metrics
    )

    false_negative = sum(
        item["false_negative"]
        for item in clip_metrics
    )

    visible_frames = sum(
        item["visible_frames"]
        for item in clip_metrics
    )

    invisible_frames = sum(
        item["invisible_frames"]
        for item in clip_metrics
    )

    predicted_frames = sum(
        item["predicted_frames"]
        for item in clip_metrics
    )

    invisible_predictions = sum(
        item[
            "invisible_predictions"
        ]
        for item in clip_metrics
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
        if (
            true_positive
            + false_positive
        ) > 0
        else 0.0
    )

    recall = (
        true_positive
        / visible_frames
        if visible_frames > 0
        else 0.0
    )

    correct_runs = [
        length
        for item in clip_metrics
        for length in item[
            "_correct_runs"
        ]
    ]

    wrong_runs = [
        length
        for item in clip_metrics
        for length in item[
            "_wrong_runs"
        ]
    ]

    predicted_runs = [
        length
        for item in clip_metrics
        for length in item[
            "_predicted_runs"
        ]
    ]

    return {
        "frames":
            sum(
                item["frames"]
                for item
                in clip_metrics
            ),
        "visible_frames":
            visible_frames,
        "invisible_frames":
            invisible_frames,
        "predicted_frames":
            predicted_frames,
        "coverage":
            ratio(
                predicted_frames,
                sum(
                    item["frames"]
                    for item
                    in clip_metrics
                ),
            ),
        "true_positive":
            true_positive,
        "false_positive":
            false_positive,
        "false_negative":
            false_negative,
        "precision":
            round(
                precision,
                6,
            ),
        "recall":
            round(
                recall,
                6,
            ),
        "f0_5":
            round(
                f_beta(
                    precision,
                    recall,
                    beta=0.5,
                ),
                6,
            ),
        "invisible_predictions":
            invisible_predictions,
        "invisible_prediction_rate":
            ratio(
                invisible_predictions,
                invisible_frames,
            ),
        "correct_run_count":
            len(correct_runs),
        "correct_run_median":
            (
                round(
                    float(
                        median(
                            correct_runs
                        )
                    ),
                    3,
                )
                if correct_runs
                else 0.0
            ),
        "longest_correct_run":
            max(
                correct_runs,
                default=0,
            ),
        "correct_runs_ge_3":
            sum(
                length >= 3
                for length
                in correct_runs
            ),
        "correct_runs_ge_5":
            sum(
                length >= 5
                for length
                in correct_runs
            ),
        "correct_runs_ge_10":
            sum(
                length >= 10
                for length
                in correct_runs
            ),
        "longest_wrong_run":
            max(
                wrong_runs,
                default=0,
            ),
        "longest_predicted_run":
            max(
                predicted_runs,
                default=0,
            ),
    }


def evaluate_many(
    clips: list[str],
    predictions_by_clip: dict[
        str,
        list[str | None],
    ],
    frame_meta_by_clip: dict[
        str,
        dict[int, dict[str, Any]],
    ],
    candidate_index: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    per_clip = {
        clip_id:
            evaluate_one_clip(
                clip_id,
                predictions_by_clip[
                    clip_id
                ],
                frame_meta_by_clip[
                    clip_id
                ],
                candidate_index,
            )
        for clip_id in clips
    }

    overall = aggregate_metrics(
        list(
            per_clip.values()
        )
    )

    public_per_clip = {
        clip_id: {
            key: value
            for key, value
            in metrics.items()
            if not key.startswith("_")
        }
        for clip_id, metrics
        in per_clip.items()
    }

    return {
        "overall":
            overall,
        "per_clip":
            public_per_clip,
        "_raw_per_clip":
            per_clip,
    }


def tuning_key(
    metrics: dict[str, Any],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    return (
        float(metrics["f0_5"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        -float(
            metrics[
                "invisible_prediction_rate"
            ]
        ),
    )


def tune_static(
    clips: list[str],
    grouped_by_clip: dict[
        str,
        dict[int, list[dict[str, Any]]],
    ],
    probabilities_by_clip: dict[
        str,
        dict[str, float],
    ],
    frame_meta_by_clip: dict[
        str,
        dict[int, dict[str, Any]],
    ],
    candidate_index: dict[
        str,
        dict[str, Any],
    ],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    best_parameters: dict[
        str,
        Any,
    ] | None = None

    best_report: dict[
        str,
        Any,
    ] | None = None

    for threshold in (
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ):
        for margin in (
            0.00,
            0.05,
            0.10,
            0.20,
        ):
            predictions = {
                clip_id:
                    static_abstention_predictions(
                        grouped_by_clip[
                            clip_id
                        ],
                        probabilities_by_clip[
                            clip_id
                        ],
                        threshold,
                        margin,
                    )
                for clip_id in clips
            }

            report = evaluate_many(
                clips,
                predictions,
                frame_meta_by_clip,
                candidate_index,
            )

            if (
                best_report is None
                or tuning_key(
                    report["overall"]
                )
                > tuning_key(
                    best_report[
                        "overall"
                    ]
                )
            ):
                best_report = report

                best_parameters = {
                    "threshold":
                        threshold,
                    "margin":
                        margin,
                }

    if (
        best_parameters is None
        or best_report is None
    ):
        raise RuntimeError(
            "Réglage statique impossible."
        )

    return (
        best_parameters,
        best_report,
    )


def tune_temporal(
    clips: list[str],
    grouped_by_clip: dict[
        str,
        dict[int, list[dict[str, Any]]],
    ],
    probabilities_by_clip: dict[
        str,
        dict[str, float],
    ],
    frame_meta_by_clip: dict[
        str,
        dict[int, dict[str, Any]],
    ],
    candidate_index: dict[
        str,
        dict[str, Any],
    ],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    best_parameters: dict[
        str,
        Any,
    ] | None = None

    best_report: dict[
        str,
        Any,
    ] | None = None

    for acquire_threshold in (
        0.45,
        0.55,
        0.65,
        0.75,
    ):
        for continue_threshold in (
            0.25,
            0.35,
            0.45,
            0.55,
        ):
            if (
                continue_threshold
                > acquire_threshold
            ):
                continue

            for acquire_margin in (
                0.00,
                0.05,
                0.10,
            ):
                for score_margin in (
                    0.00,
                    0.15,
                    0.30,
                ):
                    for minimum_run in (
                        2,
                        3,
                    ):
                        predictions = {
                            clip_id:
                                temporal_predictions(
                                    grouped_by_clip[
                                        clip_id
                                    ],
                                    probabilities_by_clip[
                                        clip_id
                                    ],
                                    acquire_threshold,
                                    continue_threshold,
                                    acquire_margin,
                                    score_margin,
                                    minimum_run,
                                )
                            for clip_id
                            in clips
                        }

                        report = evaluate_many(
                            clips,
                            predictions,
                            frame_meta_by_clip,
                            candidate_index,
                        )

                        if (
                            best_report is None
                            or tuning_key(
                                report[
                                    "overall"
                                ]
                            )
                            > tuning_key(
                                best_report[
                                    "overall"
                                ]
                            )
                        ):
                            best_report = report

                            best_parameters = {
                                "acquire_threshold":
                                    acquire_threshold,
                                "continue_threshold":
                                    continue_threshold,
                                "acquire_margin":
                                    acquire_margin,
                                "score_margin":
                                    score_margin,
                                "minimum_acquire_run":
                                    minimum_run,
                            }

    if (
        best_parameters is None
        or best_report is None
    ):
        raise RuntimeError(
            "Réglage temporel impossible."
        )

    return (
        best_parameters,
        best_report,
    )


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

    frame_meta_by_clip: dict[
        str,
        dict[int, dict[str, Any]],
    ] = defaultdict(dict)

    for row in frame_rows:
        clip_id = str(
            row["clip_id"]
        )

        local_frame = parse_int(
            row["local_frame"]
        )

        frame_meta_by_clip[
            clip_id
        ][local_frame] = {
            "gt_visible":
                parse_int(
                    row["gt_visible"]
                ),
            "gt_x":
                parse_float(
                    row.get("gt_x")
                ),
            "gt_y":
                parse_float(
                    row.get("gt_y")
                ),
        }

    clips = sorted(
        frame_meta_by_clip
    )

    if len(clips) != 3:
        raise RuntimeError(
            f"Trois clips attendus : {clips}"
        )

    for clip_id in clips:
        if (
            len(
                frame_meta_by_clip[
                    clip_id
                ]
            )
            != FRAME_COUNT_PER_CLIP
        ):
            raise RuntimeError(
                f"{clip_id} : "
                "150 frames attendues."
            )

    candidates: list[
        dict[str, Any]
    ] = []

    candidate_index: dict[
        str,
        dict[str, Any],
    ] = {}

    candidates_by_clip: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for raw in raw_candidate_rows:
        clip_id = str(
            raw["clip_id"]
        )

        local_frame = parse_int(
            raw["local_frame"]
        )

        distance = parse_float(
            raw.get(
                "distance_to_gt_px"
            )
        )

        gt_visible = parse_int(
            raw["gt_visible"]
        )

        candidate: dict[
            str,
            Any,
        ] = {
            "clip_id":
                clip_id,
            "local_frame":
                local_frame,
            "source_frame":
                parse_int(
                    raw["source_frame"]
                ),
            "candidate_id":
                str(
                    raw["candidate_id"]
                ),
            "rank":
                parse_int(
                    raw["rank"]
                ),
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
                float(
                    raw["fill_ratio"]
                ),
            "circularity":
                float(
                    raw["circularity"]
                ),
            "score":
                float(raw["score"]),
            "gt_visible":
                gt_visible,
            "distance_to_gt_px":
                distance,
            "positive":
                int(
                    gt_visible == 1
                    and distance is not None
                    and distance
                    <= RADIUS_PX
                ),
        }

        candidate["features"] = (
            feature_values(candidate)
        )

        candidates.append(
            candidate
        )

        candidate_index[
            candidate[
                "candidate_id"
            ]
        ] = candidate

        candidates_by_clip[
            clip_id
        ].append(candidate)

    grouped_by_clip = {
        clip_id:
            group_candidates(
                clip_id,
                candidates,
            )
        for clip_id in clips
    }

    all_predictions: dict[
        str,
        dict[
            str,
            list[str | None],
        ],
    ] = {
        strategy: {}
        for strategy in STRATEGIES
    }

    fold_reports: dict[
        str,
        Any,
    ] = {}

    final_probabilities_by_clip: dict[
        str,
        dict[str, float],
    ] = {}

    for held_out_clip in clips:
        training_clips = [
            clip_id
            for clip_id in clips
            if clip_id
            != held_out_clip
        ]

        nested_probabilities: dict[
            str,
            dict[str, float],
        ] = {}

        for tuning_clip in training_clips:
            nested_training_clips = [
                clip_id
                for clip_id
                in training_clips
                if clip_id
                != tuning_clip
            ]

            nested_training_rows = [
                candidate
                for candidate
                in candidates
                if (
                    candidate["clip_id"]
                    in nested_training_clips
                    and candidate[
                        "gt_visible"
                    ] == 1
                )
            ]

            nested_model = fit_logistic(
                nested_training_rows
            )

            tuning_rows = (
                candidates_by_clip[
                    tuning_clip
                ]
            )

            nested_probabilities[
                tuning_clip
            ] = predict_probabilities(
                tuning_rows,
                nested_model,
            )

        (
            static_parameters,
            static_tuning_report,
        ) = tune_static(
            training_clips,
            grouped_by_clip,
            nested_probabilities,
            frame_meta_by_clip,
            candidate_index,
        )

        (
            temporal_parameters,
            temporal_tuning_report,
        ) = tune_temporal(
            training_clips,
            grouped_by_clip,
            nested_probabilities,
            frame_meta_by_clip,
            candidate_index,
        )

        training_rows = [
            candidate
            for candidate
            in candidates
            if (
                candidate["clip_id"]
                in training_clips
                and candidate[
                    "gt_visible"
                ] == 1
            )
        ]

        final_model = fit_logistic(
            training_rows
        )

        held_out_rows = (
            candidates_by_clip[
                held_out_clip
            ]
        )

        held_out_probabilities = (
            predict_probabilities(
                held_out_rows,
                final_model,
            )
        )

        final_probabilities_by_clip[
            held_out_clip
        ] = held_out_probabilities

        all_predictions[
            "current_rank1"
        ][held_out_clip] = (
            current_rank1_predictions(
                grouped_by_clip[
                    held_out_clip
                ]
            )
        )

        all_predictions[
            "intrinsic_top1"
        ][held_out_clip] = (
            intrinsic_top1_predictions(
                grouped_by_clip[
                    held_out_clip
                ],
                held_out_probabilities,
            )
        )

        all_predictions[
            "static_abstention"
        ][held_out_clip] = (
            static_abstention_predictions(
                grouped_by_clip[
                    held_out_clip
                ],
                held_out_probabilities,
                static_parameters[
                    "threshold"
                ],
                static_parameters[
                    "margin"
                ],
            )
        )

        all_predictions[
            "temporal_abstention"
        ][held_out_clip] = (
            temporal_predictions(
                grouped_by_clip[
                    held_out_clip
                ],
                held_out_probabilities,
                temporal_parameters[
                    "acquire_threshold"
                ],
                temporal_parameters[
                    "continue_threshold"
                ],
                temporal_parameters[
                    "acquire_margin"
                ],
                temporal_parameters[
                    "score_margin"
                ],
                temporal_parameters[
                    "minimum_acquire_run"
                ],
            )
        )

        coefficients = (
            final_model[
                "coefficients"
            ]
        )

        fold_reports[
            held_out_clip
        ] = {
            "training_clips":
                training_clips,
            "static_parameters":
                static_parameters,
            "temporal_parameters":
                temporal_parameters,
            "static_tuning_metrics":
                static_tuning_report[
                    "overall"
                ],
            "temporal_tuning_metrics":
                temporal_tuning_report[
                    "overall"
                ],
            "intrinsic_model": {
                "intercept":
                    round(
                        float(
                            coefficients[0]
                        ),
                        6,
                    ),
                "standardized_coefficients":
                    {
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
                            FEATURE_NAMES
                        )
                    },
            },
        }

    strategy_reports: dict[
        str,
        Any,
    ] = {}

    for strategy in STRATEGIES:
        evaluation = evaluate_many(
            clips,
            all_predictions[
                strategy
            ],
            frame_meta_by_clip,
            candidate_index,
        )

        strategy_reports[
            strategy
        ] = {
            "overall":
                evaluation["overall"],
            "per_clip":
                evaluation["per_clip"],
        }

    prediction_rows: list[
        dict[str, Any]
    ] = []

    for strategy in STRATEGIES:
        for clip_id in clips:
            probabilities = (
                final_probabilities_by_clip[
                    clip_id
                ]
            )

            raw_metrics = evaluate_one_clip(
                clip_id,
                all_predictions[
                    strategy
                ][clip_id],
                frame_meta_by_clip[
                    clip_id
                ],
                candidate_index,
            )

            statuses = raw_metrics[
                "_statuses"
            ]

            for frame in range(
                FRAME_COUNT_PER_CLIP
            ):
                candidate_id = (
                    all_predictions[
                        strategy
                    ][clip_id][frame]
                )

                candidate = (
                    candidate_index[
                        candidate_id
                    ]
                    if candidate_id
                    is not None
                    else None
                )

                prediction_rows.append({
                    "strategy":
                        strategy,
                    "clip_id":
                        clip_id,
                    "local_frame":
                        frame,
                    "gt_visible":
                        frame_meta_by_clip[
                            clip_id
                        ][frame][
                            "gt_visible"
                        ],
                    "status":
                        statuses[frame],
                    "candidate_id":
                        candidate_id
                        or "",
                    "candidate_rank":
                        (
                            candidate["rank"]
                            if candidate
                            else ""
                        ),
                    "intrinsic_probability":
                        (
                            round(
                                probabilities.get(
                                    candidate_id,
                                    0.0,
                                ),
                                8,
                            )
                            if candidate_id
                            else ""
                        ),
                    "x":
                        (
                            candidate["x"]
                            if candidate
                            else ""
                        ),
                    "y":
                        (
                            candidate["y"]
                            if candidate
                            else ""
                        ),
                    "distance_to_gt_px":
                        (
                            round(
                                float(
                                    candidate[
                                        "distance_to_gt_px"
                                    ]
                                ),
                                6,
                            )
                            if (
                                candidate
                                and candidate[
                                    "distance_to_gt_px"
                                ]
                                is not None
                            )
                            else ""
                        ),
                    "correct_20px":
                        (
                            candidate[
                                "positive"
                            ]
                            if candidate
                            else 0
                        ),
                })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        OUTPUT_DIR
        / "i14c_temporal_predictions.csv"
    )

    report_path = (
        OUTPUT_DIR
        / "i14c_temporal_reranking_report.json"
    )

    write_csv(
        prediction_path,
        prediction_rows,
    )

    report = {
        "experiment":
            (
                "003D_I14C_"
                "temporal_reranking_abstention"
            ),
        "protocol": {
            "evaluation":
                "leave-one-clip-out",
            "temporal_parameter_selection":
                (
                    "Nested training: each of the two "
                    "training clips is scored by a model "
                    "trained on the other training clip."
                ),
            "optimization_metric":
                "F0.5, then precision, then recall",
            "positive_definition":
                "candidate within 20 px of visible GT",
            "invisible_frames_used_for_training":
                False,
        },
        "dataset": {
            "clips":
                clips,
            "frames":
                len(frame_rows),
            "visible_frames":
                sum(
                    meta[
                        "gt_visible"
                    ]
                    for clip_meta
                    in frame_meta_by_clip.values()
                    for meta
                    in clip_meta.values()
                ),
            "candidate_rows":
                len(candidates),
        },
        "intrinsic_features":
            list(FEATURE_NAMES),
        "temporal_constraints": {
            "max_gap_frames":
                MAX_GAP_FRAMES,
            "max_jump_per_frame":
                MAX_JUMP_PER_FRAME,
            "prediction_error_limit":
                PREDICTION_ERROR_LIMIT,
            "max_acceleration":
                MAX_ACCELERATION,
            "max_turn_degrees":
                MAX_TURN_DEGREES,
        },
        "folds":
            fold_reports,
        "strategies":
            strategy_reports,
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
        "I14C_TEMPORAL_RERANKING_OK"
    )
    print(
        "frames =",
        len(frame_rows),
    )
    print(
        "candidate_rows =",
        len(candidates),
    )

    print()
    print(
        "=== STRATÉGIES GLOBALES ==="
    )

    for strategy in STRATEGIES:
        metrics = strategy_reports[
            strategy
        ]["overall"]

        print()
        print(strategy)
        print(
            "  predicted =",
            metrics[
                "predicted_frames"
            ],
        )
        print(
            "  coverage =",
            f"{metrics['coverage']:.4f}",
        )
        print(
            "  tp =",
            metrics[
                "true_positive"
            ],
        )
        print(
            "  fp =",
            metrics[
                "false_positive"
            ],
        )
        print(
            "  precision =",
            f"{metrics['precision']:.4f}",
        )
        print(
            "  recall =",
            f"{metrics['recall']:.4f}",
        )
        print(
            "  f0.5 =",
            f"{metrics['f0_5']:.4f}",
        )
        print(
            "  invisible_prediction_rate =",
            (
                f"{metrics['invisible_prediction_rate']:.4f}"
            ),
        )
        print(
            "  longest_correct_run =",
            metrics[
                "longest_correct_run"
            ],
        )
        print(
            "  median_correct_run =",
            metrics[
                "correct_run_median"
            ],
        )
        print(
            "  correct_runs_ge_5 =",
            metrics[
                "correct_runs_ge_5"
            ],
        )
        print(
            "  longest_wrong_run =",
            metrics[
                "longest_wrong_run"
            ],
        )

    print()
    print(
        "=== PARAMÈTRES CHOISIS ==="
    )

    for clip_id in clips:
        fold = fold_reports[
            clip_id
        ]

        print()
        print(
            clip_id,
        )
        print(
            "  static =",
            fold[
                "static_parameters"
            ],
        )
        print(
            "  temporal =",
            fold[
                "temporal_parameters"
            ],
        )

    print()
    print(
        "=== TEMPORAL PAR CLIP ==="
    )

    temporal_per_clip = (
        strategy_reports[
            "temporal_abstention"
        ]["per_clip"]
    )

    for clip_id in clips:
        metrics = temporal_per_clip[
            clip_id
        ]

        print(
            clip_id,
            (
                f"predicted="
                f"{metrics['predicted_frames']}"
            ),
            (
                f"tp="
                f"{metrics['true_positive']}"
            ),
            (
                f"fp="
                f"{metrics['false_positive']}"
            ),
            (
                f"precision="
                f"{metrics['precision']:.4f}"
            ),
            (
                f"recall="
                f"{metrics['recall']:.4f}"
            ),
            (
                f"longest_correct="
                f"{metrics['longest_correct_run']}"
            ),
            (
                f"longest_wrong="
                f"{metrics['longest_wrong_run']}"
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
