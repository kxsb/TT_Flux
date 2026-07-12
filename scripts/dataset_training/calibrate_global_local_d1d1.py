from __future__ import annotations

import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


PREDICTIONS_PATH = Path(
    os.environ["TTFLUX_D1D0_PREDICTIONS"]
)

D1D0_SUMMARY_PATH = Path(
    os.environ["TTFLUX_D1D0_SUMMARY"]
)

OUTPUT_DIR = Path(
    os.environ["TTFLUX_D1D1_OUTPUT"]
)

DISTANCE_THRESHOLD = 20.0

GLOBAL_THRESHOLDS = (
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
)

LOCAL_THRESHOLDS = (
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
)


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def parse_float(
    value: Any,
) -> float:
    number = float(
        str(value).strip()
    )

    if not math.isfinite(number):
        raise ValueError(
            f"Nombre non fini : {value!r}"
        )

    return number


def parse_int(
    value: Any,
) -> int:
    number = parse_float(value)
    rounded = round(number)

    if abs(number - rounded) > 1e-6:
        raise ValueError(
            f"Entier non exact : {value!r}"
        )

    return int(rounded)


def harmonic_f1(
    precision: float,
    recall: float,
) -> float:
    if precision + recall == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def normalize_rows(
    source_rows: list[
        dict[str, str]
    ],
) -> list[dict[str, Any]]:
    rows = []

    for source in source_rows:
        rows.append(
            {
                "match_id":
                    source["match_id"],
                "rally_id":
                    source["rally_id"],
                "frame_index":
                    parse_int(
                        source[
                            "frame_index"
                        ]
                    ),
                "visibility":
                    parse_int(
                        source[
                            "visibility"
                        ]
                    ),
                "gt_x":
                    (
                        parse_float(
                            source["gt_x"]
                        )
                        if source[
                            "gt_x"
                        ].strip()
                        else None
                    ),
                "gt_y":
                    (
                        parse_float(
                            source["gt_y"]
                        )
                        if source[
                            "gt_y"
                        ].strip()
                        else None
                    ),
                "width":
                    parse_int(
                        source[
                            "source_width"
                        ]
                    ),
                "height":
                    parse_int(
                        source[
                            "source_height"
                        ]
                    ),
                "global_x":
                    parse_float(
                        source["global_x"]
                    ),
                "global_y":
                    parse_float(
                        source["global_y"]
                    ),
                "global_conf":
                    parse_float(
                        source[
                            "global_conf"
                        ]
                    ),
                "refined_x":
                    parse_float(
                        source["refined_x"]
                    ),
                "refined_y":
                    parse_float(
                        source["refined_y"]
                    ),
                "local_conf":
                    parse_float(
                        source[
                            "local_conf"
                        ]
                    ),
            }
        )

    return rows


def in_bounds(
    x: float,
    y: float,
    width: int,
    height: int,
) -> bool:
    return (
        0 <= x < width
        and 0 <= y < height
    )


def select_prediction(
    row: dict[str, Any],
    policy: str,
    global_threshold: float,
    local_threshold: float,
) -> tuple[
    str,
    float,
    float,
] | None:
    global_valid = (
        row["global_conf"]
        >= global_threshold
        and in_bounds(
            row["global_x"],
            row["global_y"],
            row["width"],
            row["height"],
        )
    )

    refined_valid = (
        row["global_conf"]
        >= global_threshold
        and row["local_conf"]
        >= local_threshold
        and in_bounds(
            row["refined_x"],
            row["refined_y"],
            row["width"],
            row["height"],
        )
    )

    if policy == "global_only":
        if not global_valid:
            return None

        return (
            "global",
            row["global_x"],
            row["global_y"],
        )

    if policy == "refined_only":
        if not refined_valid:
            return None

        return (
            "refined",
            row["refined_x"],
            row["refined_y"],
        )

    if policy == "refined_then_global":
        if refined_valid:
            return (
                "refined",
                row["refined_x"],
                row["refined_y"],
            )

        if global_valid:
            return (
                "global_fallback",
                row["global_x"],
                row["global_y"],
            )

        return None

    raise ValueError(
        f"Politique inconnue : {policy}"
    )


def evaluate(
    rows: list[dict[str, Any]],
    policy: str,
    global_threshold: float,
    local_threshold: float,
) -> dict[str, Any]:
    visible_count = sum(
        1
        for row in rows
        if row["visibility"] == 1
    )

    invisible_count = sum(
        1
        for row in rows
        if row["visibility"] == 0
    )

    valid_total = 0
    valid_visible = 0
    valid_invisible = 0

    hits = 0

    refined_count = 0
    fallback_count = 0

    errors = []

    for row in rows:
        selected = select_prediction(
            row,
            policy,
            global_threshold,
            local_threshold,
        )

        if selected is None:
            continue

        source, predicted_x, predicted_y = (
            selected
        )

        valid_total += 1

        if source == "refined":
            refined_count += 1
        elif source == "global_fallback":
            fallback_count += 1

        if row["visibility"] == 0:
            valid_invisible += 1
            continue

        valid_visible += 1

        if (
            row["gt_x"] is None
            or row["gt_y"] is None
        ):
            raise RuntimeError(
                "GT visible absente : "
                f"{row['match_id']}/"
                f"{row['rally_id']}/"
                f"{row['frame_index']}"
            )

        error = math.hypot(
            predicted_x - row["gt_x"],
            predicted_y - row["gt_y"],
        )

        errors.append(error)

        if error <= DISTANCE_THRESHOLD:
            hits += 1

    precision = (
        hits / valid_total
        if valid_total
        else 0.0
    )

    recall = (
        hits / visible_count
        if visible_count
        else 0.0
    )

    return {
        "policy":
            policy,
        "global_threshold":
            global_threshold,
        "local_threshold":
            (
                local_threshold
                if policy
                != "global_only"
                else None
            ),
        "visible_frames":
            visible_count,
        "invisible_frames":
            invisible_count,
        "valid_total":
            valid_total,
        "valid_visible":
            valid_visible,
        "valid_invisible":
            valid_invisible,
        "valid_rate_visible": (
            valid_visible
            / visible_count
            if visible_count
            else None
        ),
        "false_positive_rate_invisible": (
            valid_invisible
            / invisible_count
            if invisible_count
            else None
        ),
        "hits_at_20px":
            hits,
        "precision_at_20px":
            precision,
        "recall_at_20px":
            recall,
        "f1_at_20px":
            harmonic_f1(
                precision,
                recall,
            ),
        "median_valid_error_px": (
            statistics.median(errors)
            if errors
            else None
        ),
        "refined_prediction_count":
            refined_count,
        "global_fallback_count":
            fallback_count,
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
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def best_row(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["f1_at_20px"],
            row["recall_at_20px"],
            row["precision_at_20px"],
            -row[
                "false_positive_rate_invisible"
            ],
        ),
    )


def compact(
    row: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "policy",
        "global_threshold",
        "local_threshold",
        "valid_total",
        "valid_visible",
        "valid_invisible",
        "false_positive_rate_invisible",
        "hits_at_20px",
        "precision_at_20px",
        "recall_at_20px",
        "f1_at_20px",
        "median_valid_error_px",
        "refined_prediction_count",
        "global_fallback_count",
    )

    return {
        key: row[key]
        for key in keys
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_rows = read_csv(
        PREDICTIONS_PATH
    )

    rows = normalize_rows(
        source_rows
    )

    d1d0 = json.loads(
        D1D0_SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    if len(rows) != 7714:
        raise RuntimeError(
            "7714 prédictions attendues, "
            f"obtenu={len(rows)}"
        )

    current_global = evaluate(
        rows,
        "global_only",
        0.05,
        0.05,
    )

    current_refined = evaluate(
        rows,
        "refined_only",
        0.05,
        0.05,
    )

    current_fallback = evaluate(
        rows,
        "refined_then_global",
        0.05,
        0.05,
    )

    expected_global_f1 = float(
        d1d0["global"][
            "f1_at_20px"
        ]
    )

    expected_refined_f1 = float(
        d1d0[
            "refined_legacy_local"
        ]["f1_at_20px"]
    )

    if abs(
        current_global["f1_at_20px"]
        - expected_global_f1
    ) > 1e-6:
        raise RuntimeError(
            "La reproduction globale D1D0 "
            "ne correspond pas."
        )

    if abs(
        current_refined["f1_at_20px"]
        - expected_refined_f1
    ) > 1e-6:
        raise RuntimeError(
            "La reproduction raffinée D1D0 "
            "ne correspond pas."
        )

    sweep_rows = []

    for global_threshold in (
        GLOBAL_THRESHOLDS
    ):
        sweep_rows.append(
            evaluate(
                rows,
                "global_only",
                global_threshold,
                0.05,
            )
        )

        for local_threshold in (
            LOCAL_THRESHOLDS
        ):
            sweep_rows.append(
                evaluate(
                    rows,
                    "refined_only",
                    global_threshold,
                    local_threshold,
                )
            )

            sweep_rows.append(
                evaluate(
                    rows,
                    "refined_then_global",
                    global_threshold,
                    local_threshold,
                )
            )

    by_policy = {}

    for policy in (
        "global_only",
        "refined_only",
        "refined_then_global",
    ):
        policy_rows = [
            row
            for row in sweep_rows
            if row["policy"] == policy
        ]

        by_policy[policy] = best_row(
            policy_rows
        )

    conservative_candidates = [
        row
        for row in sweep_rows
        if (
            row["precision_at_20px"]
            >= 0.90
            and row[
                "false_positive_rate_invisible"
            ]
            <= current_refined[
                "false_positive_rate_invisible"
            ]
            + 1e-12
        )
    ]

    if not conservative_candidates:
        raise RuntimeError(
            "Aucune calibration conservatrice."
        )

    best_conservative = best_row(
        conservative_candidates
    )

    best_overall = best_row(
        sweep_rows
    )

    current_f1 = current_refined[
        "f1_at_20px"
    ]

    conservative_gain = (
        best_conservative[
            "f1_at_20px"
        ]
        - current_f1
    )

    fallback_gain = (
        current_fallback[
            "f1_at_20px"
        ]
        - current_f1
    )

    if conservative_gain >= 0.01:
        decision = (
            "calibration_useful_before_local_training"
        )
    elif fallback_gain >= 0.005:
        decision = (
            "fallback_policy_useful_before_local_training"
        )
    else:
        decision = (
            "legacy_local_already_near_calibrated"
        )

    sweep_path = (
        OUTPUT_DIR
        / "calibration_sweep.csv"
    )

    write_csv(
        sweep_path,
        sweep_rows,
    )

    summary = {
        "schema_version": 1,
        "experiment":
            "DSET_D1D1_global_local_calibration",
        "dataset": {
            "split":
                "validation",
            "match_ids": [
                20,
                21,
            ],
            "frames":
                len(rows),
            "test_match_ids_excluded": [
                22,
                23,
                24,
                25,
            ],
        },
        "current_threshold_005": {
            "global_only":
                compact(
                    current_global
                ),
            "refined_only":
                compact(
                    current_refined
                ),
            "refined_then_global":
                compact(
                    current_fallback
                ),
        },
        "best_by_policy": {
            policy:
                compact(row)
            for policy, row
            in by_policy.items()
        },
        "best_overall":
            compact(
                best_overall
            ),
        "best_conservative": {
            "constraints": {
                "precision_at_20px_min":
                    0.90,
                "false_positive_rate_invisible_max":
                    current_refined[
                        "false_positive_rate_invisible"
                    ],
            },
            "metrics":
                compact(
                    best_conservative
                ),
        },
        "gains_vs_current_refined": {
            "fallback_same_thresholds_f1":
                fallback_gain,
            "best_conservative_f1":
                conservative_gain,
            "best_overall_f1": (
                best_overall[
                    "f1_at_20px"
                ]
                - current_f1
            ),
        },
        "decision":
            decision,
        "warning": (
            "Seuils calibrés uniquement sur les "
            "matchs validation 20 et 21. "
            "Le split test reste gelé."
        ),
        "artifacts": {
            "calibration_sweep_csv":
                str(sweep_path),
        },
    }

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("")
    print(
        "DSET_D1D1_CALIBRATION_GENERATED"
    )

    print("")
    print(
        "=== SEUILS ACTUELS 0.05 ==="
    )

    print(
        json.dumps(
            summary[
                "current_threshold_005"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    print("")
    print(
        "=== MEILLEUR PAR POLITIQUE ==="
    )

    print(
        json.dumps(
            summary[
                "best_by_policy"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    print("")
    print(
        "=== MEILLEUR CONSERVATEUR ==="
    )

    print(
        json.dumps(
            summary[
                "best_conservative"
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    print("")
    print(
        f"decision={decision}"
    )

    print(f"summary={summary_path}")
    print(f"sweep={sweep_path}")


if __name__ == "__main__":
    main()