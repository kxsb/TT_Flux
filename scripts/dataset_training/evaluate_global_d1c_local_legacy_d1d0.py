from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(os.environ["TTFLUX_ROOT"])
CHECKPOINT_PATH = Path(os.environ["TTFLUX_TTNET_CHECKPOINT"])
I10_SCRIPT = Path(os.environ["TTFLUX_I10_SCRIPT"])
RALLIES_MANIFEST = Path(os.environ["TTFLUX_BLURBALL_RALLIES"])
FRAMES_MANIFEST = Path(os.environ["TTFLUX_BLURBALL_FRAMES"])
OUTPUT_DIR = Path(os.environ["TTFLUX_D1D0_OUTPUT"])

INPUT_WIDTH = 320
INPUT_HEIGHT = 128

CANONICAL_WIDTH = 1920
CANONICAL_HEIGHT = 1080

SEQUENCE_LENGTH = 9
HISTORY_SPAN = 8

DISTANCE_THRESHOLD = 20.0


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


def parse_float(value: Any) -> float | None:
    text = str(
        value if value is not None else ""
    ).strip()

    if text == "":
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def parse_int(value: Any) -> int:
    number = parse_float(value)

    if number is None:
        raise ValueError(
            f"Entier invalide : {value!r}"
        )

    rounded = round(number)

    if abs(number - rounded) > 1e-6:
        raise ValueError(
            f"Entier non exact : {value!r}"
        )

    return int(rounded)


def load_i10_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ttflux_i10_reference_d1d0",
        I10_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Import impossible : {I10_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def load_resized_video(
    path: Path,
) -> tuple[
    list[np.ndarray],
    int,
    int,
    float,
]:
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Vidéo illisible : {path}"
        )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    frames = []

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            resized = cv2.resize(
                rgb,
                (INPUT_WIDTH, INPUT_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )

            frames.append(resized)
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Dimensions vidéo invalides : {path}"
        )

    if len(frames) < SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Vidéo trop courte : {path}"
        )

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    return frames, width, height, fps


def rally_key(
    row: dict[str, str],
) -> tuple[int, int]:
    return (
        parse_int(row["match_id"]),
        parse_int(row["rally_id"]),
    )


def eligible_annotations(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    eligible = []

    for row in rows:
        frame_index = parse_int(
            row["frame_index"]
        )

        if frame_index < HISTORY_SPAN:
            continue

        visibility = parse_int(
            row["visibility"]
        )

        if visibility == 0:
            eligible.append(row)
            continue

        x = parse_float(row["x_px"])
        y = parse_float(row["y_px"])

        if x is None or y is None:
            continue

        width = parse_int(
            row["video_width"]
        )

        height = parse_int(
            row["video_height"]
        )

        if 0 <= x < width and 0 <= y < height:
            eligible.append(row)

    return eligible


def percentile(
    values: list[float],
    ratio: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = ratio * (len(ordered) - 1)

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


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
        / (precision + recall)
    )


def stage_metrics(
    rows: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    visible_rows = [
        row
        for row in rows
        if row["visibility"] == 1
    ]

    invisible_rows = [
        row
        for row in rows
        if row["visibility"] == 0
    ]

    valid_rows = [
        row
        for row in rows
        if row[f"{prefix}_valid"]
    ]

    valid_visible = [
        row
        for row in visible_rows
        if row[f"{prefix}_valid"]
    ]

    valid_invisible = [
        row
        for row in invisible_rows
        if row[f"{prefix}_valid"]
    ]

    raw_hits = sum(
        1
        for row in visible_rows
        if (
            row[f"{prefix}_error_px"]
            is not None
            and float(
                row[f"{prefix}_error_px"]
            ) <= DISTANCE_THRESHOLD
        )
    )

    valid_hits = sum(
        1
        for row in valid_visible
        if (
            row[f"{prefix}_error_px"]
            is not None
            and float(
                row[f"{prefix}_error_px"]
            ) <= DISTANCE_THRESHOLD
        )
    )

    precision = (
        valid_hits / len(valid_rows)
        if valid_rows
        else 0.0
    )

    recall = (
        valid_hits / len(visible_rows)
        if visible_rows
        else 0.0
    )

    raw_errors = [
        float(row[f"{prefix}_error_px"])
        for row in visible_rows
        if row[f"{prefix}_error_px"] is not None
    ]

    valid_errors = [
        float(row[f"{prefix}_error_px"])
        for row in valid_visible
        if row[f"{prefix}_error_px"] is not None
    ]

    return {
        "visible_frames": len(visible_rows),
        "invisible_frames": len(invisible_rows),
        "valid_total": len(valid_rows),
        "valid_visible": len(valid_visible),
        "valid_invisible": len(valid_invisible),
        "valid_rate_visible": (
            len(valid_visible) / len(visible_rows)
            if visible_rows
            else None
        ),
        "false_positive_rate_invisible": (
            len(valid_invisible) / len(invisible_rows)
            if invisible_rows
            else None
        ),
        "raw_hits_at_20px": raw_hits,
        "raw_recall_at_20px": (
            raw_hits / len(visible_rows)
            if visible_rows
            else 0.0
        ),
        "valid_hits_at_20px": valid_hits,
        "precision_at_20px": precision,
        "recall_at_20px": recall,
        "f1_at_20px": harmonic_f1(
            precision,
            recall,
        ),
        "median_raw_error_px": (
            statistics.median(raw_errors)
            if raw_errors
            else None
        ),
        "p90_raw_error_px": percentile(
            raw_errors,
            0.90,
        ),
        "median_valid_error_px": (
            statistics.median(valid_errors)
            if valid_errors
            else None
        ),
        "p90_valid_error_px": percentile(
            valid_errors,
            0.90,
        ),
    }


def transition_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    visible_rows = [
        row
        for row in rows
        if row["visibility"] == 1
    ]

    global_raw_hits = 0
    refined_raw_hits = 0

    global_valid_hits = 0
    refined_valid_hits = 0

    global_hit_lost = 0
    global_miss_recovered = 0

    refined_better = 0
    refined_worse = 0
    refined_equal = 0

    error_deltas = []

    for row in visible_rows:
        global_error = float(
            row["global_error_px"]
        )

        refined_error = float(
            row["refined_error_px"]
        )

        global_raw_hit = (
            global_error
            <= DISTANCE_THRESHOLD
        )

        refined_raw_hit = (
            refined_error
            <= DISTANCE_THRESHOLD
        )

        global_valid_hit = bool(
            row["global_valid"]
            and global_raw_hit
        )

        refined_valid_hit = bool(
            row["refined_valid"]
            and refined_raw_hit
        )

        if global_raw_hit:
            global_raw_hits += 1

        if refined_raw_hit:
            refined_raw_hits += 1

        if global_valid_hit:
            global_valid_hits += 1

        if refined_valid_hit:
            refined_valid_hits += 1

        if (
            global_valid_hit
            and not refined_valid_hit
        ):
            global_hit_lost += 1

        if (
            not global_valid_hit
            and refined_valid_hit
        ):
            global_miss_recovered += 1

        delta = (
            refined_error
            - global_error
        )

        error_deltas.append(delta)

        if delta < -1e-6:
            refined_better += 1
        elif delta > 1e-6:
            refined_worse += 1
        else:
            refined_equal += 1

    global_valid_refined_invalid = sum(
        1
        for row in rows
        if (
            row["global_valid"]
            and not row["refined_valid"]
        )
    )

    return {
        "visible_frames": len(visible_rows),
        "global_raw_hits_at_20px":
            global_raw_hits,
        "refined_raw_hits_at_20px":
            refined_raw_hits,
        "raw_hit_delta":
            refined_raw_hits
            - global_raw_hits,
        "global_valid_hits_at_20px":
            global_valid_hits,
        "refined_valid_hits_at_20px":
            refined_valid_hits,
        "valid_hit_delta":
            refined_valid_hits
            - global_valid_hits,
        "global_hit_lost_by_local":
            global_hit_lost,
        "global_miss_recovered_by_local":
            global_miss_recovered,
        "refined_better_error_count":
            refined_better,
        "refined_worse_error_count":
            refined_worse,
        "refined_equal_error_count":
            refined_equal,
        "median_refined_minus_global_error_px":
            (
                statistics.median(
                    error_deltas
                )
                if error_deltas
                else None
            ),
        "global_valid_refined_invalid":
            global_valid_refined_invalid,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponible."
        )

    device = torch.device("cuda:0")

    i10 = load_i10_module()

    i10.ORIGINAL_WIDTH = CANONICAL_WIDTH
    i10.ORIGINAL_HEIGHT = CANONICAL_HEIGHT

    model, metadata = i10.load_model(
        CHECKPOINT_PATH,
        device,
    )

    if (
        metadata["matched_parameters"]
        != metadata["model_parameters"]
    ):
        raise RuntimeError(
            "Chargement incomplet du checkpoint D1C."
        )

    probability_threshold = float(
        metadata["threshold"]
    )

    rally_rows = read_csv(
        RALLIES_MANIFEST
    )

    frame_rows = read_csv(
        FRAMES_MANIFEST
    )

    validation_rallies = [
        row
        for row in rally_rows
        if row["ttflux_split"]
        == "validation"
    ]

    validation_rallies.sort(
        key=rally_key
    )

    if len(validation_rallies) != 30:
        raise RuntimeError(
            "30 échanges validation attendus, "
            f"obtenu={len(validation_rallies)}"
        )

    validation_matches = {
        parse_int(row["match_id"])
        for row in validation_rallies
    }

    if validation_matches != {20, 21}:
        raise RuntimeError(
            "Matchs validation inattendus : "
            f"{sorted(validation_matches)}"
        )

    annotations_by_key: dict[
        tuple[int, int],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in frame_rows:
        annotations_by_key[
            rally_key(row)
        ].append(row)

    prediction_rows = []
    rally_summary_rows = []

    torch.backends.cudnn.benchmark = True
    torch.cuda.synchronize()

    started = time.perf_counter()

    for rally_index, rally in enumerate(
        validation_rallies,
        start=1,
    ):
        match_id, rally_id = rally_key(
            rally
        )

        video_path = (
            ROOT
            / Path(rally["video_path"])
        ).resolve()

        annotations = sorted(
            eligible_annotations(
                annotations_by_key[
                    (match_id, rally_id)
                ]
            ),
            key=lambda row:
                parse_int(
                    row["frame_index"]
                ),
        )

        (
            resized_frames,
            source_width,
            source_height,
            fps,
        ) = load_resized_video(
            video_path
        )

        annotation_max = max(
            parse_int(row["frame_index"])
            for row in annotations
        )

        if annotation_max >= len(
            resized_frames
        ):
            raise RuntimeError(
                "Annotation au-delà de la vidéo : "
                f"{match_id:02d}/{rally_id:03d}"
            )

        source_scale_x = (
            source_width
            / CANONICAL_WIDTH
        )

        source_scale_y = (
            source_height
            / CANONICAL_HEIGHT
        )

        rally_predictions = []

        for annotation in annotations:
            frame_index = parse_int(
                annotation["frame_index"]
            )

            tensor = i10.make_sequence_tensor(
                resized_frames,
                frame_index,
                1,
                device,
            )

            with torch.inference_mode():
                (
                    pred_global,
                    pred_local,
                    _pred_events,
                    _pred_seg,
                ) = model.run_demo(tensor)

            decoded = i10.decode_prediction(
                pred_global,
                pred_local,
                probability_threshold,
            )

            global_x = (
                float(decoded["global_x"])
                * source_scale_x
            )

            global_y = (
                float(decoded["global_y"])
                * source_scale_y
            )

            refined_x = (
                float(decoded["refined_x"])
                * source_scale_x
            )

            refined_y = (
                float(decoded["refined_y"])
                * source_scale_y
            )

            global_in_bounds = (
                0 <= global_x < source_width
                and 0 <= global_y < source_height
            )

            refined_in_bounds = (
                0 <= refined_x < source_width
                and 0 <= refined_y < source_height
            )

            global_valid = bool(
                decoded["global_valid"]
                and global_in_bounds
            )

            refined_valid = bool(
                decoded["refined_valid"]
                and refined_in_bounds
            )

            visibility = parse_int(
                annotation["visibility"]
            )

            gt_x = parse_float(
                annotation["x_px"]
            )

            gt_y = parse_float(
                annotation["y_px"]
            )

            global_error = None
            refined_error = None

            if visibility == 1:
                if gt_x is None or gt_y is None:
                    raise RuntimeError(
                        "GT visible absente : "
                        f"{match_id:02d}/"
                        f"{rally_id:03d}/"
                        f"{frame_index}"
                    )

                global_error = math.hypot(
                    global_x - gt_x,
                    global_y - gt_y,
                )

                refined_error = math.hypot(
                    refined_x - gt_x,
                    refined_y - gt_y,
                )

            row = {
                "match_id":
                    f"{match_id:02d}",
                "rally_id":
                    f"{rally_id:03d}",
                "frame_index":
                    frame_index,
                "fps":
                    fps,
                "source_width":
                    source_width,
                "source_height":
                    source_height,
                "visibility":
                    visibility,
                "gt_x":
                    gt_x,
                "gt_y":
                    gt_y,
                "global_x":
                    round(global_x, 4),
                "global_y":
                    round(global_y, 4),
                "global_conf":
                    round(
                        float(
                            decoded[
                                "global_conf"
                            ]
                        ),
                        8,
                    ),
                "global_valid":
                    global_valid,
                "global_error_px":
                    (
                        round(
                            global_error,
                            4,
                        )
                        if global_error
                        is not None
                        else None
                    ),
                "refined_x":
                    round(refined_x, 4),
                "refined_y":
                    round(refined_y, 4),
                "local_conf":
                    round(
                        float(
                            decoded[
                                "local_conf"
                            ]
                        ),
                        8,
                    ),
                "refined_valid":
                    refined_valid,
                "refined_error_px":
                    (
                        round(
                            refined_error,
                            4,
                        )
                        if refined_error
                        is not None
                        else None
                    ),
            }

            prediction_rows.append(row)
            rally_predictions.append(row)

        global_metrics = stage_metrics(
            rally_predictions,
            "global",
        )

        refined_metrics = stage_metrics(
            rally_predictions,
            "refined",
        )

        rally_summary_rows.append(
            {
                "match_id":
                    f"{match_id:02d}",
                "rally_id":
                    f"{rally_id:03d}",
                "frames":
                    len(
                        rally_predictions
                    ),
                "global_raw_recall_at_20px":
                    global_metrics[
                        "raw_recall_at_20px"
                    ],
                "global_recall_at_20px":
                    global_metrics[
                        "recall_at_20px"
                    ],
                "global_precision_at_20px":
                    global_metrics[
                        "precision_at_20px"
                    ],
                "global_f1_at_20px":
                    global_metrics[
                        "f1_at_20px"
                    ],
                "refined_raw_recall_at_20px":
                    refined_metrics[
                        "raw_recall_at_20px"
                    ],
                "refined_recall_at_20px":
                    refined_metrics[
                        "recall_at_20px"
                    ],
                "refined_precision_at_20px":
                    refined_metrics[
                        "precision_at_20px"
                    ],
                "refined_f1_at_20px":
                    refined_metrics[
                        "f1_at_20px"
                    ],
            }
        )

        print(
            f"[{rally_index:02d}/30] "
            f"{match_id:02d}/{rally_id:03d} "
            f"frames={len(rally_predictions):4d} "
            "G_F1="
            f"{global_metrics['f1_at_20px']:.3f} "
            "R_F1="
            f"{refined_metrics['f1_at_20px']:.3f}"
        )

        del resized_frames

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - started
    )

    global_metrics = stage_metrics(
        prediction_rows,
        "global",
    )

    refined_metrics = stage_metrics(
        prediction_rows,
        "refined",
    )

    transitions = transition_metrics(
        prediction_rows
    )

    predictions_path = (
        OUTPUT_DIR
        / "validation_predictions.csv"
    )

    per_rally_path = (
        OUTPUT_DIR
        / "validation_by_rally.csv"
    )

    write_csv(
        predictions_path,
        prediction_rows,
    )

    write_csv(
        per_rally_path,
        rally_summary_rows,
    )

    summary = {
        "schema_version": 1,
        "experiment":
            "DSET_D1D0_global_D1C_local_legacy",
        "purpose":
            "baseline_before_local_training",
        "checkpoint": {
            "path":
                str(CHECKPOINT_PATH),
            "epoch":
                metadata["epoch"],
            "matched_parameters":
                metadata[
                    "matched_parameters"
                ],
            "model_parameters":
                metadata[
                    "model_parameters"
                ],
            "probability_threshold":
                probability_threshold,
        },
        "dataset": {
            "split":
                "validation",
            "match_ids": [
                20,
                21,
            ],
            "rallies":
                len(
                    validation_rallies
                ),
            "evaluated_frames":
                len(
                    prediction_rows
                ),
            "test_match_ids_excluded": [
                22,
                23,
                24,
                25,
            ],
        },
        "global":
            global_metrics,
        "refined_legacy_local":
            refined_metrics,
        "transitions":
            transitions,
        "runtime": {
            "seconds":
                elapsed,
            "frames_per_second":
                (
                    len(
                        prediction_rows
                    )
                    / elapsed
                    if elapsed > 0
                    else None
                ),
            "device":
                torch.cuda.get_device_name(
                    0
                ),
        },
        "artifacts": {
            "predictions_csv":
                str(
                    predictions_path
                ),
            "per_rally_csv":
                str(
                    per_rally_path
                ),
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
        "DSET_D1D0_LOCAL_BASELINE_GENERATED"
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()