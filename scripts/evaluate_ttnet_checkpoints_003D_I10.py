from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch


TTNET_ROOT = Path(
    r"C:\Users\micka\Desktop\développement\ping"
    r"\TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch-master"
).resolve()

TTFLUX_ROOT = Path(
    r"C:\Users\micka\Desktop\développement\ping"
    r"\TTFlux_clean"
).resolve()

TTNET_SRC = TTNET_ROOT / "src"

sys.path.insert(
    0,
    str(TTNET_SRC),
)

from models.model_utils import create_model  # noqa: E402


BENCHMARK_DIR = (
    TTFLUX_ROOT
    / "runs"
    / "_openttgames_gt_benchmark_003D_I2"
)

MANIFEST_PATH = (
    BENCHMARK_DIR
    / "benchmark_manifest.json"
)

OUTPUT_DIR = (
    TTFLUX_ROOT
    / "runs"
    / "_ttnet_gt_evaluation_003D_I10"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "ttnet_gt_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "ttnet_gt_summary.json"
)

CHECKPOINTS = [
    {
        "model_id":
            "ttnet_120fps",
        "checkpoint_path":
            TTNET_ROOT
            / "checkpoints"
            / "ttnet_3rd_phase"
            / "ttnet_3rd_phase_best.pth",
        "temporal_stride":
            1,
    },
    {
        "model_id":
            "ttnet_30fps",
        "checkpoint_path":
            TTNET_ROOT
            / "checkpoints"
            / "ttnet_30fps_3rd_phase"
            / "ttnet_30fps_3rd_phase_best.pth",
        "temporal_stride":
            4,
    },
]

INPUT_WIDTH = 320
INPUT_HEIGHT = 128
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080
SEQUENCE_LENGTH = 9
HALF_SEQUENCE = 4

DISTANCE_THRESHOLDS = (
    5.0,
    10.0,
    20.0,
)


def read_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Objet JSON attendu : {path}"
        )

    return payload


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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            "Aucune prédiction à écrire."
        )

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


def checkpoint_value(
    configs: Any,
    name: str,
    default: Any,
) -> Any:
    if configs is None:
        return default

    if isinstance(configs, dict):
        return configs.get(
            name,
            default,
        )

    return getattr(
        configs,
        name,
        default,
    )


def torch_load_checkpoint(
    path: Path,
) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def build_configs(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> SimpleNamespace:
    stored = checkpoint.get(
        "configs"
    )

    tasks = list(
        checkpoint_value(
            stored,
            "tasks",
            [
                "global",
                "local",
                "event",
                "seg",
            ],
        )
    )

    # Les checkpoints de troisième phase
    # doivent contenir les quatre modules.
    required_tasks = {
        "global",
        "local",
        "event",
        "seg",
    }

    if not required_tasks.issubset(
        set(tasks)
    ):
        tasks = [
            "global",
            "local",
            "event",
            "seg",
        ]

    return SimpleNamespace(
        arch=checkpoint_value(
            stored,
            "arch",
            "ttnet",
        ),
        dropout_p=float(
            checkpoint_value(
                stored,
                "dropout_p",
                0.5,
            )
        ),
        multitask_learning=bool(
            checkpoint_value(
                stored,
                "multitask_learning",
                False,
            )
        ),
        tasks=tasks,
        input_size=tuple(
            checkpoint_value(
                stored,
                "input_size",
                (
                    INPUT_WIDTH,
                    INPUT_HEIGHT,
                ),
            )
        ),
        thresh_ball_pos_mask=float(
            checkpoint_value(
                stored,
                "thresh_ball_pos_mask",
                0.05,
            )
        ),
        num_frames_sequence=int(
            checkpoint_value(
                stored,
                "num_frames_sequence",
                SEQUENCE_LENGTH,
            )
        ),
        num_events=2,
        events_weights_loss=tuple(
            checkpoint_value(
                stored,
                "events_weights_loss",
                (
                    1.0,
                    3.0,
                ),
            )
        ),
        tasks_loss_weight=list(
            checkpoint_value(
                stored,
                "tasks_loss_weight",
                [
                    1.0
                    for _ in tasks
                ],
            )
        ),
        sigma=float(
            checkpoint_value(
                stored,
                "sigma",
                1.0,
            )
        ),
        device=device,
    )


def state_variants(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    variants = [
        state,
    ]

    variants.append({
        (
            key[len("module."):]
            if key.startswith("module.")
            else key
        ):
            value
        for key, value in state.items()
    })

    return variants


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
]:
    checkpoint = torch_load_checkpoint(
        checkpoint_path
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            f"Checkpoint invalide : "
            f"{checkpoint_path}"
        )

    configs = build_configs(
        checkpoint,
        device,
    )

    if (
        configs.num_frames_sequence
        != SEQUENCE_LENGTH
    ):
        raise RuntimeError(
            f"{checkpoint_path.name}: "
            f"num_frames_sequence="
            f"{configs.num_frames_sequence}, "
            f"attendu={SEQUENCE_LENGTH}"
        )

    model = create_model(
        configs
    )

    raw_state = checkpoint.get(
        "state_dict",
        checkpoint,
    )

    if not isinstance(
        raw_state,
        dict,
    ):
        raise ValueError(
            "state_dict absent du checkpoint."
        )

    target_state = model.state_dict()

    best_state: dict[str, Any] = {}
    best_match_count = -1

    for variant in state_variants(
        raw_state
    ):
        compatible = {
            key:
                value
            for key, value
            in variant.items()
            if (
                key in target_state
                and hasattr(
                    value,
                    "shape",
                )
                and tuple(
                    value.shape
                )
                == tuple(
                    target_state[
                        key
                    ].shape
                )
            )
        }

        if (
            len(compatible)
            > best_match_count
        ):
            best_match_count = len(
                compatible
            )

            best_state = compatible

    match_ratio = (
        best_match_count
        / len(target_state)
        if target_state
        else 0.0
    )

    if match_ratio < 0.90:
        raise RuntimeError(
            f"{checkpoint_path.name}: "
            f"seulement "
            f"{best_match_count}/"
            f"{len(target_state)} "
            "paramètres compatibles."
        )

    merged = dict(
        target_state
    )

    merged.update(
        best_state
    )

    model.load_state_dict(
        merged
    )

    model.to(
        device
    )

    model.eval()

    metadata = {
        "epoch":
            checkpoint.get(
                "epoch"
            ),
        "matched_parameters":
            best_match_count,
        "model_parameters":
            len(target_state),
        "match_ratio":
            match_ratio,
        "multitask_learning":
            configs.multitask_learning,
        "tasks":
            configs.tasks,
        "threshold":
            configs.thresh_ball_pos_mask,
        "input_size":
            list(
                configs.input_size
            ),
        "num_frames_sequence":
            configs.num_frames_sequence,
    }

    return (
        model,
        metadata,
    )


def load_resized_frames(
    video_path: Path,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Vidéo illisible : "
            f"{video_path}"
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
                (
                    INPUT_WIDTH,
                    INPUT_HEIGHT,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

            frames.append(
                resized
            )
    finally:
        capture.release()

    if len(frames) != 120:
        raise RuntimeError(
            f"{video_path.name}: "
            f"{len(frames)} images "
            "au lieu de 120."
        )

    return frames


def make_sequence_tensor(
    frames: list[np.ndarray],
    target_frame: int,
    temporal_stride: int,
    device: torch.device,
) -> torch.Tensor:
    # TTNet est entraîné pour prédire la balle sur la dernière
    # image des neuf images d'entrée, et non sur l'image centrale.
    indices = [
        target_frame
        - (SEQUENCE_LENGTH - 1 - sequence_index)
        * temporal_stride
        for sequence_index
        in range(SEQUENCE_LENGTH)
    ]

    selected = [
        frames[index]
        for index in indices
    ]

    stacked = np.dstack(
        selected
    ).transpose(
        2,
        0,
        1,
    )

    contiguous = np.ascontiguousarray(
        stacked
    )

    return (
        torch.from_numpy(
            contiguous
        )
        .float()
        .unsqueeze(0)
        .to(
            device,
            non_blocking=True,
        )
    )


def ball_vector(
    tensor: torch.Tensor,
) -> np.ndarray:
    return (
        tensor.detach()
        .float()
        .cpu()
        .squeeze(0)
        .numpy()
    )


def decode_prediction(
    global_tensor: torch.Tensor,
    local_tensor: torch.Tensor,
    probability_threshold: float,
) -> dict[str, Any]:
    global_values = ball_vector(
        global_tensor
    )

    local_values = ball_vector(
        local_tensor
    )

    global_x_values = (
        global_values[
            :INPUT_WIDTH
        ]
    )

    global_y_values = (
        global_values[
            INPUT_WIDTH:
        ]
    )

    local_x_values = (
        local_values[
            :INPUT_WIDTH
        ]
    )

    local_y_values = (
        local_values[
            INPUT_WIDTH:
        ]
    )

    global_x_index = int(
        np.argmax(
            global_x_values
        )
    )

    global_y_index = int(
        np.argmax(
            global_y_values
        )
    )

    local_x_index = int(
        np.argmax(
            local_x_values
        )
    )

    local_y_index = int(
        np.argmax(
            local_y_values
        )
    )

    global_x_conf = float(
        global_x_values[
            global_x_index
        ]
    )

    global_y_conf = float(
        global_y_values[
            global_y_index
        ]
    )

    local_x_conf = float(
        local_x_values[
            local_x_index
        ]
    )

    local_y_conf = float(
        local_y_values[
            local_y_index
        ]
    )

    global_valid = (
        global_x_conf
        >= probability_threshold
        and global_y_conf
        >= probability_threshold
    )

    local_valid = (
        local_x_conf
        >= probability_threshold
        and local_y_conf
        >= probability_threshold
    )

    width_ratio = (
        ORIGINAL_WIDTH
        / INPUT_WIDTH
    )

    height_ratio = (
        ORIGINAL_HEIGHT
        / INPUT_HEIGHT
    )

    global_x = (
        global_x_index
        * width_ratio
    )

    global_y = (
        global_y_index
        * height_ratio
    )

    x_center = int(
        global_x
    )

    y_center = int(
        global_y
    )

    x_min = max(
        0,
        x_center
        - INPUT_WIDTH // 2,
    )

    y_min = max(
        0,
        y_center
        - INPUT_HEIGHT // 2,
    )

    x_max = min(
        ORIGINAL_WIDTH,
        x_min + INPUT_WIDTH,
    )

    y_max = min(
        ORIGINAL_HEIGHT,
        y_min + INPUT_HEIGHT,
    )

    crop_width = (
        x_max - x_min
    )

    crop_height = (
        y_max - y_min
    )

    x_pad = int(
        (
            INPUT_WIDTH
            - crop_width
        )
        / 2
    )

    y_pad = int(
        (
            INPUT_HEIGHT
            - crop_height
        )
        / 2
    )

    refined_x = (
        x_min
        - x_pad
        + local_x_index
    )

    refined_y = (
        y_min
        - y_pad
        + local_y_index
    )

    refined_in_bounds = (
        0 <= refined_x
        < ORIGINAL_WIDTH
        and 0 <= refined_y
        < ORIGINAL_HEIGHT
    )

    refined_valid = (
        global_valid
        and local_valid
        and refined_in_bounds
    )

    return {
        "global_x":
            global_x,
        "global_y":
            global_y,
        "global_valid":
            global_valid,
        "global_conf":
            min(
                global_x_conf,
                global_y_conf,
            ),
        "refined_x":
            refined_x,
        "refined_y":
            refined_y,
        "refined_valid":
            refined_valid,
        "local_conf":
            min(
                local_x_conf,
                local_y_conf,
            ),
    }


def euclidean_distance(
    predicted_x: float,
    predicted_y: float,
    gt_x: float,
    gt_y: float,
) -> float:
    return math.hypot(
        predicted_x - gt_x,
        predicted_y - gt_y,
    )


def aggregate(
    rows: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    total = len(rows)

    valid_rows = [
        row
        for row in rows
        if bool(
            row[
                f"{prefix}_valid"
            ]
        )
    ]

    errors = [
        float(
            row[
                f"{prefix}_error_px"
            ]
        )
        for row in valid_rows
    ]

    result = {
        "gt_evaluable":
            total,
        "valid_predictions":
            len(valid_rows),
        "valid_rate":
            (
                len(valid_rows)
                / total
                if total
                else 0.0
            ),
        "median_error_px":
            (
                statistics.median(
                    errors
                )
                if errors
                else None
            ),
        "p90_error_px":
            (
                float(
                    np.percentile(
                        errors,
                        90,
                    )
                )
                if errors
                else None
            ),
    }

    for threshold in (
        DISTANCE_THRESHOLDS
    ):
        hits = sum(
            bool(
                row[
                    f"{prefix}_valid"
                ]
            )
            and float(
                row[
                    f"{prefix}_error_px"
                ]
            )
            <= threshold
            for row in rows
        )

        result[
            f"recall_{int(threshold)}"
        ] = (
            hits / total
            if total
            else 0.0
        )

    return result


def evaluate_checkpoint(
    model_spec: dict[str, Any],
    windows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    model_id = str(
        model_spec["model_id"]
    )

    checkpoint_path = Path(
        model_spec[
            "checkpoint_path"
        ]
    )

    temporal_stride = int(
        model_spec[
            "temporal_stride"
        ]
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    model, metadata = load_model(
        checkpoint_path,
        device,
    )

    history_span = (
        (SEQUENCE_LENGTH - 1)
        * temporal_stride
    )

    probability_threshold = float(
        metadata["threshold"]
    )

    prediction_rows = []

    started = time.perf_counter()

    print()
    print(
        "MODEL",
        model_id,
    )

    print(
        "  checkpoint :",
        checkpoint_path.name,
    )

    print(
        "  stride     :",
        temporal_stride,
    )

    print(
        "  paramètres :",
        (
            f"{metadata['matched_parameters']}/"
            f"{metadata['model_parameters']}"
        ),
    )

    print(
        "  epoch      :",
        metadata["epoch"],
    )

    for window_index, window in enumerate(
        windows,
        start=1,
    ):
        window_id = str(
            window["window_id"]
        )

        source_name = str(
            window["source_name"]
        )

        clip_path = Path(
            window["clip_path"]
        )

        gt_path = Path(
            window["gt_path"]
        )

        frames = load_resized_frames(
            clip_path
        )

        gt_rows = read_csv(
            gt_path
        )

        evaluated_here = 0

        for gt in gt_rows:
            local_frame = int(
                float(
                    gt[
                        "local_frame"
                    ]
                )
            )

            if (
                local_frame < history_span
                or local_frame >= len(frames)
            ):
                continue

            gt_x = float(
                gt["x"]
            )

            gt_y = float(
                gt["y"]
            )

            tensor = make_sequence_tensor(
                frames,
                local_frame,
                temporal_stride,
                device,
            )

            with torch.inference_mode():
                (
                    pred_global,
                    pred_local,
                    _pred_events,
                    _pred_seg,
                ) = model.run_demo(
                    tensor
                )

            decoded = decode_prediction(
                pred_global,
                pred_local,
                probability_threshold,
            )

            global_error = euclidean_distance(
                decoded["global_x"],
                decoded["global_y"],
                gt_x,
                gt_y,
            )

            refined_error = (
                euclidean_distance(
                    decoded["refined_x"],
                    decoded["refined_y"],
                    gt_x,
                    gt_y,
                )
            )

            prediction_rows.append({
                "model_id":
                    model_id,
                "checkpoint":
                    checkpoint_path.name,
                "temporal_stride":
                    temporal_stride,
                "window_id":
                    window_id,
                "source_name":
                    source_name,
                "local_frame":
                    local_frame,
                "sequence_first_local_frame":
                    local_frame - history_span,
                "sequence_last_local_frame":
                    local_frame,
                "target_frame_policy":
                    "last_input_frame",
                "source_frame":
                    int(
                        float(
                            gt[
                                "source_frame"
                            ]
                        )
                    ),
                "gt_x":
                    gt_x,
                "gt_y":
                    gt_y,
                "event":
                    str(
                        gt.get(
                            "event",
                            "",
                        )
                        or ""
                    ),
                "global_x":
                    round(
                        decoded["global_x"],
                        3,
                    ),
                "global_y":
                    round(
                        decoded["global_y"],
                        3,
                    ),
                "global_conf":
                    round(
                        decoded["global_conf"],
                        6,
                    ),
                "global_valid":
                    decoded["global_valid"],
                "global_error_px":
                    round(
                        global_error,
                        4,
                    ),
                "refined_x":
                    round(
                        decoded["refined_x"],
                        3,
                    ),
                "refined_y":
                    round(
                        decoded["refined_y"],
                        3,
                    ),
                "local_conf":
                    round(
                        decoded["local_conf"],
                        6,
                    ),
                "refined_valid":
                    decoded["refined_valid"],
                "refined_error_px":
                    round(
                        refined_error,
                        4,
                    ),
            })

            evaluated_here += 1

        print(
            f"  [{window_index:02d}/14] "
            f"{window_id:10} "
            f"GT={evaluated_here:3}"
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    global_summary = aggregate(
        prediction_rows,
        "global",
    )

    refined_summary = aggregate(
        prediction_rows,
        "refined",
    )

    source_summaries = []

    for source_name in sorted({
        row["source_name"]
        for row in prediction_rows
    }):
        subset = [
            row
            for row in prediction_rows
            if row["source_name"]
            == source_name
        ]

        source_summaries.append({
            "source_name":
                source_name,
            "global":
                aggregate(
                    subset,
                    "global",
                ),
            "refined":
                aggregate(
                    subset,
                    "refined",
                ),
        })

    summary = {
        "model_id":
            model_id,
        "checkpoint_path":
            str(
                checkpoint_path
            ),
        "temporal_stride":
            temporal_stride,
        "target_frame_policy":
            "last_input_frame",
        "history_span_frames":
            history_span,
        "metadata":
            metadata,
        "global":
            global_summary,
        "refined":
            refined_summary,
        "by_source":
            source_summaries,
        "runtime_seconds":
            elapsed,
    }

    del model

    torch.cuda.empty_cache()

    return (
        summary,
        prediction_rows,
    )


def format_error(
    value: float | None,
) -> str:
    return (
        f"{value:.2f}px"
        if value is not None
        else "n/a"
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA n'est pas disponible "
            "dans l'environnement TTNet."
        )

    torch.cuda.set_device(
        0
    )

    device = torch.device(
        "cuda:0"
    )

    print(
        "CUDA :",
        torch.cuda.get_device_name(0),
    )

    manifest = read_json(
        MANIFEST_PATH
    )

    windows = manifest.get(
        "windows",
        [],
    )

    if len(windows) != 14:
        raise RuntimeError(
            f"{len(windows)} fenêtres "
            "au lieu de 14."
        )

    summaries = []
    all_predictions = []

    for model_spec in CHECKPOINTS:
        summary, predictions = (
            evaluate_checkpoint(
                model_spec,
                windows,
                device,
            )
        )

        summaries.append(
            summary
        )

        all_predictions.extend(
            predictions
        )

    write_csv(
        PREDICTIONS_PATH,
        all_predictions,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "experiment":
                    "003D_I10_TTNet_last_frame_target_vs_OpenTTGames_GT",
                "benchmark":
                    str(
                        MANIFEST_PATH
                    ),
                "test_split_only":
                    True,
                "training_performed":
                    False,
                "models":
                    summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=== TTNET CHECKPOINTS VS GT ==="
    )

    for summary in summaries:
        global_metrics = summary[
            "global"
        ]

        refined_metrics = summary[
            "refined"
        ]

        print()
        print(
            summary["model_id"],
            f"(stride={summary['temporal_stride']})",
        )

        print(
            "  GT évaluables       :",
            refined_metrics[
                "gt_evaluable"
            ],
        )

        print(
            "  global valide       :",
            f"{global_metrics['valid_rate']:.3f}",
        )

        print(
            "  global rappel @5    :",
            f"{global_metrics['recall_5']:.3f}",
        )

        print(
            "  global rappel @10   :",
            f"{global_metrics['recall_10']:.3f}",
        )

        print(
            "  global rappel @20   :",
            f"{global_metrics['recall_20']:.3f}",
        )

        print(
            "  global médiane      :",
            format_error(
                global_metrics[
                    "median_error_px"
                ]
            ),
        )

        print(
            "  raffiné valide      :",
            f"{refined_metrics['valid_rate']:.3f}",
        )

        print(
            "  raffiné rappel @5   :",
            f"{refined_metrics['recall_5']:.3f}",
        )

        print(
            "  raffiné rappel @10  :",
            f"{refined_metrics['recall_10']:.3f}",
        )

        print(
            "  raffiné rappel @20  :",
            f"{refined_metrics['recall_20']:.3f}",
        )

        print(
            "  raffiné médiane     :",
            format_error(
                refined_metrics[
                    "median_error_px"
                ]
            ),
        )

        print(
            "  temps               :",
            f"{summary['runtime_seconds']:.1f}s",
        )

    print()
    print(
        "=== TTNET PAR SOURCE ==="
    )

    for summary in summaries:
        print()
        print(
            summary["model_id"]
        )

        for source in summary[
            "by_source"
        ]:
            refined = source[
                "refined"
            ]

            print(
                f"  {source['source_name']:8} "
                f"gt="
                f"{refined['gt_evaluable']:3} "
                f"valid="
                f"{refined['valid_rate']:.3f} "
                f"r5="
                f"{refined['recall_5']:.3f} "
                f"r10="
                f"{refined['recall_10']:.3f} "
                f"r20="
                f"{refined['recall_20']:.3f}"
            )

    print()
    print(
        "CSV  :",
        PREDICTIONS_PATH,
    )

    print(
        "JSON :",
        SUMMARY_PATH,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
