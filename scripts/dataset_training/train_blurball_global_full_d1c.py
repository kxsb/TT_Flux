from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


ROOT = Path(
    os.environ["TTFLUX_ROOT"]
)

CHECKPOINT_PATH = Path(
    os.environ["TTFLUX_TTNET_CHECKPOINT"]
)

I10_SCRIPT = Path(
    os.environ["TTFLUX_I10_SCRIPT"]
)

D1A_SCRIPT = Path(
    os.environ["TTFLUX_D1A_SCRIPT"]
)

RALLIES_MANIFEST = Path(
    os.environ["TTFLUX_BLURBALL_RALLIES"]
)

FRAMES_MANIFEST = Path(
    os.environ["TTFLUX_BLURBALL_FRAMES"]
)

OUTPUT_DIR = Path(
    os.environ["TTFLUX_D1C_OUTPUT"]
)

SEED = 20260712

INPUT_WIDTH = 320
INPUT_HEIGHT = 128

SEQUENCE_LENGTH = 9
HISTORY_SPAN = 8

TRAIN_EPOCHS = 1
BATCH_SIZE = 16

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

INVISIBLE_TARGET_RATIO = 0.25

VALIDATION_EVERY_STEPS = 1000
PROGRESS_EVERY_STEPS = 100

OFFICIAL_THRESHOLD = 0.05


def load_module(
    name: str,
    path: Path,
) -> Any:
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Import impossible : {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


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
            f"Aucune ligne à écrire : {path}"
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


def parse_float(
    value: Any,
) -> float | None:
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


def parse_int(
    value: Any,
) -> int:
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


def config_value(
    configs: Any,
    name: str,
) -> Any:
    if isinstance(configs, dict):
        return configs[name]

    return getattr(
        configs,
        name,
    )


def rally_key(
    row: dict[str, str],
) -> tuple[int, int]:
    return (
        parse_int(
            row["match_id"]
        ),
        parse_int(
            row["rally_id"]
        ),
    )


def eligible_rows(
    annotations: list[
        dict[str, str]
    ],
) -> list[dict[str, str]]:
    result = []

    for row in annotations:
        frame_index = parse_int(
            row["frame_index"]
        )

        if frame_index < HISTORY_SPAN:
            continue

        visibility = parse_int(
            row["visibility"]
        )

        if visibility == 0:
            result.append(row)
            continue

        x = parse_float(
            row["x_px"]
        )

        y = parse_float(
            row["y_px"]
        )

        width = parse_int(
            row["video_width"]
        )

        height = parse_int(
            row["video_height"]
        )

        if (
            x is not None
            and y is not None
            and 0 < x < width
            and 0 < y < height
        ):
            result.append(row)

    return result


def build_rally_groups(
    rally_rows: list[dict[str, str]],
    frame_rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    annotations_by_key: dict[
        tuple[int, int],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in frame_rows:
        annotations_by_key[
            rally_key(row)
        ].append(row)

    train_groups = []
    validation_groups = []

    for rally in rally_rows:
        split = rally[
            "ttflux_split"
        ]

        if split not in {
            "train",
            "validation",
        }:
            continue

        key = rally_key(rally)

        annotations = sorted(
            annotations_by_key.get(
                key,
                [],
            ),
            key=lambda row: parse_int(
                row["frame_index"]
            ),
        )

        rows = eligible_rows(
            annotations
        )

        if not rows:
            continue

        group = {
            "key": key,
            "rally": rally,
            "rows": rows,
            "visible_count": sum(
                1
                for row in rows
                if parse_int(
                    row["visibility"]
                ) == 1
            ),
            "invisible_count": sum(
                1
                for row in rows
                if parse_int(
                    row["visibility"]
                ) == 0
            ),
        }

        if split == "train":
            train_groups.append(group)
        else:
            validation_groups.append(
                group
            )

    train_groups.sort(
        key=lambda group:
            group["key"]
    )

    validation_groups.sort(
        key=lambda group:
            group["key"]
    )

    return (
        train_groups,
        validation_groups,
    )


def validate_splits(
    train_groups: list[
        dict[str, Any]
    ],
    validation_groups: list[
        dict[str, Any]
    ],
) -> None:
    train_matches = {
        group["key"][0]
        for group in train_groups
    }

    validation_matches = {
        group["key"][0]
        for group in validation_groups
    }

    if any(
        match_id >= 20
        for match_id in train_matches
    ):
        raise RuntimeError(
            "Match hors train détecté : "
            f"{sorted(train_matches)}"
        )

    if validation_matches != {
        20,
        21,
    }:
        raise RuntimeError(
            "Matchs validation inattendus : "
            f"{sorted(validation_matches)}"
        )

    all_matches = (
        train_matches
        | validation_matches
    )

    if any(
        match_id >= 22
        for match_id in all_matches
    ):
        raise RuntimeError(
            "Le split test a été utilisé."
        )

    if len(train_groups) != 353:
        raise RuntimeError(
            "353 échanges train attendus, "
            f"obtenu={len(train_groups)}"
        )

    if len(validation_groups) != 30:
        raise RuntimeError(
            "30 échanges validation attendus, "
            f"obtenu={len(validation_groups)}"
        )


def make_training_rows(
    rows: list[dict[str, str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    visible = [
        row
        for row in rows
        if parse_int(
            row["visibility"]
        ) == 1
    ]

    invisible = [
        row
        for row in rows
        if parse_int(
            row["visibility"]
        ) == 0
    ]

    selected = list(
        visible
    ) + list(
        invisible
    )

    if visible and invisible:
        desired_invisible = math.ceil(
            len(visible)
            * INVISIBLE_TARGET_RATIO
            / (
                1.0
                - INVISIBLE_TARGET_RATIO
            )
        )

        additional = max(
            0,
            desired_invisible
            - len(invisible),
        )

        for _ in range(additional):
            selected.append(
                rng.choice(
                    invisible
                )
            )

    rng.shuffle(
        selected
    )

    return selected


def make_batch(
    resized_frames: list[Any],
    rows: list[dict[str, str]],
    source_width: int,
    source_height: int,
    d1a: Any,
    create_target_ball: Any,
    sigma: float,
    target_threshold: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    input_tensors = []
    position_tensors = []

    for row in rows:
        frame_index = parse_int(
            row["frame_index"]
        )

        input_tensors.append(
            d1a.sequence_tensor_uint8(
                resized_frames,
                frame_index,
            )
        )

        visibility = parse_int(
            row["visibility"]
        )

        if visibility == 1:
            x_source = parse_float(
                row["x_px"]
            )

            y_source = parse_float(
                row["y_px"]
            )

            if (
                x_source is None
                or y_source is None
            ):
                raise RuntimeError(
                    "Coordonnée visible absente."
                )

            position_tensors.append(
                torch.tensor(
                    [
                        x_source
                        * INPUT_WIDTH
                        / source_width,
                        y_source
                        * INPUT_HEIGHT
                        / source_height,
                    ],
                    dtype=torch.float32,
                )
            )
        else:
            position_tensors.append(
                torch.tensor(
                    [
                        -1.0,
                        -1.0,
                    ],
                    dtype=torch.float32,
                )
            )

    inputs = torch.stack(
        input_tensors,
        dim=0,
    )

    targets = torch.stack(
        [
            create_target_ball(
                position,
                sigma=sigma,
                w=INPUT_WIDTH,
                h=INPUT_HEIGHT,
                thresh_mask=(
                    target_threshold
                ),
                device=torch.device(
                    "cpu"
                ),
            )
            for position
            in position_tensors
        ],
        dim=0,
    )

    return inputs, targets


def set_training_mode(
    stage: nn.Module,
) -> None:
    stage.train()

    for module in stage.modules():
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.Dropout,
                nn.Dropout2d,
            ),
        ):
            module.eval()


def normalize_batch(
    batch: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (
        batch / 255.0
        - mean
    ) / std


def forward_stage(
    stage: nn.Module,
    batch: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    prediction, *_ = stage(
        normalize_batch(
            batch,
            mean,
            std,
        )
    )

    return prediction


def percentile(
    values: list[float],
    ratio: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = ratio * (
        len(ordered) - 1
    )

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
        / (
            precision
            + recall
        )
    )


def new_metrics_accumulator() -> dict[str, Any]:
    return {
        "loss_sum": 0.0,
        "sample_count": 0,
        "visible_count": 0,
        "invisible_count": 0,
        "valid_total": 0,
        "valid_visible": 0,
        "valid_invisible": 0,
        "raw_hits_at_20px": 0,
        "valid_hits_at_20px": 0,
        "errors": [],
        "confidences": [],
    }


def update_metrics(
    accumulator: dict[str, Any],
    prediction: torch.Tensor,
    rows: list[dict[str, str]],
    source_width: int,
    source_height: int,
    loss: float,
) -> None:
    batch_size = len(rows)

    accumulator[
        "loss_sum"
    ] += (
        loss
        * batch_size
    )

    accumulator[
        "sample_count"
    ] += batch_size

    x_axis = prediction[
        :,
        :INPUT_WIDTH,
    ]

    y_axis = prediction[
        :,
        INPUT_WIDTH:,
    ]

    pred_x = torch.argmax(
        x_axis,
        dim=1,
    ).detach().cpu()

    pred_y = torch.argmax(
        y_axis,
        dim=1,
    ).detach().cpu()

    confidence = torch.minimum(
        torch.max(
            x_axis,
            dim=1,
        ).values,
        torch.max(
            y_axis,
            dim=1,
        ).values,
    ).detach().cpu()

    for index, row in enumerate(rows):
        visibility = parse_int(
            row["visibility"]
        )

        sample_confidence = float(
            confidence[index].item()
        )

        accumulator[
            "confidences"
        ].append(
            sample_confidence
        )

        valid = (
            sample_confidence
            >= OFFICIAL_THRESHOLD
        )

        if visibility == 0:
            accumulator[
                "invisible_count"
            ] += 1

            if valid:
                accumulator[
                    "valid_total"
                ] += 1

                accumulator[
                    "valid_invisible"
                ] += 1

            continue

        accumulator[
            "visible_count"
        ] += 1

        if valid:
            accumulator[
                "valid_total"
            ] += 1

            accumulator[
                "valid_visible"
            ] += 1

        gt_x = parse_float(
            row["x_px"]
        )

        gt_y = parse_float(
            row["y_px"]
        )

        if gt_x is None or gt_y is None:
            raise RuntimeError(
                "GT visible absente."
            )

        predicted_x = (
            float(
                pred_x[
                    index
                ].item()
            )
            * source_width
            / INPUT_WIDTH
        )

        predicted_y = (
            float(
                pred_y[
                    index
                ].item()
            )
            * source_height
            / INPUT_HEIGHT
        )

        error = math.hypot(
            predicted_x - gt_x,
            predicted_y - gt_y,
        )

        accumulator[
            "errors"
        ].append(error)

        if error <= 20.0:
            accumulator[
                "raw_hits_at_20px"
            ] += 1

            if valid:
                accumulator[
                    "valid_hits_at_20px"
                ] += 1


def finalize_metrics(
    accumulator: dict[str, Any],
) -> dict[str, Any]:
    sample_count = accumulator[
        "sample_count"
    ]

    visible_count = accumulator[
        "visible_count"
    ]

    invisible_count = accumulator[
        "invisible_count"
    ]

    valid_total = accumulator[
        "valid_total"
    ]

    valid_hits = accumulator[
        "valid_hits_at_20px"
    ]

    precision = (
        valid_hits
        / valid_total
        if valid_total
        else 0.0
    )

    recall = (
        valid_hits
        / visible_count
        if visible_count
        else 0.0
    )

    return {
        "loss": (
            accumulator[
                "loss_sum"
            ]
            / sample_count
            if sample_count
            else None
        ),
        "sample_count":
            sample_count,
        "visible_count":
            visible_count,
        "invisible_count":
            invisible_count,
        "valid_total":
            valid_total,
        "valid_visible":
            accumulator[
                "valid_visible"
            ],
        "valid_invisible":
            accumulator[
                "valid_invisible"
            ],
        "valid_rate_visible": (
            accumulator[
                "valid_visible"
            ]
            / visible_count
            if visible_count
            else None
        ),
        "false_positive_rate_invisible": (
            accumulator[
                "valid_invisible"
            ]
            / invisible_count
            if invisible_count
            else None
        ),
        "raw_hits_at_20px":
            accumulator[
                "raw_hits_at_20px"
            ],
        "raw_recall_at_20px": (
            accumulator[
                "raw_hits_at_20px"
            ]
            / visible_count
            if visible_count
            else 0.0
        ),
        "valid_hits_at_20px":
            valid_hits,
        "precision_at_20px":
            precision,
        "recall_at_20px":
            recall,
        "f1_at_20px":
            harmonic_f1(
                precision,
                recall,
            ),
        "median_error_px": (
            statistics.median(
                accumulator[
                    "errors"
                ]
            )
            if accumulator[
                "errors"
            ]
            else None
        ),
        "p90_error_px": (
            percentile(
                accumulator[
                    "errors"
                ],
                0.90,
            )
        ),
        "median_confidence": (
            statistics.median(
                accumulator[
                    "confidences"
                ]
            )
            if accumulator[
                "confidences"
            ]
            else None
        ),
    }


def evaluate_validation(
    stage: nn.Module,
    validation_groups: list[
        dict[str, Any]
    ],
    d1a: Any,
    create_target_ball: Any,
    criterion: nn.Module,
    sigma: float,
    target_threshold: float,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    stage.eval()

    overall = new_metrics_accumulator()

    by_fps: dict[
        str,
        dict[str, Any],
    ] = {}

    with torch.inference_mode():
        for group_index, group in enumerate(
            validation_groups,
            start=1,
        ):
            rally = group[
                "rally"
            ]

            video_path = (
                ROOT
                / Path(
                    rally["video_path"]
                )
            ).resolve()

            (
                resized_frames,
                source_width,
                source_height,
                _source_fps,
            ) = d1a.load_resized_video(
                video_path
            )

            fps_key = rally[
                "fps_nominal"
            ]

            if fps_key not in by_fps:
                by_fps[
                    fps_key
                ] = new_metrics_accumulator()

            rows = group["rows"]

            for start in range(
                0,
                len(rows),
                BATCH_SIZE,
            ):
                batch_rows = rows[
                    start:
                    start + BATCH_SIZE
                ]

                inputs, targets = make_batch(
                    resized_frames,
                    batch_rows,
                    source_width,
                    source_height,
                    d1a,
                    create_target_ball,
                    sigma,
                    target_threshold,
                )

                inputs = inputs.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                targets = targets.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                prediction = forward_stage(
                    stage,
                    inputs,
                    mean,
                    std,
                )

                loss = criterion(
                    prediction,
                    targets,
                )

                loss_value = float(
                    loss.item()
                )

                update_metrics(
                    overall,
                    prediction,
                    batch_rows,
                    source_width,
                    source_height,
                    loss_value,
                )

                update_metrics(
                    by_fps[
                        fps_key
                    ],
                    prediction,
                    batch_rows,
                    source_width,
                    source_height,
                    loss_value,
                )

            del resized_frames

            if (
                group_index % 10
                == 0
            ):
                print(
                    "  validation "
                    f"{group_index}/"
                    f"{len(validation_groups)}"
                )

    return (
        finalize_metrics(
            overall
        ),
        {
            fps:
                finalize_metrics(
                    accumulator
                )
            for fps, accumulator
            in sorted(
                by_fps.items()
            )
        },
    )


def cpu_state_dict(
    module: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name:
            tensor.detach()
            .cpu()
            .clone()
        for name, tensor
        in module.state_dict().items()
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponible."
        )

    device = torch.device(
        "cuda:0"
    )

    d1a = load_module(
        "ttflux_d1a_reference_d1c",
        D1A_SCRIPT,
    )

    i10 = load_module(
        "ttflux_i10_reference_d1c",
        I10_SCRIPT,
    )

    checkpoint = (
        i10.torch_load_checkpoint(
            CHECKPOINT_PATH
        )
    )

    configs = i10.build_configs(
        checkpoint,
        device,
    )

    wrapper, model_metadata = (
        i10.load_model(
            CHECKPOINT_PATH,
            device,
        )
    )

    if (
        model_metadata[
            "matched_parameters"
        ]
        != model_metadata[
            "model_parameters"
        ]
    ):
        raise RuntimeError(
            "Chargement incomplet du checkpoint."
        )

    if not hasattr(
        wrapper,
        "model",
    ):
        raise RuntimeError(
            "Wrapper TTNet inattendu."
        )

    ttnet = wrapper.model

    from losses.losses import Ball_Detection_Loss
    from data_process.ttnet_data_utils import create_target_ball

    criterion = Ball_Detection_Loss(
        INPUT_WIDTH,
        INPUT_HEIGHT,
    ).to(device)

    sigma = float(
        config_value(
            configs,
            "sigma",
        )
    )

    target_threshold = float(
        config_value(
            configs,
            "thresh_ball_pos_mask",
        )
    )

    rally_rows = read_csv(
        RALLIES_MANIFEST
    )

    frame_rows = read_csv(
        FRAMES_MANIFEST
    )

    (
        train_groups,
        validation_groups,
    ) = build_rally_groups(
        rally_rows,
        frame_rows,
    )

    validate_splits(
        train_groups,
        validation_groups,
    )

    train_original_samples = sum(
        len(group["rows"])
        for group in train_groups
    )

    validation_samples = sum(
        len(group["rows"])
        for group in validation_groups
    )

    train_visible = sum(
        group["visible_count"]
        for group in train_groups
    )

    train_invisible = sum(
        group["invisible_count"]
        for group in train_groups
    )

    validation_visible = sum(
        group["visible_count"]
        for group in validation_groups
    )

    validation_invisible = sum(
        group["invisible_count"]
        for group in validation_groups
    )

    print("")
    print("=== CORPUS ===")
    print(
        "train_rallies="
        f"{len(train_groups)}"
    )
    print(
        "validation_rallies="
        f"{len(validation_groups)}"
    )
    print(
        "train_samples_originaux="
        f"{train_original_samples}"
    )
    print(
        "train_visible="
        f"{train_visible}"
    )
    print(
        "train_invisible="
        f"{train_invisible}"
    )
    print(
        "validation_samples="
        f"{validation_samples}"
    )
    print(
        "validation_visible="
        f"{validation_visible}"
    )
    print(
        "validation_invisible="
        f"{validation_invisible}"
    )

    mean = torch.repeat_interleave(
        torch.tensor(
            (
                0.485,
                0.456,
                0.406,
            ),
            dtype=torch.float32,
        ).view(
            1,
            3,
            1,
            1,
        ),
        repeats=SEQUENCE_LENGTH,
        dim=1,
    ).to(device)

    std = torch.repeat_interleave(
        torch.tensor(
            (
                0.229,
                0.224,
                0.225,
            ),
            dtype=torch.float32,
        ).view(
            1,
            3,
            1,
            1,
        ),
        repeats=SEQUENCE_LENGTH,
        dim=1,
    ).to(device)

    for parameter in wrapper.parameters():
        parameter.requires_grad = False

    global_stage = (
        ttnet.ball_global_stage
    )

    for parameter in global_stage.parameters():
        parameter.requires_grad = True

    initial_digests = {
        "global":
            d1a.state_digest(
                global_stage
            ),
        "local":
            d1a.state_digest(
                ttnet.ball_local_stage
            ),
        "event":
            d1a.state_digest(
                ttnet.events_spotting
            ),
        "seg":
            d1a.state_digest(
                ttnet.segmentation
            ),
    }

    optimizer = torch.optim.AdamW(
        global_stage.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("")
    print(
        "=== VALIDATION INITIALE ==="
    )

    (
        initial_validation,
        initial_validation_by_fps,
    ) = evaluate_validation(
        global_stage,
        validation_groups,
        d1a,
        create_target_ball,
        criterion,
        sigma,
        target_threshold,
        device,
        mean,
        std,
    )

    print(
        "INITIAL "
        f"loss="
        f"{initial_validation['loss']:.6f} "
        f"raw_R20="
        f"{initial_validation['raw_recall_at_20px']:.4f} "
        f"R20="
        f"{initial_validation['recall_at_20px']:.4f} "
        f"P20="
        f"{initial_validation['precision_at_20px']:.4f} "
        f"F1="
        f"{initial_validation['f1_at_20px']:.4f}"
    )

    curve_rows = [
        {
            "epoch": 0,
            "step": 0,
            **initial_validation,
        }
    ]

    best_score = (
        initial_validation[
            "f1_at_20px"
        ],
        initial_validation[
            "raw_recall_at_20px"
        ],
        -initial_validation[
            "loss"
        ],
    )

    best_step = 0
    best_epoch = 0

    best_state = cpu_state_dict(
        global_stage
    )

    best_validation = dict(
        initial_validation
    )

    best_validation_by_fps = dict(
        initial_validation_by_fps
    )

    global_step = 0
    training_samples_seen = 0

    running_loss_sum = 0.0
    running_sample_count = 0

    last_evaluated_step = 0

    training_started = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        TRAIN_EPOCHS + 1,
    ):
        epoch_rng = random.Random(
            SEED + epoch
        )

        epoch_groups = list(
            train_groups
        )

        epoch_rng.shuffle(
            epoch_groups
        )

        print("")
        print(
            f"=== EPOCH {epoch}/"
            f"{TRAIN_EPOCHS} ==="
        )

        for group_index, group in enumerate(
            epoch_groups,
            start=1,
        ):
            rally = group[
                "rally"
            ]

            video_path = (
                ROOT
                / Path(
                    rally["video_path"]
                )
            ).resolve()

            (
                resized_frames,
                source_width,
                source_height,
                _source_fps,
            ) = d1a.load_resized_video(
                video_path
            )

            training_rows = (
                make_training_rows(
                    group["rows"],
                    epoch_rng,
                )
            )

            for start in range(
                0,
                len(training_rows),
                BATCH_SIZE,
            ):
                batch_rows = (
                    training_rows[
                        start:
                        start + BATCH_SIZE
                    ]
                )

                inputs, targets = make_batch(
                    resized_frames,
                    batch_rows,
                    source_width,
                    source_height,
                    d1a,
                    create_target_ball,
                    sigma,
                    target_threshold,
                )

                inputs = inputs.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                targets = targets.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                set_training_mode(
                    global_stage
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                prediction = forward_stage(
                    global_stage,
                    inputs,
                    mean,
                    std,
                )

                loss = criterion(
                    prediction,
                    targets,
                )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "Perte non finie : "
                        f"epoch={epoch}, "
                        f"step={global_step + 1}"
                    )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    global_stage.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

                batch_size = len(
                    batch_rows
                )

                global_step += 1

                training_samples_seen += (
                    batch_size
                )

                running_loss_sum += (
                    float(
                        loss.item()
                    )
                    * batch_size
                )

                running_sample_count += (
                    batch_size
                )

                if (
                    global_step
                    % PROGRESS_EVERY_STEPS
                    == 0
                ):
                    running_loss = (
                        running_loss_sum
                        / running_sample_count
                    )

                    print(
                        f"TRAIN "
                        f"epoch={epoch} "
                        f"step={global_step} "
                        f"rally="
                        f"{group_index}/"
                        f"{len(epoch_groups)} "
                        f"loss="
                        f"{running_loss:.6f}"
                    )

                    running_loss_sum = 0.0
                    running_sample_count = 0

                if (
                    global_step
                    % VALIDATION_EVERY_STEPS
                    != 0
                ):
                    continue

                print("")
                print(
                    "=== VALIDATION "
                    f"STEP {global_step} ==="
                )

                (
                    validation_metrics,
                    validation_by_fps,
                ) = evaluate_validation(
                    global_stage,
                    validation_groups,
                    d1a,
                    create_target_ball,
                    criterion,
                    sigma,
                    target_threshold,
                    device,
                    mean,
                    std,
                )

                last_evaluated_step = (
                    global_step
                )

                curve_rows.append(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        **validation_metrics,
                    }
                )

                score = (
                    validation_metrics[
                        "f1_at_20px"
                    ],
                    validation_metrics[
                        "raw_recall_at_20px"
                    ],
                    -validation_metrics[
                        "loss"
                    ],
                )

                if score > best_score:
                    best_score = score
                    best_step = global_step
                    best_epoch = epoch

                    best_state = (
                        cpu_state_dict(
                            global_stage
                        )
                    )

                    best_validation = dict(
                        validation_metrics
                    )

                    best_validation_by_fps = dict(
                        validation_by_fps
                    )

                print(
                    "VALIDATION "
                    f"step={global_step} "
                    f"loss="
                    f"{validation_metrics['loss']:.6f} "
                    f"raw_R20="
                    f"{validation_metrics['raw_recall_at_20px']:.4f} "
                    f"R20="
                    f"{validation_metrics['recall_at_20px']:.4f} "
                    f"P20="
                    f"{validation_metrics['precision_at_20px']:.4f} "
                    f"F1="
                    f"{validation_metrics['f1_at_20px']:.4f} "
                    f"best_step={best_step}"
                )

            del resized_frames

            if (
                group_index % 25
                == 0
            ):
                print(
                    "RALLIES "
                    f"{group_index}/"
                    f"{len(epoch_groups)}"
                )

                torch.cuda.empty_cache()

    if last_evaluated_step != global_step:
        print("")
        print(
            "=== VALIDATION FINALE "
            f"STEP {global_step} ==="
        )

        (
            validation_metrics,
            validation_by_fps,
        ) = evaluate_validation(
            global_stage,
            validation_groups,
            d1a,
            create_target_ball,
            criterion,
            sigma,
            target_threshold,
            device,
            mean,
            std,
        )

        curve_rows.append(
            {
                "epoch":
                    TRAIN_EPOCHS,
                "step":
                    global_step,
                **validation_metrics,
            }
        )

        score = (
            validation_metrics[
                "f1_at_20px"
            ],
            validation_metrics[
                "raw_recall_at_20px"
            ],
            -validation_metrics[
                "loss"
            ],
        )

        if score > best_score:
            best_score = score
            best_step = global_step
            best_epoch = TRAIN_EPOCHS

            best_state = cpu_state_dict(
                global_stage
            )

            best_validation = dict(
                validation_metrics
            )

            best_validation_by_fps = dict(
                validation_by_fps
            )

        print(
            "VALIDATION "
            f"step={global_step} "
            f"loss="
            f"{validation_metrics['loss']:.6f} "
            f"raw_R20="
            f"{validation_metrics['raw_recall_at_20px']:.4f} "
            f"R20="
            f"{validation_metrics['recall_at_20px']:.4f} "
            f"P20="
            f"{validation_metrics['precision_at_20px']:.4f} "
            f"F1="
            f"{validation_metrics['f1_at_20px']:.4f} "
            f"best_step={best_step}"
        )

    training_seconds = (
        time.perf_counter()
        - training_started
    )

    global_stage.load_state_dict(
        best_state
    )

    final_digests = {
        "global":
            d1a.state_digest(
                global_stage
            ),
        "local":
            d1a.state_digest(
                ttnet.ball_local_stage
            ),
        "event":
            d1a.state_digest(
                ttnet.events_spotting
            ),
        "seg":
            d1a.state_digest(
                ttnet.segmentation
            ),
    }

    isolation_checks = {
        "global_changed": (
            initial_digests[
                "global"
            ]
            != final_digests[
                "global"
            ]
        ),
        "local_unchanged": (
            initial_digests[
                "local"
            ]
            == final_digests[
                "local"
            ]
        ),
        "event_unchanged": (
            initial_digests[
                "event"
            ]
            == final_digests[
                "event"
            ]
        ),
        "seg_unchanged": (
            initial_digests[
                "seg"
            ]
            == final_digests[
                "seg"
            ]
        ),
    }

    isolation_ok = all(
        isolation_checks.values()
    )

    validation_loss_ratio = (
        best_validation["loss"]
        / initial_validation["loss"]
    )

    raw_recall_gain = (
        best_validation[
            "raw_recall_at_20px"
        ]
        - initial_validation[
            "raw_recall_at_20px"
        ]
    )

    f1_gain = (
        best_validation[
            "f1_at_20px"
        ]
        - initial_validation[
            "f1_at_20px"
        ]
    )

    training_success = bool(
        isolation_ok
        and best_step > 0
        and validation_loss_ratio < 0.90
        and raw_recall_gain >= 0.05
        and f1_gain >= 0.05
    )

    curve_path = (
        OUTPUT_DIR
        / "validation_curve.csv"
    )

    write_csv(
        curve_path,
        curve_rows,
    )

    full_checkpoint_path = (
        OUTPUT_DIR
        / "d1c_ttnet_global_best.pth"
    )

    output_checkpoint = {
        "epoch":
            best_epoch,
        "state_dict":
            cpu_state_dict(
                wrapper
            ),
        "best_val_loss":
            best_validation["loss"],
        "configs":
            checkpoint.get(
                "configs"
            ),
        "ttflux_metadata": {
            "schema_version": 1,
            "experiment":
                "DSET_D1C_BlurBall_global_full",
            "source_checkpoint":
                str(CHECKPOINT_PATH),
            "trained_module":
                "model.ball_global_stage",
            "best_step":
                best_step,
            "best_epoch":
                best_epoch,
            "train_rallies":
                len(train_groups),
            "validation_rallies":
                len(validation_groups),
            "test_matches_excluded": [
                22,
                23,
                24,
                25,
            ],
            "validation_metrics":
                best_validation,
        },
    }

    torch.save(
        output_checkpoint,
        full_checkpoint_path,
    )

    summary = {
        "schema_version": 1,
        "experiment":
            "DSET_D1C_BlurBall_global_full",
        "purpose":
            "full_corpus_global_stage_phase_1",
        "source_checkpoint": {
            "path":
                str(CHECKPOINT_PATH),
            "epoch":
                model_metadata["epoch"],
            "matched_parameters":
                model_metadata[
                    "matched_parameters"
                ],
            "model_parameters":
                model_metadata[
                    "model_parameters"
                ],
        },
        "dataset": {
            "train_rallies":
                len(train_groups),
            "validation_rallies":
                len(validation_groups),
            "train_original_samples":
                train_original_samples,
            "train_visible":
                train_visible,
            "train_invisible":
                train_invisible,
            "validation_samples":
                validation_samples,
            "validation_visible":
                validation_visible,
            "validation_invisible":
                validation_invisible,
            "train_match_ids":
                list(range(20)),
            "validation_match_ids": [
                20,
                21,
            ],
            "test_match_ids_excluded": [
                22,
                23,
                24,
                25,
            ],
        },
        "configuration": {
            "epochs":
                TRAIN_EPOCHS,
            "batch_size":
                BATCH_SIZE,
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "invisible_target_ratio":
                INVISIBLE_TARGET_RATIO,
            "validation_every_steps":
                VALIDATION_EVERY_STEPS,
            "sequence_length":
                SEQUENCE_LENGTH,
            "target_frame_policy":
                "last_input_frame",
            "trained_module":
                "ball_global_stage",
            "batchnorm_running_stats":
                "frozen",
            "dropout":
                "disabled",
            "selection_metric":
                "validation_f1_at_20px",
        },
        "runtime": {
            "global_steps":
                global_step,
            "training_samples_seen":
                training_samples_seen,
            "training_seconds":
                training_seconds,
            "device":
                torch.cuda.get_device_name(
                    0
                ),
        },
        "initial_validation":
            initial_validation,
        "initial_validation_by_fps":
            initial_validation_by_fps,
        "best": {
            "step":
                best_step,
            "epoch":
                best_epoch,
            "validation":
                best_validation,
            "validation_by_fps":
                best_validation_by_fps,
        },
        "diagnostics": {
            "validation_loss_ratio":
                validation_loss_ratio,
            "raw_recall_gain_at_20px":
                raw_recall_gain,
            "f1_gain_at_20px":
                f1_gain,
            "isolation_checks":
                isolation_checks,
            "isolation_ok":
                isolation_ok,
            "training_success":
                training_success,
        },
        "artifacts": {
            "validation_curve_csv":
                str(curve_path),
            "best_checkpoint":
                str(
                    full_checkpoint_path
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
        "DSET_D1C_GLOBAL_FULL_GENERATED"
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