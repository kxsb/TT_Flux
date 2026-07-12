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
    os.environ["TTFLUX_D1B_OUTPUT"]
)

SEED = 20260712

INPUT_WIDTH = 320
INPUT_HEIGHT = 128

SEQUENCE_LENGTH = 9
HISTORY_SPAN = 8

TRAIN_RALLY_COUNT = 12
VALIDATION_RALLY_COUNT = 4

SAMPLES_PER_RALLY = 32
MAX_INVISIBLE_PER_RALLY = 8

BATCH_SIZE = 8
VISIBLE_PER_BATCH = 6

TRAINING_STEPS = 800
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

OFFICIAL_THRESHOLD = 0.05

EVALUATION_STEPS = {
    0,
    100,
    200,
    400,
    600,
    800,
}


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


def evenly_spaced(
    rows: list[dict[str, str]],
    count: int,
) -> list[dict[str, str]]:
    if count <= 0:
        return []

    if len(rows) <= count:
        return list(rows)

    indices = []

    if count == 1:
        indices = [
            len(rows) // 2
        ]
    else:
        for index in range(count):
            selected = round(
                index
                * (len(rows) - 1)
                / (count - 1)
            )

            if selected not in indices:
                indices.append(selected)

    if len(indices) < count:
        for index in range(len(rows)):
            if index not in indices:
                indices.append(index)

            if len(indices) == count:
                break

    return [
        rows[index]
        for index in sorted(
            indices[:count]
        )
    ]


def annotation_key(
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


def eligible_annotations(
    annotations: list[
        dict[str, str]
    ],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
]:
    visible = []
    invisible = []

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
            invisible.append(row)
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
            visible.append(row)

    return visible, invisible


def build_candidates(
    rally_rows: list[dict[str, str]],
    annotations_by_key: dict[
        tuple[int, int],
        list[dict[str, str]],
    ],
) -> list[dict[str, Any]]:
    candidates = []

    for rally in rally_rows:
        split = rally[
            "ttflux_split"
        ]

        if split not in {
            "train",
            "validation",
        }:
            continue

        key = annotation_key(rally)

        annotations = sorted(
            annotations_by_key.get(
                key,
                [],
            ),
            key=lambda row: parse_int(
                row["frame_index"]
            ),
        )

        visible, invisible = (
            eligible_annotations(
                annotations
            )
        )

        if len(visible) < 24:
            continue

        candidates.append(
            {
                "key": key,
                "rally": rally,
                "annotations":
                    annotations,
                "visible":
                    visible,
                "invisible":
                    invisible,
                "fps_nominal":
                    rally[
                        "fps_nominal"
                    ],
                "resolution": (
                    parse_int(
                        rally[
                            "video_width"
                        ]
                    ),
                    parse_int(
                        rally[
                            "video_height"
                        ]
                    ),
                ),
                "score": (
                    -min(
                        len(invisible),
                        MAX_INVISIBLE_PER_RALLY,
                    ),
                    abs(
                        len(annotations)
                        - 160
                    ),
                    key[1],
                ),
            }
        )

    return candidates


def choose_train_candidates(
    candidates: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    by_match: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for candidate in candidates:
        if (
            candidate[
                "rally"
            ]["ttflux_split"]
            != "train"
        ):
            continue

        by_match[
            candidate["key"][0]
        ].append(candidate)

    best_per_match = []

    for match_id, items in sorted(
        by_match.items()
    ):
        items.sort(
            key=lambda item:
                item["score"]
        )

        best_per_match.append(
            items[0]
        )

    buckets: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for candidate in best_per_match:
        buckets[
            candidate["fps_nominal"]
        ].append(candidate)

    for items in buckets.values():
        items.sort(
            key=lambda item: (
                item["resolution"],
                item["key"],
            )
        )

    fps_order = [
        "24",
        "25",
        "30",
        "50",
        "60",
        "other",
        "unknown",
    ]

    selected = []

    while (
        len(selected)
        < TRAIN_RALLY_COUNT
    ):
        added = False

        dynamic_order = (
            fps_order
            + sorted(
                key
                for key in buckets
                if key not in fps_order
            )
        )

        for fps in dynamic_order:
            items = buckets.get(
                fps,
                [],
            )

            if not items:
                continue

            selected.append(
                items.pop(0)
            )

            added = True

            if (
                len(selected)
                == TRAIN_RALLY_COUNT
            ):
                break

        if not added:
            break

    if len(selected) != TRAIN_RALLY_COUNT:
        raise RuntimeError(
            "Nombre insuffisant d'échanges train : "
            f"{len(selected)}"
        )

    return selected


def choose_validation_candidates(
    candidates: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    by_match: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for candidate in candidates:
        if (
            candidate[
                "rally"
            ]["ttflux_split"]
            != "validation"
        ):
            continue

        by_match[
            candidate["key"][0]
        ].append(candidate)

    expected_matches = {
        20,
        21,
    }

    if set(by_match) != expected_matches:
        raise RuntimeError(
            "Matchs de validation inattendus : "
            f"{sorted(by_match)}"
        )

    selected = []

    for match_id in sorted(
        expected_matches
    ):
        items = by_match[
            match_id
        ]

        items.sort(
            key=lambda item: (
                item["score"],
                item["fps_nominal"],
                item["resolution"],
            )
        )

        selected.extend(
            items[:2]
        )

    if (
        len(selected)
        != VALIDATION_RALLY_COUNT
    ):
        raise RuntimeError(
            "Quatre échanges validation "
            "étaient attendus."
        )

    return selected


def sample_candidate(
    candidate: dict[str, Any],
) -> list[dict[str, str]]:
    invisible_count = min(
        MAX_INVISIBLE_PER_RALLY,
        len(
            candidate["invisible"]
        ),
    )

    visible_count = (
        SAMPLES_PER_RALLY
        - invisible_count
    )

    visible = evenly_spaced(
        candidate["visible"],
        visible_count,
    )

    invisible = evenly_spaced(
        candidate["invisible"],
        invisible_count,
    )

    samples = sorted(
        visible + invisible,
        key=lambda row: parse_int(
            row["frame_index"]
        ),
    )

    if (
        len(samples)
        != SAMPLES_PER_RALLY
    ):
        raise RuntimeError(
            "Échantillonnage incomplet : "
            f"{candidate['key']} "
            f"count={len(samples)}"
        )

    return samples


def build_dataset(
    candidates: list[
        dict[str, Any]
    ],
    d1a: Any,
    create_target_ball: Any,
    sigma: float,
    target_threshold: float,
) -> dict[str, Any]:
    inputs = []
    positions_stage = []
    positions_source = []
    visibility_values = []
    widths = []
    heights = []
    metadata = []

    for rally_index, candidate in enumerate(
        candidates,
        start=1,
    ):
        match_id, rally_id = (
            candidate["key"]
        )

        rally = candidate[
            "rally"
        ]

        video_path = (
            ROOT
            / Path(
                rally["video_path"]
            )
        ).resolve()

        print(
            "LOAD "
            f"{rally_index}/"
            f"{len(candidates)} "
            f"{rally['ttflux_split']} "
            f"{match_id:02d}/{rally_id:03d}"
        )

        (
            resized_frames,
            source_width,
            source_height,
            source_fps,
        ) = d1a.load_resized_video(
            video_path
        )

        samples = sample_candidate(
            candidate
        )

        for sample in samples:
            frame_index = parse_int(
                sample["frame_index"]
            )

            visibility = parse_int(
                sample["visibility"]
            )

            sequence = (
                d1a.sequence_tensor_uint8(
                    resized_frames,
                    frame_index,
                )
            )

            if visibility == 1:
                x_source = parse_float(
                    sample["x_px"]
                )

                y_source = parse_float(
                    sample["y_px"]
                )

                if (
                    x_source is None
                    or y_source is None
                ):
                    raise RuntimeError(
                        "Coordonnée visible absente."
                    )

                x_stage = (
                    x_source
                    * INPUT_WIDTH
                    / source_width
                )

                y_stage = (
                    y_source
                    * INPUT_HEIGHT
                    / source_height
                )
            else:
                x_source = -1.0
                y_source = -1.0
                x_stage = -1.0
                y_stage = -1.0

            inputs.append(
                sequence
            )

            positions_stage.append(
                [
                    x_stage,
                    y_stage,
                ]
            )

            positions_source.append(
                [
                    x_source,
                    y_source,
                ]
            )

            visibility_values.append(
                visibility == 1
            )

            widths.append(
                source_width
            )

            heights.append(
                source_height
            )

            metadata.append(
                {
                    "split":
                        rally[
                            "ttflux_split"
                        ],
                    "match_id":
                        match_id,
                    "rally_id":
                        rally_id,
                    "frame_index":
                        frame_index,
                    "fps_nominal":
                        rally[
                            "fps_nominal"
                        ],
                    "fps_actual":
                        source_fps,
                    "width":
                        source_width,
                    "height":
                        source_height,
                }
            )

        del resized_frames

    input_tensor = torch.stack(
        inputs,
        dim=0,
    )

    stage_position_tensor = (
        torch.tensor(
            positions_stage,
            dtype=torch.float32,
        )
    )

    source_position_tensor = (
        torch.tensor(
            positions_source,
            dtype=torch.float32,
        )
    )

    visibility_tensor = torch.tensor(
        visibility_values,
        dtype=torch.bool,
    )

    width_tensor = torch.tensor(
        widths,
        dtype=torch.float32,
    )

    height_tensor = torch.tensor(
        heights,
        dtype=torch.float32,
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
            in stage_position_tensor
        ],
        dim=0,
    )

    return {
        "inputs":
            input_tensor,
        "targets":
            targets,
        "positions_stage":
            stage_position_tensor,
        "positions_source":
            source_position_tensor,
        "visibility":
            visibility_tensor,
        "widths":
            width_tensor,
        "heights":
            height_tensor,
        "metadata":
            metadata,
    }


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


def evaluate(
    stage: nn.Module,
    dataset: dict[str, Any],
    criterion: nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, Any]:
    stage.eval()

    inputs = dataset[
        "inputs"
    ]

    targets = dataset[
        "targets"
    ]

    positions_source = dataset[
        "positions_source"
    ]

    visibility = dataset[
        "visibility"
    ]

    widths = dataset[
        "widths"
    ]

    heights = dataset[
        "heights"
    ]

    total_loss = 0.0
    total_count = 0

    valid_total = 0
    valid_visible = 0
    valid_invisible = 0

    raw_hits_20 = 0
    valid_hits_20 = 0

    visible_count = int(
        visibility.sum().item()
    )

    invisible_count = int(
        (
            ~visibility
        ).sum().item()
    )

    errors = []
    confidences = []

    with torch.inference_mode():
        for start in range(
            0,
            len(inputs),
            BATCH_SIZE,
        ):
            end = min(
                start + BATCH_SIZE,
                len(inputs),
            )

            batch_input = (
                inputs[start:end]
                .to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            )

            batch_target = (
                targets[start:end]
                .to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            )

            prediction = forward_stage(
                stage,
                batch_input,
                mean,
                std,
            )

            loss = criterion(
                prediction,
                batch_target,
            )

            batch_size = end - start

            total_loss += (
                float(loss.item())
                * batch_size
            )

            total_count += batch_size

            pred_x_axis = prediction[
                :,
                :INPUT_WIDTH,
            ]

            pred_y_axis = prediction[
                :,
                INPUT_WIDTH:,
            ]

            pred_x = torch.argmax(
                pred_x_axis,
                dim=1,
            ).detach().cpu().float()

            pred_y = torch.argmax(
                pred_y_axis,
                dim=1,
            ).detach().cpu().float()

            confidence = torch.minimum(
                torch.max(
                    pred_x_axis,
                    dim=1,
                ).values,
                torch.max(
                    pred_y_axis,
                    dim=1,
                ).values,
            ).detach().cpu()

            for local_index in range(
                batch_size
            ):
                index = (
                    start
                    + local_index
                )

                sample_visible = bool(
                    visibility[
                        index
                    ].item()
                )

                sample_confidence = float(
                    confidence[
                        local_index
                    ].item()
                )

                confidences.append(
                    sample_confidence
                )

                valid = (
                    sample_confidence
                    >= OFFICIAL_THRESHOLD
                )

                if valid:
                    valid_total += 1

                    if sample_visible:
                        valid_visible += 1
                    else:
                        valid_invisible += 1

                if not sample_visible:
                    continue

                predicted_source_x = (
                    float(
                        pred_x[
                            local_index
                        ].item()
                    )
                    * float(
                        widths[
                            index
                        ].item()
                    )
                    / INPUT_WIDTH
                )

                predicted_source_y = (
                    float(
                        pred_y[
                            local_index
                        ].item()
                    )
                    * float(
                        heights[
                            index
                        ].item()
                    )
                    / INPUT_HEIGHT
                )

                gt_x = float(
                    positions_source[
                        index,
                        0,
                    ].item()
                )

                gt_y = float(
                    positions_source[
                        index,
                        1,
                    ].item()
                )

                error = math.hypot(
                    predicted_source_x
                    - gt_x,
                    predicted_source_y
                    - gt_y,
                )

                errors.append(error)

                if error <= 20.0:
                    raw_hits_20 += 1

                    if valid:
                        valid_hits_20 += 1

    precision_20 = (
        valid_hits_20
        / valid_total
        if valid_total > 0
        else 0.0
    )

    recall_20 = (
        valid_hits_20
        / visible_count
        if visible_count > 0
        else 0.0
    )

    raw_recall_20 = (
        raw_hits_20
        / visible_count
        if visible_count > 0
        else 0.0
    )

    return {
        "loss": (
            total_loss
            / total_count
            if total_count
            else None
        ),
        "sample_count":
            total_count,
        "visible_count":
            visible_count,
        "invisible_count":
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
        "raw_hits_at_20px":
            raw_hits_20,
        "raw_recall_at_20px":
            raw_recall_20,
        "valid_hits_at_20px":
            valid_hits_20,
        "precision_at_20px":
            precision_20,
        "recall_at_20px":
            recall_20,
        "f1_at_20px":
            harmonic_f1(
                precision_20,
                recall_20,
            ),
        "median_error_px": (
            statistics.median(
                errors
            )
            if errors
            else None
        ),
        "p90_error_px": (
            percentile(
                errors,
                0.90,
            )
        ),
        "median_confidence": (
            statistics.median(
                confidences
            )
            if confidences
            else None
        ),
    }


def sample_batch_indices(
    visibility: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    visible_indices = torch.nonzero(
        visibility,
        as_tuple=False,
    ).flatten()

    invisible_indices = torch.nonzero(
        ~visibility,
        as_tuple=False,
    ).flatten()

    selected = []

    visible_count = min(
        VISIBLE_PER_BATCH,
        BATCH_SIZE,
    )

    visible_choices = torch.randint(
        low=0,
        high=len(
            visible_indices
        ),
        size=(
            visible_count,
        ),
        generator=generator,
    )

    selected.extend(
        visible_indices[
            visible_choices
        ].tolist()
    )

    remaining = (
        BATCH_SIZE
        - len(selected)
    )

    if (
        remaining > 0
        and len(
            invisible_indices
        ) > 0
    ):
        invisible_choices = torch.randint(
            low=0,
            high=len(
                invisible_indices
            ),
            size=(remaining,),
            generator=generator,
        )

        selected.extend(
            invisible_indices[
                invisible_choices
            ].tolist()
        )

    while len(selected) < BATCH_SIZE:
        extra = torch.randint(
            low=0,
            high=len(
                visible_indices
            ),
            size=(1,),
            generator=generator,
        )

        selected.append(
            int(
                visible_indices[
                    extra
                ].item()
            )
        )

    permutation = torch.randperm(
        len(selected),
        generator=generator,
    )

    return torch.tensor(
        selected,
        dtype=torch.long,
    )[permutation]


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
        "ttflux_d1a_reference",
        D1A_SCRIPT,
    )

    i10 = load_module(
        "ttflux_i10_reference_d1b",
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

    rally_rows = read_csv(
        RALLIES_MANIFEST
    )

    frame_rows = read_csv(
        FRAMES_MANIFEST
    )

    annotations_by_key: dict[
        tuple[int, int],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in frame_rows:
        annotations_by_key[
            annotation_key(row)
        ].append(row)

    candidates = build_candidates(
        rally_rows,
        annotations_by_key,
    )

    train_candidates = (
        choose_train_candidates(
            candidates
        )
    )

    validation_candidates = (
        choose_validation_candidates(
            candidates
        )
    )

    all_selected_matches = {
        candidate["key"][0]
        for candidate in (
            train_candidates
            + validation_candidates
        )
    }

    if any(
        match_id >= 22
        for match_id
        in all_selected_matches
    ):
        raise RuntimeError(
            "Le split test a été utilisé."
        )

    print("")
    print(
        "=== ÉCHANGES TRAIN ==="
    )

    for candidate in train_candidates:
        print(
            f"  {candidate['key'][0]:02d}/"
            f"{candidate['key'][1]:03d} "
            f"fps={candidate['fps_nominal']} "
            f"resolution="
            f"{candidate['resolution'][0]}x"
            f"{candidate['resolution'][1]}"
        )

    print("")
    print(
        "=== ÉCHANGES VALIDATION ==="
    )

    for candidate in validation_candidates:
        print(
            f"  {candidate['key'][0]:02d}/"
            f"{candidate['key'][1]:03d} "
            f"fps={candidate['fps_nominal']} "
            f"resolution="
            f"{candidate['resolution'][0]}x"
            f"{candidate['resolution'][1]}"
        )

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

    print("")
    print(
        "=== CONSTRUCTION TRAIN ==="
    )

    train_dataset = build_dataset(
        train_candidates,
        d1a,
        create_target_ball,
        sigma,
        target_threshold,
    )

    print("")
    print(
        "=== CONSTRUCTION VALIDATION ==="
    )

    validation_dataset = build_dataset(
        validation_candidates,
        d1a,
        create_target_ball,
        sigma,
        target_threshold,
    )

    expected_train_samples = (
        TRAIN_RALLY_COUNT
        * SAMPLES_PER_RALLY
    )

    expected_validation_samples = (
        VALIDATION_RALLY_COUNT
        * SAMPLES_PER_RALLY
    )

    if (
        len(
            train_dataset[
                "inputs"
            ]
        )
        != expected_train_samples
    ):
        raise RuntimeError(
            "Nombre train incorrect."
        )

    if (
        len(
            validation_dataset[
                "inputs"
            ]
        )
        != expected_validation_samples
    ):
        raise RuntimeError(
            "Nombre validation incorrect."
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

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        SEED
    )

    curve_rows = []

    initial_train = evaluate(
        global_stage,
        train_dataset,
        criterion,
        device,
        mean,
        std,
    )

    initial_validation = evaluate(
        global_stage,
        validation_dataset,
        criterion,
        device,
        mean,
        std,
    )

    curve_rows.append(
        {
            "step": 0,
            "split": "train",
            **initial_train,
        }
    )

    curve_rows.append(
        {
            "step": 0,
            "split": "validation",
            **initial_validation,
        }
    )

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

    best_state = cpu_state_dict(
        global_stage
    )

    print("")
    print(
        "INITIAL "
        f"train_loss="
        f"{initial_train['loss']:.6f} "
        f"val_loss="
        f"{initial_validation['loss']:.6f} "
        f"val_raw_R20="
        f"{initial_validation['raw_recall_at_20px']:.3f} "
        f"val_F1="
        f"{initial_validation['f1_at_20px']:.3f}"
    )

    started = time.perf_counter()

    for step in range(
        1,
        TRAINING_STEPS + 1,
    ):
        set_training_mode(
            global_stage
        )

        batch_indices = (
            sample_batch_indices(
                train_dataset[
                    "visibility"
                ],
                generator,
            )
        )

        batch_input = (
            train_dataset[
                "inputs"
            ][batch_indices]
            .to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )

        batch_target = (
            train_dataset[
                "targets"
            ][batch_indices]
            .to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = forward_stage(
            global_stage,
            batch_input,
            mean,
            std,
        )

        loss = criterion(
            prediction,
            batch_target,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Perte non finie à l'étape {step}."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            global_stage.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        if step not in EVALUATION_STEPS:
            continue

        train_metrics = evaluate(
            global_stage,
            train_dataset,
            criterion,
            device,
            mean,
            std,
        )

        validation_metrics = evaluate(
            global_stage,
            validation_dataset,
            criterion,
            device,
            mean,
            std,
        )

        curve_rows.append(
            {
                "step": step,
                "split": "train",
                **train_metrics,
            }
        )

        curve_rows.append(
            {
                "step": step,
                "split": "validation",
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
            best_step = step
            best_state = cpu_state_dict(
                global_stage
            )

        print(
            f"STEP {step:03d} "
            f"train_loss="
            f"{train_metrics['loss']:.6f} "
            f"val_loss="
            f"{validation_metrics['loss']:.6f} "
            f"val_raw_R20="
            f"{validation_metrics['raw_recall_at_20px']:.3f} "
            f"val_R20="
            f"{validation_metrics['recall_at_20px']:.3f} "
            f"val_P20="
            f"{validation_metrics['precision_at_20px']:.3f} "
            f"val_F1="
            f"{validation_metrics['f1_at_20px']:.3f}"
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    global_stage.load_state_dict(
        best_state
    )

    final_train = evaluate(
        global_stage,
        train_dataset,
        criterion,
        device,
        mean,
        std,
    )

    final_validation = evaluate(
        global_stage,
        validation_dataset,
        criterion,
        device,
        mean,
        std,
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
        final_validation["loss"]
        / initial_validation["loss"]
    )

    validation_raw_gain = (
        final_validation[
            "raw_recall_at_20px"
        ]
        - initial_validation[
            "raw_recall_at_20px"
        ]
    )

    pilot_success = bool(
        isolation_ok
        and validation_loss_ratio < 0.95
        and validation_raw_gain >= 0.05
    )

    curve_path = (
        OUTPUT_DIR
        / "training_curve.csv"
    )

    write_csv(
        curve_path,
        curve_rows,
    )

    selected_path = (
        OUTPUT_DIR
        / "selected_rallies.csv"
    )

    selected_rows = []

    for candidate in (
        train_candidates
        + validation_candidates
    ):
        rally = candidate[
            "rally"
        ]

        selected_rows.append(
            {
                "split":
                    rally[
                        "ttflux_split"
                    ],
                "match_id":
                    f"{candidate['key'][0]:02d}",
                "rally_id":
                    f"{candidate['key'][1]:03d}",
                "fps_nominal":
                    candidate[
                        "fps_nominal"
                    ],
                "width":
                    candidate[
                        "resolution"
                    ][0],
                "height":
                    candidate[
                        "resolution"
                    ][1],
                "visible_available":
                    len(
                        candidate[
                            "visible"
                        ]
                    ),
                "invisible_available":
                    len(
                        candidate[
                            "invisible"
                        ]
                    ),
                "samples_used":
                    SAMPLES_PER_RALLY,
                "video_path":
                    rally[
                        "video_path"
                    ],
            }
        )

    write_csv(
        selected_path,
        selected_rows,
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "d1b_global_best.pth"
    )

    torch.save(
        {
            "schema_version": 1,
            "purpose":
                "global_stage_multirally_pilot",
            "source_checkpoint":
                str(CHECKPOINT_PATH),
            "best_step":
                best_step,
            "global_state_dict":
                best_state,
            "validation_metrics":
                final_validation,
            "train_matches": sorted(
                {
                    candidate[
                        "key"
                    ][0]
                    for candidate
                    in train_candidates
                }
            ),
            "validation_matches": [
                20,
                21,
            ],
            "test_matches_excluded": [
                22,
                23,
                24,
                25,
            ],
        },
        checkpoint_path,
    )

    summary = {
        "schema_version": 1,
        "experiment":
            "DSET_D1B_BlurBall_global_multirally_pilot",
        "purpose":
            "pilot_before_full_training",
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
                len(
                    train_candidates
                ),
            "validation_rallies":
                len(
                    validation_candidates
                ),
            "train_samples":
                len(
                    train_dataset[
                        "inputs"
                    ]
                ),
            "validation_samples":
                len(
                    validation_dataset[
                        "inputs"
                    ]
                ),
            "train_match_ids": sorted(
                {
                    candidate[
                        "key"
                    ][0]
                    for candidate
                    in train_candidates
                }
            ),
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
            "training_steps":
                TRAINING_STEPS,
            "batch_size":
                BATCH_SIZE,
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "sequence_length":
                SEQUENCE_LENGTH,
            "target_frame_policy":
                "last_input_frame",
            "samples_per_rally":
                SAMPLES_PER_RALLY,
            "batchnorm_running_stats":
                "frozen",
            "dropout":
                "disabled",
            "trained_module":
                "ball_global_stage",
        },
        "initial": {
            "train":
                initial_train,
            "validation":
                initial_validation,
        },
        "best": {
            "step":
                best_step,
            "train":
                final_train,
            "validation":
                final_validation,
        },
        "diagnostics": {
            "validation_loss_ratio":
                validation_loss_ratio,
            "validation_raw_recall_gain_at_20px":
                validation_raw_gain,
            "training_seconds":
                elapsed,
            "isolation_checks":
                isolation_checks,
            "isolation_ok":
                isolation_ok,
            "pilot_success":
                pilot_success,
        },
        "artifacts": {
            "training_curve_csv":
                str(curve_path),
            "selected_rallies_csv":
                str(selected_path),
            "best_checkpoint":
                str(checkpoint_path),
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
        "DSET_D1B_GLOBAL_PILOT_GENERATED"
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