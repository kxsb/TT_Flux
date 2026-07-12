from __future__ import annotations

import csv
import hashlib
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

import cv2
import numpy as np
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

RALLIES_MANIFEST = Path(
    os.environ["TTFLUX_BLURBALL_RALLIES"]
)

FRAMES_MANIFEST = Path(
    os.environ["TTFLUX_BLURBALL_FRAMES"]
)

OUTPUT_DIR = Path(
    os.environ["TTFLUX_D1A_OUTPUT"]
)

EXPECTED_CHECKPOINT_SHA256 = (
    "c28cb6e720267c96eed6f2aefbe80c1e"
    "13929e6d26171aa6dcfd3d7e58d1df69"
)

SEED = 20260712

INPUT_WIDTH = 320
INPUT_HEIGHT = 128

CANONICAL_WIDTH = 1920
CANONICAL_HEIGHT = 1080

SEQUENCE_LENGTH = 9
HISTORY_SPAN = SEQUENCE_LENGTH - 1

VISIBLE_SAMPLE_TARGET = 32
INVISIBLE_SAMPLE_TARGET = 8

BATCH_SIZE = 8
VISIBLE_PER_BATCH = 6
INVISIBLE_PER_BATCH = 2

GLOBAL_STEPS = 300
LOCAL_STEPS = 300

GLOBAL_LR = 3e-4
LOCAL_LR = 3e-4

EVALUATION_STEPS = {
    0,
    10,
    25,
    50,
    100,
    150,
    200,
    300,
}

OFFICIAL_THRESHOLD = 0.05


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def state_digest(
    module: nn.Module | None,
) -> str | None:
    if module is None:
        return None

    digest = hashlib.sha256()

    for name, tensor in sorted(
        module.state_dict().items()
    ):
        digest.update(
            name.encode("utf-8")
        )

        contiguous = (
            tensor.detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            contiguous.numpy().tobytes()
        )

    return digest.hexdigest()


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


def load_i10_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ttflux_i10_reference_d1a",
        I10_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Import impossible : {I10_SCRIPT}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def evenly_spaced(
    rows: list[dict[str, str]],
    count: int,
) -> list[dict[str, str]]:
    if count <= 0:
        return []

    if len(rows) <= count:
        return list(rows)

    selected_indices = []

    if count == 1:
        selected_indices = [
            len(rows) // 2
        ]
    else:
        for index in range(count):
            position = round(
                index
                * (len(rows) - 1)
                / (count - 1)
            )

            selected_indices.append(
                position
            )

    unique_indices = []

    for index in selected_indices:
        if index not in unique_indices:
            unique_indices.append(index)

    if len(unique_indices) < count:
        for index in range(len(rows)):
            if index not in unique_indices:
                unique_indices.append(index)

            if len(unique_indices) == count:
                break

    return [
        rows[index]
        for index in sorted(
            unique_indices[:count]
        )
    ]


def select_training_rally(
    rally_rows: list[dict[str, str]],
    annotations_by_key: dict[
        tuple[int, int],
        list[dict[str, str]],
    ],
) -> tuple[
    dict[str, str],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    candidates = []

    for rally in rally_rows:
        if rally["ttflux_split"] != "train":
            continue

        key = (
            parse_int(
                rally["match_id"]
            ),
            parse_int(
                rally["rally_id"]
            ),
        )

        annotations = sorted(
            annotations_by_key.get(
                key,
                [],
            ),
            key=lambda row: parse_int(
                row["frame_index"]
            ),
        )

        eligible = [
            row
            for row in annotations
            if parse_int(
                row["frame_index"]
            ) >= HISTORY_SPAN
        ]

        visible = []

        for row in eligible:
            if parse_int(
                row["visibility"]
            ) != 1:
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

        invisible = [
            row
            for row in eligible
            if parse_int(
                row["visibility"]
            ) == 0
        ]

        if len(visible) < VISIBLE_SAMPLE_TARGET:
            continue

        candidates.append(
            {
                "rally": rally,
                "annotations": annotations,
                "visible": visible,
                "invisible": invisible,
                "score": (
                    -min(
                        len(invisible),
                        INVISIBLE_SAMPLE_TARGET,
                    ),
                    abs(
                        len(annotations) - 160
                    ),
                    key[0],
                    key[1],
                ),
            }
        )

    if not candidates:
        raise RuntimeError(
            "Aucun échange train compatible "
            "avec le micro-overfit."
        )

    candidates.sort(
        key=lambda candidate:
            candidate["score"]
    )

    selected = candidates[0]

    visible_samples = evenly_spaced(
        selected["visible"],
        VISIBLE_SAMPLE_TARGET,
    )

    invisible_samples = evenly_spaced(
        selected["invisible"],
        min(
            INVISIBLE_SAMPLE_TARGET,
            len(
                selected["invisible"]
            ),
        ),
    )

    samples = sorted(
        visible_samples
        + invisible_samples,
        key=lambda row: parse_int(
            row["frame_index"]
        ),
    )

    return (
        selected["rally"],
        selected["annotations"],
        samples,
    )


def load_resized_video(
    path: Path,
) -> tuple[
    list[np.ndarray],
    int,
    int,
    float,
]:
    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Vidéo illisible : {path}"
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

    if len(frames) < SEQUENCE_LENGTH:
        raise RuntimeError(
            "Vidéo trop courte."
        )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "Dimensions vidéo invalides."
        )

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    return (
        frames,
        width,
        height,
        fps,
    )


def sequence_tensor_uint8(
    frames: list[np.ndarray],
    target_frame: int,
) -> torch.Tensor:
    first_frame = (
        target_frame
        - HISTORY_SPAN
    )

    selected = frames[
        first_frame:
        target_frame + 1
    ]

    if len(selected) != SEQUENCE_LENGTH:
        raise RuntimeError(
            "Séquence temporelle incomplète : "
            f"target={target_frame}"
        )

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

    return torch.from_numpy(
        contiguous
    )


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


def set_micro_train_mode(
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


def evaluate_stage(
    stage_name: str,
    stage: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    positions_xy: torch.Tensor,
    visibility: torch.Tensor,
    frames: list[int],
    criterion: nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    stage.eval()

    total_loss = 0.0
    total_rows = 0
    details = []

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

            total_rows += batch_size

            x_values = prediction[
                :,
                :INPUT_WIDTH,
            ]

            y_values = prediction[
                :,
                INPUT_WIDTH:,
            ]

            x_indices = torch.argmax(
                x_values,
                dim=1,
            )

            y_indices = torch.argmax(
                y_values,
                dim=1,
            )

            x_confidences = torch.max(
                x_values,
                dim=1,
            ).values

            y_confidences = torch.max(
                y_values,
                dim=1,
            ).values

            confidences = torch.minimum(
                x_confidences,
                y_confidences,
            )

            for local_index in range(
                batch_size
            ):
                source_index = (
                    start + local_index
                )

                visible = bool(
                    visibility[
                        source_index
                    ].item()
                )

                predicted_x = float(
                    x_indices[
                        local_index
                    ].item()
                )

                predicted_y = float(
                    y_indices[
                        local_index
                    ].item()
                )

                confidence = float(
                    confidences[
                        local_index
                    ].item()
                )

                valid = (
                    confidence
                    >= OFFICIAL_THRESHOLD
                )

                target_x = float(
                    positions_xy[
                        source_index,
                        0,
                    ].item()
                )

                target_y = float(
                    positions_xy[
                        source_index,
                        1,
                    ].item()
                )

                error = None

                if visible:
                    error = math.hypot(
                        predicted_x - target_x,
                        predicted_y - target_y,
                    )

                details.append(
                    {
                        "stage": stage_name,
                        "frame_index": (
                            frames[source_index]
                        ),
                        "visibility": (
                            1 if visible else 0
                        ),
                        "target_x": (
                            round(target_x, 4)
                        ),
                        "target_y": (
                            round(target_y, 4)
                        ),
                        "predicted_x": (
                            round(
                                predicted_x,
                                4,
                            )
                        ),
                        "predicted_y": (
                            round(
                                predicted_y,
                                4,
                            )
                        ),
                        "confidence": (
                            round(
                                confidence,
                                8,
                            )
                        ),
                        "valid_at_005": (
                            valid
                        ),
                        "error_stage_px": (
                            round(error, 4)
                            if error is not None
                            else None
                        ),
                    }
                )

    visible_details = [
        row
        for row in details
        if row["visibility"] == 1
    ]

    invisible_details = [
        row
        for row in details
        if row["visibility"] == 0
    ]

    errors = [
        float(
            row["error_stage_px"]
        )
        for row in visible_details
    ]

    valid_visible = [
        row
        for row in visible_details
        if row["valid_at_005"]
    ]

    valid_invisible = [
        row
        for row in invisible_details
        if row["valid_at_005"]
    ]

    raw_hits_2 = sum(
        1
        for row in visible_details
        if float(
            row["error_stage_px"]
        ) <= 2.0
    )

    raw_hits_5 = sum(
        1
        for row in visible_details
        if float(
            row["error_stage_px"]
        ) <= 5.0
    )

    raw_hits_10 = sum(
        1
        for row in visible_details
        if float(
            row["error_stage_px"]
        ) <= 10.0
    )

    valid_hits_5 = sum(
        1
        for row in visible_details
        if (
            row["valid_at_005"]
            and float(
                row["error_stage_px"]
            ) <= 5.0
        )
    )

    summary = {
        "loss": (
            total_loss / total_rows
            if total_rows
            else None
        ),
        "sample_count": len(
            details
        ),
        "visible_count": len(
            visible_details
        ),
        "invisible_count": len(
            invisible_details
        ),
        "valid_visible": len(
            valid_visible
        ),
        "valid_invisible": len(
            valid_invisible
        ),
        "valid_rate_visible": (
            len(valid_visible)
            / len(visible_details)
            if visible_details
            else None
        ),
        "false_positive_rate_invisible": (
            len(valid_invisible)
            / len(invisible_details)
            if invisible_details
            else None
        ),
        "raw_recall_at_2_stage_px": (
            raw_hits_2
            / len(visible_details)
            if visible_details
            else None
        ),
        "raw_recall_at_5_stage_px": (
            raw_hits_5
            / len(visible_details)
            if visible_details
            else None
        ),
        "raw_recall_at_10_stage_px": (
            raw_hits_10
            / len(visible_details)
            if visible_details
            else None
        ),
        "valid_recall_at_5_stage_px": (
            valid_hits_5
            / len(visible_details)
            if visible_details
            else None
        ),
        "median_error_stage_px": (
            statistics.median(
                errors
            )
            if errors
            else None
        ),
        "p90_error_stage_px": (
            percentile(
                errors,
                0.90,
            )
        ),
    }

    return summary, details


def sample_batch_indices(
    visible_indices: list[int],
    invisible_indices: list[int],
    generator: torch.Generator,
) -> torch.Tensor:
    if not visible_indices:
        raise RuntimeError(
            "Aucun échantillon visible."
        )

    visible_tensor = torch.tensor(
        visible_indices,
        dtype=torch.long,
    )

    visible_choice = torch.randint(
        low=0,
        high=len(visible_indices),
        size=(
            min(
                VISIBLE_PER_BATCH,
                BATCH_SIZE,
            ),
        ),
        generator=generator,
    )

    selected = visible_tensor[
        visible_choice
    ].tolist()

    remaining = (
        BATCH_SIZE
        - len(selected)
    )

    if (
        remaining > 0
        and invisible_indices
    ):
        invisible_tensor = torch.tensor(
            invisible_indices,
            dtype=torch.long,
        )

        invisible_choice = torch.randint(
            low=0,
            high=len(
                invisible_indices
            ),
            size=(remaining,),
            generator=generator,
        )

        selected.extend(
            invisible_tensor[
                invisible_choice
            ].tolist()
        )

    while len(selected) < BATCH_SIZE:
        extra_choice = torch.randint(
            low=0,
            high=len(
                visible_indices
            ),
            size=(1,),
            generator=generator,
        )

        selected.append(
            visible_indices[
                int(
                    extra_choice.item()
                )
            ]
        )

    permutation = torch.randperm(
        len(selected),
        generator=generator,
    )

    return torch.tensor(
        selected,
        dtype=torch.long,
    )[permutation]


def train_stage(
    stage_name: str,
    stage: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    positions_xy: torch.Tensor,
    visibility: torch.Tensor,
    frames: list[int],
    steps: int,
    learning_rate: float,
    criterion: nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    for parameter in stage.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        stage.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    visible_indices = [
        index
        for index, value in enumerate(
            visibility.tolist()
        )
        if bool(value)
    ]

    invisible_indices = [
        index
        for index, value in enumerate(
            visibility.tolist()
        )
        if not bool(value)
    ]

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        SEED
        + (
            1
            if stage_name == "global"
            else 2
        )
    )

    curve = []

    initial_summary, initial_details = (
        evaluate_stage(
            stage_name,
            stage,
            inputs,
            targets,
            positions_xy,
            visibility,
            frames,
            criterion,
            device,
            mean,
            std,
        )
    )

    curve.append(
        {
            "stage": stage_name,
            "step": 0,
            **initial_summary,
        }
    )

    started = time.perf_counter()

    for step in range(
        1,
        steps + 1,
    ):
        set_micro_train_mode(
            stage
        )

        batch_indices = sample_batch_indices(
            visible_indices,
            invisible_indices,
            generator,
        )

        batch_input = (
            inputs[
                batch_indices
            ]
            .to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )

        batch_target = (
            targets[
                batch_indices
            ]
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
            stage,
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
                f"Perte non finie : "
                f"stage={stage_name}, "
                f"step={step}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            stage.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        if step in EVALUATION_STEPS:
            summary, _ = evaluate_stage(
                stage_name,
                stage,
                inputs,
                targets,
                positions_xy,
                visibility,
                frames,
                criterion,
                device,
                mean,
                std,
            )

            curve.append(
                {
                    "stage": stage_name,
                    "step": step,
                    **summary,
                }
            )

            print(
                f"{stage_name.upper()} "
                f"step={step:03d} "
                f"loss={summary['loss']:.6f} "
                "R5raw="
                f"{summary['raw_recall_at_5_stage_px']:.3f} "
                "R5valid="
                f"{summary['valid_recall_at_5_stage_px']:.3f} "
                "median="
                f"{summary['median_error_stage_px']:.3f}"
            )

    elapsed = (
        time.perf_counter()
        - started
    )

    final_summary, final_details = (
        evaluate_stage(
            stage_name,
            stage,
            inputs,
            targets,
            positions_xy,
            visibility,
            frames,
            criterion,
            device,
            mean,
            std,
        )
    )

    final_summary[
        "training_seconds"
    ] = elapsed

    return (
        curve,
        initial_summary,
        final_summary,
        initial_details,
        final_details,
    )


def cpu_state_dict(
    module: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach()
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
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponible."
        )

    checkpoint_sha256 = sha256_file(
        CHECKPOINT_PATH
    )

    if (
        checkpoint_sha256
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Le checkpoint TTNet a changé : "
            f"{checkpoint_sha256}"
        )

    device = torch.device(
        "cuda:0"
    )

    i10 = load_i10_module()

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

    if not hasattr(
        wrapper,
        "model",
    ):
        raise RuntimeError(
            "Wrapper TTNet inattendu."
        )

    ttnet = wrapper.model

    if ttnet.ball_local_stage is None:
        raise RuntimeError(
            "Étage local absent."
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
        key = (
            parse_int(
                row["match_id"]
            ),
            parse_int(
                row["rally_id"]
            ),
        )

        annotations_by_key[
            key
        ].append(row)

    (
        selected_rally,
        selected_annotations,
        selected_samples,
    ) = select_training_rally(
        rally_rows,
        annotations_by_key,
    )

    match_id = parse_int(
        selected_rally["match_id"]
    )

    rally_id = parse_int(
        selected_rally["rally_id"]
    )

    if (
        selected_rally[
            "ttflux_split"
        ]
        != "train"
    ):
        raise RuntimeError(
            "L'échange sélectionné n'est "
            "pas dans le split train."
        )

    video_path = (
        ROOT
        / Path(
            selected_rally[
                "video_path"
            ]
        )
    ).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            video_path
        )

    print(
        "SELECTED "
        f"match={match_id:02d} "
        f"rally={rally_id:03d} "
        f"samples={len(selected_samples)}"
    )

    print(f"video={video_path}")

    (
        resized_frames,
        source_width,
        source_height,
        source_fps,
    ) = load_resized_video(
        video_path
    )

    annotation_max = max(
        parse_int(
            row["frame_index"]
        )
        for row in selected_annotations
    )

    if annotation_max >= len(
        resized_frames
    ):
        raise RuntimeError(
            "Les annotations dépassent la vidéo."
        )

    global_inputs_list = []
    global_positions = []
    canonical_positions = []
    visibility_values = []
    selected_frame_indices = []

    for sample in selected_samples:
        frame_index = parse_int(
            sample["frame_index"]
        )

        visibility = parse_int(
            sample["visibility"]
        )

        sequence = sequence_tensor_uint8(
            resized_frames,
            frame_index,
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

            global_x = (
                x_source
                * INPUT_WIDTH
                / source_width
            )

            global_y = (
                y_source
                * INPUT_HEIGHT
                / source_height
            )

            canonical_x = (
                x_source
                * CANONICAL_WIDTH
                / source_width
            )

            canonical_y = (
                y_source
                * CANONICAL_HEIGHT
                / source_height
            )
        else:
            global_x = -1.0
            global_y = -1.0

            canonical_x = -1.0
            canonical_y = -1.0

        global_inputs_list.append(
            sequence
        )

        global_positions.append(
            [
                global_x,
                global_y,
            ]
        )

        canonical_positions.append(
            [
                canonical_x,
                canonical_y,
            ]
        )

        visibility_values.append(
            visibility == 1
        )

        selected_frame_indices.append(
            frame_index
        )

    global_inputs = torch.stack(
        global_inputs_list,
        dim=0,
    )

    global_positions_tensor = torch.tensor(
        global_positions,
        dtype=torch.float32,
    )

    canonical_positions_tensor = torch.tensor(
        canonical_positions,
        dtype=torch.float32,
    )

    visibility_tensor = torch.tensor(
        visibility_values,
        dtype=torch.bool,
    )

    target_device = torch.device(
        "cpu"
    )

    global_targets = torch.stack(
        [
            create_target_ball(
                position,
                sigma=float(
                    configs.sigma
                ),
                w=INPUT_WIDTH,
                h=INPUT_HEIGHT,
                thresh_mask=float(
                    configs[
                        "thresh_ball_pos_mask"
                    ]
                    if isinstance(
                        configs,
                        dict,
                    )
                    else configs.thresh_ball_pos_mask
                ),
                device=target_device,
            )
            for position
            in global_positions_tensor
        ],
        dim=0,
    )

    print(
        "Préparation des crops locaux "
        "teacher-forced..."
    )

    local_inputs_list = []
    local_positions_list = []

    ttnet.eval()

    crop_method = getattr(
        ttnet,
        "__crop_original_batch__",
    )

    local_gt_method = getattr(
        ttnet,
        "__get_groundtruth_local_ball_pos__",
    )

    for sample_index in range(
        len(global_inputs)
    ):
        if (
            sample_index == 0
            or (
                sample_index + 1
            ) % 10 == 0
        ):
            print(
                "LOCAL CROP "
                f"{sample_index + 1}/"
                f"{len(global_inputs)}"
            )

        input_batch = (
            global_inputs[
                sample_index:
                sample_index + 1
            ]
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        teacher_prediction = torch.zeros(
            (
                1,
                INPUT_WIDTH
                + INPUT_HEIGHT,
            ),
            dtype=torch.float32,
            device=device,
        )

        if bool(
            visibility_tensor[
                sample_index
            ].item()
        ):
            global_x = float(
                global_positions_tensor[
                    sample_index,
                    0,
                ].item()
            )

            global_y = float(
                global_positions_tensor[
                    sample_index,
                    1,
                ].item()
            )

            x_index = max(
                0,
                min(
                    INPUT_WIDTH - 1,
                    round(global_x),
                ),
            )

            y_index = max(
                0,
                min(
                    INPUT_HEIGHT - 1,
                    round(global_y),
                ),
            )

            teacher_prediction[
                0,
                x_index,
            ] = 1.0

            teacher_prediction[
                0,
                INPUT_WIDTH
                + y_index,
            ] = 1.0

        original_position = (
            canonical_positions_tensor[
                sample_index:
                sample_index + 1
            ]
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        with torch.inference_mode():
            (
                local_input,
                crop_parameters,
            ) = crop_method(
                input_batch,
                teacher_prediction,
            )

            local_position = (
                local_gt_method(
                    original_position,
                    crop_parameters,
                )
            )

        local_inputs_list.append(
            local_input[
                0
            ]
            .detach()
            .cpu()
            .to(
                dtype=torch.float16
            )
        )

        local_positions_list.append(
            local_position[
                0
            ]
            .detach()
            .cpu()
            .to(
                dtype=torch.float32
            )
        )

        del (
            input_batch,
            teacher_prediction,
            original_position,
            local_input,
            local_position,
        )

        if (
            sample_index + 1
        ) % 8 == 0:
            torch.cuda.empty_cache()

    local_inputs = torch.stack(
        local_inputs_list,
        dim=0,
    )

    local_positions_tensor = torch.stack(
        local_positions_list,
        dim=0,
    )

    local_targets = torch.stack(
        [
            create_target_ball(
                position,
                sigma=float(
                    configs.sigma
                ),
                w=INPUT_WIDTH,
                h=INPUT_HEIGHT,
                thresh_mask=float(
                    configs[
                        "thresh_ball_pos_mask"
                    ]
                    if isinstance(
                        configs,
                        dict,
                    )
                    else configs.thresh_ball_pos_mask
                ),
                device=target_device,
            )
            for position
            in local_positions_tensor
        ],
        dim=0,
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

    initial_digests = {
        "global":
            state_digest(
                ttnet.ball_global_stage
            ),
        "local":
            state_digest(
                ttnet.ball_local_stage
            ),
        "event":
            state_digest(
                ttnet.events_spotting
            ),
        "seg":
            state_digest(
                ttnet.segmentation
            ),
    }

    print("")
    print(
        "=== PHASE GLOBALE ==="
    )

    (
        global_curve,
        global_initial,
        global_final,
        global_initial_details,
        global_final_details,
    ) = train_stage(
        "global",
        ttnet.ball_global_stage,
        global_inputs,
        global_targets,
        global_positions_tensor,
        visibility_tensor,
        selected_frame_indices,
        GLOBAL_STEPS,
        GLOBAL_LR,
        criterion,
        device,
        mean,
        std,
    )

    after_global_digests = {
        "global":
            state_digest(
                ttnet.ball_global_stage
            ),
        "local":
            state_digest(
                ttnet.ball_local_stage
            ),
        "event":
            state_digest(
                ttnet.events_spotting
            ),
        "seg":
            state_digest(
                ttnet.segmentation
            ),
    }

    for parameter in wrapper.parameters():
        parameter.requires_grad = False

    print("")
    print(
        "=== PHASE LOCALE TEACHER-FORCED ==="
    )

    (
        local_curve,
        local_initial,
        local_final,
        local_initial_details,
        local_final_details,
    ) = train_stage(
        "local",
        ttnet.ball_local_stage,
        local_inputs,
        local_targets,
        local_positions_tensor,
        visibility_tensor,
        selected_frame_indices,
        LOCAL_STEPS,
        LOCAL_LR,
        criterion,
        device,
        mean,
        std,
    )

    final_digests = {
        "global":
            state_digest(
                ttnet.ball_global_stage
            ),
        "local":
            state_digest(
                ttnet.ball_local_stage
            ),
        "event":
            state_digest(
                ttnet.events_spotting
            ),
        "seg":
            state_digest(
                ttnet.segmentation
            ),
    }

    isolation_checks = {
        "global_changed":
            (
                initial_digests["global"]
                != after_global_digests[
                    "global"
                ]
            ),
        "local_unchanged_during_global":
            (
                initial_digests["local"]
                == after_global_digests[
                    "local"
                ]
            ),
        "global_unchanged_during_local":
            (
                after_global_digests[
                    "global"
                ]
                == final_digests[
                    "global"
                ]
            ),
        "local_changed":
            (
                after_global_digests[
                    "local"
                ]
                != final_digests[
                    "local"
                ]
            ),
        "event_unchanged":
            (
                initial_digests["event"]
                == final_digests["event"]
            ),
        "seg_unchanged":
            (
                initial_digests["seg"]
                == final_digests["seg"]
            ),
    }

    global_loss_ratio = (
        float(global_final["loss"])
        / float(global_initial["loss"])
    )

    local_loss_ratio = (
        float(local_final["loss"])
        / float(local_initial["loss"])
    )

    global_learning_ok = bool(
        global_loss_ratio < 0.70
        and global_final[
            "raw_recall_at_5_stage_px"
        ] >= max(
            0.50,
            global_initial[
                "raw_recall_at_5_stage_px"
            ]
            + 0.25,
        )
    )

    local_learning_ok = bool(
        local_loss_ratio < 0.70
        and local_final[
            "raw_recall_at_5_stage_px"
        ] >= max(
            0.50,
            local_initial[
                "raw_recall_at_5_stage_px"
            ]
            + 0.25,
        )
    )

    isolation_ok = all(
        isolation_checks.values()
    )

    overfit_success = bool(
        global_learning_ok
        and local_learning_ok
        and isolation_ok
    )

    curve_rows = (
        global_curve
        + local_curve
    )

    curve_path = (
        OUTPUT_DIR
        / "training_curve.csv"
    )

    write_csv(
        curve_path,
        curve_rows,
    )

    detail_rows = []

    for snapshot, rows in (
        (
            "before",
            global_initial_details,
        ),
        (
            "after",
            global_final_details,
        ),
        (
            "before",
            local_initial_details,
        ),
        (
            "after",
            local_final_details,
        ),
    ):
        for row in rows:
            detail_rows.append(
                {
                    "snapshot": snapshot,
                    **row,
                }
            )

    predictions_path = (
        OUTPUT_DIR
        / "sample_predictions.csv"
    )

    write_csv(
        predictions_path,
        detail_rows,
    )

    diagnostic_checkpoint_path = (
        OUTPUT_DIR
        / "d1a_ball_stages_overfit.pth"
    )

    torch.save(
        {
            "schema_version": 1,
            "purpose":
                "diagnostic_micro_overfit_only",
            "source_checkpoint":
                str(CHECKPOINT_PATH),
            "source_checkpoint_sha256":
                checkpoint_sha256,
            "selected_rally": {
                "match_id":
                    f"{match_id:02d}",
                "rally_id":
                    f"{rally_id:03d}",
                "split":
                    "train",
            },
            "global_state_dict":
                cpu_state_dict(
                    ttnet.ball_global_stage
                ),
            "local_state_dict":
                cpu_state_dict(
                    ttnet.ball_local_stage
                ),
            "global_metrics":
                global_final,
            "local_metrics":
                local_final,
        },
        diagnostic_checkpoint_path,
    )

    summary = {
        "schema_version": 1,
        "experiment":
            "DSET_D1A_BlurBall_micro_overfit",
        "purpose":
            "diagnostic_only_not_final_model",
        "source_checkpoint": {
            "path":
                str(CHECKPOINT_PATH),
            "sha256":
                checkpoint_sha256,
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
        "selected_rally": {
            "match_id":
                f"{match_id:02d}",
            "rally_id":
                f"{rally_id:03d}",
            "split":
                selected_rally[
                    "ttflux_split"
                ],
            "video_path":
                str(video_path),
            "video_width":
                source_width,
            "video_height":
                source_height,
            "video_fps":
                source_fps,
            "annotation_rows":
                len(
                    selected_annotations
                ),
        },
        "samples": {
            "total":
                len(selected_samples),
            "visible":
                int(
                    visibility_tensor.sum()
                    .item()
                ),
            "invisible":
                int(
                    (
                        ~visibility_tensor
                    ).sum().item()
                ),
            "frame_indices":
                selected_frame_indices,
        },
        "configuration": {
            "sequence_length":
                SEQUENCE_LENGTH,
            "target_frame_policy":
                "last_input_frame",
            "global_steps":
                GLOBAL_STEPS,
            "local_steps":
                LOCAL_STEPS,
            "batch_size":
                BATCH_SIZE,
            "global_learning_rate":
                GLOBAL_LR,
            "local_learning_rate":
                LOCAL_LR,
            "sigma":
                float(
                    configs.sigma
                ),
            "target_mask_threshold":
                float(
                    configs[
                        "thresh_ball_pos_mask"
                    ]
                    if isinstance(
                        configs,
                        dict,
                    )
                    else configs.thresh_ball_pos_mask
                ),
            "prediction_threshold":
                OFFICIAL_THRESHOLD,
            "local_crop_policy":
                "official_teacher_forced",
            "batchnorm_running_stats":
                "frozen",
            "dropout":
                "disabled",
        },
        "global": {
            "initial":
                global_initial,
            "final":
                global_final,
            "loss_ratio":
                global_loss_ratio,
            "learning_ok":
                global_learning_ok,
        },
        "local": {
            "initial":
                local_initial,
            "final":
                local_final,
            "loss_ratio":
                local_loss_ratio,
            "learning_ok":
                local_learning_ok,
        },
        "isolation_checks":
            isolation_checks,
        "isolation_ok":
            isolation_ok,
        "overfit_success":
            overfit_success,
        "artifacts": {
            "training_curve_csv":
                str(curve_path),
            "sample_predictions_csv":
                str(predictions_path),
            "diagnostic_checkpoint":
                str(
                    diagnostic_checkpoint_path
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
        "DSET_D1A_MICRO_OVERFIT_GENERATED"
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