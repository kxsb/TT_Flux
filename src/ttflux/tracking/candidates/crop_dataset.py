from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


LOCAL_SIZE_PX = 48
CONTEXT_SIZE_PX = 96
TEMPORAL_OFFSETS = (-1, 0, 1)

LABEL_TO_ID = {
    "not_ball": 0,
    "ball": 1,
    "ignore": -1,
}

SHARD_KEYS = (
    "local_rgb",
    "context_rgb",
    "label",
    "hard_negative",
    "manifest_index",
    "candidate_id",
    "clip_id",
    "local_frame",
    "source_frame",
    "rank",
    "x",
    "y",
)


def center_pixel(
    x: float,
    y: float,
) -> tuple[int, int]:
    return (
        int(math.floor(float(x) + 0.5)),
        int(math.floor(float(y) + 0.5)),
    )


def square_crop_bounds(
    x: float,
    y: float,
    size: int,
) -> tuple[int, int, int, int]:
    if size <= 0:
        raise ValueError("Crop size must be positive.")

    center_x, center_y = center_pixel(x, y)

    x0 = center_x - size // 2
    y0 = center_y - size // 2

    return (
        x0,
        y0,
        x0 + size,
        y0 + size,
    )


def extract_square_rgb(
    frame_bgr: np.ndarray,
    x: float,
    y: float,
    size: int,
) -> np.ndarray:
    if (
        frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
    ):
        raise ValueError(
            "Expected a BGR image with shape [H, W, 3]."
        )

    if frame_bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 video frames.")

    x0, y0, x1, y1 = square_crop_bounds(
        x,
        y,
        size,
    )

    frame_height, frame_width = frame_bgr.shape[:2]

    source_x0 = max(0, x0)
    source_y0 = max(0, y0)
    source_x1 = min(frame_width, x1)
    source_y1 = min(frame_height, y1)

    crop_rgb = np.zeros(
        (size, size, 3),
        dtype=np.uint8,
    )

    if (
        source_x1 <= source_x0
        or source_y1 <= source_y0
    ):
        return crop_rgb

    destination_x0 = source_x0 - x0
    destination_y0 = source_y0 - y0
    destination_x1 = (
        destination_x0
        + source_x1
        - source_x0
    )
    destination_y1 = (
        destination_y0
        + source_y1
        - source_y0
    )

    source_bgr = frame_bgr[
        source_y0:source_y1,
        source_x0:source_x1,
    ]

    crop_rgb[
        destination_y0:destination_y1,
        destination_x0:destination_x1,
    ] = source_bgr[..., ::-1]

    return crop_rgb


def stack_temporal_rgb(
    frames_bgr: Sequence[np.ndarray],
    x: float,
    y: float,
    size: int,
) -> np.ndarray:
    if len(frames_bgr) != len(TEMPORAL_OFFSETS):
        raise ValueError(
            "Expected exactly three temporal frames."
        )

    return np.stack(
        [
            extract_square_rgb(
                frame,
                x,
                y,
                size,
            )
            for frame in frames_bgr
        ],
        axis=0,
    )


def label_id(label: str) -> int:
    try:
        return LABEL_TO_ID[label]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported candidate label: {label!r}"
        ) from exc


def fixed_unicode_array(
    values: Sequence[str],
    width: int,
) -> np.ndarray:
    if width <= 0:
        raise ValueError(
            "Unicode width must be positive."
        )

    normalized = [
        str(value)
        for value in values
    ]

    too_long = [
        value
        for value in normalized
        if len(value) > width
    ]

    if too_long:
        raise ValueError(
            f"Unicode value exceeds width {width}: "
            f"{too_long[0]!r}"
        )

    return np.asarray(
        normalized,
        dtype=f"<U{width}",
    )


def validate_shard_arrays(
    arrays: Mapping[str, np.ndarray],
) -> int:
    if tuple(arrays) != SHARD_KEYS:
        raise ValueError(
            "Unexpected shard key order: "
            f"{tuple(arrays)}"
        )

    local = arrays["local_rgb"]
    context = arrays["context_rgb"]

    if local.ndim != 5:
        raise ValueError(
            "local_rgb must have five dimensions."
        )

    row_count = int(local.shape[0])

    expected_shapes = {
        "local_rgb": (
            row_count,
            3,
            LOCAL_SIZE_PX,
            LOCAL_SIZE_PX,
            3,
        ),
        "context_rgb": (
            row_count,
            3,
            CONTEXT_SIZE_PX,
            CONTEXT_SIZE_PX,
            3,
        ),
        "label": (row_count,),
        "hard_negative": (row_count,),
        "manifest_index": (row_count,),
        "candidate_id": (row_count,),
        "clip_id": (row_count,),
        "local_frame": (row_count,),
        "source_frame": (row_count,),
        "rank": (row_count,),
        "x": (row_count,),
        "y": (row_count,),
    }

    expected_dtypes = {
        "local_rgb": np.dtype(np.uint8),
        "context_rgb": np.dtype(np.uint8),
        "label": np.dtype(np.int8),
        "hard_negative": np.dtype(np.uint8),
        "manifest_index": np.dtype(np.int32),
        "local_frame": np.dtype(np.int32),
        "source_frame": np.dtype(np.int32),
        "rank": np.dtype(np.int16),
        "x": np.dtype(np.float32),
        "y": np.dtype(np.float32),
    }

    for key in SHARD_KEYS:
        array = arrays[key]

        if array.shape != expected_shapes[key]:
            raise ValueError(
                f"{key}: unexpected shape "
                f"{array.shape}, expected "
                f"{expected_shapes[key]}."
            )

        if array.dtype.hasobject:
            raise ValueError(
                f"{key}: object dtype is forbidden."
            )

        expected_dtype = expected_dtypes.get(key)

        if (
            expected_dtype is not None
            and array.dtype != expected_dtype
        ):
            raise ValueError(
                f"{key}: unexpected dtype "
                f"{array.dtype}, expected "
                f"{expected_dtype}."
            )

    if arrays["candidate_id"].dtype.kind != "U":
        raise ValueError(
            "candidate_id must use a fixed unicode dtype."
        )

    if arrays["clip_id"].dtype.kind != "U":
        raise ValueError(
            "clip_id must use a fixed unicode dtype."
        )

    if not set(
        arrays["label"].tolist()
    ).issubset({-1, 0, 1}):
        raise ValueError(
            "Invalid label values in shard."
        )

    if not set(
        arrays["hard_negative"].tolist()
    ).issubset({0, 1}):
        raise ValueError(
            "Invalid hard_negative values."
        )

    return row_count


def write_npz_compressed(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    ordered = {
        key: arrays[key]
        for key in SHARD_KEYS
    }

    validate_shard_arrays(ordered)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **ordered,
        )

    temporary.replace(path)
