from __future__ import annotations

from pathlib import Path

import numpy as np

from ttflux.tracking.candidates.crop_dataset import (
    CONTEXT_SIZE_PX,
    LOCAL_SIZE_PX,
    SHARD_KEYS,
    center_pixel,
    extract_square_rgb,
    fixed_unicode_array,
    label_id,
    stack_temporal_rgb,
    validate_shard_arrays,
    write_npz_compressed,
)


def test_center_pixel_uses_half_up_rounding() -> None:
    assert center_pixel(
        10.49,
        20.49,
    ) == (10, 20)

    assert center_pixel(
        10.50,
        20.50,
    ) == (11, 21)


def test_extract_square_rgb_converts_and_pads() -> None:
    frame = np.zeros(
        (3, 3, 3),
        dtype=np.uint8,
    )
    frame[0, 0] = (1, 2, 3)

    crop = extract_square_rgb(
        frame,
        x=0.0,
        y=0.0,
        size=4,
    )

    assert crop.shape == (4, 4, 3)
    assert crop.dtype == np.uint8

    assert np.array_equal(
        crop[2, 2],
        np.array(
            [3, 2, 1],
            dtype=np.uint8,
        ),
    )

    assert np.count_nonzero(
        crop[:2]
    ) == 0

    assert np.count_nonzero(
        crop[:, :2]
    ) == 0


def test_temporal_stack_preserves_order() -> None:
    frames = []

    for value in (10, 20, 30):
        frame = np.zeros(
            (3, 3, 3),
            dtype=np.uint8,
        )
        frame[:] = (
            value,
            value + 1,
            value + 2,
        )
        frames.append(frame)

    stack = stack_temporal_rgb(
        frames,
        x=1.0,
        y=1.0,
        size=2,
    )

    assert stack.shape == (
        3,
        2,
        2,
        3,
    )

    assert stack[:, 0, 0, 0].tolist() == [
        12,
        22,
        32,
    ]


def test_npz_schema_without_pickle(
    tmp_path: Path,
) -> None:
    row_count = 2

    arrays = {
        "local_rgb": np.zeros(
            (
                row_count,
                3,
                LOCAL_SIZE_PX,
                LOCAL_SIZE_PX,
                3,
            ),
            dtype=np.uint8,
        ),
        "context_rgb": np.zeros(
            (
                row_count,
                3,
                CONTEXT_SIZE_PX,
                CONTEXT_SIZE_PX,
                3,
            ),
            dtype=np.uint8,
        ),
        "label": np.asarray(
            [
                label_id("ball"),
                label_id("not_ball"),
            ],
            dtype=np.int8,
        ),
        "hard_negative": np.asarray(
            [0, 1],
            dtype=np.uint8,
        ),
        "manifest_index": np.asarray(
            [0, 1],
            dtype=np.int32,
        ),
        "candidate_id": fixed_unicode_array(
            ["candidate_0", "candidate_1"],
            width=64,
        ),
        "clip_id": fixed_unicode_array(
            ["clip_a", "clip_b"],
            width=32,
        ),
        "local_frame": np.asarray(
            [10, 20],
            dtype=np.int32,
        ),
        "source_frame": np.asarray(
            [1010, 1020],
            dtype=np.int32,
        ),
        "rank": np.asarray(
            [1, 2],
            dtype=np.int16,
        ),
        "x": np.asarray(
            [10.5, 20.5],
            dtype=np.float32,
        ),
        "y": np.asarray(
            [30.5, 40.5],
            dtype=np.float32,
        ),
    }

    assert tuple(arrays) == SHARD_KEYS
    assert validate_shard_arrays(arrays) == 2

    path = tmp_path / "test_shard.npz"

    write_npz_compressed(
        path,
        arrays,
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        loaded = {
            key: archive[key]
            for key in SHARD_KEYS
        }

    assert validate_shard_arrays(loaded) == 2
    assert loaded["candidate_id"].tolist() == [
        "candidate_0",
        "candidate_1",
    ]
