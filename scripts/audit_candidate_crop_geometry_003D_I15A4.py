from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EXPECTED_ROWS = 10299
EXPECTED_LABELS = {
    "ball": 322,
    "ignore": 37,
    "not_ball": 9940,
}
EXPECTED_HARD_NEGATIVES = 81

GEOMETRY_SIZES = (
    24,
    32,
    48,
    64,
    96,
    128,
)

PILOT_LOCAL_SIZE = 48
PILOT_CONTEXT_SIZE = 96
TEMPORAL_OFFSETS = (-1, 0, 1)

OUTPUT_REPORT = "i15a4_crop_geometry_report.json"
OUTPUT_GALLERY = "i15a4_crop_geometry_gallery.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=Path(
            "runs/_ball_candidate_labels_003D_I15A3/"
            "i15a3_candidate_label_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/_ball_crop_geometry_003D_I15A4"
        ),
    )
    parser.add_argument(
        "--compression-samples",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any) -> float:
    text = str(value or "").strip().replace(",", ".")

    if not text:
        raise ValueError(f"Missing float value: {value!r}")

    number = float(text)

    if not math.isfinite(number):
        raise ValueError(f"Non-finite float: {value!r}")

    return number


def parse_optional_float(
    value: Any,
) -> float | None:
    text = str(value or "").strip().replace(",", ".")

    if not text:
        return None

    number = float(text)

    if not math.isfinite(number):
        return None

    return number


def parse_int(value: Any) -> int:
    return int(round(parse_float(value)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    return result.stdout.strip()


def percentile(
    values: list[float],
    fraction: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return float(ordered[lower])

    weight = position - lower

    return (
        float(ordered[lower]) * (1.0 - weight)
        + float(ordered[upper]) * weight
    )


def format_number(
    value: float | None,
    digits: int = 6,
) -> float | None:
    if value is None:
        return None

    return round(float(value), digits)


def mib(byte_count: float) -> float:
    return float(byte_count) / (1024.0 * 1024.0)


def normalized_row(
    row: dict[str, str],
) -> dict[str, Any]:
    return {
        **row,
        "local_frame_i": parse_int(row["local_frame"]),
        "source_frame_i": parse_int(row["source_frame"]),
        "rank_i": parse_int(row["rank"]),
        "x_f": parse_float(row["x"]),
        "y_f": parse_float(row["y"]),
        "bbox_x_i": parse_int(row["bbox_x"]),
        "bbox_y_i": parse_int(row["bbox_y"]),
        "bbox_w_i": parse_int(row["bbox_w"]),
        "bbox_h_i": parse_int(row["bbox_h"]),
        "source_width_i": parse_int(
            row["source_width"]
        ),
        "source_height_i": parse_int(
            row["source_height"]
        ),
        "source_fps_f": parse_float(
            row["source_fps"]
        ),
        "hard_negative_i": parse_int(
            row["hard_negative"]
        ),
    }


def center_pixel(
    row: dict[str, Any],
) -> tuple[int, int]:
    return (
        int(math.floor(float(row["x_f"]) + 0.5)),
        int(math.floor(float(row["y_f"]) + 0.5)),
    )


def virtual_crop_box(
    row: dict[str, Any],
    size: int,
) -> tuple[int, int, int, int]:
    center_x, center_y = center_pixel(row)

    x0 = center_x - size // 2
    y0 = center_y - size // 2

    return (
        x0,
        y0,
        x0 + size,
        y0 + size,
    )


def geometry_for_row(
    row: dict[str, Any],
    size: int,
) -> dict[str, Any]:
    width = int(row["source_width_i"])
    height = int(row["source_height_i"])

    x0, y0, x1, y1 = virtual_crop_box(
        row,
        size,
    )

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)

    valid_x0 = max(0, x0)
    valid_y0 = max(0, y0)
    valid_x1 = min(width, x1)
    valid_y1 = min(height, y1)

    valid_width = max(0, valid_x1 - valid_x0)
    valid_height = max(0, valid_y1 - valid_y0)

    valid_pixels = valid_width * valid_height
    total_pixels = size * size
    padded_pixels = total_pixels - valid_pixels

    bbox_x0 = int(row["bbox_x_i"])
    bbox_y0 = int(row["bbox_y_i"])
    bbox_x1 = bbox_x0 + int(row["bbox_w_i"])
    bbox_y1 = bbox_y0 + int(row["bbox_h_i"])

    margins = (
        bbox_x0 - x0,
        x1 - bbox_x1,
        bbox_y0 - y0,
        y1 - bbox_y1,
    )

    return {
        "padding": (
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
        ),
        "padded_pixels": padded_pixels,
        "pad_fraction": (
            float(padded_pixels) / float(total_pixels)
        ),
        "bbox_contained": min(margins) >= 0,
        "minimum_bbox_margin": min(margins),
    }


def summarize_geometry(
    rows: list[dict[str, Any]],
    size: int,
) -> dict[str, Any]:
    padded_rows = 0
    bbox_outside_rows = 0

    side_counts = {
        "left": 0,
        "top": 0,
        "right": 0,
        "bottom": 0,
    }

    pad_fractions: list[float] = []
    bbox_margins: list[float] = []

    for row in rows:
        measurement = geometry_for_row(
            row,
            size,
        )

        padding = measurement["padding"]

        if any(padding):
            padded_rows += 1

        for side, value in zip(
            ("left", "top", "right", "bottom"),
            padding,
        ):
            if value > 0:
                side_counts[side] += 1

        if not measurement["bbox_contained"]:
            bbox_outside_rows += 1

        pad_fractions.append(
            float(measurement["pad_fraction"])
        )
        bbox_margins.append(
            float(
                measurement[
                    "minimum_bbox_margin"
                ]
            )
        )

    return {
        "size_px": size,
        "rows": len(rows),
        "padded_rows": padded_rows,
        "padded_fraction": (
            float(padded_rows) / float(len(rows))
        ),
        "padding_side_counts": side_counts,
        "mean_pad_fraction": (
            sum(pad_fractions)
            / float(len(pad_fractions))
        ),
        "p95_pad_fraction": percentile(
            pad_fractions,
            0.95,
        ),
        "max_pad_fraction": max(pad_fractions),
        "bbox_outside_rows": bbox_outside_rows,
        "minimum_bbox_margin_px": min(
            bbox_margins
        ),
        "p05_bbox_margin_px": percentile(
            bbox_margins,
            0.05,
        ),
        "median_bbox_margin_px": percentile(
            bbox_margins,
            0.50,
        ),
    }


def inspect_video(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open source video: {path}"
        )

    frame_count = int(round(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    ))
    width = int(round(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    ))
    height = int(round(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ))
    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    capture.release()

    return {
        "path": str(path),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
    }


def even_sample(
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["clip_id"]),
            int(row["local_frame_i"]),
            int(row["rank_i"]),
            str(row["candidate_id"]),
        ),
    )

    if count <= 0:
        return []

    if count >= len(ordered):
        return ordered

    if count == 1:
        return [ordered[len(ordered) // 2]]

    indices = [
        (index * (len(ordered) - 1))
        // (count - 1)
        for index in range(count)
    ]

    return [
        ordered[index]
        for index in indices
    ]


def extract_square(
    frame: np.ndarray,
    row: dict[str, Any],
    size: int,
) -> np.ndarray:
    x0, y0, x1, y1 = virtual_crop_box(
        row,
        size,
    )

    frame_height, frame_width = frame.shape[:2]

    source_x0 = max(0, x0)
    source_y0 = max(0, y0)
    source_x1 = min(frame_width, x1)
    source_y1 = min(frame_height, y1)

    crop = np.zeros(
        (size, size, 3),
        dtype=frame.dtype,
    )

    if (
        source_x1 > source_x0
        and source_y1 > source_y0
    ):
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

        crop[
            destination_y0:destination_y1,
            destination_x0:destination_x1,
        ] = frame[
            source_y0:source_y1,
            source_x0:source_x1,
        ]

    return crop


def load_requested_crops(
    rows: list[dict[str, Any]],
    sizes: tuple[int, ...],
) -> dict[tuple[str, int, int], np.ndarray]:
    rows_by_id = {
        str(row["candidate_id"]): row
        for row in rows
    }

    requests: dict[
        str,
        dict[int, set[tuple[str, int]]],
    ] = defaultdict(
        lambda: defaultdict(set)
    )

    for row in rows:
        video_path = str(row["source_video"])
        candidate_id = str(row["candidate_id"])
        source_frame = int(row["source_frame_i"])

        for offset in TEMPORAL_OFFSETS:
            requests[video_path][
                source_frame + offset
            ].add(
                (candidate_id, offset)
            )

    crop_cache: dict[
        tuple[str, int, int],
        np.ndarray,
    ] = {}

    for video_text in sorted(requests):
        video_path = Path(video_text)
        frame_requests = requests[video_text]

        minimum_frame = min(frame_requests)
        maximum_frame = max(frame_requests)

        if minimum_frame < 0:
            raise RuntimeError(
                f"Negative requested frame in {video_path}"
            )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Cannot open video: {video_path}"
            )

        frame_index = 0

        while frame_index <= maximum_frame:
            ok, frame = capture.read()

            if not ok:
                capture.release()
                raise RuntimeError(
                    f"Decode stopped at frame "
                    f"{frame_index} in {video_path}"
                )

            for candidate_id, offset in sorted(
                frame_requests.get(
                    frame_index,
                    set(),
                )
            ):
                row = rows_by_id[candidate_id]

                for size in sizes:
                    crop_cache[
                        (
                            candidate_id,
                            size,
                            offset,
                        )
                    ] = extract_square(
                        frame,
                        row,
                        size,
                    )

            frame_index += 1

        capture.release()

    expected_count = (
        len(rows)
        * len(sizes)
        * len(TEMPORAL_OFFSETS)
    )

    if len(crop_cache) != expected_count:
        raise RuntimeError(
            "Crop request count mismatch: "
            f"{len(crop_cache)} != "
            f"{expected_count}"
        )

    return crop_cache


def png_size(image: np.ndarray) -> int:
    ok, encoded = cv2.imencode(
        ".png",
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ],
    )

    if not ok:
        raise RuntimeError("PNG encoding failed.")

    return int(encoded.size)


def compression_summary(
    rows: list[dict[str, Any]],
    crop_cache: dict[
        tuple[str, int, int],
        np.ndarray,
    ],
    size: int,
    total_rows: int,
) -> dict[str, Any]:
    bytes_per_candidate: list[int] = []

    for row in rows:
        candidate_id = str(row["candidate_id"])

        encoded_bytes = sum(
            png_size(
                crop_cache[
                    (
                        candidate_id,
                        size,
                        offset,
                    )
                ]
            )
            for offset in TEMPORAL_OFFSETS
        )

        bytes_per_candidate.append(
            encoded_bytes
        )

    mean_bytes = (
        sum(bytes_per_candidate)
        / float(len(bytes_per_candidate))
    )

    return {
        "size_px": size,
        "sample_rows": len(rows),
        "mean_png_bytes_per_candidate_triplet":
            mean_bytes,
        "median_png_bytes_per_candidate_triplet":
            percentile(
                [
                    float(value)
                    for value in bytes_per_candidate
                ],
                0.50,
            ),
        "p95_png_bytes_per_candidate_triplet":
            percentile(
                [
                    float(value)
                    for value in bytes_per_candidate
                ],
                0.95,
            ),
        "estimated_total_png_bytes":
            mean_bytes * float(total_rows),
        "estimated_total_png_mib":
            mib(mean_bytes * float(total_rows)),
    }


def annotate_crop(
    crop: np.ndarray,
    row: dict[str, Any],
    size: int,
    draw_bbox: bool,
    color: tuple[int, int, int],
) -> np.ndarray:
    result = crop.copy()

    x0, y0, _, _ = virtual_crop_box(
        row,
        size,
    )

    center_x, center_y = center_pixel(row)

    local_center_x = center_x - x0
    local_center_y = center_y - y0

    cv2.drawMarker(
        result,
        (
            local_center_x,
            local_center_y,
        ),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=max(5, size // 8),
        thickness=1,
    )

    if draw_bbox:
        bbox_x0 = int(row["bbox_x_i"]) - x0
        bbox_y0 = int(row["bbox_y_i"]) - y0
        bbox_x1 = (
            bbox_x0
            + int(row["bbox_w_i"])
            - 1
        )
        bbox_y1 = (
            bbox_y0
            + int(row["bbox_h_i"])
            - 1
        )

        cv2.rectangle(
            result,
            (bbox_x0, bbox_y0),
            (bbox_x1, bbox_y1),
            color,
            1,
        )

    return result


def build_gallery(
    rows: list[dict[str, Any]],
    categories: dict[str, str],
    crop_cache: dict[
        tuple[str, int, int],
        np.ndarray,
    ],
) -> np.ndarray:
    tile_width = 256
    tile_height = 318
    column_count = 4
    row_count = int(math.ceil(
        len(rows) / float(column_count)
    ))

    sheet = np.full(
        (
            row_count * tile_height,
            column_count * tile_width,
            3,
        ),
        24,
        dtype=np.uint8,
    )

    colors = {
        "ball": (40, 220, 40),
        "ignore": (0, 220, 220),
        "hard_negative": (40, 40, 240),
        "not_ball": (220, 220, 220),
    }

    for index, row in enumerate(rows):
        category = categories[
            str(row["candidate_id"])
        ]
        color = colors[category]

        tile = np.full(
            (tile_height, tile_width, 3),
            16,
            dtype=np.uint8,
        )

        header = (
            f"{category} "
            f"{row['clip_id']} "
            f"f{row['local_frame_i']} "
            f"r{row['rank_i']}"
        )

        distance = parse_optional_float(
            row.get("distance_to_gt_px")
        )

        distance_text = (
            "-"
            if distance is None
            else f"{distance:.1f}px"
        )

        subheader = (
            f"{str(row['candidate_id'])[-20:]} "
            f"d={distance_text}"
        )

        cv2.putText(
            tile,
            header[:38],
            (8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            subheader[:38],
            (8, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        local_x = 32
        local_y = 42
        local_display_size = 64

        for offset_index, offset in enumerate(
            TEMPORAL_OFFSETS
        ):
            crop = crop_cache[
                (
                    str(row["candidate_id"]),
                    PILOT_LOCAL_SIZE,
                    offset,
                )
            ]

            annotated = annotate_crop(
                crop,
                row,
                PILOT_LOCAL_SIZE,
                draw_bbox=(offset == 0),
                color=color,
            )

            resized = cv2.resize(
                annotated,
                (
                    local_display_size,
                    local_display_size,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

            target_x = (
                local_x
                + offset_index
                * local_display_size
            )

            tile[
                local_y:
                    local_y + local_display_size,
                target_x:
                    target_x + local_display_size,
            ] = resized

            cv2.putText(
                tile,
                f"t{offset:+d}",
                (
                    target_x + 4,
                    local_y + 12,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                1,
                cv2.LINE_AA,
            )

        context = crop_cache[
            (
                str(row["candidate_id"]),
                PILOT_CONTEXT_SIZE,
                0,
            )
        ]

        annotated_context = annotate_crop(
            context,
            row,
            PILOT_CONTEXT_SIZE,
            draw_bbox=True,
            color=color,
        )

        context_resized = cv2.resize(
            annotated_context,
            (192, 192),
            interpolation=cv2.INTER_NEAREST,
        )

        context_x = 32
        context_y = 116

        tile[
            context_y:context_y + 192,
            context_x:context_x + 192,
        ] = context_resized

        cv2.rectangle(
            tile,
            (0, 0),
            (
                tile_width - 1,
                tile_height - 1,
            ),
            color,
            2,
        )

        sheet_row = index // column_count
        sheet_column = index % column_count

        y0 = sheet_row * tile_height
        x0 = sheet_column * tile_width

        sheet[
            y0:y0 + tile_height,
            x0:x0 + tile_width,
        ] = tile

    return sheet


def write_png(
    path: Path,
    image: np.ndarray,
) -> None:
    ok, encoded = cv2.imencode(
        ".png",
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ],
    )

    if not ok:
        raise RuntimeError(
            f"Cannot encode PNG: {path}"
        )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    encoded.tofile(str(temporary))
    temporary.replace(path)


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )

    temporary.replace(path)


def main() -> None:
    args = parse_args()

    manifest_path = (
        args.candidate_manifest.resolve()
    )
    output_dir = args.output_dir.resolve()

    raw_rows = read_csv(manifest_path)

    if len(raw_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"got {len(raw_rows)}."
        )

    rows = [
        normalized_row(row)
        for row in raw_rows
    ]

    candidate_ids = [
        str(row["candidate_id"])
        for row in rows
    ]

    if len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise RuntimeError(
            "Duplicate candidate_id values."
        )

    label_counts = Counter(
        str(row["label"])
        for row in rows
    )

    if dict(label_counts) != EXPECTED_LABELS:
        raise RuntimeError(
            f"Unexpected labels: {dict(label_counts)}"
        )

    hard_negative_count = sum(
        int(row["hard_negative_i"])
        for row in rows
    )

    if (
        hard_negative_count
        != EXPECTED_HARD_NEGATIVES
    ):
        raise RuntimeError(
            "Unexpected hard negative count: "
            f"{hard_negative_count}"
        )

    video_rows: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        video_rows[
            str(row["source_video"])
        ].append(row)

    video_information: list[dict[str, Any]] = []
    video_info_by_path: dict[str, dict[str, Any]] = {}

    for video_text in sorted(video_rows):
        information = inspect_video(
            Path(video_text)
        )

        sample_row = video_rows[video_text][0]

        if (
            information["width"]
            != int(sample_row["source_width_i"])
            or information["height"]
            != int(sample_row["source_height_i"])
        ):
            raise RuntimeError(
                f"Video geometry mismatch: {video_text}"
            )

        expected_fps = float(
            sample_row["source_fps_f"]
        )

        if abs(
            float(information["fps"])
            - expected_fps
        ) > 0.01:
            raise RuntimeError(
                f"Video fps mismatch: {video_text}"
            )

        video_information.append(
            information
        )
        video_info_by_path[
            video_text
        ] = information

    unique_frames = {
        (
            str(row["source_video"]),
            int(row["source_frame_i"]),
        )
        for row in rows
    }

    source_triplet_frames = sum(
        1
        for video_text, source_frame
        in unique_frames
        if (
            source_frame - 1 >= 0
            and source_frame + 1
            < int(
                video_info_by_path[
                    video_text
                ]["frame_count"]
            )
        )
    )

    clip_local_triplet_frames = len({
        (
            str(row["clip_id"]),
            int(row["local_frame_i"]),
        )
        for row in rows
        if 1 <= int(row["local_frame_i"]) <= 148
    })

    source_triplet_rows = sum(
        1
        for row in rows
        if (
            int(row["source_frame_i"]) - 1 >= 0
            and int(row["source_frame_i"]) + 1
            < int(
                video_info_by_path[
                    str(row["source_video"])
                ]["frame_count"]
            )
        )
    )

    if source_triplet_frames != len(
        unique_frames
    ):
        raise RuntimeError(
            "Some source t-1/t/t+1 triplets "
            "are unavailable."
        )

    geometry_options = {
        str(size): summarize_geometry(
            rows,
            size,
        )
        for size in GEOMETRY_SIZES
    }

    compression_sample_rows = even_sample(
        rows,
        min(
            args.compression_samples,
            len(rows),
        ),
    )

    gallery_groups = (
        (
            "ball",
            [
                row
                for row in rows
                if row["label"] == "ball"
            ],
            8,
        ),
        (
            "ignore",
            [
                row
                for row in rows
                if row["label"] == "ignore"
            ],
            6,
        ),
        (
            "hard_negative",
            [
                row
                for row in rows
                if row["hard_negative_i"] == 1
            ],
            8,
        ),
        (
            "not_ball",
            [
                row
                for row in rows
                if (
                    row["label"] == "not_ball"
                    and row["hard_negative_i"] == 0
                )
            ],
            10,
        ),
    )

    gallery_rows: list[dict[str, Any]] = []
    gallery_categories: dict[str, str] = {}

    for category, group_rows, count in (
        gallery_groups
    ):
        selected = even_sample(
            group_rows,
            min(count, len(group_rows)),
        )

        for row in selected:
            candidate_id = str(
                row["candidate_id"]
            )

            gallery_rows.append(row)
            gallery_categories[
                candidate_id
            ] = category

    rows_for_images: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in (
        compression_sample_rows
        + gallery_rows
    ):
        rows_for_images[
            str(row["candidate_id"])
        ] = row

    image_rows = sorted(
        rows_for_images.values(),
        key=lambda row: (
            str(row["clip_id"]),
            int(row["local_frame_i"]),
            int(row["rank_i"]),
            str(row["candidate_id"]),
        ),
    )

    crop_cache = load_requested_crops(
        image_rows,
        (
            PILOT_LOCAL_SIZE,
            PILOT_CONTEXT_SIZE,
        ),
    )

    local_compression = compression_summary(
        compression_sample_rows,
        crop_cache,
        PILOT_LOCAL_SIZE,
        len(rows),
    )
    context_compression = compression_summary(
        compression_sample_rows,
        crop_cache,
        PILOT_CONTEXT_SIZE,
        len(rows),
    )

    gallery = build_gallery(
        gallery_rows,
        gallery_categories,
        crop_cache,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gallery_path = (
        output_dir / OUTPUT_GALLERY
    )
    report_path = (
        output_dir / OUTPUT_REPORT
    )

    write_png(
        gallery_path,
        gallery,
    )

    all_rows_raw_bytes = (
        len(rows)
        * len(TEMPORAL_OFFSETS)
        * (
            PILOT_LOCAL_SIZE
            * PILOT_LOCAL_SIZE
            + PILOT_CONTEXT_SIZE
            * PILOT_CONTEXT_SIZE
        )
        * 3
    )

    trainable_rows = sum(
        1
        for row in rows
        if row["label"] != "ignore"
    )

    trainable_raw_bytes = (
        trainable_rows
        * len(TEMPORAL_OFFSETS)
        * (
            PILOT_LOCAL_SIZE
            * PILOT_LOCAL_SIZE
            + PILOT_CONTEXT_SIZE
            * PILOT_CONTEXT_SIZE
        )
        * 3
    )

    report = {
        "experiment":
            "003D_I15A4_crop_geometry_audit",
        "schema_version": 1,
        "source_commit": current_commit(),
        "opencv_version": cv2.__version__,
        "input": {
            "candidate_manifest":
                str(manifest_path),
            "candidate_manifest_sha256":
                sha256_file(manifest_path),
        },
        "dataset": {
            "rows": len(rows),
            "trainable_rows": trainable_rows,
            "labels": dict(label_counts),
            "hard_negatives":
                hard_negative_count,
            "unique_source_frames":
                len(unique_frames),
            "source_videos":
                len(video_information),
            "maximum_bbox_width_px": max(
                int(row["bbox_w_i"])
                for row in rows
            ),
            "maximum_bbox_height_px": max(
                int(row["bbox_h_i"])
                for row in rows
            ),
        },
        "videos": video_information,
        "temporal_context": {
            "offsets": list(
                TEMPORAL_OFFSETS
            ),
            "source_triplet_frames_available":
                source_triplet_frames,
            "source_triplet_frames_total":
                len(unique_frames),
            "source_triplet_rows_available":
                source_triplet_rows,
            "source_triplet_rows_total":
                len(rows),
            "clip_local_triplet_frames_available":
                clip_local_triplet_frames,
            "clip_local_triplet_frames_total":
                len(unique_frames),
            "source_video_required_at_clip_edges":
                True,
        },
        "geometry_options": {
            size: {
                key: format_number(value)
                if isinstance(value, float)
                else value
                for key, value
                in summary.items()
            }
            for size, summary
            in geometry_options.items()
        },
        "pilot_geometry": {
            "local_size_px":
                PILOT_LOCAL_SIZE,
            "context_size_px":
                PILOT_CONTEXT_SIZE,
            "centering_rule":
                "floor(candidate_coordinate + 0.5)",
            "padding_rule":
                "constant_zero_outside_source",
            "temporal_offsets":
                list(TEMPORAL_OFFSETS),
            "status":
                "recommended_for_i15a5",
        },
        "storage": {
            "channels": 3,
            "dtype": "uint8",
            "all_rows_raw_bytes":
                all_rows_raw_bytes,
            "all_rows_raw_mib":
                format_number(
                    mib(all_rows_raw_bytes),
                    3,
                ),
            "trainable_rows_raw_bytes":
                trainable_raw_bytes,
            "trainable_rows_raw_mib":
                format_number(
                    mib(trainable_raw_bytes),
                    3,
                ),
            "float32_training_working_set_mib":
                format_number(
                    mib(
                        all_rows_raw_bytes * 4
                    ),
                    3,
                ),
            "png_pilot": {
                "sample_rows":
                    len(compression_sample_rows),
                "sample_labels": dict(
                    Counter(
                        str(row["label"])
                        for row
                        in compression_sample_rows
                    )
                ),
                "local": {
                    key: format_number(value)
                    if isinstance(value, float)
                    else value
                    for key, value
                    in local_compression.items()
                },
                "context": {
                    key: format_number(value)
                    if isinstance(value, float)
                    else value
                    for key, value
                    in context_compression.items()
                },
                "estimated_combined_png_mib":
                    format_number(
                        float(
                            local_compression[
                                "estimated_total_png_mib"
                            ]
                        )
                        + float(
                            context_compression[
                                "estimated_total_png_mib"
                            ]
                        ),
                        3,
                    ),
            },
        },
        "gallery": {
            "rows": len(gallery_rows),
            "categories": dict(
                Counter(
                    gallery_categories[
                        str(row["candidate_id"])
                    ]
                    for row in gallery_rows
                )
            ),
            "local_temporal_strip_px":
                PILOT_LOCAL_SIZE,
            "context_center_frame_px":
                PILOT_CONTEXT_SIZE,
            "filename":
                gallery_path.name,
            "sha256":
                sha256_file(gallery_path),
        },
        "artifacts": {
            "report_json":
                report_path.name,
            "gallery_png":
                gallery_path.name,
        },
    }

    atomic_write_json(
        report_path,
        report,
    )

    if not args.quiet:
        print()
        print("I15A4_GEOMETRY_AUDIT_OK")

        print()
        print("=== DATASET ===")
        print("rows =", len(rows))
        print("trainable_rows =", trainable_rows)
        print("labels =", dict(label_counts))
        print(
            "hard_negatives =",
            hard_negative_count,
        )
        print(
            "max_bbox =",
            (
                report["dataset"][
                    "maximum_bbox_width_px"
                ],
                report["dataset"][
                    "maximum_bbox_height_px"
                ],
            ),
        )

        print()
        print("=== TEMPORAL CONTEXT ===")
        print(
            "source_triplet_frames =",
            f"{source_triplet_frames}/"
            f"{len(unique_frames)}",
        )
        print(
            "source_triplet_rows =",
            f"{source_triplet_rows}/"
            f"{len(rows)}",
        )
        print(
            "clip_local_triplet_frames =",
            f"{clip_local_triplet_frames}/"
            f"{len(unique_frames)}",
        )

        print()
        print("=== GEOMETRY OPTIONS ===")

        for size in GEOMETRY_SIZES:
            summary = geometry_options[
                str(size)
            ]

            print(
                f"{size:>3}px",
                "padded=",
                summary["padded_rows"],
                "bbox_out=",
                summary["bbox_outside_rows"],
                "mean_pad=",
                f"{summary['mean_pad_fraction']:.6f}",
                "min_margin=",
                summary[
                    "minimum_bbox_margin_px"
                ],
            )

        print()
        print("=== PILOT 32 / 96 ===")
        print(
            "raw_all_mib =",
            report["storage"][
                "all_rows_raw_mib"
            ],
        )
        print(
            "raw_trainable_mib =",
            report["storage"][
                "trainable_rows_raw_mib"
            ],
        )
        print(
            "png_estimated_mib =",
            report["storage"][
                "png_pilot"
            ]["estimated_combined_png_mib"],
        )
        print(
            "compression_sample_rows =",
            len(compression_sample_rows),
        )

        print()
        print("=== GALLERY ===")
        print(
            "rows =",
            len(gallery_rows),
        )
        print(
            "categories =",
            report["gallery"]["categories"],
        )
        print(
            "gallery =",
            gallery_path,
        )
        print(
            "gallery_sha256 =",
            report["gallery"]["sha256"],
        )
        print(
            "report =",
            report_path,
        )


if __name__ == "__main__":
    main()
