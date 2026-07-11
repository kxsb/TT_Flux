from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np


CLIP_ID = "i12a_wide_v61_4"
STRATEGY = "temporal_abstention"

MANIFEST_PATH = Path(
    "runs/_ball_gt_seed_003D_I12A/"
    "i12a_frozen_manifest.csv"
)

FRAME_PATH = Path(
    "runs/_ball_candidate_oracle_003D_I13B/"
    "i13b_frame_oracle.csv"
)

PREDICTION_PATH = Path(
    "runs/_ball_temporal_reranking_003D_I14C/"
    "i14c_temporal_predictions.csv"
)

OUTPUT_DIR = Path(
    "runs/_ball_false_positive_gallery_003D_I14D"
)

DETAIL_DIR = OUTPUT_DIR / "wide_false_positive_runs"

FALSE_POSITIVE_STATUSES = {
    "wrong_visible",
    "false_positive_invisible",
}

OVERVIEW_COLUMNS = 4
OVERVIEW_CELL_WIDTH = 360
OVERVIEW_CELL_HEIGHT = 285

DETAIL_SAMPLE_COUNT = 5
DETAIL_COLUMN_WIDTH = 300
DETAIL_HEADER_HEIGHT = 72
DETAIL_FULL_HEIGHT = 169
DETAIL_ZOOM_HEIGHT = 180
DETAIL_FOOTER_HEIGHT = 52

PREDICTION_COLOR = (40, 40, 245)
GT_COLOR = (245, 235, 40)
TEXT_COLOR = (235, 235, 235)
MUTED_COLOR = (170, 170, 170)
BACKGROUND_COLOR = (24, 24, 24)
PANEL_COLOR = (38, 38, 38)


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
    raw: Any,
) -> float | None:
    text = str(
        raw or ""
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
    raw: Any,
) -> int:
    result = parse_float(raw)

    if result is None:
        raise ValueError(
            f"Entier invalide : {raw!r}"
        )

    return int(round(result))


def safe_write_image(
    path: Path,
    image: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extension = path.suffix.lower()

    if extension not in {
        ".png",
        ".jpg",
        ".jpeg",
    }:
        raise ValueError(
            f"Extension image invalide : {path}"
        )

    success, encoded = cv2.imencode(
        extension,
        image,
    )

    if not success:
        raise RuntimeError(
            f"Encodage impossible : {path}"
        )

    encoded.tofile(str(path))


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.48,
    color: tuple[int, int, int] = TEXT_COLOR,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def resize_letterbox(
    image: np.ndarray,
    width: int,
    height: int,
    background: tuple[int, int, int] = BACKGROUND_COLOR,
) -> tuple[
    np.ndarray,
    float,
    int,
    int,
]:
    source_height, source_width = (
        image.shape[:2]
    )

    scale = min(
        width / source_width,
        height / source_height,
    )

    resized_width = max(
        1,
        int(round(source_width * scale)),
    )

    resized_height = max(
        1,
        int(round(source_height * scale)),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (height, width, 3),
        background,
        dtype=np.uint8,
    )

    offset_x = (
        width - resized_width
    ) // 2

    offset_y = (
        height - resized_height
    ) // 2

    canvas[
        offset_y:
        offset_y + resized_height,
        offset_x:
        offset_x + resized_width,
    ] = resized

    return (
        canvas,
        scale,
        offset_x,
        offset_y,
    )


def transform_point(
    x: float,
    y: float,
    scale: float,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int]:
    return (
        int(round(
            x * scale + offset_x
        )),
        int(round(
            y * scale + offset_y
        )),
    )


def annotate_full_frame(
    frame: np.ndarray,
    prediction_x: float,
    prediction_y: float,
    gt_x: float | None,
    gt_y: float | None,
    width: int,
    height: int,
) -> np.ndarray:
    resized, scale, offset_x, offset_y = (
        resize_letterbox(
            frame,
            width,
            height,
        )
    )

    prediction_point = transform_point(
        prediction_x,
        prediction_y,
        scale,
        offset_x,
        offset_y,
    )

    cv2.circle(
        resized,
        prediction_point,
        10,
        PREDICTION_COLOR,
        3,
        cv2.LINE_AA,
    )

    cv2.drawMarker(
        resized,
        prediction_point,
        PREDICTION_COLOR,
        cv2.MARKER_CROSS,
        16,
        2,
        cv2.LINE_AA,
    )

    if (
        gt_x is not None
        and gt_y is not None
    ):
        gt_point = transform_point(
            gt_x,
            gt_y,
            scale,
            offset_x,
            offset_y,
        )

        cv2.drawMarker(
            resized,
            gt_point,
            GT_COLOR,
            cv2.MARKER_TILTED_CROSS,
            20,
            3,
            cv2.LINE_AA,
        )

        cv2.line(
            resized,
            prediction_point,
            gt_point,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

    return resized


def crop_with_padding(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    size: int,
) -> tuple[
    np.ndarray,
    int,
    int,
]:
    half = size // 2

    center_x_int = int(round(center_x))
    center_y_int = int(round(center_y))

    x0 = center_x_int - half
    y0 = center_y_int - half
    x1 = x0 + size
    y1 = y0 + size

    frame_height, frame_width = (
        frame.shape[:2]
    )

    canvas = np.full(
        (size, size, 3),
        BACKGROUND_COLOR,
        dtype=np.uint8,
    )

    source_x0 = max(0, x0)
    source_y0 = max(0, y0)
    source_x1 = min(frame_width, x1)
    source_y1 = min(frame_height, y1)

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

    if (
        source_x1 > source_x0
        and source_y1 > source_y0
    ):
        canvas[
            destination_y0:
            destination_y1,
            destination_x0:
            destination_x1,
        ] = frame[
            source_y0:source_y1,
            source_x0:source_x1,
        ]

    return canvas, x0, y0


def annotate_zoom(
    frame: np.ndarray,
    prediction_x: float,
    prediction_y: float,
    gt_x: float | None,
    gt_y: float | None,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    crop_size = 180

    crop, crop_x0, crop_y0 = (
        crop_with_padding(
            frame,
            prediction_x,
            prediction_y,
            crop_size,
        )
    )

    prediction_point = (
        int(round(
            prediction_x - crop_x0
        )),
        int(round(
            prediction_y - crop_y0
        )),
    )

    cv2.circle(
        crop,
        prediction_point,
        12,
        PREDICTION_COLOR,
        3,
        cv2.LINE_AA,
    )

    cv2.drawMarker(
        crop,
        prediction_point,
        PREDICTION_COLOR,
        cv2.MARKER_CROSS,
        22,
        2,
        cv2.LINE_AA,
    )

    if (
        gt_x is not None
        and gt_y is not None
    ):
        gt_point = (
            int(round(
                gt_x - crop_x0
            )),
            int(round(
                gt_y - crop_y0
            )),
        )

        if (
            0 <= gt_point[0] < crop_size
            and 0 <= gt_point[1] < crop_size
        ):
            cv2.drawMarker(
                crop,
                gt_point,
                GT_COLOR,
                cv2.MARKER_TILTED_CROSS,
                22,
                3,
                cv2.LINE_AA,
            )

    return cv2.resize(
        crop,
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    )


def sample_run_rows(
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows

    indices = np.linspace(
        0,
        len(rows) - 1,
        count,
    )

    unique_indices: list[int] = []

    for index in indices:
        rounded = int(round(float(index)))

        if rounded not in unique_indices:
            unique_indices.append(rounded)

    return [
        rows[index]
        for index in unique_indices
    ]


def group_consecutive_rows(
    rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not rows:
        return []

    ordered = sorted(
        rows,
        key=lambda row:
            int(row["local_frame"]),
    )

    groups: list[
        list[dict[str, Any]]
    ] = []

    current = [
        ordered[0]
    ]

    for row in ordered[1:]:
        previous_frame = int(
            current[-1]["local_frame"]
        )

        current_frame = int(
            row["local_frame"]
        )

        if (
            current_frame
            == previous_frame + 1
        ):
            current.append(row)
            continue

        groups.append(current)
        current = [row]

    groups.append(current)

    return groups


def decode_clip_segment(
    video_path: Path,
    start_frame: int,
    frame_count: int,
) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Vidéo inaccessible : {video_path}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        float(start_frame),
    )

    frames: dict[
        int,
        np.ndarray
    ] = {}

    try:
        for local_frame in range(
            frame_count
        ):
            success, frame = capture.read()

            if not success:
                raise RuntimeError(
                    f"Lecture interrompue à la "
                    f"frame locale {local_frame}"
                )

            frames[local_frame] = frame
    finally:
        capture.release()

    return frames


def build_detail_gallery(
    run_id: str,
    rows: list[dict[str, Any]],
    frames: dict[int, np.ndarray],
) -> np.ndarray:
    samples = sample_run_rows(
        rows,
        DETAIL_SAMPLE_COUNT,
    )

    width = (
        DETAIL_COLUMN_WIDTH
        * len(samples)
    )

    height = (
        DETAIL_HEADER_HEIGHT
        + DETAIL_FULL_HEIGHT
        + DETAIL_ZOOM_HEIGHT
        + DETAIL_FOOTER_HEIGHT
    )

    canvas = np.full(
        (height, width, 3),
        BACKGROUND_COLOR,
        dtype=np.uint8,
    )

    distances = [
        float(row["distance_to_gt_px"])
        for row in rows
        if row["distance_to_gt_px"]
        is not None
    ]

    header = (
        f"{run_id} | frames "
        f"{rows[0]['local_frame']}-"
        f"{rows[-1]['local_frame']} | "
        f"length={len(rows)} | "
        f"visible_fp="
        f"{sum(row['gt_visible'] == 1 for row in rows)} | "
        f"invisible_fp="
        f"{sum(row['gt_visible'] == 0 for row in rows)}"
    )

    draw_text(
        canvas,
        header,
        (12, 25),
        scale=0.58,
        thickness=1,
    )

    distance_text = (
        "median_distance="
        + (
            f"{median(distances):.1f}px"
            if distances
            else "n/a"
        )
        + " | rouge=prediction | cyan=GT"
    )

    draw_text(
        canvas,
        distance_text,
        (12, 52),
        scale=0.48,
        color=MUTED_COLOR,
    )

    for column, row in enumerate(samples):
        x0 = (
            column
            * DETAIL_COLUMN_WIDTH
        )

        local_frame = int(
            row["local_frame"]
        )

        frame = frames[local_frame]

        full = annotate_full_frame(
            frame,
            float(row["x"]),
            float(row["y"]),
            row["gt_x"],
            row["gt_y"],
            DETAIL_COLUMN_WIDTH,
            DETAIL_FULL_HEIGHT,
        )

        canvas[
            DETAIL_HEADER_HEIGHT:
            DETAIL_HEADER_HEIGHT
            + DETAIL_FULL_HEIGHT,
            x0:
            x0 + DETAIL_COLUMN_WIDTH,
        ] = full

        zoom = annotate_zoom(
            frame,
            float(row["x"]),
            float(row["y"]),
            row["gt_x"],
            row["gt_y"],
            DETAIL_COLUMN_WIDTH,
            DETAIL_ZOOM_HEIGHT,
        )

        zoom_y0 = (
            DETAIL_HEADER_HEIGHT
            + DETAIL_FULL_HEIGHT
        )

        canvas[
            zoom_y0:
            zoom_y0 + DETAIL_ZOOM_HEIGHT,
            x0:
            x0 + DETAIL_COLUMN_WIDTH,
        ] = zoom

        distance = (
            f"{row['distance_to_gt_px']:.1f}px"
            if row["distance_to_gt_px"]
            is not None
            else "GT invisible"
        )

        line1 = (
            f"f={local_frame:03d} "
            f"rank={row['candidate_rank']} "
            f"p={row['intrinsic_probability']:.3f}"
        )

        line2 = (
            f"{row['status']} | {distance}"
        )

        draw_text(
            canvas,
            line1,
            (
                x0 + 8,
                zoom_y0
                + DETAIL_ZOOM_HEIGHT
                + 20,
            ),
            scale=0.43,
        )

        draw_text(
            canvas,
            line2,
            (
                x0 + 8,
                zoom_y0
                + DETAIL_ZOOM_HEIGHT
                + 42,
            ),
            scale=0.40,
            color=MUTED_COLOR,
        )

        if column > 0:
            cv2.line(
                canvas,
                (x0, 0),
                (x0, height),
                (70, 70, 70),
                1,
            )

    return canvas


def build_overview_cell(
    run_id: str,
    rows: list[dict[str, Any]],
    frames: dict[int, np.ndarray],
) -> np.ndarray:
    cell = np.full(
        (
            OVERVIEW_CELL_HEIGHT,
            OVERVIEW_CELL_WIDTH,
            3,
        ),
        PANEL_COLOR,
        dtype=np.uint8,
    )

    middle_row = rows[
        len(rows) // 2
    ]

    local_frame = int(
        middle_row["local_frame"]
    )

    frame = frames[local_frame]

    preview = annotate_full_frame(
        frame,
        float(middle_row["x"]),
        float(middle_row["y"]),
        middle_row["gt_x"],
        middle_row["gt_y"],
        OVERVIEW_CELL_WIDTH - 16,
        190,
    )

    cell[
        42:232,
        8:
        OVERVIEW_CELL_WIDTH - 8,
    ] = preview

    draw_text(
        cell,
        (
            f"{run_id} | "
            f"f{rows[0]['local_frame']}-"
            f"f{rows[-1]['local_frame']} | "
            f"len={len(rows)}"
        ),
        (10, 25),
        scale=0.49,
    )

    distances = [
        float(row["distance_to_gt_px"])
        for row in rows
        if row["distance_to_gt_px"]
        is not None
    ]

    distance_text = (
        f"med_dist={median(distances):.1f}px"
        if distances
        else "GT invisible"
    )

    second_line = (
        f"visible={sum(row['gt_visible'] == 1 for row in rows)} "
        f"invisible={sum(row['gt_visible'] == 0 for row in rows)} "
        f"{distance_text}"
    )

    draw_text(
        cell,
        second_line,
        (10, 255),
        scale=0.42,
        color=MUTED_COLOR,
    )

    draw_text(
        cell,
        "label: ____________________",
        (10, 277),
        scale=0.42,
        color=TEXT_COLOR,
    )

    return cell


def main() -> None:
    manifest_rows = read_csv(
        MANIFEST_PATH
    )

    prediction_rows = read_csv(
        PREDICTION_PATH
    )

    frame_rows = read_csv(
        FRAME_PATH
    )

    manifest_matches = [
        row
        for row in manifest_rows
        if (
            str(row.get("clip_id", ""))
            == CLIP_ID
            and str(
                row.get(
                    "review_status",
                    "",
                )
            ).strip().lower()
            == "accepted"
        )
    ]

    if len(manifest_matches) != 1:
        raise RuntimeError(
            f"Manifest introuvable ou ambigu "
            f"pour {CLIP_ID}: "
            f"{len(manifest_matches)} entrée(s)"
        )

    manifest = manifest_matches[0]

    video_path = Path(
        manifest["source_video"]
    ).resolve()

    start_frame = parse_int(
        manifest["start_frame"]
    )

    end_frame = parse_int(
        manifest[
            "end_frame_exclusive"
        ]
    )

    frame_count = (
        end_frame - start_frame
    )

    if frame_count != 150:
        raise RuntimeError(
            f"150 frames attendues, "
            f"obtenu {frame_count}"
        )

    frame_gt: dict[
        int,
        dict[str, Any],
    ] = {}

    for row in frame_rows:
        if str(row["clip_id"]) != CLIP_ID:
            continue

        local_frame = parse_int(
            row["local_frame"]
        )

        frame_gt[local_frame] = {
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

    if len(frame_gt) != 150:
        raise RuntimeError(
            f"150 lignes GT attendues pour "
            f"{CLIP_ID}, obtenu {len(frame_gt)}"
        )

    false_positive_rows: list[
        dict[str, Any]
    ] = []

    for row in prediction_rows:
        if (
            str(row["strategy"])
            != STRATEGY
            or str(row["clip_id"])
            != CLIP_ID
        ):
            continue

        status = str(
            row["status"]
        ).strip()

        candidate_id = str(
            row.get(
                "candidate_id",
                "",
            )
        ).strip()

        if (
            status
            not in FALSE_POSITIVE_STATUSES
            or not candidate_id
        ):
            continue

        local_frame = parse_int(
            row["local_frame"]
        )

        metadata = frame_gt[
            local_frame
        ]

        false_positive_rows.append({
            "clip_id":
                CLIP_ID,
            "strategy":
                STRATEGY,
            "local_frame":
                local_frame,
            "source_frame":
                start_frame
                + local_frame,
            "status":
                status,
            "gt_visible":
                metadata[
                    "gt_visible"
                ],
            "gt_x":
                metadata["gt_x"],
            "gt_y":
                metadata["gt_y"],
            "candidate_id":
                candidate_id,
            "candidate_rank":
                parse_int(
                    row["candidate_rank"]
                ),
            "intrinsic_probability":
                (
                    parse_float(
                        row[
                            "intrinsic_probability"
                        ]
                    )
                    or 0.0
                ),
            "x":
                float(row["x"]),
            "y":
                float(row["y"]),
            "distance_to_gt_px":
                parse_float(
                    row.get(
                        "distance_to_gt_px"
                    )
                ),
        })

    false_positive_rows.sort(
        key=lambda row:
            row["local_frame"]
    )

    if len(false_positive_rows) != 48:
        raise RuntimeError(
            "48 faux positifs temporels "
            f"attendus pour wide, obtenu "
            f"{len(false_positive_rows)}"
        )

    runs = group_consecutive_rows(
        false_positive_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DETAIL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for stale_image in DETAIL_DIR.glob(
        "*.png"
    ):
        stale_image.unlink()

    frames = decode_clip_segment(
        video_path,
        start_frame,
        frame_count,
    )

    frame_export_rows: list[
        dict[str, Any]
    ] = []

    run_export_rows: list[
        dict[str, Any]
    ] = []

    for run_index, rows in enumerate(
        runs,
        start=1,
    ):
        run_id = f"R{run_index:03d}"

        distances = [
            float(row["distance_to_gt_px"])
            for row in rows
            if row["distance_to_gt_px"]
            is not None
        ]

        ranks = [
            int(row["candidate_rank"])
            for row in rows
        ]

        probabilities = [
            float(
                row[
                    "intrinsic_probability"
                ]
            )
            for row in rows
        ]

        detail_gallery = (
            build_detail_gallery(
                run_id,
                rows,
                frames,
            )
        )

        detail_filename = (
            f"{run_id}_"
            f"f{rows[0]['local_frame']:03d}_"
            f"f{rows[-1]['local_frame']:03d}.png"
        )

        safe_write_image(
            DETAIL_DIR / detail_filename,
            detail_gallery,
        )

        run_export_rows.append({
            "run_id":
                run_id,
            "clip_id":
                CLIP_ID,
            "strategy":
                STRATEGY,
            "start_local_frame":
                rows[0]["local_frame"],
            "end_local_frame":
                rows[-1]["local_frame"],
            "length":
                len(rows),
            "visible_false_positives":
                sum(
                    row["gt_visible"] == 1
                    for row in rows
                ),
            "invisible_false_positives":
                sum(
                    row["gt_visible"] == 0
                    for row in rows
                ),
            "median_distance_to_gt_px":
                (
                    round(
                        float(
                            median(distances)
                        ),
                        3,
                    )
                    if distances
                    else ""
                ),
            "median_candidate_rank":
                round(
                    float(
                        median(ranks)
                    ),
                    3,
                ),
            "median_intrinsic_probability":
                round(
                    float(
                        median(probabilities)
                    ),
                    6,
                ),
            "mean_prediction_x":
                round(
                    float(
                        np.mean(
                            [
                                row["x"]
                                for row in rows
                            ]
                        )
                    ),
                    3,
                ),
            "mean_prediction_y":
                round(
                    float(
                        np.mean(
                            [
                                row["y"]
                                for row in rows
                            ]
                        )
                    ),
                    3,
                ),
            "detail_image":
                str(
                    Path(
                        DETAIL_DIR.name
                    )
                    / detail_filename
                ),
            "human_label":
                "",
            "human_notes":
                "",
        })

        for row in rows:
            frame_export_rows.append({
                "run_id":
                    run_id,
                "clip_id":
                    CLIP_ID,
                "strategy":
                    STRATEGY,
                "local_frame":
                    row["local_frame"],
                "source_frame":
                    row["source_frame"],
                "status":
                    row["status"],
                "gt_visible":
                    row["gt_visible"],
                "gt_x":
                    (
                        row["gt_x"]
                        if row["gt_x"]
                        is not None
                        else ""
                    ),
                "gt_y":
                    (
                        row["gt_y"]
                        if row["gt_y"]
                        is not None
                        else ""
                    ),
                "candidate_id":
                    row["candidate_id"],
                "candidate_rank":
                    row["candidate_rank"],
                "intrinsic_probability":
                    round(
                        row[
                            "intrinsic_probability"
                        ],
                        8,
                    ),
                "prediction_x":
                    row["x"],
                "prediction_y":
                    row["y"],
                "distance_to_gt_px":
                    (
                        round(
                            row[
                                "distance_to_gt_px"
                            ],
                            6,
                        )
                        if row[
                            "distance_to_gt_px"
                        ] is not None
                        else ""
                    ),
                "human_label":
                    "",
                "human_notes":
                    "",
            })

    overview_rows = math.ceil(
        len(runs)
        / OVERVIEW_COLUMNS
    )

    overview = np.full(
        (
            overview_rows
            * OVERVIEW_CELL_HEIGHT,
            OVERVIEW_COLUMNS
            * OVERVIEW_CELL_WIDTH,
            3,
        ),
        BACKGROUND_COLOR,
        dtype=np.uint8,
    )

    for run_index, rows in enumerate(
        runs,
        start=1,
    ):
        run_id = f"R{run_index:03d}"

        cell = build_overview_cell(
            run_id,
            rows,
            frames,
        )

        grid_index = run_index - 1

        column = (
            grid_index
            % OVERVIEW_COLUMNS
        )

        row_index = (
            grid_index
            // OVERVIEW_COLUMNS
        )

        x0 = (
            column
            * OVERVIEW_CELL_WIDTH
        )

        y0 = (
            row_index
            * OVERVIEW_CELL_HEIGHT
        )

        overview[
            y0:
            y0 + OVERVIEW_CELL_HEIGHT,
            x0:
            x0 + OVERVIEW_CELL_WIDTH,
        ] = cell

    overview_path = (
        OUTPUT_DIR
        / "i14d_wide_fp_overview.png"
    )

    frame_csv_path = (
        OUTPUT_DIR
        / "i14d_wide_fp_frames.csv"
    )

    run_csv_path = (
        OUTPUT_DIR
        / "i14d_wide_fp_runs.csv"
    )

    report_path = (
        OUTPUT_DIR
        / "i14d_wide_fp_report.json"
    )

    safe_write_image(
        overview_path,
        overview,
    )

    write_csv(
        frame_csv_path,
        frame_export_rows,
    )

    write_csv(
        run_csv_path,
        run_export_rows,
    )

    visible_false_positives = sum(
        row["gt_visible"] == 1
        for row in false_positive_rows
    )

    invisible_false_positives = sum(
        row["gt_visible"] == 0
        for row in false_positive_rows
    )

    report = {
        "experiment":
            (
                "003D_I14D_"
                "wide_false_positive_gallery"
            ),
        "clip_id":
            CLIP_ID,
        "strategy":
            STRATEGY,
        "source_video":
            str(video_path),
        "start_frame":
            start_frame,
        "end_frame_exclusive":
            end_frame,
        "false_positive_frames":
            len(false_positive_rows),
        "visible_false_positives":
            visible_false_positives,
        "invisible_false_positives":
            invisible_false_positives,
        "run_count":
            len(runs),
        "longest_run":
            max(
                (
                    len(run)
                    for run in runs
                ),
                default=0,
            ),
        "human_label_vocabulary": [
            "player_head",
            "player_hand_arm",
            "player_torso",
            "player_leg_shoe",
            "table_edge_line",
            "net",
            "reflection_light",
            "background_object",
            "ball_like_but_wrong",
            "other",
            "uncertain",
        ],
        "artifacts": {
            "overview":
                overview_path.name,
            "run_csv":
                run_csv_path.name,
            "frame_csv":
                frame_csv_path.name,
            "detail_directory":
                DETAIL_DIR.name,
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
        "I14D_WIDE_FP_GALLERY_OK"
    )
    print(
        "clip_id =",
        CLIP_ID,
    )
    print(
        "false_positive_frames =",
        len(false_positive_rows),
    )
    print(
        "visible_false_positives =",
        visible_false_positives,
    )
    print(
        "invisible_false_positives =",
        invisible_false_positives,
    )
    print(
        "run_count =",
        len(runs),
    )
    print(
        "longest_run =",
        max(
            (
                len(run)
                for run in runs
            ),
            default=0,
        ),
    )

    print()
    print(
        "=== SÉQUENCES ==="
    )

    for row in run_export_rows:
        print(
            row["run_id"],
            (
                f"frames="
                f"{row['start_local_frame']}-"
                f"{row['end_local_frame']}"
            ),
            (
                f"length="
                f"{row['length']}"
            ),
            (
                f"visible="
                f"{row['visible_false_positives']}"
            ),
            (
                f"invisible="
                f"{row['invisible_false_positives']}"
            ),
            (
                f"median_distance="
                f"{row['median_distance_to_gt_px']}"
            ),
            (
                f"median_rank="
                f"{row['median_candidate_rank']}"
            ),
        )

    print()
    print(
        "overview =",
        overview_path.resolve(),
    )
    print(
        "run_csv =",
        run_csv_path.resolve(),
    )
    print(
        "frame_csv =",
        frame_csv_path.resolve(),
    )
    print(
        "detail_dir =",
        DETAIL_DIR.resolve(),
    )
    print(
        "report =",
        report_path.resolve(),
    )


if __name__ == "__main__":
    main()
