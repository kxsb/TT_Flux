from __future__ import annotations

import csv
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLIP_ID = "i12a_wide_v61_4"
STRATEGY = "temporal_abstention"
FRAME_COUNT = 150
TRAIL_LENGTH = 24

MANIFEST_PATH = Path(
    "runs/_ball_gt_seed_003D_I12A/"
    "i12a_frozen_manifest.csv"
)

GT_PATH = Path(
    "runs/_ball_candidate_oracle_003D_I13B/"
    "i13b_frame_oracle.csv"
)

PREDICTION_PATH = Path(
    "runs/_ball_temporal_reranking_003D_I14C/"
    "i14c_temporal_predictions.csv"
)

OUTPUT_PATH = Path(
    "runs/_ball_candidate_global_occupancy_003D_I14F/"
    "i14fv2_wide_trajectory_compare.mp4"
)

GT_COLOR = (255, 255, 0)
CORRECT_COLOR = (40, 230, 40)
WRONG_COLOR = (40, 40, 245)
ERROR_COLOR = (0, 220, 255)
TEXT_COLOR = (245, 245, 245)
MUTED_COLOR = (175, 175, 175)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", ".")

    if not text:
        return None

    try:
        result = float(text)
    except ValueError:
        return None

    if not math.isfinite(result):
        return None

    return result


def parse_int(value: Any) -> int:
    result = parse_float(value)

    if result is None:
        raise ValueError(
            f"Entier invalide : {value!r}"
        )

    return int(round(result))


def point(
    x: float | None,
    y: float | None,
) -> tuple[int, int] | None:
    if x is None or y is None:
        return None

    return (
        int(round(x)),
        int(round(y)),
    )


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
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


def scaled_color(
    color: tuple[int, int, int],
    factor: float,
) -> tuple[int, int, int]:
    return tuple(
        int(round(channel * factor))
        for channel in color
    )


def draw_trail(
    image: np.ndarray,
    history: list[
        tuple[
            tuple[int, int],
            tuple[int, int, int],
        ]
    ],
) -> None:
    if not history:
        return

    visible = history[-TRAIL_LENGTH:]

    for index in range(1, len(visible)):
        previous_point, previous_color = (
            visible[index - 1]
        )

        current_point, current_color = (
            visible[index]
        )

        age_factor = (
            index / max(1, len(visible) - 1)
        )

        color = scaled_color(
            current_color,
            0.25 + 0.75 * age_factor,
        )

        thickness = (
            1
            if age_factor < 0.5
            else 2
        )

        cv2.line(
            image,
            previous_point,
            current_point,
            color,
            thickness,
            cv2.LINE_AA,
        )

    for index, (trail_point, color) in enumerate(
        visible
    ):
        age_factor = (
            (index + 1) / len(visible)
        )

        radius = (
            2
            if age_factor < 0.65
            else 3
        )

        cv2.circle(
            image,
            trail_point,
            radius,
            scaled_color(
                color,
                0.30 + 0.70 * age_factor,
            ),
            -1,
            cv2.LINE_AA,
        )


def create_writer(
    path: Path,
    fps: float,
    size: tuple[int, int],
) -> cv2.VideoWriter:
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(
                *codec
            ),
            fps,
            size,
        )

        if writer.isOpened():
            print(
                "codec =",
                codec,
            )

            return writer

        writer.release()

    raise RuntimeError(
        "Aucun codec MP4 disponible."
    )


def main() -> None:
    manifest_matches = [
        row
        for row in read_csv(
            MANIFEST_PATH
        )
        if (
            row.get("clip_id") == CLIP_ID
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
            f"Manifest ambigu : "
            f"{len(manifest_matches)}"
        )

    manifest = manifest_matches[0]

    video_path = Path(
        manifest["source_video"]
    ).resolve()

    start_frame = parse_int(
        manifest["start_frame"]
    )

    end_frame = parse_int(
        manifest["end_frame_exclusive"]
    )

    if end_frame - start_frame != FRAME_COUNT:
        raise RuntimeError(
            "150 frames attendues."
        )

    gt_by_frame: dict[
        int,
        dict[str, Any],
    ] = {}

    for row in read_csv(GT_PATH):
        if row.get("clip_id") != CLIP_ID:
            continue

        local_frame = parse_int(
            row["local_frame"]
        )

        gt_by_frame[local_frame] = {
            "visible":
                parse_int(
                    row["gt_visible"]
                ) == 1,
            "x":
                parse_float(
                    row.get("gt_x")
                ),
            "y":
                parse_float(
                    row.get("gt_y")
                ),
        }

    if len(gt_by_frame) != FRAME_COUNT:
        raise RuntimeError(
            f"GT incomplet : {len(gt_by_frame)}"
        )

    prediction_by_frame: dict[
        int,
        dict[str, Any],
    ] = {}

    for row in read_csv(
        PREDICTION_PATH
    ):
        if (
            row.get("clip_id") != CLIP_ID
            or row.get("strategy")
            != STRATEGY
        ):
            continue

        local_frame = parse_int(
            row["local_frame"]
        )

        candidate_id = str(
            row.get(
                "candidate_id",
                "",
            )
        ).strip()

        prediction_by_frame[
            local_frame
        ] = {
            "candidate_id":
                candidate_id,
            "status":
                str(
                    row.get(
                        "status",
                        "",
                    )
                ),
            "x":
                parse_float(
                    row.get("x")
                ),
            "y":
                parse_float(
                    row.get("y")
                ),
            "rank":
                parse_float(
                    row.get(
                        "candidate_rank"
                    )
                ),
            "probability":
                parse_float(
                    row.get(
                        "intrinsic_probability"
                    )
                ),
            "distance":
                parse_float(
                    row.get(
                        "distance_to_gt_px"
                    )
                ),
        }

    if len(prediction_by_frame) != FRAME_COUNT:
        raise RuntimeError(
            "Prédictions I14C incomplètes : "
            f"{len(prediction_by_frame)}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Vidéo inaccessible : {video_path}"
        )

    source_fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    if (
        not math.isfinite(source_fps)
        or source_fps <= 0
    ):
        source_fps = 50.0

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

    output_fps = max(
        12.0,
        source_fps / 2.0,
    )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        float(start_frame),
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        Path(tempfile.gettempdir())
        / "ttflux_i14fv2_wide.mp4"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    writer = create_writer(
        temporary_path,
        output_fps,
        (width, height),
    )

    gt_history: list[
        tuple[
            tuple[int, int],
            tuple[int, int, int],
        ]
    ] = []

    prediction_history: list[
        tuple[
            tuple[int, int],
            tuple[int, int, int],
        ]
    ] = []

    predicted_count = 0
    correct_count = 0
    wrong_count = 0
    abstention_count = 0

    try:
        for local_frame in range(
            FRAME_COUNT
        ):
            ok, frame = capture.read()

            if not ok:
                raise RuntimeError(
                    "Lecture interrompue à "
                    f"la frame {local_frame}"
                )

            gt = gt_by_frame[
                local_frame
            ]

            prediction = (
                prediction_by_frame[
                    local_frame
                ]
            )

            gt_point = point(
                gt["x"],
                gt["y"],
            )

            prediction_point = point(
                prediction["x"],
                prediction["y"],
            )

            if gt_point is not None:
                gt_history.append(
                    (
                        gt_point,
                        GT_COLOR,
                    )
                )

            status = prediction["status"]

            if prediction_point is None:
                abstention_count += 1
            else:
                predicted_count += 1

                is_correct = (
                    status == "correct"
                )

                if is_correct:
                    prediction_color = (
                        CORRECT_COLOR
                    )

                    correct_count += 1
                else:
                    prediction_color = (
                        WRONG_COLOR
                    )

                    wrong_count += 1

                prediction_history.append(
                    (
                        prediction_point,
                        prediction_color,
                    )
                )

            draw_trail(
                frame,
                gt_history,
            )

            draw_trail(
                frame,
                prediction_history,
            )

            if gt_point is not None:
                cv2.circle(
                    frame,
                    gt_point,
                    9,
                    GT_COLOR,
                    3,
                    cv2.LINE_AA,
                )

                cv2.drawMarker(
                    frame,
                    gt_point,
                    GT_COLOR,
                    cv2.MARKER_TILTED_CROSS,
                    18,
                    2,
                    cv2.LINE_AA,
                )

            if prediction_point is not None:
                prediction_color = (
                    CORRECT_COLOR
                    if status == "correct"
                    else WRONG_COLOR
                )

                cv2.circle(
                    frame,
                    prediction_point,
                    12,
                    prediction_color,
                    3,
                    cv2.LINE_AA,
                )

                cv2.drawMarker(
                    frame,
                    prediction_point,
                    prediction_color,
                    cv2.MARKER_CROSS,
                    20,
                    2,
                    cv2.LINE_AA,
                )

                if (
                    gt_point is not None
                    and status != "correct"
                ):
                    cv2.line(
                        frame,
                        prediction_point,
                        gt_point,
                        ERROR_COLOR,
                        2,
                        cv2.LINE_AA,
                    )

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (width, 94),
                (12, 12, 12),
                -1,
            )

            cv2.addWeighted(
                overlay,
                0.78,
                frame,
                0.22,
                0,
                frame,
            )

            draw_text(
                frame,
                (
                    "I14F-V2 — trajectoire GT "
                    "contre reranking temporel"
                ),
                (18, 27),
                scale=0.65,
                thickness=2,
            )

            draw_text(
                frame,
                (
                    f"frame {local_frame:03d}/149"
                    f" | statut={status}"
                    f" | rang="
                    f"{prediction['rank'] if prediction['rank'] is not None else '-'}"
                    f" | p="
                    f"{prediction['probability']:.3f}"
                    if prediction[
                        "probability"
                    ] is not None
                    else (
                        f"frame {local_frame:03d}/149"
                        f" | statut={status}"
                    )
                ),
                (18, 54),
                scale=0.50,
                color=MUTED_COLOR,
            )

            distance_text = (
                f"{prediction['distance']:.1f}px"
                if prediction[
                    "distance"
                ] is not None
                else "-"
            )

            draw_text(
                frame,
                (
                    f"distance GT={distance_text}"
                    " | cyan=GT"
                    " | vert=correct"
                    " | rouge=erreur"
                    " | jaune=écart"
                ),
                (18, 79),
                scale=0.48,
                color=MUTED_COLOR,
            )

            writer.write(frame)

    finally:
        capture.release()
        writer.release()

    if not temporary_path.exists():
        raise RuntimeError(
            "Le fichier vidéo temporaire "
            "n'a pas été créé."
        )

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    shutil.move(
        str(temporary_path),
        str(OUTPUT_PATH),
    )

    print()
    print(
        "I14FV2_TRAJECTORY_VIDEO_OK"
    )
    print(
        "clip_id =",
        CLIP_ID,
    )
    print(
        "frames =",
        FRAME_COUNT,
    )
    print(
        "source_fps =",
        round(source_fps, 3),
    )
    print(
        "output_fps =",
        round(output_fps, 3),
    )
    print(
        "predicted_frames =",
        predicted_count,
    )
    print(
        "correct_frames =",
        correct_count,
    )
    print(
        "wrong_frames =",
        wrong_count,
    )
    print(
        "abstention_frames =",
        abstention_count,
    )
    print(
        "output =",
        OUTPUT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
