from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLIP_ID = "i12a_wide_v61_4"
REFERENCE_LOCAL_FRAME = 123

MANIFEST_PATH = Path(
    "runs/_ball_gt_seed_003D_I12A/"
    "i12a_frozen_manifest.csv"
)

CANDIDATE_PATH = Path(
    "runs/_ball_candidate_oracle_003D_I13B/"
    "i13b_candidates_with_gt_distance.csv"
)

OCCUPANCY_PATH = Path(
    "runs/_ball_candidate_global_occupancy_003D_I14F/"
    "i14f_occupancy_predictions.csv"
)

TEMPORAL_PATH = Path(
    "runs/_ball_temporal_reranking_003D_I14C/"
    "i14c_temporal_predictions.csv"
)

OUTPUT_PATH = Path(
    "runs/_ball_candidate_global_occupancy_003D_I14F/"
    "i14f_wide_visual.png"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", ".")

    if not text:
        return None

    try:
        result = float(text)
    except ValueError:
        return None

    return result if math.isfinite(result) else None


def as_int(value: Any) -> int:
    result = as_float(value)

    if result is None:
        raise ValueError(
            f"Entier invalide : {value!r}"
        )

    return int(round(result))


def safe_write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok, encoded = cv2.imencode(
        path.suffix,
        image,
    )

    if not ok:
        raise RuntimeError(
            f"Encodage impossible : {path}"
        )

    encoded.tofile(str(path))


def put_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float = 0.58,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def load_reference_frame(
    video_path: Path,
    source_frame: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Vidéo inaccessible : {video_path}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        float(source_frame),
    )

    ok, frame = capture.read()
    capture.release()

    if not ok:
        raise RuntimeError(
            f"Frame inaccessible : {source_frame}"
        )

    return frame


def main() -> None:
    manifest_rows = read_csv(
        MANIFEST_PATH
    )

    manifests = [
        row
        for row in manifest_rows
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

    if len(manifests) != 1:
        raise RuntimeError(
            f"Manifest ambigu : {len(manifests)}"
        )

    manifest = manifests[0]

    video_path = Path(
        manifest["source_video"]
    ).resolve()

    start_frame = as_int(
        manifest["start_frame"]
    )

    source_frame = (
        start_frame
        + REFERENCE_LOCAL_FRAME
    )

    frame = load_reference_frame(
        video_path,
        source_frame,
    )

    height, width = frame.shape[:2]

    candidate_rows = [
        row
        for row in read_csv(
            CANDIDATE_PATH
        )
        if row.get("clip_id") == CLIP_ID
    ]

    candidate_lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in candidate_rows:
        candidate_id = str(
            row["candidate_id"]
        )

        candidate_lookup[candidate_id] = {
            "candidate_id":
                candidate_id,
            "local_frame":
                as_int(
                    row["local_frame"]
                ),
            "x":
                float(row["x"]),
            "y":
                float(row["y"]),
        }

    occupancy_rows = [
        row
        for row in read_csv(
            OCCUPANCY_PATH
        )
        if (
            row.get("clip_id") == CLIP_ID
            and row.get("model")
            == "intrinsic_plus_occupancy"
        )
    ]

    heat = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    weighted_points: list[
        tuple[int, int, float]
    ] = []

    ball_candidate_ids: set[str] = set()

    for row in occupancy_rows:
        candidate_id = str(
            row["candidate_id"]
        )

        candidate = candidate_lookup.get(
            candidate_id
        )

        if candidate is None:
            continue

        density = as_float(
            row.get(
                "global_density_64"
            )
        )

        if density is None:
            continue

        x = int(round(
            candidate["x"]
        ))

        y = int(round(
            candidate["y"]
        ))

        weight = math.log1p(
            max(0.0, density)
        )

        weighted_points.append(
            (x, y, weight)
        )

        if as_int(
            row["positive_20px"]
        ) == 1:
            ball_candidate_ids.add(
                candidate_id
            )

    if not weighted_points:
        raise RuntimeError(
            "Aucun point d'occupation."
        )

    weights = np.asarray(
        [
            point[2]
            for point in weighted_points
        ],
        dtype=np.float32,
    )

    normalization = float(
        np.percentile(
            weights,
            98,
        )
    )

    normalization = max(
        normalization,
        1e-6,
    )

    for x, y, weight in weighted_points:
        normalized_weight = min(
            1.0,
            weight / normalization,
        )

        cv2.circle(
            heat,
            (x, y),
            10,
            float(normalized_weight),
            -1,
            cv2.LINE_AA,
        )

    heat = cv2.GaussianBlur(
        heat,
        (0, 0),
        sigmaX=22,
        sigmaY=22,
    )

    heat_max = float(
        np.max(heat)
    )

    if heat_max > 0:
        heat /= heat_max

    heat_uint8 = np.clip(
        heat * 255.0,
        0,
        255,
    ).astype(np.uint8)

    heat_color = cv2.applyColorMap(
        heat_uint8,
        cv2.COLORMAP_TURBO,
    )

    alpha = (
        heat[:, :, None]
        * 0.72
    )

    heat_panel = np.clip(
        frame.astype(np.float32)
        * (
            1.0 - alpha
        )
        + heat_color.astype(
            np.float32
        )
        * alpha,
        0,
        255,
    ).astype(np.uint8)

    marker_panel = frame.copy()

    for candidate_id in ball_candidate_ids:
        candidate = candidate_lookup.get(
            candidate_id
        )

        if candidate is None:
            continue

        point = (
            int(round(
                candidate["x"]
            )),
            int(round(
                candidate["y"]
            )),
        )

        cv2.circle(
            marker_panel,
            point,
            3,
            (255, 255, 0),
            -1,
            cv2.LINE_AA,
        )

    temporal_rows = [
        row
        for row in read_csv(
            TEMPORAL_PATH
        )
        if (
            row.get("strategy")
            == "temporal_abstention"
            and row.get("clip_id")
            == CLIP_ID
            and row.get("status")
            in {
                "wrong_visible",
                "false_positive_invisible",
            }
        )
    ]

    false_positive_points: list[
        tuple[int, int, int]
    ] = []

    for row in temporal_rows:
        candidate_id = str(
            row.get(
                "candidate_id",
                "",
            )
        ).strip()

        candidate = candidate_lookup.get(
            candidate_id
        )

        if candidate is None:
            continue

        local_frame = as_int(
            row["local_frame"]
        )

        x = int(round(
            candidate["x"]
        ))

        y = int(round(
            candidate["y"]
        ))

        false_positive_points.append(
            (local_frame, x, y)
        )

        cv2.circle(
            marker_panel,
            (x, y),
            7,
            (35, 35, 245),
            2,
            cv2.LINE_AA,
        )

    r014_points = [
        (x, y)
        for local_frame, x, y
        in false_positive_points
        if 117 <= local_frame <= 129
    ]

    if len(r014_points) >= 2:
        cv2.polylines(
            marker_panel,
            [
                np.asarray(
                    r014_points,
                    dtype=np.int32,
                )
            ],
            False,
            (0, 230, 255),
            3,
            cv2.LINE_AA,
        )

    header_height = 62

    left = np.full(
        (
            height + header_height,
            width,
            3,
        ),
        22,
        dtype=np.uint8,
    )

    right = left.copy()

    left[
        header_height:
        header_height + height
    ] = heat_panel

    right[
        header_height:
        header_height + height
    ] = marker_panel

    put_text(
        left,
        "Occupation globale des candidats — densité 64 px",
        (18, 28),
        scale=0.66,
        thickness=2,
    )

    put_text(
        left,
        (
            "Zones rouges/jaunes = "
            "activité persistante élevée"
        ),
        (18, 51),
        scale=0.48,
        color=(185, 185, 185),
    )

    put_text(
        right,
        "Balle GT contre faux positifs temporels",
        (18, 28),
        scale=0.66,
        thickness=2,
    )

    put_text(
        right,
        (
            f"cyan = balle ({len(ball_candidate_ids)} candidats) | "
            f"rouge = FP ({len(false_positive_points)}) | "
            "jaune = dérive R014"
        ),
        (18, 51),
        scale=0.45,
        color=(185, 185, 185),
    )

    separator = np.full(
        (
            height + header_height,
            12,
            3,
        ),
        10,
        dtype=np.uint8,
    )

    output = np.hstack(
        (
            left,
            separator,
            right,
        )
    )

    safe_write(
        OUTPUT_PATH,
        output,
    )

    print()
    print(
        "I14FV_OCCUPANCY_VISUAL_OK"
    )
    print(
        "clip_id =",
        CLIP_ID,
    )
    print(
        "reference_local_frame =",
        REFERENCE_LOCAL_FRAME,
    )
    print(
        "occupancy_points =",
        len(weighted_points),
    )
    print(
        "ball_candidate_points =",
        len(ball_candidate_ids),
    )
    print(
        "false_positive_points =",
        len(false_positive_points),
    )
    print(
        "output =",
        OUTPUT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
