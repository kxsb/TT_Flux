from __future__ import annotations

import collections
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(os.environ["TTFLUX_ROOT"]).resolve()
PING_ROOT = Path(os.environ["TTFLUX_PING_ROOT"]).resolve()
CHECKPOINT = Path(os.environ["TTFLUX_CHECKPOINT"]).resolve()
POLICY_PATH = Path(os.environ["TTFLUX_POLICY"]).resolve()
I10_PATH = Path(os.environ["TTFLUX_I10"]).resolve()
OUTPUT_DIR = Path(os.environ["TTFLUX_OUTPUT"]).resolve()
FFMPEG = Path(os.environ["TTFLUX_FFMPEG"]).resolve()

DATASET_OVERRIDE = os.environ.get(
    "TTFLUX_DATASET_VIDEO",
    "",
).strip()

RAW_OVERRIDE = os.environ.get(
    "TTFLUX_RAW_VIDEO",
    "",
).strip()

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
}

EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    ".venv_torch",
    "__pycache__",
    "checkpoints",
    "node_modules",
    "runs",
}

CANONICAL_WIDTH = 1920
CANONICAL_HEIGHT = 1080

INPUT_WIDTH = 320
INPUT_HEIGHT = 128

SEQUENCE_LENGTH = 9
CLIP_SECONDS = 30.0
TRAIL_SECONDS = 1.35


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


def is_inside(
    path: Path,
    parent: Path,
) -> bool:
    try:
        path.resolve().relative_to(
            parent.resolve()
        )
        return True
    except ValueError:
        return False


def resolve_override(
    value: str,
) -> Path | None:
    if not value:
        return None

    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    path = path.resolve()

    if not path.is_file():
        raise RuntimeError(
            f"Vidéo forcée absente : {path}"
        )

    return path


def probe_video(
    path: Path,
) -> dict[str, Any] | None:
    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():
        return None

    try:
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

        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )
    finally:
        capture.release()

    if (
        width <= 0
        or height <= 0
        or frame_count <= 0
        or not math.isfinite(fps)
        or fps <= 0
    ):
        return None

    duration = frame_count / fps

    return {
        "path": path.resolve(),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
    }


def collect_videos(
    root: Path,
) -> list[Path]:
    results = []

    if not root.exists():
        return results

    for current_root, directories, files in os.walk(
        root
    ):
        directories[:] = [
            name
            for name in directories
            if name.lower()
            not in EXCLUDED_DIRECTORIES
        ]

        current = Path(current_root)

        for filename in files:
            path = current / filename

            if (
                path.suffix.lower()
                in VIDEO_EXTENSIONS
            ):
                results.append(
                    path.resolve()
                )

    return results


def select_dataset_video() -> dict[str, Any]:
    override = resolve_override(
        DATASET_OVERRIDE
    )

    if override is not None:
        info = probe_video(override)

        if info is None:
            raise RuntimeError(
                "Vidéo dataset forcée illisible."
            )

        return info

    dataset_root = (
        ROOT
        / "data"
        / "blurball_dataset"
    )

    candidates = []

    for path in collect_videos(
        dataset_root
    ):
        info = probe_video(path)

        if info is not None:
            candidates.append(info)

    if not candidates:
        raise RuntimeError(
            "Aucune vidéo BlurBall lisible."
        )

    def score(
        info: dict[str, Any],
    ) -> tuple[Any, ...]:
        path = info["path"]

        reduced_full_video = (
            path.name.lower()
            == "reduced_fps.mp4"
        )

        long_enough = (
            info["duration"]
            >= CLIP_SECONDS
        )

        return (
            1 if long_enough else 0,
            1 if reduced_full_video else 0,
            info["duration"],
        )

    return max(
        candidates,
        key=score,
    )


def select_raw_video(
    dataset_path: Path,
) -> dict[str, Any]:
    override = resolve_override(
        RAW_OVERRIDE
    )

    if override is not None:
        info = probe_video(override)

        if info is None:
            raise RuntimeError(
                "Vidéo raw forcée illisible."
            )

        return info

    dataset_root = (
        ROOT
        / "data"
        / "blurball_dataset"
    ).resolve()

    candidates = []

    for path in collect_videos(
        PING_ROOT
    ):
        if is_inside(
            path,
            dataset_root,
        ):
            continue

        if path.resolve() == dataset_path.resolve():
            continue

        info = probe_video(path)

        if info is not None:
            candidates.append(info)

    if not candidates:
        raise RuntimeError(
            "Aucune vidéo raw trouvée. "
            "Renseigner $rawVideo au début du patch."
        )

    keywords = (
        "raw",
        "competition",
        "match",
        "game",
        "tournament",
        "wtt",
        "ittf",
        "open",
        "final",
    )

    def score(
        info: dict[str, Any],
    ) -> tuple[Any, ...]:
        path_text = str(
            info["path"]
        ).lower()

        keyword_score = sum(
            1
            for keyword in keywords
            if keyword in path_text
        )

        long_enough = (
            info["duration"]
            >= CLIP_SECONDS
        )

        return (
            keyword_score,
            1 if long_enough else 0,
            info["duration"],
        )

    return max(
        candidates,
        key=score,
    )


def choose_motion_start(
    info: dict[str, Any],
) -> tuple[float, float]:
    duration = float(
        info["duration"]
    )

    if duration <= CLIP_SECONDS + 0.25:
        return 0.0, 0.0

    interval = max(
        0.75,
        duration / 240.0,
    )

    sample_times = np.arange(
        0.0,
        duration,
        interval,
        dtype=np.float64,
    )

    capture = cv2.VideoCapture(
        str(info["path"])
    )

    previous = None
    values = []
    times = []

    try:
        for sample_time in sample_times:
            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                float(
                    sample_time * 1000.0
                ),
            )

            ok, frame = capture.read()

            if not ok:
                continue

            small = cv2.resize(
                frame,
                (160, 90),
                interpolation=cv2.INTER_AREA,
            )

            gray = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2GRAY,
            )

            gray = cv2.GaussianBlur(
                gray,
                (5, 5),
                0,
            )

            if previous is None:
                movement = 0.0
            else:
                movement = float(
                    np.mean(
                        cv2.absdiff(
                            gray,
                            previous,
                        )
                    )
                )

            previous = gray
            values.append(movement)
            times.append(
                float(sample_time)
            )
    finally:
        capture.release()

    if len(values) < 3:
        return 0.0, 0.0

    window_size = max(
        1,
        int(
            round(
                CLIP_SECONDS
                / interval
            )
        ),
    )

    if len(values) <= window_size:
        return 0.0, float(
            np.mean(values)
        )

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    cumulative = np.concatenate(
        (
            np.array([0.0]),
            np.cumsum(array),
        )
    )

    best_index = 0
    best_score = -1.0

    for index in range(
        0,
        len(array)
        - window_size
        + 1,
    ):
        score = float(
            (
                cumulative[
                    index + window_size
                ]
                - cumulative[index]
            )
            / window_size
        )

        if score > best_score:
            best_score = score
            best_index = index

    start = min(
        times[best_index],
        duration - CLIP_SECONDS,
    )

    return (
        max(0.0, start),
        best_score,
    )


def put_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX

    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        font,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    trail: collections.deque[
        tuple[int, float, float]
    ],
    raw_point: tuple[float, float] | None,
    global_conf: float,
    local_conf: float,
    source_kind: str,
    accepted: int,
    processed: int,
) -> None:
    points = list(trail)

    if len(points) >= 2:
        for index in range(
            1,
            len(points),
        ):
            progress = (
                index
                / max(
                    1,
                    len(points) - 1,
                )
            )

            previous = points[
                index - 1
            ]

            current = points[index]

            color = (
                int(
                    255
                    - 180 * progress
                ),
                int(
                    130
                    + 125 * progress
                ),
                int(
                    40
                    + 200 * progress
                ),
            )

            cv2.line(
                frame,
                (
                    int(previous[1]),
                    int(previous[2]),
                ),
                (
                    int(current[1]),
                    int(current[2]),
                ),
                color,
                max(
                    2,
                    int(
                        2 + 3 * progress
                    ),
                ),
                cv2.LINE_AA,
            )

    if points:
        current = points[-1]

        center = (
            int(current[1]),
            int(current[2]),
        )

        cv2.circle(
            frame,
            center,
            11,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            center,
            8,
            (70, 255, 255),
            3,
            cv2.LINE_AA,
        )

    if raw_point is not None:
        cv2.drawMarker(
            frame,
            (
                int(raw_point[0]),
                int(raw_point[1]),
            ),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            10,
            2,
            cv2.LINE_AA,
        )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (15, 15),
        (500, 122),
        (10, 15, 22),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0.0,
        frame,
    )

    scale = max(
        0.52,
        min(
            0.82,
            frame.shape[1] / 1800.0,
        ),
    )

    acceptance = (
        accepted / processed
        if processed
        else 0.0
    )

    status = (
        "LOCKED"
        if raw_point is not None
        else "SEARCH"
    )

    put_text(
        frame,
        "TTFlux D1 | TTR TRAJECTORY",
        30,
        45,
        scale,
        (245, 245, 245),
    )

    put_text(
        frame,
        (
            f"{source_kind} | "
            f"{status} | "
            f"G {global_conf:.3f} "
            f"L {local_conf:.3f}"
        ),
        30,
        78,
        scale * 0.78,
        (120, 230, 255),
    )

    put_text(
        frame,
        (
            "refined_only 0.05/0.05 | "
            f"accept {acceptance * 100:.1f}%"
        ),
        30,
        106,
        scale * 0.70,
        (190, 205, 220),
    )


def transcode_with_audio(
    temporary_path: Path,
    source_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    command = [
        str(FFMPEG),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(temporary_path),
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        message = result.stderr.decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            "ffmpeg a échoué : "
            f"{message}"
        )


def render(
    model: Any,
    i10: Any,
    info: dict[str, Any],
    start_seconds: float,
    source_kind: str,
    output_path: Path,
) -> dict[str, Any]:
    source_path = info["path"]
    width = int(info["width"])
    height = int(info["height"])
    fps = float(info["fps"])

    duration = min(
        CLIP_SECONDS,
        info["duration"]
        - start_seconds,
    )

    start_frame = int(
        round(
            start_seconds * fps
        )
    )

    end_frame = min(
        int(info["frame_count"]),
        start_frame
        + int(
            round(
                duration * fps
            )
        ),
    )

    warmup_frame = max(
        0,
        start_frame
        - (
            SEQUENCE_LENGTH - 1
        ),
    )

    capture = cv2.VideoCapture(
        str(source_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Vidéo illisible : {source_path}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        warmup_frame,
    )

    temporary_path = (
        output_path.parent
        / (
            output_path.stem
            + "_temporary.mp4"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    if output_path.exists():
        output_path.unlink()

    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            "VideoWriter OpenCV indisponible."
        )

    frame_buffer: collections.deque[
        np.ndarray
    ] = collections.deque(
        maxlen=SEQUENCE_LENGTH
    )

    trail: collections.deque[
        tuple[int, float, float]
    ] = collections.deque()

    maximum_trail_frames = max(
        10,
        int(
            round(
                TRAIL_SECONDS * fps
            )
        ),
    )

    current_frame = warmup_frame

    previous_point = None
    previous_smoothed = None
    previous_valid_frame = None

    processed = 0
    accepted = 0

    try:
        while current_frame < end_frame:
            ok, frame = capture.read()

            if not ok:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            resized = cv2.resize(
                rgb,
                (INPUT_WIDTH, INPUT_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )

            frame_buffer.append(
                resized
            )

            if current_frame < start_frame:
                current_frame += 1
                continue

            processed += 1

            global_conf = 0.0
            local_conf = 0.0
            raw_point = None

            if (
                len(frame_buffer)
                == SEQUENCE_LENGTH
            ):
                tensor = (
                    i10.make_sequence_tensor(
                        list(frame_buffer),
                        SEQUENCE_LENGTH - 1,
                        1,
                        torch.device("cuda:0"),
                    )
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

                decoded = (
                    i10.decode_prediction(
                        pred_global,
                        pred_local,
                        0.0,
                    )
                )

                global_conf = float(
                    decoded["global_conf"]
                )

                local_conf = float(
                    decoded["local_conf"]
                )

                x = (
                    float(
                        decoded["refined_x"]
                    )
                    * width
                    / CANONICAL_WIDTH
                )

                y = (
                    float(
                        decoded["refined_y"]
                    )
                    * height
                    / CANONICAL_HEIGHT
                )

                valid = (
                    global_conf >= 0.05
                    and local_conf >= 0.05
                    and 0 <= x < width
                    and 0 <= y < height
                )

                if valid:
                    raw_point = (x, y)
                    accepted += 1

                    reset = False

                    if (
                        previous_point is None
                        or previous_valid_frame
                        is None
                    ):
                        reset = True
                    else:
                        gap = (
                            current_frame
                            - previous_valid_frame
                        )

                        jump = math.hypot(
                            x - previous_point[0],
                            y - previous_point[1],
                        )

                        jump_limit = max(
                            140.0,
                            width * 0.18,
                        )

                        if (
                            gap > 4
                            or jump > jump_limit
                        ):
                            reset = True

                    if reset:
                        trail.clear()
                        smoothed = (x, y)
                    else:
                        alpha = 0.72

                        smoothed = (
                            alpha * x
                            + (
                                1.0 - alpha
                            )
                            * previous_smoothed[0],
                            alpha * y
                            + (
                                1.0 - alpha
                            )
                            * previous_smoothed[1],
                        )

                    trail.append(
                        (
                            current_frame,
                            smoothed[0],
                            smoothed[1],
                        )
                    )

                    previous_point = (x, y)
                    previous_smoothed = smoothed
                    previous_valid_frame = (
                        current_frame
                    )

            while trail:
                age = (
                    current_frame
                    - trail[0][0]
                )

                if age <= maximum_trail_frames:
                    break

                trail.popleft()

            output_frame = frame.copy()

            draw_overlay(
                output_frame,
                trail,
                raw_point,
                global_conf,
                local_conf,
                source_kind,
                accepted,
                processed,
            )

            writer.write(output_frame)

            progress_interval = max(
                1,
                int(round(fps * 5.0)),
            )

            if processed % progress_interval == 0:
                print(
                    f"  {source_kind} "
                    f"{processed / fps:.1f}s/"
                    f"{duration:.1f}s "
                    f"accept={accepted}/{processed}"
                )

            current_frame += 1
    finally:
        capture.release()
        writer.release()

    actual_duration = (
        processed / fps
        if processed
        else 0.0
    )

    if processed == 0:
        raise RuntimeError(
            f"Aucune frame rendue : {source_path}"
        )

    transcode_with_audio(
        temporary_path,
        source_path,
        output_path,
        start_frame / fps,
        actual_duration,
    )

    if temporary_path.exists():
        temporary_path.unlink()

    if not output_path.exists():
        raise RuntimeError(
            f"Sortie absente : {output_path}"
        )

    return {
        "source_path": str(source_path),
        "source_width": width,
        "source_height": height,
        "source_fps": fps,
        "source_duration_seconds": (
            float(info["duration"])
        ),
        "clip_start_seconds": (
            start_frame / fps
        ),
        "clip_duration_seconds": (
            actual_duration
        ),
        "processed_frames": processed,
        "accepted_frames": accepted,
        "acceptance_rate": (
            accepted / processed
        ),
        "output_path": str(output_path),
        "output_size_bytes": (
            output_path.stat().st_size
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponible."
        )

    policy = json.loads(
        POLICY_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        policy["prediction_policy"]
        != "refined_only"
    ):
        raise RuntimeError(
            "Politique D1 inattendue."
        )

    if (
        float(
            policy["global_threshold"]
        )
        != 0.05
    ):
        raise RuntimeError(
            "Seuil global inattendu."
        )

    if (
        float(
            policy["local_threshold"]
        )
        != 0.05
    ):
        raise RuntimeError(
            "Seuil local inattendu."
        )

    dataset_info = (
        select_dataset_video()
    )

    raw_info = select_raw_video(
        dataset_info["path"]
    )

    dataset_start, dataset_motion = (
        choose_motion_start(
            dataset_info
        )
    )

    raw_start, raw_motion = (
        choose_motion_start(
            raw_info
        )
    )

    print("")
    print(
        "=== SOURCES SÉLECTIONNÉES ==="
    )

    print(
        "DATASET"
        f"\n  {dataset_info['path']}"
        f"\n  durée="
        f"{dataset_info['duration']:.2f}s"
        f"\n  début="
        f"{dataset_start:.2f}s"
        f"\n  mouvement="
        f"{dataset_motion:.3f}"
    )

    print("")
    print(
        "RAW COMPETITION"
        f"\n  {raw_info['path']}"
        f"\n  durée="
        f"{raw_info['duration']:.2f}s"
        f"\n  début="
        f"{raw_start:.2f}s"
        f"\n  mouvement="
        f"{raw_motion:.3f}"
    )

    i10 = load_module(
        "ttflux_i10_d1f_r2",
        I10_PATH,
    )

    i10.ORIGINAL_WIDTH = (
        CANONICAL_WIDTH
    )

    i10.ORIGINAL_HEIGHT = (
        CANONICAL_HEIGHT
    )

    model, metadata = i10.load_model(
        CHECKPOINT,
        torch.device("cuda:0"),
    )

    if (
        metadata["matched_parameters"]
        != metadata["model_parameters"]
    ):
        raise RuntimeError(
            "Chargement incomplet du checkpoint."
        )

    dataset_output = (
        OUTPUT_DIR
        / "ttflux_d1_dataset_ttr.mp4"
    )

    raw_output = (
        OUTPUT_DIR
        / "ttflux_d1_raw_competition_ttr.mp4"
    )

    print("")
    print(
        "=== RENDU DATASET ==="
    )

    dataset_result = render(
        model,
        i10,
        dataset_info,
        dataset_start,
        "DATASET",
        dataset_output,
    )

    print("")
    print(
        "=== RENDU RAW COMPETITION ==="
    )

    raw_result = render(
        model,
        i10,
        raw_info,
        raw_start,
        "RAW COMPETITION",
        raw_output,
    )

    summary = {
        "schema_version": 1,
        "experiment":
            "DSET_D1F_R2_TTR_visuals",
        "policy": {
            "prediction_policy":
                "refined_only",
            "global_threshold":
                0.05,
            "local_threshold":
                0.05,
        },
        "dataset_video": {
            "motion_score":
                dataset_motion,
            **dataset_result,
        },
        "raw_competition_video": {
            "motion_score":
                raw_motion,
            **raw_result,
        },
    }

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("")
    print(
        "DSET_D1F_R2_TTR_VISUALS_GENERATED"
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()