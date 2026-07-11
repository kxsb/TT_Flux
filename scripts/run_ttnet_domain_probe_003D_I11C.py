from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


SEQUENCE_LENGTH = 9
TEMPORAL_STRIDE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--i10-script",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def load_i10_module(path: Path) -> Any:
    path = path.resolve()

    spec = importlib.util.spec_from_file_location(
        "ttflux_i10_reference",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Impossible de charger I10 : {path}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def read_selected_manifest(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    selected = [
        row
        for row in rows
        if str(
            row.get("selected_for_i11") or ""
        ).strip().lower()
        == "yes"
    ]

    if len(selected) != 6:
        raise RuntimeError(
            "Six clips sélectionnés étaient attendus, "
            f"{len(selected)} trouvés."
        )

    return selected


def sanitize_name(value: str) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )

    return cleaned.strip("_")


def percentile(
    values: list[float],
    percentile_value: float,
) -> float | None:
    if not values:
        return None

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            percentile_value,
        )
    )


def median_or_none(
    values: list[float],
) -> float | None:
    return percentile(
        values,
        50.0,
    )


def round_or_none(
    value: float | None,
    digits: int = 4,
) -> float | None:
    if value is None:
        return None

    return round(
        float(value),
        digits,
    )


def longest_false_run(
    flags: list[bool],
) -> int:
    longest = 0
    current = 0

    for flag in flags:
        if flag:
            current = 0
        else:
            current += 1
            longest = max(
                longest,
                current,
            )

    return longest


class FFmpegVideoWriter:
    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        ffmpeg = shutil.which(
            "ffmpeg"
        )

        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg est introuvable dans le PATH."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.8f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

        self.output_path = output_path

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if self.process.stdin is None:
            raise RuntimeError(
                "Canal stdin ffmpeg indisponible."
            )

    def write(
        self,
        frame: np.ndarray,
    ) -> None:
        if self.process.stdin is None:
            raise RuntimeError(
                "Writer ffmpeg déjà fermé."
            )

        try:
            self.process.stdin.write(
                frame.tobytes()
            )
        except BrokenPipeError as error:
            stderr = b""

            if self.process.stderr is not None:
                stderr = self.process.stderr.read()

            raise RuntimeError(
                "ffmpeg a interrompu l'encodage : "
                + stderr.decode(
                    "utf-8",
                    errors="replace",
                )
            ) from error

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None

        stderr = b""

        if self.process.stderr is not None:
            stderr = self.process.stderr.read()

        return_code = self.process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg a échoué pour "
                f"{self.output_path} : "
                + stderr.decode(
                    "utf-8",
                    errors="replace",
                )
            )


def put_text(
    frame: np.ndarray,
    text: str,
    y: int,
    color: tuple[int, int, int] = (
        255,
        255,
        255,
    ),
) -> None:
    cv2.putText(
        frame,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


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
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def process_clip(
    row: dict[str, str],
    model: torch.nn.Module,
    metadata: dict[str, Any],
    device: torch.device,
    i10: Any,
    output_dir: Path,
) -> tuple[
    dict[str, Any],
    list[float],
]:
    source_key = str(
        row["source_key"]
    )

    video_path = Path(
        row["path"]
    ).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            video_path
        )

    safe_name = sanitize_name(
        source_key
    )

    csv_path = (
        output_dir
        / f"{safe_name}_predictions.csv"
    )

    overlay_path = (
        output_dir
        / f"{safe_name}_overlay.mp4"
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Vidéo illisible : {video_path}"
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

    declared_frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if width <= 0 or height <= 0:
        capture.release()

        raise RuntimeError(
            f"Dimensions vidéo invalides : {video_path}"
        )

    if not math.isfinite(fps) or fps <= 0:
        fps = float(
            str(
                row.get("fps") or "50"
            ).replace(",", ".")
        )

    writer = FFmpegVideoWriter(
        overlay_path,
        width,
        height,
        fps,
    )

    probability_threshold = float(
        metadata["threshold"]
    )

    source_scale_x = (
        width
        / float(i10.ORIGINAL_WIDTH)
    )

    source_scale_y = (
        height
        / float(i10.ORIGINAL_HEIGHT)
    )

    resized_window: deque[np.ndarray] = deque(
        maxlen=SEQUENCE_LENGTH
    )

    prediction_rows: list[
        dict[str, Any]
    ] = []

    refined_flags: list[bool] = []
    global_flags: list[bool] = []

    refined_steps: list[float] = []
    global_confidences: list[float] = []
    local_confidences: list[float] = []

    previous_refined: tuple[
        int,
        float,
        float,
    ] | None = None

    frame_index = -1
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            frame_index += 1

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            resized = cv2.resize(
                rgb,
                (
                    i10.INPUT_WIDTH,
                    i10.INPUT_HEIGHT,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

            resized_window.append(
                resized
            )

            overlay = frame.copy()

            if len(resized_window) < SEQUENCE_LENGTH:
                put_text(
                    overlay,
                    (
                        f"{source_key}  "
                        f"frame={frame_index}  "
                        "warmup"
                    ),
                    28,
                    (
                        180,
                        180,
                        180,
                    ),
                )

                writer.write(
                    overlay
                )

                continue

            window_frames = list(
                resized_window
            )

            tensor = i10.make_sequence_tensor(
                window_frames,
                SEQUENCE_LENGTH - 1,
                TEMPORAL_STRIDE,
                device,
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

            decoded = i10.decode_prediction(
                pred_global,
                pred_local,
                probability_threshold,
            )

            global_valid = bool(
                decoded["global_valid"]
            )

            refined_valid = bool(
                decoded["refined_valid"]
            )

            global_x_canonical = float(
                decoded["global_x"]
            )

            global_y_canonical = float(
                decoded["global_y"]
            )

            refined_x_canonical = float(
                decoded["refined_x"]
            )

            refined_y_canonical = float(
                decoded["refined_y"]
            )

            global_x_source = (
                global_x_canonical
                * source_scale_x
            )

            global_y_source = (
                global_y_canonical
                * source_scale_y
            )

            refined_x_source = (
                refined_x_canonical
                * source_scale_x
            )

            refined_y_source = (
                refined_y_canonical
                * source_scale_y
            )

            global_conf = float(
                decoded["global_conf"]
            )

            local_conf = float(
                decoded["local_conf"]
            )

            refined_step_px: float | None = None

            if refined_valid:
                if (
                    previous_refined is not None
                    and previous_refined[0]
                    == frame_index - 1
                ):
                    refined_step_px = math.hypot(
                        refined_x_source
                        - previous_refined[1],
                        refined_y_source
                        - previous_refined[2],
                    )

                    refined_steps.append(
                        refined_step_px
                    )

                previous_refined = (
                    frame_index,
                    refined_x_source,
                    refined_y_source,
                )
            else:
                previous_refined = None

            global_flags.append(
                global_valid
            )

            refined_flags.append(
                refined_valid
            )

            global_confidences.append(
                global_conf
            )

            local_confidences.append(
                local_conf
            )

            prediction_rows.append({
                "source_key":
                    source_key,
                "video_name":
                    video_path.name,
                "frame":
                    frame_index,
                "time_s":
                    round(
                        frame_index / fps,
                        6,
                    ),
                "target_frame_policy":
                    "last_input_frame",
                "sequence_first_frame":
                    frame_index
                    - SEQUENCE_LENGTH
                    + 1,
                "sequence_last_frame":
                    frame_index,
                "global_x_canonical":
                    round(
                        global_x_canonical,
                        3,
                    ),
                "global_y_canonical":
                    round(
                        global_y_canonical,
                        3,
                    ),
                "global_x_source":
                    round(
                        global_x_source,
                        3,
                    ),
                "global_y_source":
                    round(
                        global_y_source,
                        3,
                    ),
                "global_conf":
                    round(
                        global_conf,
                        6,
                    ),
                "global_valid":
                    int(
                        global_valid
                    ),
                "refined_x_canonical":
                    round(
                        refined_x_canonical,
                        3,
                    ),
                "refined_y_canonical":
                    round(
                        refined_y_canonical,
                        3,
                    ),
                "refined_x_source":
                    round(
                        refined_x_source,
                        3,
                    ),
                "refined_y_source":
                    round(
                        refined_y_source,
                        3,
                    ),
                "local_conf":
                    round(
                        local_conf,
                        6,
                    ),
                "refined_valid":
                    int(
                        refined_valid
                    ),
                "refined_step_px":
                    (
                        round(
                            refined_step_px,
                            3,
                        )
                        if refined_step_px
                        is not None
                        else ""
                    ),
            })

            if global_valid:
                cv2.circle(
                    overlay,
                    (
                        int(
                            round(
                                global_x_source
                            )
                        ),
                        int(
                            round(
                                global_y_source
                            )
                        ),
                    ),
                    8,
                    (
                        0,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

            if refined_valid:
                cv2.circle(
                    overlay,
                    (
                        int(
                            round(
                                refined_x_source
                            )
                        ),
                        int(
                            round(
                                refined_y_source
                            )
                        ),
                    ),
                    6,
                    (
                        0,
                        255,
                        0,
                    ),
                    2,
                    cv2.LINE_AA,
                )

            put_text(
                overlay,
                (
                    f"{source_key}  "
                    f"frame={frame_index}"
                ),
                28,
            )

            put_text(
                overlay,
                (
                    f"global={int(global_valid)} "
                    f"conf={global_conf:.3f}  "
                    f"refined={int(refined_valid)} "
                    f"conf={local_conf:.3f}"
                ),
                54,
                (
                    0,
                    255,
                    0,
                )
                if refined_valid
                else (
                    0,
                    0,
                    255,
                ),
            )

            put_text(
                overlay,
                (
                    "yellow=global  green=refined"
                ),
                80,
                (
                    220,
                    220,
                    220,
                ),
            )

            writer.write(
                overlay
            )
    finally:
        capture.release()
        writer.close()

    elapsed = (
        time.perf_counter()
        - started
    )

    target_frames = len(
        prediction_rows
    )

    if target_frames <= 0:
        raise RuntimeError(
            f"Aucune inférence produite : {video_path}"
        )

    write_csv(
        csv_path,
        prediction_rows,
    )

    global_valid_count = sum(
        global_flags
    )

    refined_valid_count = sum(
        refined_flags
    )

    possible_adjacent_pairs = max(
        0,
        target_frames - 1,
    )

    summary = {
        "source_key":
            source_key,
        "video_path":
            str(video_path),
        "video_name":
            video_path.name,
        "width":
            width,
        "height":
            height,
        "fps":
            fps,
        "declared_frame_count":
            declared_frame_count,
        "decoded_frame_count":
            frame_index + 1,
        "warmup_frames":
            SEQUENCE_LENGTH - 1,
        "target_frames":
            target_frames,
        "global_valid_count":
            global_valid_count,
        "global_valid_rate":
            global_valid_count
            / target_frames,
        "refined_valid_count":
            refined_valid_count,
        "refined_valid_rate":
            refined_valid_count
            / target_frames,
        "median_global_conf":
            round_or_none(
                median_or_none(
                    global_confidences
                )
            ),
        "median_local_conf":
            round_or_none(
                median_or_none(
                    local_confidences
                )
            ),
        "adjacent_refined_pairs":
            len(
                refined_steps
            ),
        "possible_adjacent_pairs":
            possible_adjacent_pairs,
        "adjacent_refined_pair_rate":
            (
                len(refined_steps)
                / possible_adjacent_pairs
                if possible_adjacent_pairs > 0
                else 0.0
            ),
        "refined_step_median_px":
            round_or_none(
                percentile(
                    refined_steps,
                    50.0,
                )
            ),
        "refined_step_p90_px":
            round_or_none(
                percentile(
                    refined_steps,
                    90.0,
                )
            ),
        "refined_step_p99_px":
            round_or_none(
                percentile(
                    refined_steps,
                    99.0,
                )
            ),
        "refined_step_max_px":
            round_or_none(
                max(refined_steps)
                if refined_steps
                else None
            ),
        "max_invalid_run_frames":
            longest_false_run(
                refined_flags
            ),
        "runtime_seconds":
            elapsed,
        "inference_fps":
            target_frames
            / elapsed,
        "predictions_csv":
            str(csv_path),
        "overlay_video":
            str(overlay_path),
    }

    return (
        summary,
        refined_steps,
    )


def main() -> None:
    args = parse_args()

    manifest_path = args.manifest.resolve()
    i10_script = args.i10_script.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_rows = read_selected_manifest(
        manifest_path
    )

    i10 = load_i10_module(
        i10_script
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponible."
        )

    device = torch.device(
        "cuda:0"
    )

    model_spec = next(
        item
        for item in i10.CHECKPOINTS
        if item["model_id"]
        == "ttnet_120fps"
    )

    checkpoint_path = Path(
        model_spec["checkpoint_path"]
    )

    model, metadata = i10.load_model(
        checkpoint_path,
        device,
    )

    model.eval()

    print(
        "CUDA:",
        torch.cuda.get_device_name(0),
    )

    print(
        "CHECKPOINT:",
        checkpoint_path,
    )

    print(
        "THRESHOLD:",
        metadata["threshold"],
    )

    clip_summaries: list[
        dict[str, Any]
    ] = []

    all_refined_steps: list[
        float
    ] = []

    global_started = time.perf_counter()

    for clip_index, row in enumerate(
        selected_rows,
        start=1,
    ):
        print()
        print(
            f"[{clip_index}/"
            f"{len(selected_rows)}] "
            f"{row['source_key']}"
        )

        clip_summary, refined_steps = process_clip(
            row,
            model,
            metadata,
            device,
            i10,
            output_dir,
        )

        clip_summaries.append(
            clip_summary
        )

        all_refined_steps.extend(
            refined_steps
        )

        print(
            "  target_frames =",
            clip_summary[
                "target_frames"
            ],
        )

        print(
            "  global_valid  =",
            (
                f"{clip_summary['global_valid_rate']:.3f}"
            ),
        )

        print(
            "  refined_valid =",
            (
                f"{clip_summary['refined_valid_rate']:.3f}"
            ),
        )

        print(
            "  step_median   =",
            clip_summary[
                "refined_step_median_px"
            ],
        )

        print(
            "  inference_fps =",
            (
                f"{clip_summary['inference_fps']:.1f}"
            ),
        )

    global_elapsed = (
        time.perf_counter()
        - global_started
    )

    total_target_frames = sum(
        int(
            summary["target_frames"]
        )
        for summary in clip_summaries
    )

    total_global_valid = sum(
        int(
            summary["global_valid_count"]
        )
        for summary in clip_summaries
    )

    total_refined_valid = sum(
        int(
            summary["refined_valid_count"]
        )
        for summary in clip_summaries
    )

    total_possible_pairs = sum(
        int(
            summary[
                "possible_adjacent_pairs"
            ]
        )
        for summary in clip_summaries
    )

    total_adjacent_pairs = sum(
        int(
            summary[
                "adjacent_refined_pairs"
            ]
        )
        for summary in clip_summaries
    )

    summary_payload = {
        "experiment":
            "003D_I11C_TTNet_120fps_domain_probe",
        "accuracy_evaluated":
            False,
        "ground_truth_available":
            False,
        "interpretation":
            (
                "Validity and continuity are diagnostics only. "
                "They do not establish tracking accuracy."
            ),
        "model_id":
            "ttnet_120fps",
        "checkpoint_path":
            str(checkpoint_path),
        "target_frame_policy":
            "last_input_frame",
        "sequence_length":
            SEQUENCE_LENGTH,
        "temporal_stride":
            TEMPORAL_STRIDE,
        "selected_clip_count":
            len(clip_summaries),
        "total_target_frames":
            total_target_frames,
        "global_valid_rate":
            (
                total_global_valid
                / total_target_frames
            ),
        "refined_valid_rate":
            (
                total_refined_valid
                / total_target_frames
            ),
        "adjacent_refined_pair_rate":
            (
                total_adjacent_pairs
                / total_possible_pairs
                if total_possible_pairs > 0
                else 0.0
            ),
        "refined_step_median_px":
            round_or_none(
                percentile(
                    all_refined_steps,
                    50.0,
                )
            ),
        "refined_step_p90_px":
            round_or_none(
                percentile(
                    all_refined_steps,
                    90.0,
                )
            ),
        "refined_step_p99_px":
            round_or_none(
                percentile(
                    all_refined_steps,
                    99.0,
                )
            ),
        "runtime_seconds":
            global_elapsed,
        "inference_fps":
            (
                total_target_frames
                / global_elapsed
            ),
        "clips":
            clip_summaries,
    }

    summary_path = (
        output_dir
        / "i11c_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("I11C_DOMAIN_PROBE_OK")
    print(
        "summary =",
        summary_path,
    )
    print(
        "target_frames =",
        total_target_frames,
    )
    print(
        "global_valid_rate =",
        (
            f"{summary_payload['global_valid_rate']:.4f}"
        ),
    )
    print(
        "refined_valid_rate =",
        (
            f"{summary_payload['refined_valid_rate']:.4f}"
        ),
    )
    print(
        "inference_fps =",
        (
            f"{summary_payload['inference_fps']:.1f}"
        ),
    )


if __name__ == "__main__":
    main()
