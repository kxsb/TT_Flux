from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from ttflux.pipeline.contracts import (
    CLIP_SCHEMA_VERSION,
    RunPayload,
)


ProbeVideo = Callable[
    [Path],
    dict[str, Any],
]


def extract_clip(
    source_path: Path,
    destination_path: Path,
    start_s: float,
    duration_s: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg est introuvable dans le PATH."
        )

    temporary_path = destination_path.with_name(
        destination_path.stem
        + ".partial"
        + destination_path.suffix
    )
    temporary_path.unlink(missing_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration_s:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        temporary_path.replace(
            destination_path
        )
    except subprocess.CalledProcessError as exc:
        temporary_path.unlink(
            missing_ok=True
        )
        message = (
            exc.stderr.strip()
            or "\u00c9chec inconnu de ffmpeg"
        )
        raise RuntimeError(
            "Extraction FFmpeg impossible : "
            f"{message}"
        ) from exc
    except Exception:
        temporary_path.unlink(
            missing_ok=True
        )
        raise


def build_clip_payload(
    run_payload: RunPayload,
    clip_path: Path,
    *,
    probe_video_fn: ProbeVideo,
) -> dict[str, Any]:
    probe = probe_video_fn(
        clip_path
    )
    configuration = run_payload[
        "configuration"
    ]

    return {
        "schema_version": CLIP_SCHEMA_VERSION,
        "run_id": run_payload["run_id"],
        "video_id": run_payload["video_id"],
        "source_video_filename": run_payload[
            "video_filename"
        ],
        "artifact": "source_clip.mp4",
        "requested_start_s": configuration[
            "clip_start_s"
        ],
        "requested_duration_s": configuration[
            "clip_duration_s"
        ],
        "size_mb": round(
            clip_path.stat().st_size
            / (1024 * 1024),
            3,
        ),
        **probe,
    }


__all__ = [
    "ProbeVideo",
    "build_clip_payload",
    "extract_clip",
]
