from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from ttflux.core.paths import (
    VIDEO_CATALOG_PATH,
    VIDEOS_DIR,
    ensure_project_layout,
)


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".webm",
}


def stable_video_id(relative_path: Path | str) -> str:
    normalized = Path(relative_path).as_posix().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def parse_frame_rate(value: Any) -> float | None:
    if value in (None, "", "N/A", "0/0"):
        return None

    try:
        return round(float(Fraction(str(value))), 3)
    except (ValueError, ZeroDivisionError):
        return None


def parse_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None

    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def parse_integer(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def probe_video(path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        return {
            "probe_error": "ffprobe est introuvable dans le PATH",
        }

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=width,height,r_frame_rate,avg_frame_rate,"
            "nb_frames,duration,codec_name"
            ":format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        payload = json.loads(completed.stdout)
    except subprocess.CalledProcessError as exc:
        return {
            "probe_error": exc.stderr.strip() or "Échec de ffprobe",
        }
    except json.JSONDecodeError:
        return {
            "probe_error": "Réponse JSON invalide de ffprobe",
        }

    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    format_data = payload.get("format") or {}

    fps = (
        parse_frame_rate(stream.get("avg_frame_rate"))
        or parse_frame_rate(stream.get("r_frame_rate"))
    )

    duration_s = (
        parse_float(stream.get("duration"))
        or parse_float(format_data.get("duration"))
    )

    frame_count = parse_integer(stream.get("nb_frames"))

    if frame_count is None and fps and duration_s:
        frame_count = round(fps * duration_s)

    return {
        "width": parse_integer(stream.get("width")),
        "height": parse_integer(stream.get("height")),
        "fps": fps,
        "duration_s": duration_s,
        "frame_count": frame_count,
        "codec": stream.get("codec_name"),
        "probe_error": None,
    }


def inspect_video(path: Path) -> dict[str, Any]:
    try:
        relative_path = path.relative_to(VIDEOS_DIR)
    except ValueError:
        relative_path = Path(path.name)

    record: dict[str, Any] = {
        "id": stable_video_id(relative_path),
        "filename": path.name,
        "relative_path": relative_path.as_posix(),
        "absolute_path": str(path.resolve()),
        "extension": path.suffix.lower(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }

    record.update(probe_video(path))
    return record


def scan_videos() -> list[dict[str, Any]]:
    ensure_project_layout()

    paths = sorted(
        (
            path
            for path in VIDEOS_DIR.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )

    return [inspect_video(path) for path in paths]


def write_catalog() -> list[dict[str, Any]]:
    videos = scan_videos()

    payload = {
        "schema_version": 1,
        "video_count": len(videos),
        "videos": videos,
    }

    VIDEO_CATALOG_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return videos


def find_video(video_id: str) -> dict[str, Any] | None:
    for video in scan_videos():
        if video["id"] == video_id:
            return video

    return None