from __future__ import annotations

import json
import re
import shutil
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ttflux.core.paths import RUNS_DIR, ensure_project_layout
from ttflux.video.catalog import find_video, probe_video


class UnknownVideoError(LookupError):
    """La vidéo demandée n'existe pas dans la bibliothèque locale."""


class UnknownRunError(LookupError):
    """Le run demandé n'existe pas."""


class InvalidRunStateError(RuntimeError):
    """Le run ne peut pas être exécuté depuis son état courant."""


class InvalidClipRangeError(ValueError):
    """La plage temporelle demandée est invalide."""


_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_VIDEO_SNAPSHOT_KEYS = (
    "id",
    "filename",
    "relative_path",
    "extension",
    "size_mb",
    "width",
    "height",
    "fps",
    "duration_s",
    "frame_count",
    "codec",
    "probe_error",
)


def slugify(value: str, fallback: str = "video") -> str:
    """Produit un fragment lisible et sûr pour un identifiant de run."""

    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = _SLUG_PATTERN.sub("-", ascii_value.lower()).strip("-")
    return slug[:48] or fallback


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise UnknownRunError(path.parent.name) from None

    if not isinstance(payload, dict):
        raise ValueError(f"Objet JSON attendu dans {path.name}")

    return payload


def _video_snapshot(video: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        **{key: video.get(key) for key in _VIDEO_SNAPSHOT_KEYS},
    }


def _new_run_id(video: dict[str, Any], created_at: datetime) -> str:
    timestamp = created_at.strftime("%Y%m%dT%H%M%S")
    video_slug = slugify(Path(str(video["filename"])).stem)
    token = uuid4().hex[:6]
    return f"{timestamp}_{video_slug}_{token}"


def _resolve_run_dir(run_id: str) -> Path:
    if (
        not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or "/" in run_id
        or "\\" in run_id
    ):
        raise UnknownRunError(run_id)

    run_dir = RUNS_DIR / run_id

    if not run_dir.is_dir():
        raise UnknownRunError(run_id)

    return run_dir


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    number = float(value)
    return number if number > 0 else None


def validate_clip_range(
    video: dict[str, Any],
    start_s: float,
    duration_s: float,
) -> tuple[float, float]:
    """Valide et normalise une plage temporelle d'extraction."""

    try:
        start = float(start_s)
        duration = float(duration_s)
    except (TypeError, ValueError):
        raise InvalidClipRangeError(
            "Le début et la durée doivent être numériques."
        ) from None

    if start < 0:
        raise InvalidClipRangeError("Le début ne peut pas être négatif.")

    if duration < 1:
        raise InvalidClipRangeError(
            "La durée minimale d'un segment est de 1 seconde."
        )

    if duration > 60:
        raise InvalidClipRangeError(
            "La durée maximale d'un segment est de 60 secondes."
        )

    source_duration = _positive_number(video.get("duration_s"))

    if source_duration is not None:
        if start >= source_duration:
            raise InvalidClipRangeError(
                "Le début du segment est situé après la fin de la vidéo."
            )

        if start + duration > source_duration + 0.01:
            remaining = max(0.0, source_duration - start)
            raise InvalidClipRangeError(
                "Le segment dépasse la fin de la vidéo "
                f"({remaining:.2f} s disponibles)."
            )

    return round(start, 3), round(duration, 3)


def create_run(
    video_id: str,
    clip_start_s: float = 0.0,
    clip_duration_s: float = 15.0,
) -> dict[str, Any]:
    """Crée un run d'extraction de segment dans l'état created."""

    ensure_project_layout()
    video = find_video(video_id)

    if video is None:
        raise UnknownVideoError(video_id)

    start_s, duration_s = validate_clip_range(
        video,
        clip_start_s,
        clip_duration_s,
    )
    created_at = datetime.now().astimezone()

    for _ in range(10):
        run_id = _new_run_id(video, created_at)
        run_dir = RUNS_DIR / run_id

        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("Impossible de générer un identifiant de run unique")

    video_payload = _video_snapshot(video)
    run_payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": video["id"],
        "video_filename": video["filename"],
        "video_relative_path": video["relative_path"],
        "created_at": created_at.isoformat(timespec="seconds"),
        "status": "created",
        "pipeline": {
            "name": "clip_extract",
            "version": 1,
        },
        "configuration": {
            "clip_start_s": start_s,
            "clip_duration_s": duration_s,
            "ball_tracking_enabled": False,
            "table_context_enabled": False,
            "pose_enabled": False,
        },
        "artifacts": {
            "run": "run.json",
            "video": "video.json",
        },
    }

    try:
        _write_json_atomic(run_dir / "video.json", video_payload)
        _write_json_atomic(run_dir / "run.json", run_payload)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    return run_payload


def _extract_clip(
    source_path: Path,
    destination_path: Path,
    start_s: float,
    duration_s: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError("ffmpeg est introuvable dans le PATH.")

    temporary_path = destination_path.with_name(
        destination_path.stem + ".partial" + destination_path.suffix
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
        temporary_path.replace(destination_path)
    except subprocess.CalledProcessError as exc:
        temporary_path.unlink(missing_ok=True)
        message = exc.stderr.strip() or "Échec inconnu de ffmpeg"
        raise RuntimeError(f"Extraction FFmpeg impossible : {message}") from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _build_clip_payload(
    run_payload: dict[str, Any],
    clip_path: Path,
) -> dict[str, Any]:
    probe = probe_video(clip_path)
    configuration = run_payload["configuration"]

    return {
        "schema_version": 1,
        "run_id": run_payload["run_id"],
        "video_id": run_payload["video_id"],
        "source_video_filename": run_payload["video_filename"],
        "artifact": "source_clip.mp4",
        "requested_start_s": configuration["clip_start_s"],
        "requested_duration_s": configuration["clip_duration_s"],
        "size_mb": round(
            clip_path.stat().st_size / (1024 * 1024),
            3,
        ),
        **probe,
    }


def _build_analysis(
    run_payload: dict[str, Any],
    video_payload: dict[str, Any],
    clip_payload: dict[str, Any],
    generated_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": run_payload["run_id"],
        "video_id": run_payload["video_id"],
        "pipeline": run_payload["pipeline"],
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "source": {
            "filename": run_payload["video_filename"],
            "frame_count": video_payload.get("frame_count"),
            "duration_s": video_payload.get("duration_s"),
            "fps": video_payload.get("fps"),
            "resolution": {
                "width": video_payload.get("width"),
                "height": video_payload.get("height"),
            },
        },
        "clip": {
            "artifact": clip_payload["artifact"],
            "requested_start_s": clip_payload["requested_start_s"],
            "requested_duration_s": clip_payload["requested_duration_s"],
            "actual_duration_s": clip_payload.get("duration_s"),
            "frame_count": clip_payload.get("frame_count"),
            "fps": clip_payload.get("fps"),
            "resolution": {
                "width": clip_payload.get("width"),
                "height": clip_payload.get("height"),
            },
        },
    }


def execute_run(run_id: str) -> dict[str, Any]:
    """Extrait le segment demandé et persiste le cycle d'état."""

    ensure_project_layout()
    run_dir = _resolve_run_dir(run_id)
    run_path = run_dir / "run.json"
    video_path = run_dir / "video.json"

    run_payload = _read_json_object(run_path)
    video_payload = _read_json_object(video_path)

    current_status = run_payload.get("status")

    if current_status != "created":
        raise InvalidRunStateError(
            f"Run {run_id} non exécutable depuis l'état {current_status!r}"
        )

    video = find_video(str(run_payload["video_id"]))

    if video is None:
        raise UnknownVideoError(str(run_payload["video_id"]))

    source_path = Path(str(video["absolute_path"]))

    if not source_path.is_file():
        raise FileNotFoundError(f"Vidéo source absente : {source_path}")

    started_at = datetime.now().astimezone()
    run_payload["status"] = "running"
    run_payload["started_at"] = started_at.isoformat(timespec="seconds")
    run_payload.pop("completed_at", None)
    run_payload.pop("failed_at", None)
    run_payload.pop("error", None)
    _write_json_atomic(run_path, run_payload)

    clip_path = run_dir / "source_clip.mp4"
    clip_json_path = run_dir / "clip.json"
    analysis_path = run_dir / "analysis.json"

    try:
        configuration = run_payload["configuration"]
        _extract_clip(
            source_path,
            clip_path,
            float(configuration["clip_start_s"]),
            float(configuration["clip_duration_s"]),
        )

        clip_payload = _build_clip_payload(
            run_payload,
            clip_path,
        )
        _write_json_atomic(clip_json_path, clip_payload)

        generated_at = datetime.now().astimezone()
        analysis_payload = _build_analysis(
            run_payload,
            video_payload,
            clip_payload,
            generated_at,
        )
        _write_json_atomic(analysis_path, analysis_payload)

        completed_at = datetime.now().astimezone()
        run_payload["status"] = "completed"
        run_payload["completed_at"] = completed_at.isoformat(timespec="seconds")
        run_payload["artifacts"].update(
            {
                "source_clip": "source_clip.mp4",
                "clip": "clip.json",
                "analysis": "analysis.json",
            }
        )
        _write_json_atomic(run_path, run_payload)
    except Exception as exc:
        failed_at = datetime.now().astimezone()
        run_payload["status"] = "failed"
        run_payload["failed_at"] = failed_at.isoformat(timespec="seconds")
        run_payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json_atomic(run_path, run_payload)
        raise

    return run_payload


def create_and_execute_clip_run(
    video_id: str,
    clip_start_s: float,
    clip_duration_s: float,
) -> dict[str, Any]:
    """Crée puis exécute immédiatement un run d'extraction."""

    created_run = create_run(
        video_id,
        clip_start_s=clip_start_s,
        clip_duration_s=clip_duration_s,
    )
    return execute_run(str(created_run["run_id"]))


def get_run_clip_path(run_id: str) -> Path:
    """Retourne le clip d'un run terminé."""

    run_dir = _resolve_run_dir(run_id)
    run_payload = _read_json_object(run_dir / "run.json")
    artifact_name = (
        run_payload.get("artifacts", {}).get("source_clip")
    )

    if artifact_name != "source_clip.mp4":
        raise FileNotFoundError(
            f"Aucun segment vidéo disponible pour le run {run_id}."
        )

    clip_path = run_dir / artifact_name

    if not clip_path.is_file():
        raise FileNotFoundError(
            f"Artefact absent pour le run {run_id}."
        )

    return clip_path


def _created_label(value: Any) -> str:
    if not isinstance(value, str):
        return "Date inconnue"

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value

    return parsed.strftime("%d/%m/%Y %H:%M:%S")


def list_runs() -> list[dict[str, Any]]:
    """Relit les runs locaux valides, du plus récent au plus ancien."""

    ensure_project_layout()
    runs: list[dict[str, Any]] = []

    for run_json_path in RUNS_DIR.glob("*/run.json"):
        try:
            payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict) or not payload.get("run_id"):
            continue

        summary = dict(payload)
        summary["created_label"] = _created_label(payload.get("created_at"))
        summary["directory"] = run_json_path.parent.name
        runs.append(summary)

    runs.sort(
        key=lambda item: str(
            item.get("created_at")
            or item.get("run_id")
            or ""
        ),
        reverse=True,
    )
    return runs
