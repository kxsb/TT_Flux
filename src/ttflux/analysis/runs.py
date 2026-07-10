from __future__ import annotations

import json
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ttflux.core.paths import RUNS_DIR, ensure_project_layout
from ttflux.video.catalog import find_video


class UnknownVideoError(LookupError):
    """La vidéo demandée n'existe pas dans la bibliothèque locale."""


class UnknownRunError(LookupError):
    """Le run demandé n'existe pas."""


class InvalidRunStateError(RuntimeError):
    """Le run ne peut pas être exécuté depuis son état courant."""


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


def create_run(video_id: str) -> dict[str, Any]:
    """Crée un run metadata-only dans l'état created."""

    ensure_project_layout()
    video = find_video(video_id)

    if video is None:
        raise UnknownVideoError(video_id)

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
            "name": "metadata_only",
            "version": 2,
        },
        "configuration": {
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


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    number = float(value)
    return number if number > 0 else None


def _build_analysis(
    run_payload: dict[str, Any],
    video_payload: dict[str, Any],
    generated_at: datetime,
) -> dict[str, Any]:
    fps = _positive_number(video_payload.get("fps"))
    frame_interval_ms = round(1000.0 / fps, 3) if fps else None

    return {
        "schema_version": 1,
        "run_id": run_payload["run_id"],
        "video_id": run_payload["video_id"],
        "video_filename": run_payload["video_filename"],
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "frame_count": video_payload.get("frame_count"),
        "duration_s": video_payload.get("duration_s"),
        "fps": video_payload.get("fps"),
        "resolution": {
            "width": video_payload.get("width"),
            "height": video_payload.get("height"),
        },
        "estimated_frame_interval_ms": frame_interval_ms,
    }


def execute_run(run_id: str) -> dict[str, Any]:
    """Exécute le pipeline metadata-only et persiste son cycle d'état."""

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

    started_at = datetime.now().astimezone()
    run_payload["status"] = "running"
    run_payload["started_at"] = started_at.isoformat(timespec="seconds")
    run_payload.pop("completed_at", None)
    run_payload.pop("failed_at", None)
    run_payload.pop("error", None)
    _write_json_atomic(run_path, run_payload)

    try:
        generated_at = datetime.now().astimezone()
        analysis_payload = _build_analysis(
            run_payload,
            video_payload,
            generated_at,
        )
        _write_json_atomic(run_dir / "analysis.json", analysis_payload)

        completed_at = datetime.now().astimezone()
        run_payload["status"] = "completed"
        run_payload["completed_at"] = completed_at.isoformat(timespec="seconds")
        run_payload["artifacts"]["analysis"] = "analysis.json"
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


def create_and_execute_run(video_id: str) -> dict[str, Any]:
    """Crée puis exécute immédiatement un run metadata-only."""

    created_run = create_run(video_id)
    return execute_run(str(created_run["run_id"]))


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
