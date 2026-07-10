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

    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_PATTERN.sub("-", ascii_value.lower()).strip("-")
    return slug[:48] or fallback


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


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


def create_run(video_id: str) -> dict[str, Any]:
    """Crée un run metadata-only et ses deux contrats JSON."""

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
            "version": 1,
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
        key=lambda item: str(item.get("created_at") or item.get("run_id") or ""),
        reverse=True,
    )
    return runs
