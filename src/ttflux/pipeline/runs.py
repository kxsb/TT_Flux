from __future__ import annotations

import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ttflux.core.paths import RUNS_DIR, ensure_project_layout
from ttflux.pipeline.clips import (
    build_clip_payload as _build_clip_payload_impl,
    extract_clip as _extract_clip_impl,
)
from ttflux.pipeline.catalog import (
    created_label as _created_label_impl,
    list_runs as _list_runs_impl,
)
from ttflux.pipeline.contracts import (
    PIPELINE_NAME,
    PIPELINE_VERSION,
    RUN_SCHEMA_VERSION,
    VIDEO_SNAPSHOT_KEYS,
    VIDEO_SNAPSHOT_SCHEMA_VERSION,
    RunPayload,
)
from ttflux.pipeline.errors import (
    InvalidClipRangeError,
    InvalidRunStateError,
    UnknownRunError,
    UnknownVideoError,
)
from ttflux.pipeline.states import (
    RUN_COMPLETED,
    RUN_CREATED,
    RUN_FAILED,
    RUN_RUNNING,
)
from ttflux.pipeline.storage import (
    read_json_object as _read_json_object,
    resolve_run_dir,
    try_read_json_object as _try_read_json_object,
    write_json_atomic as _write_json_atomic,
)
from ttflux.pipeline.artifacts import (
    get_run_clip_path as _get_run_clip_path,
    get_run_overlay_path as _get_run_overlay_path,
    get_run_tracks_overlay_path as _get_run_tracks_overlay_path,
)
from ttflux.pipeline.maintenance import (
    delete_run as _delete_run,
)
from ttflux.pipeline.reporting import (
    build_analysis as _build_analysis_impl,
    tracking_descriptor as _tracking_descriptor_impl,
)
from ttflux.tracking import BallTrackingEngine
from ttflux.video.catalog import find_video, probe_video


_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify(value: str, fallback: str = "video") -> str:
    """Produit un fragment lisible et sûr pour un identifiant de run."""

    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = _SLUG_PATTERN.sub("-", ascii_value.lower()).strip("-")
    return slug[:48] or fallback


def _video_snapshot(video: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": VIDEO_SNAPSHOT_SCHEMA_VERSION,
        **{key: video.get(key) for key in VIDEO_SNAPSHOT_KEYS},
    }


def _new_run_id(video: dict[str, Any], created_at: datetime) -> str:
    timestamp = created_at.strftime("%Y%m%dT%H%M%S")
    video_slug = slugify(Path(str(video["filename"])).stem)
    token = uuid4().hex[:6]
    return f"{timestamp}_{video_slug}_{token}"


def _resolve_run_dir(run_id: str) -> Path:
    return resolve_run_dir(
        run_id,
        RUNS_DIR,
    )


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
) -> RunPayload:
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
    run_payload: RunPayload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "video_id": video["id"],
        "video_filename": video["filename"],
        "video_relative_path": video["relative_path"],
        "created_at": created_at.isoformat(timespec="seconds"),
        "status": RUN_CREATED,
        "pipeline": {
            "name": PIPELINE_NAME,
            "version": PIPELINE_VERSION,
        },
        "configuration": {
            "clip_start_s": start_s,
            "clip_duration_s": duration_s,
            "candidate_detection_enabled": True,
            "track_probe_enabled": True,
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
    _extract_clip_impl(
        source_path,
        destination_path,
        start_s,
        duration_s,
    )


def _build_clip_payload(
    run_payload: RunPayload,
    clip_path: Path,
) -> dict[str, Any]:
    return _build_clip_payload_impl(
        run_payload,
        clip_path,
        probe_video_fn=probe_video,
    )


def _tracking_descriptor(
    tracking_result: Any,
) -> dict[str, Any]:
    return _tracking_descriptor_impl(
        tracking_result
    )


def _build_analysis(
    run_payload: RunPayload,
    video_payload: dict[str, Any],
    clip_payload: dict[str, Any],
    candidate_metrics: dict[str, Any],
    track_metrics: dict[str, Any],
    tracking: dict[str, Any],
    generated_at: datetime,
) -> dict[str, Any]:
    return _build_analysis_impl(
        run_payload,
        video_payload,
        clip_payload,
        candidate_metrics,
        track_metrics,
        tracking,
        generated_at,
    )


def execute_run(run_id: str) -> RunPayload:
    """Extrait le segment demandé et persiste le cycle d'état."""

    ensure_project_layout()
    run_dir = _resolve_run_dir(run_id)
    run_path = run_dir / "run.json"
    video_path = run_dir / "video.json"

    run_payload = _read_json_object(run_path)
    video_payload = _read_json_object(video_path)

    current_status = run_payload.get("status")

    if current_status != RUN_CREATED:
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
    run_payload["status"] = RUN_RUNNING
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

        tracking_result = BallTrackingEngine().run(
            clip_path=clip_path,
            output_dir=run_dir,
        )
        candidate_metrics = (
            tracking_result.candidate_metrics
        )
        track_metrics = (
            tracking_result.track_metrics
        )
        tracking_descriptor = _tracking_descriptor(
            tracking_result
        )

        generated_at = datetime.now().astimezone()
        analysis_payload = _build_analysis(
            run_payload,
            video_payload,
            clip_payload,
            candidate_metrics,
            track_metrics,
            tracking_descriptor,
            generated_at,
        )
        _write_json_atomic(analysis_path, analysis_payload)

        completed_at = datetime.now().astimezone()
        run_payload["status"] = RUN_COMPLETED
        run_payload["completed_at"] = completed_at.isoformat(timespec="seconds")
        run_payload["artifacts"].update(
            {
                "source_clip": "source_clip.mp4",
                "clip": "clip.json",
                "candidates": "candidates.csv",
                "candidate_metrics": "candidates_metrics.json",
                "candidate_overlay": "overlay_candidates.mp4",
                "tracks": "tracks_probe.csv",
                "track_metrics": "tracks_metrics.json",
                "track_overlay": "overlay_tracks_probe.mp4",
                "analysis": "analysis.json",
            }
        )
        run_payload["tracking"] = tracking_descriptor
        run_payload["metrics"] = candidate_metrics["summary"]
        run_payload["track_metrics"] = track_metrics["summary"]
        _write_json_atomic(run_path, run_payload)
    except Exception as exc:
        failed_at = datetime.now().astimezone()
        run_payload["status"] = RUN_FAILED
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
) -> RunPayload:
    """Crée puis exécute immédiatement un run d'extraction."""

    created_run = create_run(
        video_id,
        clip_start_s=clip_start_s,
        clip_duration_s=clip_duration_s,
    )
    return execute_run(str(created_run["run_id"]))


def get_run_clip_path(run_id: str) -> Path:
    return _get_run_clip_path(
        run_id,
        RUNS_DIR,
    )


def get_run_overlay_path(run_id: str) -> Path:
    return _get_run_overlay_path(
        run_id,
        RUNS_DIR,
    )


def get_run_tracks_overlay_path(
    run_id: str,
) -> Path:
    return _get_run_tracks_overlay_path(
        run_id,
        RUNS_DIR,
    )


def delete_run(run_id: str) -> dict[str, Any]:
    ensure_project_layout()
    return _delete_run(
        run_id,
        RUNS_DIR,
    )


def _created_label(value: Any) -> str:
    return _created_label_impl(value)


def list_runs() -> list[dict[str, Any]]:
    """Relit les runs locaux valides, du plus r?cent au plus ancien."""

    ensure_project_layout()
    return _list_runs_impl(
        RUNS_DIR
    )
