from __future__ import annotations

from pathlib import Path

from ttflux.pipeline.storage import (
    read_json_object,
    resolve_run_dir,
)


def resolve_run_artifact_path(
    run_id: str,
    runs_dir: Path,
    *,
    artifact_key: str,
    expected_filename: str,
    missing_reference_message: str,
    missing_file_message: str,
) -> Path:
    run_dir = resolve_run_dir(
        run_id,
        runs_dir,
    )
    run_payload = read_json_object(
        run_dir / "run.json"
    )

    artifacts = run_payload.get("artifacts")
    artifact_name = (
        artifacts.get(artifact_key)
        if isinstance(artifacts, dict)
        else None
    )

    if artifact_name != expected_filename:
        raise FileNotFoundError(
            missing_reference_message.format(
                run_id=run_id
            )
        )

    artifact_path = run_dir / expected_filename

    if not artifact_path.is_file():
        raise FileNotFoundError(
            missing_file_message.format(
                run_id=run_id
            )
        )

    return artifact_path


def get_run_clip_path(
    run_id: str,
    runs_dir: Path,
) -> Path:
    return resolve_run_artifact_path(
        run_id,
        runs_dir,
        artifact_key="source_clip",
        expected_filename="source_clip.mp4",
        missing_reference_message=(
            "Aucun segment vid\u00e9o disponible "
            "pour le run {run_id}."
        ),
        missing_file_message=(
            "Artefact absent pour le run {run_id}."
        ),
    )


def get_run_overlay_path(
    run_id: str,
    runs_dir: Path,
) -> Path:
    return resolve_run_artifact_path(
        run_id,
        runs_dir,
        artifact_key="candidate_overlay",
        expected_filename="overlay_candidates.mp4",
        missing_reference_message=(
            "Aucun overlay candidat disponible "
            "pour le run {run_id}."
        ),
        missing_file_message=(
            "Overlay candidat absent "
            "pour le run {run_id}."
        ),
    )


def get_run_tracks_overlay_path(
    run_id: str,
    runs_dir: Path,
) -> Path:
    return resolve_run_artifact_path(
        run_id,
        runs_dir,
        artifact_key="track_overlay",
        expected_filename="overlay_tracks_probe.mp4",
        missing_reference_message=(
            "Aucun overlay de pistes disponible "
            "pour le run {run_id}."
        ),
        missing_file_message=(
            "Overlay de pistes absent "
            "pour le run {run_id}."
        ),
    )


__all__ = [
    "get_run_clip_path",
    "get_run_overlay_path",
    "get_run_tracks_overlay_path",
    "resolve_run_artifact_path",
]
