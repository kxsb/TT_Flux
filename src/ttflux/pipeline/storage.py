from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ttflux.pipeline.errors import UnknownRunError


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_json_object(
    path: Path,
) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise UnknownRunError(
            path.parent.name
        ) from None

    if not isinstance(payload, dict):
        raise ValueError(
            f"Objet JSON attendu dans {path.name}"
        )

    return payload


def try_read_json_object(
    path: Path,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(payload, dict):
        return None

    return payload


def resolve_run_dir(
    run_id: str,
    runs_dir: Path,
) -> Path:
    if (
        not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or "/" in run_id
        or "\\" in run_id
    ):
        raise UnknownRunError(run_id)

    run_dir = runs_dir / run_id

    if not run_dir.is_dir():
        raise UnknownRunError(run_id)

    return run_dir


__all__ = [
    "read_json_object",
    "resolve_run_dir",
    "try_read_json_object",
    "write_json_atomic",
]
