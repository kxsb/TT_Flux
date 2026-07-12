from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ttflux.pipeline.storage import (
    try_read_json_object,
)


def created_label(value: Any) -> str:
    if not isinstance(value, str):
        return "Date inconnue"

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value

    return parsed.strftime(
        "%d/%m/%Y %H:%M:%S"
    )


def list_runs(
    runs_dir: Path,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []

    for run_json_path in runs_dir.glob(
        "*/run.json"
    ):
        payload = try_read_json_object(
            run_json_path
        )

        if (
            payload is None
            or not payload.get("run_id")
        ):
            continue

        summary = dict(payload)
        summary["created_label"] = (
            created_label(
                payload.get("created_at")
            )
        )
        summary["directory"] = (
            run_json_path.parent.name
        )
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


__all__ = [
    "created_label",
    "list_runs",
]
