from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ttflux.pipeline.errors import (
    InvalidRunStateError,
)
from ttflux.pipeline.states import RUN_RUNNING
from ttflux.pipeline.storage import (
    read_json_object,
    resolve_run_dir,
)


def delete_run(
    run_id: str,
    runs_dir: Path,
) -> dict[str, Any]:
    run_dir = resolve_run_dir(
        run_id,
        runs_dir,
    )
    run_payload = read_json_object(
        run_dir / "run.json"
    )
    current_status = run_payload.get("status")

    if current_status == RUN_RUNNING:
        raise InvalidRunStateError(
            f"Le run {run_id} est encore "
            "en cours d'ex\u00e9cution."
        )

    deleted = {
        "run_id": run_id,
        "status": current_status,
        "video_id": run_payload.get("video_id"),
        "video_filename": run_payload.get(
            "video_filename"
        ),
    }

    shutil.rmtree(run_dir)
    return deleted


__all__ = [
    "delete_run",
]
