from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import time
import uuid


@dataclass
class RunManifest:
    run_id: str
    patch: str
    status: str
    created_at: float
    params: dict[str, Any]
    outputs: dict[str, str]
    notes: str = ""


def new_run_id(patch: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{stamp}_{patch}_{suffix}"


def write_manifest(run_dir: Path, patch: str, params: dict[str, Any] | None = None, notes: str = "") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(
        run_id=run_dir.name,
        patch=patch,
        status="created",
        created_at=time.time(),
        params=params or {},
        outputs={},
        notes=notes,
    )
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return out
