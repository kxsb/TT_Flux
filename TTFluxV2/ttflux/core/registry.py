from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import time


@dataclass
class ArtifactRecord:
    artifact_id: str
    kind: str
    path: str
    source: str = "ttfluxv2"
    status: str = "draft"
    notes: str = ""


class ArtifactRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write({"created_at": time.time(), "artifacts": []})

    def _read(self) -> dict[str, Any]:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def list(self) -> list[dict[str, Any]]:
        return self._read().get("artifacts", [])

    def add(self, record: ArtifactRecord) -> None:
        payload = self._read()
        artifacts = payload.setdefault("artifacts", [])
        artifacts = [a for a in artifacts if a.get("artifact_id") != record.artifact_id]
        artifacts.append(asdict(record))
        payload["artifacts"] = artifacts
        payload["updated_at"] = time.time()
        self._write(payload)
