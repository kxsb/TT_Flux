from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def legacy_root() -> Path:
    env_value = os.environ.get("TTFLUX_LEGACY_ROOT", "").strip()

    candidates = []
    if env_value:
        try:
            candidates.append(Path(env_value).expanduser())
        except Exception:
            pass

    candidates.append(project_root().parent / "TTFlux")

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate

        if resolved.exists():
            return resolved

    # Dernier fallback : sibling attendu, même s'il n'existe pas.
    return project_root().parent / "TTFlux"


def data_dir() -> Path:
    return project_root() / "data"


def runs_dir() -> Path:
    return project_root() / "runs"


def configs_dir() -> Path:
    return project_root() / "configs"


def legacy_imports_dir() -> Path:
    return project_root() / "legacy_imports"


def ensure_base_dirs() -> None:
    for p in [
        data_dir(),
        data_dir() / "videos",
        data_dir() / "frames",
        data_dir() / "labels",
        data_dir() / "external",
        runs_dir(),
        configs_dir(),
        legacy_imports_dir(),
    ]:
        p.mkdir(parents=True, exist_ok=True)
