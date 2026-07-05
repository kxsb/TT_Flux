from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the TTFluxV4 project root from the installed source layout."""
    return Path(__file__).resolve().parents[3]


def ensure_project_layout(root: Path | None = None) -> dict[str, Path]:
    root = root or project_root()
    paths = {
        "data": root / "data",
        "runs": root / "runs",
        "videos_raw": root / "data" / "videos_raw",
        "datasets": root / "data" / "datasets",
        "reports": root / "runs" / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
