from __future__ import annotations

import os
from pathlib import Path


def find_project_root() -> Path:
    """Retrouve la racine du projet sans dépendre du dossier courant."""

    override = os.environ.get("TTFLUX_ROOT")

    if override:
        return Path(override).expanduser().resolve()

    current_file = Path(__file__).resolve()

    for parent in current_file.parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
RUNS_DIR = PROJECT_ROOT / "runs"
DOCS_DIR = PROJECT_ROOT / "docs"

VIDEO_CATALOG_PATH = DATA_DIR / "video_catalog.json"


def ensure_project_layout() -> None:
    """Crée uniquement les dossiers de données manquants."""

    for directory in (
        DATA_DIR,
        VIDEOS_DIR,
        ANNOTATIONS_DIR,
        RUNS_DIR,
        DOCS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)