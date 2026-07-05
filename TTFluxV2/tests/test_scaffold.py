from pathlib import Path


def test_project_scaffold_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app" / "backend" / "main.py").exists()
    assert (root / "app" / "static" / "index.html").exists()
    assert (root / "ttflux" / "core" / "paths.py").exists()
    assert (root / "legacy_imports" / "legacy_artifacts_index.json").exists()
