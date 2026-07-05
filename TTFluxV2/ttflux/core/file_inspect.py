from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha1_short(path: Path, max_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(65536, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()[:12]


def inspect_csv(path: Path, sample_rows: int = 3) -> dict[str, Any]:
    out: dict[str, Any] = {
        "columns": [],
        "sample": [],
        "row_count_estimate": 0,
        "csv_error": "",
    }

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            out["columns"] = list(reader.fieldnames or [])
            rows = []
            count = 0
            for row in reader:
                count += 1
                if len(rows) < sample_rows:
                    rows.append({k: row.get(k, "") for k in out["columns"][:20]})
            out["sample"] = rows
            out["row_count_estimate"] = count
    except UnicodeDecodeError:
        try:
            with path.open("r", encoding="cp1252", newline="") as f:
                reader = csv.DictReader(f)
                out["columns"] = list(reader.fieldnames or [])
                rows = []
                count = 0
                for row in reader:
                    count += 1
                    if len(rows) < sample_rows:
                        rows.append({k: row.get(k, "") for k in out["columns"][:20]})
                out["sample"] = rows
                out["row_count_estimate"] = count
        except Exception as exc:
            out["csv_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        out["csv_error"] = f"{type(exc).__name__}: {exc}"

    return out


def inspect_json(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "json_type": "",
        "json_keys": [],
        "json_len": None,
        "json_error": "",
    }

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        out["json_type"] = type(data).__name__
        if isinstance(data, dict):
            out["json_keys"] = list(data.keys())[:50]
            out["json_len"] = len(data)
        elif isinstance(data, list):
            out["json_len"] = len(data)
            if data and isinstance(data[0], dict):
                out["json_keys"] = list(data[0].keys())[:50]
    except Exception as exc:
        out["json_error"] = f"{type(exc).__name__}: {exc}"

    return out


def inspect_artifact(path: Path) -> dict[str, Any]:
    path = Path(path)
    out: dict[str, Any] = {
        "absolute_path": str(path),
        "exists": path.exists(),
        "kind": "missing",
        "size_bytes": 0,
        "size_mb": 0.0,
        "mtime": None,
        "sha1_1mb": "",
        "suffix": path.suffix.lower(),
        "columns": [],
        "row_count_estimate": None,
        "json_type": "",
        "json_keys": [],
        "json_len": None,
        "error": "",
    }

    if not path.exists():
        return out

    try:
        stat = path.stat()
        out["mtime"] = stat.st_mtime
        out["kind"] = "dir" if path.is_dir() else "file" if path.is_file() else "other"

        if path.is_file():
            out["size_bytes"] = stat.st_size
            out["size_mb"] = round(stat.st_size / 1024 / 1024, 3)
            out["sha1_1mb"] = sha1_short(path)

            if path.suffix.lower() == ".csv":
                csv_info = inspect_csv(path)
                out["columns"] = csv_info.get("columns", [])
                out["row_count_estimate"] = csv_info.get("row_count_estimate")
                if csv_info.get("csv_error"):
                    out["error"] = csv_info["csv_error"]

            if path.suffix.lower() == ".json":
                json_info = inspect_json(path)
                out["json_type"] = json_info.get("json_type", "")
                out["json_keys"] = json_info.get("json_keys", [])
                out["json_len"] = json_info.get("json_len")
                if json_info.get("json_error"):
                    out["error"] = json_info["json_error"]

        elif path.is_dir():
            children = list(path.iterdir())
            out["dir_child_count"] = len(children)
            out["dir_sample"] = [c.name for c in children[:20]]

    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"

    return out
