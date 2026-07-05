from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ttflux.core.paths import project_root, legacy_root
from ttflux.video.reader import read_video_meta

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}


def video_id_for(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:16]


def is_inside_dir(path: Path, dirname: str) -> bool:
    return dirname.lower() in [p.lower() for p in path.parts]


def classify_video(path: Path, legacy: Path) -> str:
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()

    if "runs" in parts:
        return "generated_run_video"
    if "overlay" in name or "contact" in name or "debug" in name or "preview" in name:
        return "generated_or_debug"
    if "clip" in parts or "clips" in parts:
        return "source_clip"
    if "video" in parts or "videos" in parts:
        return "source_video"
    if str(path).lower().startswith(str(legacy).lower()):
        return "legacy_unknown_video"
    return "project_video"


def default_scan_roots(include_unknown_roots: bool = False) -> list[tuple[str, Path]]:
    root = project_root()
    legacy = legacy_root()

    candidates = [
        ("v2_data_videos", root / "data" / "videos"),
        ("legacy_data_videos", legacy / "data" / "videos"),
        ("legacy_videos", legacy / "videos"),
        ("legacy_clips", legacy / "clips"),
        ("legacy_data", legacy / "data"),
    ]

    if include_unknown_roots:
        candidates.append(("legacy_root_wide", legacy))

    out: list[tuple[str, Path]] = []
    seen = set()
    for label, p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            continue
        key = str(rp).lower()
        if key not in seen:
            seen.add(key)
            out.append((label, rp))
    return out


def iter_video_files(
    include_generated: bool = False,
    include_unknown: bool = False,
    max_results: int = 500,
) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    seen = set()

    source_roles = {"source_video", "source_clip", "project_video"}

    for root_label, root in default_scan_roots(include_unknown_roots=include_unknown):
        if not root.exists():
            continue

        try:
            for p in root.rglob("*"):
                if len(out) >= max_results:
                    return out

                if any(part in SKIP_DIR_NAMES for part in p.parts):
                    continue

                if not p.is_file():
                    continue

                if p.suffix.lower() not in VIDEO_EXTS:
                    continue

                rp = p.resolve()
                key = str(rp).lower()
                if key in seen:
                    continue

                role = classify_video(rp, legacy_root())
                if not include_generated and role.startswith("generated"):
                    continue

                if not include_unknown and role not in source_roles:
                    continue

                seen.add(key)
                out.append((root_label, rp))
        except Exception:
            continue

    return out


def build_video_library(
    output_dir: Path | None = None,
    include_generated: bool = False,
    include_unknown: bool = False,
    max_results: int = 500,
) -> dict[str, Any]:
    root = project_root()
    legacy = legacy_root()

    if output_dir is None:
        output_dir = root / "runs" / "007C_video_library"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for root_label, path in iter_video_files(
        include_generated=include_generated,
        include_unknown=include_unknown,
        max_results=max_results,
    ):
        try:
            stat = path.stat()
            role = classify_video(path, legacy)
            meta = read_video_meta(path)
            record = {
                "id": video_id_for(path),
                "root_label": root_label,
                "role": role,
                "name": path.name,
                "path": str(path),
                "suffix": path.suffix.lower(),
                "size_mb": round(stat.st_size / 1024 / 1024, 3),
                "mtime": stat.st_mtime,
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
                "fps": meta.get("fps", 0.0),
                "frame_count": meta.get("frame_count", 0),
                "duration_sec": meta.get("duration_sec", 0.0),
                "backend_ok": meta.get("backend_ok", False),
                "error": meta.get("error", ""),
            }
        except Exception as exc:
            record = {
                "id": video_id_for(path),
                "root_label": root_label,
                "role": "inspect_error",
                "name": path.name,
                "path": str(path),
                "suffix": path.suffix.lower(),
                "size_mb": 0,
                "mtime": 0,
                "width": 0,
                "height": 0,
                "fps": 0,
                "frame_count": 0,
                "duration_sec": 0,
                "backend_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        records.append(record)

    records.sort(
        key=lambda r: (
            0 if r.get("role") in {"source_video", "source_clip", "project_video"} else 1,
            -float(r.get("duration_sec") or 0),
            str(r.get("name") or ""),
        )
    )

    summary = {
        "patch": "007C_TTFluxV2_video_frame_browser",
        "created_at": time.time(),
        "project_root": str(root),
        "legacy_root": str(legacy),
        "include_generated": include_generated,
        "include_unknown": include_unknown,
        "max_results": max_results,
        "records_total": len(records),
        "backend_ok_count": sum(1 for r in records if r.get("backend_ok")),
        "source_like_count": sum(1 for r in records if r.get("role") in {"source_video", "source_clip", "project_video"}),
        "generated_count": sum(1 for r in records if str(r.get("role", "")).startswith("generated")),
        "records": records,
    }

    json_path = output_dir / "video_library_007C.json"
    csv_path = output_dir / "video_library_007C.csv"
    summary_path = output_dir / "video_library_summary_007C.json"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    compact = {k: v for k, v in summary.items() if k != "records"}
    summary_path.write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "id",
        "root_label",
        "role",
        "name",
        "path",
        "suffix",
        "size_mb",
        "width",
        "height",
        "fps",
        "frame_count",
        "duration_sec",
        "backend_ok",
        "error",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    summary["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "summary_json": str(summary_path),
    }

    return summary


if __name__ == "__main__":
    report = build_video_library(include_generated=False, include_unknown=False, max_results=500)
    print("OK 007C")
    print(f"records_total={report['records_total']}")
    print(f"backend_ok_count={report['backend_ok_count']}")
    print(f"source_like_count={report['source_like_count']}")
    print(f"generated_count={report['generated_count']}")
    print(f"json={report['outputs']['json']}")
    print(f"csv={report['outputs']['csv']}")
    print(f"summary_json={report['outputs']['summary_json']}")
