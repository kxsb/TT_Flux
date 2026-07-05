from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data_external"
REG = DATA / "registry"
OUT_MD = DATA / "DATASET_INDEX.md"
OUT_CSV = REG / "dataset_assets_index.csv"

TTNET_DATASET_JUNCTION = DATA / "code" / "ttnet_pytorch" / "dataset"

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def is_junction_like(p: Path) -> bool:
    if hasattr(p, "is_junction"):
        try:
            return p.is_junction()
        except Exception:
            return False
    return False

def dir_size(path: Path, skip_junctions: bool = True):
    total = 0
    count = 0

    if not path.exists():
        return 0, 0

    if path.is_file():
        return path.stat().st_size, 1

    for dirpath, dirnames, filenames in os.walk(path):
        d = Path(dirpath)

        if skip_junctions:
            kept = []
            for name in dirnames:
                child = d / name
                if is_junction_like(child):
                    continue
                if child == TTNET_DATASET_JUNCTION:
                    continue
                kept.append(name)
            dirnames[:] = kept

        for fn in filenames:
            p = d / fn
            try:
                total += p.stat().st_size
                count += 1
            except Exception:
                pass

    return total, count

def gb(n: int) -> float:
    return round(n / 1024 / 1024 / 1024, 3)

assets_spec = [
    ("openttgames_sources", "dataset_source", DATA / "sources" / "openttgames", "canonical_source", "ready", "12 videos + annotations + masks"),
    ("openttgames_raw_videos", "video_source", DATA / "sources" / "openttgames" / "raw_videos", "canonical_videos", "ready", "OpenTTGames MP4 1920x1080 120fps"),
    ("openttgames_annotations", "annotation_source", DATA / "sources" / "openttgames" / "annotations", "canonical_annotations", "ready", "ball_markup/events/masks per sample"),
    ("ttnet_layout", "derived_layout", DATA / "derived" / "ttnet_layout" / "dataset", "tool_layout", "ready", "TTNet-compatible dataset layout; videos are hardlinks"),
    ("ttnet_repo", "code", DATA / "code" / "ttnet_pytorch", "training_code_reference", "ready", "TTNet PyTorch repo; dataset path is a junction and is not counted here"),
    ("tt3d_code", "code", DATA / "code" / "tt3d", "3d_reconstruction_reference", "ready", "TT3D code and examples"),
    ("tt3d_raw", "dataset_source", DATA / "raw" / "tt3d_raw", "raw_reference_data", "ready", "TT3D raw archive"),
    ("tt3d_raw_extracted", "dataset_source", DATA / "extracted" / "tt3d_raw", "extracted_reference_data", "ready", "TT3D raw extracted"),
    ("blurball_code", "code", DATA / "code" / "blurball", "ball_tracking_reference", "ready", "Blur-aware ball tracking repo"),
    ("sportsvideo_code", "code", DATA / "code" / "sportsvideo", "event_detection_reference", "ready", "SportsVideo/MediaEval reference"),
    ("registry", "index", DATA / "registry", "central_registry", "ready", "dataset/video/annotation/link indexes"),
]

assets = []
for asset_id, kind, path, role, status, note in assets_spec:
    size, count = dir_size(path, skip_junctions=True)
    assets.append({
        "asset_id": asset_id,
        "kind": kind,
        "path": rel(path),
        "exists": path.exists(),
        "role": role,
        "status": status if path.exists() else "missing",
        "file_count": count,
        "logical_size_gb": gb(size),
        "note": note,
    })

write_csv(OUT_CSV, assets)

links = read_csv(REG / "link_index.csv")

md = []
md.append("# TTFlux - Dataset Index")
md.append("")
md.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
md.append("")
md.append("## Usage rules")
md.append("")
md.append("- Use `data_external/sources/` as the canonical source area.")
md.append("- Use `data_external/derived/` for tool-specific layouts.")
md.append("- Use `data_external/registry/` to link videos, annotations, archives and derived layouts.")
md.append("- Do not use old paths `data_external/raw/openttgames` or `data_external/extracted/openttgames`.")
md.append("- `data_external/code/ttnet_pytorch/dataset` is a Windows junction to the derived TTNet layout.")
md.append("")
md.append("## Active assets")
md.append("")
for a in assets:
    md.append(f"- `{a['asset_id']}`")
    md.append(f"  - path: `{a['path']}`")
    md.append(f"  - status: `{a['status']}`")
    md.append(f"  - files: `{a['file_count']}`")
    md.append(f"  - logical size: `{a['logical_size_gb']} GB`")
    md.append(f"  - note: {a['note']}")
md.append("")
md.append("## OpenTTGames canonical links")
md.append("")
for r in links:
    md.append(f"- `{r['sample_id']}` - split `{r['split']}`")
    md.append(f"  - video: `{r['video_path']}`")
    md.append(f"  - annotations: `{r['annotation_path']}`")
    md.append(f"  - archive: `{r['archive_path']}`")
md.append("")
md.append("## TTNet layout")
md.append("")
md.append("Recommended path for TTNet scripts:")
md.append("")
md.append("```text")
md.append("data_external/code/ttnet_pytorch/dataset")
md.append("```")
md.append("")
md.append("This path points to:")
md.append("")
md.append("```text")
md.append("data_external/derived/ttnet_layout/dataset")
md.append("```")
md.append("")
md.append("## Validated state")
md.append("")
md.append("- Cleanup quarantine deleted.")
md.append("- Active audit: 34.7 GB, 55,820 files, 566 videos.")
md.append("- OpenTTGames integrity: 12/12 samples OK.")
md.append("- Smoke test: game_1 video and annotations readable.")
md.append("")
md.append("## Next recommended step")
md.append("")
md.append("Run a controlled TTNet image extraction on one sample first, then expand to all 12 samples if disk growth is acceptable.")

OUT_MD.write_text("\n".join(md), encoding="utf-8")

print("")
print("=== DATASET INDEX 006L2 DONE ===")
print("Markdown:", OUT_MD)
print("CSV:", OUT_CSV)
