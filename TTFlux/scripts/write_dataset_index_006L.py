from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data_external"
REG = DATA / "registry"
OUT_MD = DATA / "DATASET_INDEX.md"
OUT_CSV = REG / "dataset_assets_index.csv"

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
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def dir_size(path: Path):
    total = 0
    count = 0
    if not path.exists():
        return 0, 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            count += 1
    return total, count

def gb(n: int) -> float:
    return round(n / 1024 / 1024 / 1024, 3)

assets = []

for asset_id, kind, path, role, status, note in [
    ("openttgames_sources", "dataset_source", DATA / "sources" / "openttgames", "canonical_source", "ready", "12 videos + annotations + masks"),
    ("openttgames_raw_videos", "video_source", DATA / "sources" / "openttgames" / "raw_videos", "canonical_videos", "ready", "OpenTTGames MP4 1920x1080 120fps"),
    ("openttgames_annotations", "annotation_source", DATA / "sources" / "openttgames" / "annotations", "canonical_annotations", "ready", "ball_markup/events/masks per sample"),
    ("ttnet_layout", "derived_layout", DATA / "derived" / "ttnet_layout" / "dataset", "tool_layout", "ready", "TTNet-compatible dataset layout"),
    ("ttnet_repo", "code", DATA / "code" / "ttnet_pytorch", "training_code_reference", "ready", "TTNet PyTorch repo; dataset path is junction to derived layout"),
    ("tt3d_code", "code", DATA / "code" / "tt3d", "3d_reconstruction_reference", "ready", "TT3D code and examples"),
    ("tt3d_raw", "dataset_source", DATA / "raw" / "tt3d_raw", "raw_reference_data", "ready", "TT3D raw archive"),
    ("tt3d_raw_extracted", "dataset_source", DATA / "extracted" / "tt3d_raw", "extracted_reference_data", "ready", "TT3D raw extracted"),
    ("blurball_code", "code", DATA / "code" / "blurball", "ball_tracking_reference", "ready", "Blur-aware ball tracking repo"),
    ("sportsvideo_code", "code", DATA / "code" / "sportsvideo", "event_detection_reference", "ready", "SportsVideo/MediaEval reference"),
    ("registry", "index", DATA / "registry", "central_registry", "ready", "dataset/video/annotation/link indexes"),
]:
    size, count = dir_size(path)
    assets.append({
        "asset_id": asset_id,
        "kind": kind,
        "path": rel(path),
        "exists": path.exists(),
        "role": role,
        "status": status if path.exists() else "missing",
        "file_count": count,
        "size_gb": gb(size),
        "note": note,
    })

write_csv(OUT_CSV, assets)

links = read_csv(REG / "link_index.csv")
ttnet = read_csv(REG / "ttnet_layout_index.csv")

md = []
md.append("# TTFlux — Dataset Index")
md.append("")
md.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
md.append("")
md.append("## Règle d’usage")
md.append("")
md.append("- Utiliser `data_external/sources/` comme source canonique.")
md.append("- Utiliser `data_external/derived/` pour les formats propres à un outil.")
md.append("- Utiliser `data_external/registry/` pour relier vidéos, annotations, archives et layouts.")
md.append("- Ne plus utiliser les anciens chemins `data_external/raw/openttgames` et `data_external/extracted/openttgames`.")
md.append("")
md.append("## Assets actifs")
md.append("")
for a in assets:
    md.append(f"- `{a['asset_id']}` — `{a['path']}` — {a['size_gb']} GB — {a['status']} — {a['note']}")
md.append("")
md.append("## OpenTTGames — liens canoniques")
md.append("")
for r in links:
    md.append(f"- `{r['sample_id']}` — split `{r['split']}`")
    md.append(f"  - video: `{r['video_path']}`")
    md.append(f"  - annotations: `{r['annotation_path']}`")
    md.append(f"  - archive: `{r['archive_path']}`")
md.append("")
md.append("## Layout TTNet")
md.append("")
md.append("Chemin recommandé pour scripts TTNet :")
md.append("")
md.append("```text")
md.append("data_external/code/ttnet_pytorch/dataset")
md.append("```")
md.append("")
md.append("Ce chemin est une jonction vers :")
md.append("")
md.append("```text")
md.append("data_external/derived/ttnet_layout/dataset")
md.append("```")
md.append("")
md.append("## Prochaine étape recommandée")
md.append("")
md.append("Faire une extraction contrôlée des images annotées sur 1 sample, puis sur les 12 samples si tout est stable.")
md.append("")
md.append("Ne pas lancer l'entraînement TTNet complet avant d'avoir validé :")
md.append("")
md.append("- extraction images ;")
md.append("- compatibilité Windows des scripts ;")
md.append("- chemins avec accent ;")
md.append("- taille disque générée par les images.")

OUT_MD.write_text("\n".join(md), encoding="utf-8")

print("")
print("=== DATASET INDEX 006L DONE ===")
print("Markdown:", OUT_MD)
print("CSV:", OUT_CSV)
