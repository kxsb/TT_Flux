from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
DATA = ROOT / "data_external"
REG = DATA / "registry"

SRC_OPENTT = DATA / "sources" / "openttgames"
DERIVED = DATA / "derived" / "ttnet_layout"
TTNET_DATASET = DERIVED / "dataset"
REPORT = ROOT / "runs" / "external_data_reorg_006F"

LINK_INDEX = REG / "link_index.csv"

for p in [TTNET_DATASET, REPORT]:
    p.mkdir(parents=True, exist_ok=True)

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")

def read_csv(path: Path) -> list[dict]:
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
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def hardlink_or_copy(src: Path, dst: Path) -> str:
    ensure_dir(dst.parent)

    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "exists_same_size"
        dst.unlink()

    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy_fallback"

def copy_tree_clean(src: Path, dst: Path) -> str:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return "copytree"

if not LINK_INDEX.exists():
    raise SystemExit(f"Missing registry: {LINK_INDEX}")

links = read_csv(LINK_INDEX)

rows = []
problems = []

for item in links:
    sample_id = item["sample_id"]
    split = item["split"]

    src_video = ROOT / item["video_path"]
    src_ann = ROOT / item["annotation_path"]

    dst_video = TTNET_DATASET / split / "videos" / f"{sample_id}.mp4"
    dst_ann = TTNET_DATASET / split / "annotations" / sample_id
    dst_img = TTNET_DATASET / split / "images" / sample_id

    ensure_dir(dst_video.parent)
    ensure_dir(dst_ann.parent)
    ensure_dir(dst_img)

    video_mode = "missing"
    ann_mode = "missing"

    if src_video.exists():
        video_mode = hardlink_or_copy(src_video, dst_video)
    else:
        problems.append({
            "sample_id": sample_id,
            "type": "missing_source_video",
            "path": rel(src_video),
        })

    if src_ann.exists():
        ann_mode = copy_tree_clean(src_ann, dst_ann)
    else:
        problems.append({
            "sample_id": sample_id,
            "type": "missing_source_annotation",
            "path": rel(src_ann),
        })

    ball = dst_ann / "ball_markup.json"
    events = dst_ann / "events_markup.json"
    masks = dst_ann / "segmentation_masks"

    rows.append({
        "sample_id": sample_id,
        "dataset_id": "openttgames",
        "tool_layout": "ttnet",
        "split": split,
        "video_path": rel(dst_video),
        "annotation_path": rel(dst_ann),
        "images_path": rel(dst_img),
        "video_link_mode": video_mode,
        "annotation_mode": ann_mode,
        "has_video": dst_video.exists(),
        "has_ball_markup": ball.exists(),
        "has_events_markup": events.exists(),
        "has_segmentation_masks": masks.exists(),
        "segmentation_mask_files": sum(1 for p in masks.rglob("*") if p.is_file()) if masks.exists() else 0,
    })

write_csv(REG / "ttnet_layout_index.csv", rows)
write_csv(REPORT / "problems.csv", problems)

ok = [
    r for r in rows
    if str(r["has_video"]).lower() == "true"
    and str(r["has_ball_markup"]).lower() == "true"
    and str(r["has_events_markup"]).lower() == "true"
]

md = []
md.append("# TTFlux derived TTNet layout 006F")
md.append("")
md.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
md.append(f"- Derived layout: `{rel(TTNET_DATASET)}`")
md.append(f"- Registry: `data_external/registry/ttnet_layout_index.csv`")
md.append(f"- Samples expected: **{len(rows)}**")
md.append(f"- Samples video + core annotations OK: **{len(ok)}**")
md.append(f"- Problems: **{len(problems)}**")
md.append("")
md.append("## Layout")
md.append("")
md.append("```text")
md.append("data_external/derived/ttnet_layout/dataset/")
md.append("├── training/")
md.append("│   ├── videos/")
md.append("│   ├── annotations/")
md.append("│   └── images/")
md.append("└── test/")
md.append("    ├── videos/")
md.append("    ├── annotations/")
md.append("    └── images/")
md.append("```")
md.append("")
md.append("## Samples")
md.append("")
for r in rows:
    md.append(
        f"- `{r['sample_id']}` — {r['split']} — video={r['video_link_mode']} — "
        f"ball={r['has_ball_markup']} — events={r['has_events_markup']} — masks={r['segmentation_mask_files']}"
    )
md.append("")
md.append("No original file was deleted or moved.")

(REPORT / "ttnet_layout_summary.md").write_text("\n".join(md), encoding="utf-8")

print("")
print("=== REORG 006F DONE ===")
print(f"Summary: {REPORT / 'ttnet_layout_summary.md'}")
print(f"Index:   {REG / 'ttnet_layout_index.csv'}")
print(f"OK:      {len(ok)} / {len(rows)}")
print(f"Problems:{len(problems)}")
