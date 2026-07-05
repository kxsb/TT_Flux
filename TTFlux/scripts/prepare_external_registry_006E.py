from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data_external"

RAW_OPENTT = DATA / "raw" / "openttgames"
SRC = DATA / "sources" / "openttgames"
SRC_VIDEOS = SRC / "raw_videos"
SRC_ARCHIVES = SRC / "raw_archives"
SRC_ANN = SRC / "annotations"

REG = DATA / "registry"
REPORT = ROOT / "runs" / "external_data_reorg_006E"

IDS = [f"game_{i}" for i in range(1, 6)] + [f"test_{i}" for i in range(1, 8)]

for p in [SRC_VIDEOS, SRC_ARCHIVES, SRC_ANN, REG, REPORT]:
    p.mkdir(parents=True, exist_ok=True)

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")

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

def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of", "json",
        str(path),
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if cp.returncode != 0:
            return {"ffprobe_ok": False, "ffprobe_error": cp.stderr[:300]}
        data = json.loads(cp.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        return {
            "ffprobe_ok": True,
            "width": stream.get("width", ""),
            "height": stream.get("height", ""),
            "r_frame_rate": stream.get("r_frame_rate", ""),
            "avg_frame_rate": stream.get("avg_frame_rate", ""),
            "nb_frames": stream.get("nb_frames", ""),
            "duration": stream.get("duration", ""),
        }
    except Exception as e:
        return {"ffprobe_ok": False, "ffprobe_error": repr(e)[:300]}

def hardlink_or_copy(src: Path, dst: Path) -> str:
    """
    Hardlink = pas de doublon disque sur même volume NTFS.
    Si impossible, fallback copie.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

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

dataset_rows = []
video_rows = []
archive_rows = []
annotation_rows = []
link_rows = []
problems = []

dataset_rows.append({
    "dataset_id": "openttgames",
    "dataset_name": "OpenTTGames",
    "source_family": "lab.osai.ai",
    "local_root": rel(SRC),
    "status": "prepared_non_destructive",
    "created_at": datetime.now().isoformat(timespec="seconds"),
})

for oid in IDS:
    split = "training" if oid.startswith("game_") else "test"

    raw_mp4 = RAW_OPENTT / f"{oid}.mp4"
    raw_zip = RAW_OPENTT / f"{oid}.zip"

    clean_mp4 = SRC_VIDEOS / f"{oid}.mp4"
    clean_zip = SRC_ARCHIVES / f"{oid}.zip"
    clean_ann = SRC_ANN / oid

    video_status = "missing"
    archive_status = "missing"
    annotation_status = "missing"

    if raw_mp4.exists():
        mode = hardlink_or_copy(raw_mp4, clean_mp4)
        video_status = mode
        vrow = {
            "video_id": oid,
            "dataset_id": "openttgames",
            "split": split,
            "path": rel(clean_mp4),
            "source_path": rel(raw_mp4),
            "size_bytes": clean_mp4.stat().st_size,
            "link_mode": mode,
        }
        vrow.update(ffprobe(clean_mp4))
        video_rows.append(vrow)
    else:
        problems.append({"id": oid, "type": "missing_video", "path": rel(raw_mp4)})

    if raw_zip.exists():
        mode = hardlink_or_copy(raw_zip, clean_zip)
        archive_status = mode

        zip_ok = False
        zip_count = 0
        has_ball = False
        has_events = False
        has_masks = False

        try:
            with zipfile.ZipFile(clean_zip, "r") as zf:
                names = zf.namelist()
                zip_ok = True
                zip_count = len(names)
                has_ball = any(n.endswith("ball_markup.json") for n in names)
                has_events = any(n.endswith("events_markup.json") for n in names)
                has_masks = any("segmentation_masks/" in n or "segmentation_masks\\" in n for n in names)

                clean_ann.mkdir(parents=True, exist_ok=True)

                # Nettoyage ciblé du dossier d'annotation propre uniquement.
                for child in clean_ann.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()

                zf.extractall(clean_ann)

            archive_rows.append({
                "archive_id": oid,
                "dataset_id": "openttgames",
                "split": split,
                "path": rel(clean_zip),
                "source_path": rel(raw_zip),
                "size_bytes": clean_zip.stat().st_size,
                "link_mode": mode,
                "zip_ok": zip_ok,
                "zip_file_count": zip_count,
                "zip_has_ball_markup": has_ball,
                "zip_has_events_markup": has_events,
                "zip_has_segmentation_masks": has_masks,
            })
        except Exception as e:
            problems.append({"id": oid, "type": "zip_extract_error", "path": rel(clean_zip), "error": repr(e)[:300]})
    else:
        problems.append({"id": oid, "type": "missing_archive", "path": rel(raw_zip)})

    ball = clean_ann / "ball_markup.json"
    events = clean_ann / "events_markup.json"
    masks = clean_ann / "segmentation_masks"

    mask_count = 0
    if masks.exists():
        mask_count = sum(1 for p in masks.rglob("*") if p.is_file())

    if ball.exists() or events.exists() or mask_count:
        annotation_status = "ok"

    annotation_rows.append({
        "annotation_id": oid,
        "dataset_id": "openttgames",
        "split": split,
        "path": rel(clean_ann),
        "has_ball_markup": ball.exists(),
        "ball_markup_path": rel(ball) if ball.exists() else "",
        "has_events_markup": events.exists(),
        "events_markup_path": rel(events) if events.exists() else "",
        "has_segmentation_masks": masks.exists(),
        "segmentation_masks_path": rel(masks) if masks.exists() else "",
        "segmentation_mask_files": mask_count,
    })

    link_rows.append({
        "sample_id": oid,
        "dataset_id": "openttgames",
        "split": split,
        "video_id": oid,
        "annotation_id": oid,
        "archive_id": oid,
        "video_path": rel(clean_mp4) if clean_mp4.exists() else "",
        "annotation_path": rel(clean_ann) if clean_ann.exists() else "",
        "archive_path": rel(clean_zip) if clean_zip.exists() else "",
        "video_status": video_status,
        "archive_status": archive_status,
        "annotation_status": annotation_status,
    })

write_csv(REG / "dataset_index.csv", dataset_rows)
write_csv(REG / "video_index.csv", video_rows)
write_csv(REG / "archive_index.csv", archive_rows)
write_csv(REG / "annotation_index.csv", annotation_rows)
write_csv(REG / "link_index.csv", link_rows)
write_csv(REPORT / "problems.csv", problems)

# Rapport markdown
ok_links = [r for r in link_rows if r["video_status"] != "missing" and r["annotation_status"] == "ok"]
bad_links = [r for r in link_rows if not (r["video_status"] != "missing" and r["annotation_status"] == "ok")]

md = []
md.append("# TTFlux external data reorg 006E")
md.append("")
md.append("Non-destructive preparation. No original file was deleted or moved.")
md.append("")
md.append(f"- Source root: `{rel(SRC)}`")
md.append(f"- Registry root: `{rel(REG)}`")
md.append(f"- OpenTTGames expected IDs: **{len(IDS)}**")
md.append(f"- Linked video + annotations OK: **{len(ok_links)}**")
md.append(f"- Problem links: **{len(bad_links)}**")
md.append("")
md.append("## Generated registry files")
md.append("")
md.append("- `data_external/registry/dataset_index.csv`")
md.append("- `data_external/registry/video_index.csv`")
md.append("- `data_external/registry/archive_index.csv`")
md.append("- `data_external/registry/annotation_index.csv`")
md.append("- `data_external/registry/link_index.csv`")
md.append("")
md.append("## OpenTTGames links")
md.append("")
for r in link_rows:
    md.append(
        f"- `{r['sample_id']}` — video={r['video_status']} — archive={r['archive_status']} — annotations={r['annotation_status']}"
    )
md.append("")
md.append("## Next possible cleanup")
md.append("")
md.append("If hardlinks worked, `sources/openttgames/raw_videos` does not duplicate disk blocks.")
md.append("After validation, the duplicated TTNet-layout MP4 copies can be replaced by hardlinks or symlinks.")

(REPORT / "reorg_summary.md").write_text("\n".join(md), encoding="utf-8")

print("")
print("=== REORG 006E DONE ===")
print(f"Summary: {REPORT / 'reorg_summary.md'}")
print(f"Registry: {REG}")
print("")
print("Links OK:", len(ok_links), "/", len(IDS))
if problems:
    print("Problems:", len(problems))
    for p in problems[:20]:
        print(p)
