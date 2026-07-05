from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data_external"
OUT = ROOT / "runs" / "external_data_audit_006D"

OUT.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ARCHIVE_EXTS = {".zip", ".7z", ".tar", ".gz", ".bz2", ".xz", ".rar"}
ANNOT_EXTS = {".json", ".csv", ".txt", ".xml", ".yaml", ".yml", ".pkl", ".npy", ".npz"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DOC_EXTS = {".pdf", ".md", ".html", ".htm", ".txt"}

OPENTT_IDS = [f"game_{i}" for i in range(1, 6)] + [f"test_{i}" for i in range(1, 8)]

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")

def size_mb(n: int) -> float:
    return round(n / 1024 / 1024, 3)

def safe_read_head(path: Path, n: int = 512) -> bytes:
    try:
        with path.open("rb") as f:
            return f.read(n)
    except Exception:
        return b""

def quick_hash(path: Path) -> str:
    """
    Hash léger : taille + premiers/derniers 1 Mo.
    Suffisant pour repérer doublons probables sans hasher 30 Go.
    """
    try:
        h = hashlib.sha256()
        size = path.stat().st_size
        h.update(str(size).encode("ascii"))
        with path.open("rb") as f:
            h.update(f.read(1024 * 1024))
            if size > 1024 * 1024:
                f.seek(max(0, size - 1024 * 1024))
                h.update(f.read(1024 * 1024))
        return h.hexdigest()
    except Exception:
        return ""

def classify_file(path: Path) -> str:
    s = path.suffix.lower()
    parts = [x.lower() for x in path.parts]

    if ".git" in parts:
        return "git_internal"
    if s in VIDEO_EXTS:
        return "video"
    if s in ARCHIVE_EXTS:
        return "archive"
    if s in IMAGE_EXTS:
        return "image"
    if s in ANNOT_EXTS:
        name = path.name.lower()
        if name in {"ball_markup.json", "events_markup.json"}:
            return "opentt_annotation"
        if "segmentation" in rel(path).lower():
            return "segmentation_annotation"
        if name in {"data.yaml", "dataset.yaml"}:
            return "dataset_config"
        return "annotation_or_data"
    if s in DOC_EXTS:
        return "doc"
    if s in {".py", ".ps1", ".sh", ".ipynb"}:
        return "code"
    return "other"

def detect_dataset(path: Path) -> str:
    r = rel(path).lower()

    if "openttgames" in r:
        return "openttgames"
    if "ttnet_pytorch" in r:
        return "ttnet_pytorch"
    if "tt3d" in r:
        return "tt3d"
    if "blurball" in r:
        return "blurball"
    if "sportsvideo" in r:
        return "sportsvideo"
    if "zenodo_t3set" in r or "t3set" in r:
        return "t3set"
    if "kaggle_ball_position" in r:
        return "kaggle_ball_position"
    if "kaggle_ttnet" in r:
        return "kaggle_ttnet"
    if "roboflow" in r:
        return "roboflow_tabletennis"
    if "dtu_table_tennis_data" in r or "table_tennis_data" in r:
        return "extended_openttgames_dtu"
    if "rtmpose" in r:
        return "rtmpose"
    if "motionbert" in r:
        return "motionbert"
    return "unknown"

def opentt_id_from_path(path: Path) -> str:
    r = rel(path).lower()
    for oid in OPENTT_IDS:
        if re.search(rf"(^|[/\\_\-]){re.escape(oid)}($|[/\\_.\-])", r):
            return oid
    return ""

def split_from_opentt_id(oid: str) -> str:
    if oid.startswith("game_"):
        return "training"
    if oid.startswith("test_"):
        return "test"
    return ""

def ffprobe_video(path: Path) -> dict:
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
            return {"ffprobe_ok": False, "ffprobe_error": cp.stderr.strip()[:300]}
        data = json.loads(cp.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return {"ffprobe_ok": False, "ffprobe_error": "no video stream"}
        st = streams[0]
        return {
            "ffprobe_ok": True,
            "width": st.get("width", ""),
            "height": st.get("height", ""),
            "r_frame_rate": st.get("r_frame_rate", ""),
            "avg_frame_rate": st.get("avg_frame_rate", ""),
            "nb_frames": st.get("nb_frames", ""),
            "duration": st.get("duration", ""),
        }
    except FileNotFoundError:
        return {"ffprobe_ok": False, "ffprobe_error": "ffprobe not found"}
    except Exception as e:
        return {"ffprobe_ok": False, "ffprobe_error": repr(e)[:300]}

def inspect_zip(path: Path) -> dict:
    info = {
        "zip_ok": "",
        "zip_file_count": "",
        "zip_top_dirs": "",
        "zip_contains_ball_markup": "",
        "zip_contains_events_markup": "",
        "zip_contains_segmentation_masks": "",
        "zip_first_files": "",
        "bad_download_hint": "",
    }

    head = safe_read_head(path, 128)
    if head.lstrip().startswith(b"<"):
        info["bad_download_hint"] = "looks_like_html"
        info["zip_ok"] = False
        return info

    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            tops = sorted({n.split("/")[0] for n in names if n.strip("/")})
            info["zip_ok"] = True
            info["zip_file_count"] = len(names)
            info["zip_top_dirs"] = "|".join(tops[:20])
            info["zip_contains_ball_markup"] = any(n.endswith("ball_markup.json") for n in names)
            info["zip_contains_events_markup"] = any(n.endswith("events_markup.json") for n in names)
            info["zip_contains_segmentation_masks"] = any("segmentation_masks" in n for n in names)
            info["zip_first_files"] = "|".join(names[:20])
    except Exception as e:
        info["zip_ok"] = False
        info["bad_download_hint"] = repr(e)[:250]

    return info

def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fields = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

def find_annotation_roots(files: list[dict]) -> list[dict]:
    roots = defaultdict(lambda: {
        "path": "",
        "dataset_guess": "",
        "opentt_id": "",
        "has_ball_markup": False,
        "has_events_markup": False,
        "segmentation_mask_files": 0,
        "json_files": 0,
        "csv_files": 0,
        "txt_files": 0,
        "yaml_files": 0,
        "total_files": 0,
        "total_size_bytes": 0,
    })

    for row in files:
        p = Path(row["abs_path"])
        name = p.name.lower()
        r = rel(p).lower()

        root = None

        if name in {"ball_markup.json", "events_markup.json"}:
            root = p.parent
        elif "segmentation_masks" in r:
            # root dataset = parent before segmentation_masks if possible
            parts = list(p.parts)
            try:
                idx = [x.lower() for x in parts].index("segmentation_masks")
                root = Path(*parts[:idx])
            except Exception:
                root = p.parent
        elif name in {"data.yaml", "dataset.yaml"}:
            root = p.parent
        elif p.suffix.lower() in {".json", ".csv", ".txt", ".yaml", ".yml"} and detect_dataset(p) != "unknown":
            oid = opentt_id_from_path(p)
            if oid:
                # nearest parent containing oid
                cur = p.parent
                while cur != cur.parent:
                    if oid in cur.name.lower():
                        root = cur
                        break
                    cur = cur.parent
                if root is None:
                    root = p.parent

        if root is None:
            continue

        key = str(root)
        rec = roots[key]
        rec["path"] = rel(root)
        rec["dataset_guess"] = detect_dataset(root)
        rec["opentt_id"] = opentt_id_from_path(root)
        rec["total_files"] += 1
        rec["total_size_bytes"] += int(row["size_bytes"] or 0)

        if name == "ball_markup.json":
            rec["has_ball_markup"] = True
        elif name == "events_markup.json":
            rec["has_events_markup"] = True
        elif "segmentation_masks" in r:
            rec["segmentation_mask_files"] += 1

        s = p.suffix.lower()
        if s == ".json":
            rec["json_files"] += 1
        elif s == ".csv":
            rec["csv_files"] += 1
        elif s == ".txt":
            rec["txt_files"] += 1
        elif s in {".yaml", ".yml"}:
            rec["yaml_files"] += 1

    out = []
    for rec in roots.values():
        rec["total_size_mb"] = size_mb(rec["total_size_bytes"])
        out.append(dict(rec))
    return sorted(out, key=lambda x: (x.get("opentt_id", ""), x["path"]))

def scan() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    if not DATA.exists():
        raise SystemExit(f"Missing {DATA}")

    files = []
    videos = []
    archives = []
    errors = []

    all_paths = []
    for dirpath, dirnames, filenames in os.walk(DATA):
        # évite de scanner les entrailles .git
        dirnames[:] = [d for d in dirnames if d != ".git"]
        d = Path(dirpath)
        for fn in filenames:
            all_paths.append(d / fn)

    total = len(all_paths)
    print(f"[audit] files found under data_external: {total}")

    for i, p in enumerate(all_paths, 1):
        try:
            st = p.stat()
            cls = classify_file(p)
            dataset = detect_dataset(p)
            oid = opentt_id_from_path(p)
            row = {
                "rel_path": rel(p),
                "abs_path": str(p),
                "name": p.name,
                "suffix": p.suffix.lower(),
                "class": cls,
                "dataset_guess": dataset,
                "opentt_id": oid,
                "opentt_split": split_from_opentt_id(oid),
                "size_bytes": st.st_size,
                "size_mb": size_mb(st.st_size),
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            }

            if cls in {"video", "archive", "opentt_annotation", "segmentation_annotation", "dataset_config"}:
                row["quick_hash"] = quick_hash(p)

            files.append(row)

            if cls == "video":
                v = dict(row)
                v.update(ffprobe_video(p))
                videos.append(v)

            if cls == "archive":
                a = dict(row)
                if p.suffix.lower() == ".zip":
                    a.update(inspect_zip(p))
                archives.append(a)

        except Exception as e:
            errors.append({"path": rel(p), "error": repr(e)})

        if i % 5000 == 0:
            print(f"[audit] scanned {i}/{total}")

    return files, videos, archives, errors

files, videos, archives, errors = scan()
annotations = find_annotation_roots(files)

# Duplicates probables
dup_groups = defaultdict(list)
for row in files:
    if row["class"] == "git_internal":
        continue
    key = (row["name"].lower(), row["size_bytes"])
    dup_groups[key].append(row)

duplicates = []
for (name, sz), group in dup_groups.items():
    if len(group) <= 1:
        continue
    for row in group:
        duplicates.append({
            "dup_key": f"{name}|{sz}",
            "count": len(group),
            "name": row["name"],
            "size_bytes": row["size_bytes"],
            "size_mb": row["size_mb"],
            "class": row["class"],
            "dataset_guess": row["dataset_guess"],
            "rel_path": row["rel_path"],
        })

# Matrix OpenTTGames
by_video_id = defaultdict(list)
for v in videos:
    oid = v.get("opentt_id", "")
    if oid:
        by_video_id[oid].append(v)

by_ann_id = defaultdict(list)
for a in annotations:
    oid = a.get("opentt_id", "")
    if oid:
        by_ann_id[oid].append(a)

by_archive_id = defaultdict(list)
for a in archives:
    oid = a.get("opentt_id", "")
    if oid:
        by_archive_id[oid].append(a)

matrix = []
for oid in OPENTT_IDS:
    raw_videos = [v for v in by_video_id[oid] if "/raw/openttgames/" in v["rel_path"].lower()]
    ttnet_videos = [v for v in by_video_id[oid] if "/code/ttnet_pytorch/dataset/" in v["rel_path"].lower()]
    ann = by_ann_id[oid]
    arc = by_archive_id[oid]

    matrix.append({
        "opentt_id": oid,
        "split": split_from_opentt_id(oid),
        "raw_video_count": len(raw_videos),
        "ttnet_video_count": len(ttnet_videos),
        "archive_count": len(arc),
        "annotation_root_count": len(ann),
        "has_ball_markup_anywhere": any(x.get("has_ball_markup") for x in ann),
        "has_events_markup_anywhere": any(x.get("has_events_markup") for x in ann),
        "segmentation_mask_files_total": sum(int(x.get("segmentation_mask_files") or 0) for x in ann),
        "raw_video_paths": " | ".join(v["rel_path"] for v in raw_videos),
        "ttnet_video_paths": " | ".join(v["rel_path"] for v in ttnet_videos),
        "archive_paths": " | ".join(a["rel_path"] for a in arc),
        "annotation_roots": " | ".join(a["path"] for a in ann),
    })

# Summary
class_counts = Counter(row["class"] for row in files)
dataset_counts = Counter(row["dataset_guess"] for row in files)
dataset_sizes = defaultdict(int)
for row in files:
    dataset_sizes[row["dataset_guess"]] += int(row["size_bytes"] or 0)

top_dirs = defaultdict(int)
top_files = defaultdict(int)
for row in files:
    parts = Path(row["rel_path"]).parts
    key = "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts)
    top_dirs[key] += int(row["size_bytes"] or 0)
    top_files[key] += 1

summary = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "root": str(ROOT),
    "data_external": str(DATA),
    "file_count": len(files),
    "video_count": len(videos),
    "archive_count": len(archives),
    "annotation_root_count": len(annotations),
    "duplicate_rows": len(duplicates),
    "error_count": len(errors),
    "total_size_bytes": sum(int(row["size_bytes"] or 0) for row in files),
    "total_size_gb": round(sum(int(row["size_bytes"] or 0) for row in files) / 1024 / 1024 / 1024, 3),
    "class_counts": dict(class_counts),
    "dataset_counts": dict(dataset_counts),
    "dataset_sizes_gb": {k: round(v / 1024 / 1024 / 1024, 3) for k, v in sorted(dataset_sizes.items())},
    "top_dirs_gb": [
        {
            "dir": k,
            "files": top_files[k],
            "size_gb": round(v / 1024 / 1024 / 1024, 3),
        }
        for k, v in sorted(top_dirs.items(), key=lambda kv: kv[1], reverse=True)[:40]
    ],
}

# CSV outputs
write_csv(OUT / "file_inventory.csv", files)
write_csv(OUT / "video_inventory.csv", videos)
write_csv(OUT / "archive_inventory.csv", archives)
write_csv(OUT / "annotation_roots.csv", annotations)
write_csv(OUT / "duplicates_by_name_size.csv", duplicates)
write_csv(OUT / "openttgames_matrix.csv", matrix)
write_csv(OUT / "scan_errors.csv", errors)

with (OUT / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# Markdown report
missing_ann = [m for m in matrix if not m["has_ball_markup_anywhere"] and not m["has_events_markup_anywhere"] and m["segmentation_mask_files_total"] == 0]
missing_raw_video = [m for m in matrix if m["raw_video_count"] == 0]
missing_ttnet_video = [m for m in matrix if m["ttnet_video_count"] == 0]

md = []
md.append("# TTFlux external data audit 006D")
md.append("")
md.append(f"- Generated: `{summary['created_at']}`")
md.append(f"- Root: `{summary['root']}`")
md.append(f"- Total files: **{summary['file_count']}**")
md.append(f"- Total size: **{summary['total_size_gb']} GB**")
md.append(f"- Videos: **{summary['video_count']}**")
md.append(f"- Archives: **{summary['archive_count']}**")
md.append(f"- Annotation roots: **{summary['annotation_root_count']}**")
md.append(f"- Duplicate rows by name+size: **{summary['duplicate_rows']}**")
md.append("")

md.append("## Immediate findings")
md.append("")
if missing_raw_video:
    md.append("### OpenTTGames raw videos missing")
    for m in missing_raw_video:
        md.append(f"- `{m['opentt_id']}`")
    md.append("")
else:
    md.append("- OpenTTGames raw MP4: **all expected IDs have at least one raw video candidate**.")
    md.append("")

if missing_ttnet_video:
    md.append("### OpenTTGames TTNet-layout videos missing")
    for m in missing_ttnet_video:
        md.append(f"- `{m['opentt_id']}`")
    md.append("")
else:
    md.append("- TTNet-layout MP4: **all expected IDs have at least one TTNet video candidate**.")
    md.append("")

if missing_ann:
    md.append("### OpenTTGames annotation roots not linked/found")
    md.append("These IDs have no detected `ball_markup.json`, `events_markup.json`, or `segmentation_masks` root.")
    for m in missing_ann:
        md.append(f"- `{m['opentt_id']}`")
    md.append("")
else:
    md.append("- OpenTTGames annotations: **all expected IDs have at least one annotation candidate**.")
    md.append("")

md.append("## Dataset size summary")
md.append("")
for k, gb in summary["dataset_sizes_gb"].items():
    md.append(f"- `{k}`: {gb} GB")
md.append("")

md.append("## Largest folders")
md.append("")
for row in summary["top_dirs_gb"]:
    md.append(f"- `{row['dir']}` — {row['size_gb']} GB — {row['files']} files")
md.append("")

md.append("## Files generated")
md.append("")
for name in [
    "summary.json",
    "file_inventory.csv",
    "video_inventory.csv",
    "archive_inventory.csv",
    "annotation_roots.csv",
    "duplicates_by_name_size.csv",
    "openttgames_matrix.csv",
    "scan_errors.csv",
]:
    md.append(f"- `{name}`")
md.append("")

md.append("## Proposed next step, not executed")
md.append("")
md.append("Use `openttgames_matrix.csv` and `duplicates_by_name_size.csv` to design a clean layout with:")
md.append("")
md.append("```text")
md.append("data_external/")
md.append("  sources/")
md.append("    openttgames/")
md.append("      raw_videos/")
md.append("      raw_archives/")
md.append("      annotations/")
md.append("    blurball/")
md.append("    tt3d/")
md.append("    t3set/")
md.append("  derived/")
md.append("    ttnet_layout/")
md.append("    yolo_exports/")
md.append("    scene_state_exports/")
md.append("  registry/")
md.append("    dataset_index.csv")
md.append("    video_index.csv")
md.append("    annotation_index.csv")
md.append("    link_index.csv")
md.append("```")
md.append("")
md.append("No file was moved by this audit.")

(OUT / "audit_summary.md").write_text("\n".join(md), encoding="utf-8")

print("")
print("=== AUDIT 006D DONE ===")
print(f"Report dir: {OUT}")
print(f"Summary:    {OUT / 'audit_summary.md'}")
print(f"Matrix:     {OUT / 'openttgames_matrix.csv'}")
print(f"Videos:     {OUT / 'video_inventory.csv'}")
print(f"Ann roots:  {OUT / 'annotation_roots.csv'}")
print(f"Duplicates: {OUT / 'duplicates_by_name_size.csv'}")
print("")
print("Quick summary:")
print(json.dumps({
    "file_count": summary["file_count"],
    "total_size_gb": summary["total_size_gb"],
    "video_count": summary["video_count"],
    "archive_count": summary["archive_count"],
    "annotation_root_count": summary["annotation_root_count"],
    "missing_opentt_annotation_ids": [m["opentt_id"] for m in missing_ann],
}, ensure_ascii=False, indent=2))
