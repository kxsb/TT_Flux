from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data_external"
REG = DATA / "registry"
DERIVED_TTNET = DATA / "derived" / "ttnet_layout" / "dataset"
TTNET_REPO_DATASET = DATA / "code" / "ttnet_pytorch" / "dataset"
QUARANTINE = DATA / "_quarantine" / "006G_cleanup"
REPORT = ROOT / "runs" / "external_data_cleanup_006G"

APPLY = "--apply" in sys.argv

REPORT.mkdir(parents=True, exist_ok=True)

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
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def dir_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1

    total = 0
    count = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
                count += 1
            except Exception:
                pass
    return total, count

def gb(n: int) -> float:
    return round(n / 1024 / 1024 / 1024, 3)

def validate_registry() -> tuple[bool, list[str]]:
    errors = []

    link_rows = read_csv(REG / "link_index.csv")
    ttnet_rows = read_csv(REG / "ttnet_layout_index.csv")

    if len(link_rows) != 12:
        errors.append(f"link_index.csv expected 12 rows, got {len(link_rows)}")

    if len(ttnet_rows) != 12:
        errors.append(f"ttnet_layout_index.csv expected 12 rows, got {len(ttnet_rows)}")

    for r in link_rows:
        sid = r.get("sample_id", "")
        video = ROOT / r.get("video_path", "")
        ann = ROOT / r.get("annotation_path", "")
        if not video.exists():
            errors.append(f"{sid}: missing source video {video}")
        if not ann.exists():
            errors.append(f"{sid}: missing source annotation {ann}")
        if not (ann / "ball_markup.json").exists():
            errors.append(f"{sid}: missing ball_markup.json")
        if not (ann / "events_markup.json").exists():
            errors.append(f"{sid}: missing events_markup.json")

    for r in ttnet_rows:
        sid = r.get("sample_id", "")
        video = ROOT / r.get("video_path", "")
        ann = ROOT / r.get("annotation_path", "")
        if not video.exists():
            errors.append(f"{sid}: missing derived ttnet video {video}")
        if not ann.exists():
            errors.append(f"{sid}: missing derived ttnet annotation {ann}")
        if not (ann / "ball_markup.json").exists():
            errors.append(f"{sid}: derived missing ball_markup.json")
        if not (ann / "events_markup.json").exists():
            errors.append(f"{sid}: derived missing events_markup.json")

    if not DERIVED_TTNET.exists():
        errors.append(f"missing derived TTNet dataset: {DERIVED_TTNET}")

    return len(errors) == 0, errors

def unique_quarantine_path(src: Path) -> Path:
    dst = QUARANTINE / rel(src)
    if not dst.exists():
        return dst

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dst.with_name(dst.name + "_" + stamp)

def move_to_quarantine(src: Path) -> dict:
    dst = unique_quarantine_path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return {
        "action": "moved_to_quarantine",
        "src": rel(src),
        "dst": rel(dst),
    }

def create_ttnet_junction() -> dict:
    if TTNET_REPO_DATASET.exists():
        return {
            "action": "junction_skipped",
            "reason": "dataset path already exists",
            "path": rel(TTNET_REPO_DATASET),
        }

    TTNET_REPO_DATASET.parent.mkdir(parents=True, exist_ok=True)

    if os.name != "nt":
        try:
            os.symlink(DERIVED_TTNET, TTNET_REPO_DATASET, target_is_directory=True)
            return {
                "action": "symlink_created",
                "link": rel(TTNET_REPO_DATASET),
                "target": rel(DERIVED_TTNET),
            }
        except Exception as e:
            return {
                "action": "symlink_failed",
                "error": repr(e),
                "link": rel(TTNET_REPO_DATASET),
                "target": rel(DERIVED_TTNET),
            }

    cmd = [
        "cmd",
        "/c",
        "mklink",
        "/J",
        str(TTNET_REPO_DATASET),
        str(DERIVED_TTNET),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode == 0:
        return {
            "action": "junction_created",
            "link": rel(TTNET_REPO_DATASET),
            "target": rel(DERIVED_TTNET),
            "stdout": cp.stdout.strip(),
        }
    return {
        "action": "junction_failed",
        "link": rel(TTNET_REPO_DATASET),
        "target": rel(DERIVED_TTNET),
        "stderr": cp.stderr.strip(),
        "stdout": cp.stdout.strip(),
    }

ok, validation_errors = validate_registry()

if not ok:
    write_csv(REPORT / "validation_errors.csv", [{"error": e} for e in validation_errors])
    print("ABORT: registry/layout validation failed")
    for e in validation_errors:
        print(" -", e)
    raise SystemExit(2)

candidates = [
    {
        "path": DATA / "code" / "ttnet_pytorch" / "dataset",
        "reason": "duplicate_old_ttnet_dataset_replaced_by_data_external_derived_ttnet_layout",
        "priority": "high",
        "after_move": "create_junction_to_derived_layout",
    },
    {
        "path": DATA / "raw" / "openttgames",
        "reason": "superseded_by_clean_data_external_sources_openttgames",
        "priority": "high",
        "after_move": "none",
    },
    {
        "path": DATA / "extracted" / "openttgames",
        "reason": "old_broken_extraction_merged_annotations_without_sample_dirs",
        "priority": "high",
        "after_move": "none",
    },
    {
        "path": DATA / "index",
        "reason": "superseded_by_data_external_registry",
        "priority": "low",
        "after_move": "none",
    },
    {
        "path": DATA / "_download_log.txt",
        "reason": "old_download_log",
        "priority": "low",
        "after_move": "none",
    },
    {
        "path": DATA / "_openttgames_ttnet_sync_006C.log",
        "reason": "old_sync_log",
        "priority": "low",
        "after_move": "none",
    },
]

candidate_rows = []
actions = []

for c in candidates:
    p = c["path"]
    size, count = dir_size(p)
    exists = p.exists()

    row = {
        "path": rel(p),
        "exists": exists,
        "type": "dir" if exists and p.is_dir() else "file" if exists else "missing",
        "file_count": count,
        "size_bytes": size,
        "size_gb": gb(size),
        "reason": c["reason"],
        "priority": c["priority"],
        "after_move": c["after_move"],
        "planned_action": "move_to_quarantine" if exists else "skip_missing",
    }
    candidate_rows.append(row)

write_csv(REPORT / "cleanup_candidates.csv", candidate_rows)

if APPLY:
    QUARANTINE.mkdir(parents=True, exist_ok=True)

    for row in candidate_rows:
        src = ROOT / row["path"]
        if not src.exists():
            actions.append({
                "action": "skip_missing",
                "src": row["path"],
            })
            continue

        actions.append(move_to_quarantine(src))

    actions.append(create_ttnet_junction())

write_csv(REPORT / "cleanup_actions.csv", actions)

active_size_after = 0
active_files_after = 0
if DATA.exists():
    active_size_after, active_files_after = dir_size(DATA)

md = []
md.append("# TTFlux external data cleanup 006G")
md.append("")
md.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
md.append(f"- Mode: `{'APPLY' if APPLY else 'DRY_RUN'}`")
md.append(f"- Validation: `OK`")
md.append(f"- Quarantine: `{rel(QUARANTINE)}`")
md.append("")
md.append("## Candidates")
md.append("")
for r in candidate_rows:
    md.append(
        f"- `{r['path']}` — {r['size_gb']} GB — {r['file_count']} files — "
        f"{r['planned_action']} — {r['reason']}"
    )
md.append("")
md.append("## Actions")
md.append("")
if actions:
    for a in actions:
        md.append(f"- `{a.get('action')}` — {a}")
else:
    md.append("- No action executed. Dry-run only.")
md.append("")
md.append("## Post-state")
md.append("")
md.append(f"- Active `data_external` size after current run: **{gb(active_size_after)} GB**")
md.append(f"- Active `data_external` files after current run: **{active_files_after}**")
md.append("")
md.append("## Safety")
md.append("")
md.append("No permanent deletion is performed by this script.")
md.append("To recover, move files back from `data_external/_quarantine/006G_cleanup/`.")
md.append("Permanent deletion should only happen after a fresh audit confirms the dataset still works.")

(REPORT / "cleanup_summary.md").write_text("\n".join(md), encoding="utf-8")

print("")
print("=== CLEANUP 006G ===")
print("Mode:", "APPLY" if APPLY else "DRY_RUN")
print("Summary:", REPORT / "cleanup_summary.md")
print("Candidates:", REPORT / "cleanup_candidates.csv")
print("Actions:", REPORT / "cleanup_actions.csv")
print("")
for r in candidate_rows:
    print(f"{r['planned_action']:18} {r['size_gb']:8} GB  {r['path']}")
print("")
if APPLY:
    print("Applied. Re-run audit 006D after this.")
else:
    print("Dry-run only. To apply: python scripts\\cleanup_external_data_006G.py --apply")
