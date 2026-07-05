from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data_external"
Q = DATA / "_quarantine" / "006G_cleanup"
OUT = ROOT / "runs" / "external_data_delete_quarantine_006K"
APPLY = "--apply" in sys.argv

OUT.mkdir(parents=True, exist_ok=True)

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")

def dir_size(path: Path):
    total = 0
    count = 0
    if not path.exists():
        return 0, 0
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

def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

errors = []

required = [
    DATA / "sources" / "openttgames" / "raw_videos" / "game_1.mp4",
    DATA / "sources" / "openttgames" / "annotations" / "game_1" / "ball_markup.json",
    DATA / "sources" / "openttgames" / "annotations" / "test_1" / "ball_markup.json",
    DATA / "derived" / "ttnet_layout" / "dataset" / "training" / "videos" / "game_1.mp4",
    DATA / "code" / "ttnet_pytorch" / "dataset" / "training" / "videos" / "game_1.mp4",
    DATA / "registry" / "link_index.csv",
    DATA / "registry" / "ttnet_layout_index.csv",
    ROOT / "runs" / "external_data_integrity_006I" / "integrity_problems.csv",
    ROOT / "runs" / "ttnet_dataset_smoke_006J" / "game_1" / "summary.json",
    ROOT / "runs" / "ttnet_dataset_smoke_006J" / "game_1" / "contact_sheet.jpg",
]

for p in required:
    if not p.exists():
        errors.append(f"missing required file/path: {rel(p)}")

problem_rows = read_csv_rows(ROOT / "runs" / "external_data_integrity_006I" / "integrity_problems.csv")
if problem_rows:
    errors.append(f"integrity_problems.csv is not empty: {len(problem_rows)} rows")

smoke_json = ROOT / "runs" / "ttnet_dataset_smoke_006J" / "game_1" / "summary.json"
if smoke_json.exists():
    data = json.loads(smoke_json.read_text(encoding="utf-8"))
    if data.get("sampled_read_ok") != data.get("sampled_frames"):
        errors.append("smoke test did not read all sampled frames")
    if int(data.get("ball_points_collected", 0)) <= 0:
        errors.append("smoke test collected zero ball points")
    if int(data.get("mask_files", 0)) <= 0:
        errors.append("smoke test found zero masks")

if not Q.exists():
    errors.append(f"quarantine not found: {rel(Q)}")

q_size, q_files = dir_size(Q)

md = []
md.append("# TTFlux delete quarantine 006K")
md.append("")
md.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
md.append(f"- Mode: `{'APPLY' if APPLY else 'DRY_RUN'}`")
md.append(f"- Quarantine: `{rel(Q)}`")
md.append(f"- Quarantine files: **{q_files}**")
md.append(f"- Quarantine logical size: **{gb(q_size)} GB**")
md.append(f"- Validation errors: **{len(errors)}**")
md.append("")

if errors:
    md.append("## Errors")
    md.append("")
    for e in errors:
        md.append(f"- {e}")
    md.append("")
    md.append("Deletion aborted.")
    (OUT / "delete_quarantine_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("ABORT: validation errors")
    for e in errors:
        print(" -", e)
    raise SystemExit(2)

if APPLY:
    shutil.rmtree(Q)
    action = "deleted"
else:
    action = "dry_run_only"

md.append("## Action")
md.append("")
md.append(f"- `{action}`")
md.append("")
md.append("## Note")
md.append("")
md.append("This operation is permanent when run with `--apply`.")

(OUT / "delete_quarantine_summary.md").write_text("\n".join(md), encoding="utf-8")

print("")
print("=== DELETE QUARANTINE 006K ===")
print("Mode:", "APPLY" if APPLY else "DRY_RUN")
print("Quarantine:", Q)
print("Files:", q_files)
print("Logical size:", gb(q_size), "GB")
print("Action:", action)
print("Summary:", OUT / "delete_quarantine_summary.md")
print("")
if not APPLY:
    print('To apply: python "scripts\\delete_quarantine_006K.py" --apply')
