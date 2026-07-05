from __future__ import annotations

import csv
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

ROOT = Path.cwd()
DATA = ROOT / "data_external"
OUT = ROOT / "runs" / "external_data_active_audit_006H"
OUT.mkdir(parents=True, exist_ok=True)

SKIP_PARTS = {
    "_quarantine",
}

SKIP_EXACT = {
    (DATA / "code" / "ttnet_pytorch" / "dataset").resolve(),
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")

def gb(n: int) -> float:
    return round(n / 1024 / 1024 / 1024, 3)

def is_junction_like(p: Path) -> bool:
    if not p.exists():
        return False
    if hasattr(p, "is_junction"):
        try:
            return p.is_junction()
        except Exception:
            return False
    return False

rows = []
dir_sizes = defaultdict(int)
dir_files = defaultdict(int)
video_rows = []
annotation_roots = []

for dirpath, dirnames, filenames in os.walk(DATA):
    d = Path(dirpath)

    # Ne pas descendre dans la quarantaine.
    dirnames[:] = [x for x in dirnames if x not in SKIP_PARTS]

    # Ne pas descendre dans la jonction TTNet dataset.
    filtered = []
    for x in dirnames:
        child = d / x
        try:
            child_resolved = child.resolve()
        except Exception:
            child_resolved = child

        if child_resolved in SKIP_EXACT or is_junction_like(child):
            continue
        filtered.append(x)

    dirnames[:] = filtered

    for fn in filenames:
        p = d / fn
        try:
            st = p.stat()
        except Exception:
            continue

        suffix = p.suffix.lower()
        r = rel(p)

        top = "/".join(Path(r).parts[:3])
        dir_sizes[top] += st.st_size
        dir_files[top] += 1

        row = {
            "path": r,
            "name": p.name,
            "suffix": suffix,
            "size_bytes": st.st_size,
            "size_gb": gb(st.st_size),
        }
        rows.append(row)

        if suffix in VIDEO_EXTS:
            video_rows.append(row)

        if p.name in {"ball_markup.json", "events_markup.json"}:
            annotation_roots.append({
                "root": rel(p.parent),
                "file": p.name,
                "size_bytes": st.st_size,
            })

def write_csv(path: Path, rows: list[dict]):
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

write_csv(OUT / "active_file_inventory.csv", rows)
write_csv(OUT / "active_video_inventory.csv", video_rows)
write_csv(OUT / "active_annotation_roots.csv", annotation_roots)

summary_dirs = [
    {
        "dir": k,
        "files": dir_files[k],
        "size_gb": gb(v),
    }
    for k, v in sorted(dir_sizes.items(), key=lambda kv: kv[1], reverse=True)
]

write_csv(OUT / "active_folder_sizes.csv", summary_dirs)

total = sum(r["size_bytes"] for r in rows)

md = []
md.append("# TTFlux active data audit 006H")
md.append("")
md.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
md.append("- Excluded: `data_external/_quarantine`")
md.append("- Excluded: TTNet `dataset` junction traversal")
md.append(f"- Active logical files: **{len(rows)}**")
md.append(f"- Active logical size: **{gb(total)} GB**")
md.append(f"- Active videos: **{len(video_rows)}**")
md.append(f"- Annotation marker files: **{len(annotation_roots)}**")
md.append("")
md.append("## Largest active folders")
md.append("")
for d in summary_dirs[:30]:
    md.append(f"- `{d['dir']}` — {d['size_gb']} GB — {d['files']} files")
md.append("")
md.append("## Generated files")
md.append("")
md.append("- `active_file_inventory.csv`")
md.append("- `active_video_inventory.csv`")
md.append("- `active_annotation_roots.csv`")
md.append("- `active_folder_sizes.csv`")

(OUT / "active_audit_summary.md").write_text("\n".join(md), encoding="utf-8")

print("")
print("=== ACTIVE AUDIT 006H DONE ===")
print("Summary:", OUT / "active_audit_summary.md")
print("Active logical size:", gb(total), "GB")
print("Files:", len(rows))
print("Videos:", len(video_rows))
