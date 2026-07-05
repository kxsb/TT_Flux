from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
REG = ROOT / "data_external" / "registry"
OUT = ROOT / "runs" / "external_data_integrity_006I"
OUT.mkdir(parents=True, exist_ok=True)

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
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json",
        str(path),
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if cp.returncode != 0:
            return {"video_ok": False, "video_error": cp.stderr[:300]}
        data = json.loads(cp.stdout or "{}")
        st = (data.get("streams") or [{}])[0]
        return {
            "video_ok": True,
            "width": st.get("width", ""),
            "height": st.get("height", ""),
            "avg_frame_rate": st.get("avg_frame_rate", ""),
            "nb_frames": st.get("nb_frames", ""),
            "duration": st.get("duration", ""),
        }
    except Exception as e:
        return {"video_ok": False, "video_error": repr(e)[:300]}

def load_json(path: Path) -> tuple[bool, int, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return True, len(data), "list"
        if isinstance(data, dict):
            return True, len(data), "dict"
        return True, 1, type(data).__name__
    except Exception as e:
        return False, 0, repr(e)[:300]

rows = []
problems = []

for r in read_csv(REG / "link_index.csv"):
    sid = r["sample_id"]
    video = ROOT / r["video_path"]
    ann = ROOT / r["annotation_path"]

    ball = ann / "ball_markup.json"
    events = ann / "events_markup.json"
    masks = ann / "segmentation_masks"

    row = {
        "sample_id": sid,
        "split": r["split"],
        "video_path": rel(video),
        "annotation_path": rel(ann),
        "video_exists": video.exists(),
        "annotation_exists": ann.exists(),
        "ball_exists": ball.exists(),
        "events_exists": events.exists(),
        "masks_exists": masks.exists(),
        "mask_files": sum(1 for p in masks.rglob("*") if p.is_file()) if masks.exists() else 0,
    }

    if video.exists():
        row.update(ffprobe(video))
    else:
        row["video_ok"] = False
        row["video_error"] = "missing"

    if ball.exists():
        ok, n, typ = load_json(ball)
        row["ball_json_ok"] = ok
        row["ball_json_count"] = n
        row["ball_json_type"] = typ
    else:
        row["ball_json_ok"] = False

    if events.exists():
        ok, n, typ = load_json(events)
        row["events_json_ok"] = ok
        row["events_json_count"] = n
        row["events_json_type"] = typ
    else:
        row["events_json_ok"] = False

    bad = []
    for k in ["video_exists", "annotation_exists", "ball_exists", "events_exists", "masks_exists", "video_ok", "ball_json_ok", "events_json_ok"]:
        if not row.get(k):
            bad.append(k)
    if row["mask_files"] <= 0:
        bad.append("mask_files_zero")

    row["status"] = "ok" if not bad else "problem"
    row["problem_fields"] = "|".join(bad)

    if bad:
        problems.append(row)

    rows.append(row)

write_csv(OUT / "integrity_samples.csv", rows)
write_csv(OUT / "integrity_problems.csv", problems)

ok_count = sum(1 for r in rows if r["status"] == "ok")

md = []
md.append("# TTFlux external data integrity 006I")
md.append("")
md.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
md.append(f"- Samples checked: **{len(rows)}**")
md.append(f"- OK: **{ok_count}**")
md.append(f"- Problems: **{len(problems)}**")
md.append("")
md.append("## Samples")
md.append("")
for r in rows:
    md.append(
        f"- `{r['sample_id']}` — {r['status']} — "
        f"{r.get('width')}x{r.get('height')} — frames={r.get('nb_frames')} — "
        f"ball={r.get('ball_json_count')} — events={r.get('events_json_count')} — masks={r.get('mask_files')}"
    )
md.append("")
md.append("## Files")
md.append("")
md.append("- `integrity_samples.csv`")
md.append("- `integrity_problems.csv`")

(OUT / "integrity_summary.md").write_text("\n".join(md), encoding="utf-8")

print("")
print("=== INTEGRITY 006I DONE ===")
print("Summary:", OUT / "integrity_summary.md")
print("OK:", ok_count, "/", len(rows))
print("Problems:", len(problems))
