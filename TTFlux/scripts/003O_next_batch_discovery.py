from pathlib import Path
import json
import os

ROOT = Path.cwd()
OUT = ROOT / "runs" / "next_batch_discovery_003O.json"

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)

def file_info(p: Path):
    return {
        "path": rel(p),
        "size": p.stat().st_size,
    }

scripts = []
for p in sorted((ROOT / "scripts").glob("*.py")):
    txt = ""
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass

    scripts.append({
        "path": rel(p),
        "mentions_batch": "batch" in txt.lower(),
        "mentions_001E": "001E" in txt,
        "mentions_config": "config" in txt.lower(),
        "mentions_run": "--run" in txt or "argparse" in txt,
    })

csvs = []
for p in sorted((ROOT / "runs").rglob("*.csv")):
    csvs.append(file_info(p))

jsons = []
for p in sorted((ROOT / "runs").rglob("*.json")):
    jsons.append(file_info(p))

videos = []
for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
    for p in sorted((ROOT / "runs").rglob(ext)):
        videos.append(file_info(p))

root_configs = []
for ext in ("*.yaml", "*.yml", "*.json", "*.toml", "*.ini"):
    for p in sorted(ROOT.rglob(ext)):
        if ".git" in p.parts or "__pycache__" in p.parts:
            continue
        if "runs" in p.parts and len(root_configs) > 200:
            continue
        root_configs.append(file_info(p))

interesting = {
    "scripts_batch_like": [s for s in scripts if s["mentions_batch"] or s["mentions_001E"] or s["mentions_config"]],
    "all_scripts_count": len(scripts),
    "csv_count": len(csvs),
    "json_count": len(jsons),
    "video_count": len(videos),
    "candidate_csvs": csvs[:120],
    "candidate_jsons": jsons[:120],
    "candidate_videos": videos[:80],
    "candidate_configs": root_configs[:120],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(interesting, ensure_ascii=False, indent=2), encoding="utf-8")

print("003O discovery wrote", OUT)
print("scripts_batch_like:")
for s in interesting["scripts_batch_like"]:
    print(" -", s["path"])
print("csv_count=", len(csvs), "video_count=", len(videos), "json_count=", len(jsons))
