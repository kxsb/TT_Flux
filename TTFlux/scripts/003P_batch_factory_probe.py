from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path.cwd()
OUT = ROOT / "runs" / "batch_factory_probe_003P.json"

SCRIPT_PATHS = [
    ROOT / "scripts" / "make_batch_configs_001E.py",
    ROOT / "scripts" / "run_batch_001E.py",
    ROOT / "scripts" / "run_batch_with_arbiter_001T.py",
    ROOT / "scripts" / "run_batch_with_arbiter_001T2.py",
    ROOT / "scripts" / "inventory_clips_001E.py",
]

JSON_PATHS = [
    ROOT / "configs" / "batch_001E" / "batch_001E.json",
    ROOT / "runs" / "inventory_001E" / "clip_inventory_001E.json",
    ROOT / "videos_dataset2_v60" / "dataset2_manifest_v60.json",
    ROOT / "runs" / "batch_001E" / "batch_summary_001E.json",
]


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def load_json_preview(path: Path):
    if not path.is_file():
        return {"exists": False}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"exists": True, "error": str(e)}

    info = {
        "exists": True,
        "type": type(data).__name__,
    }

    if isinstance(data, dict):
        info["keys"] = list(data.keys())[:80]

        for k, v in data.items():
            if isinstance(v, list):
                info[f"{k}_type"] = "list"
                info[f"{k}_len"] = len(v)
                if v:
                    info[f"{k}_first_type"] = type(v[0]).__name__
                    if isinstance(v[0], dict):
                        info[f"{k}_first_keys"] = list(v[0].keys())[:80]
                        info[f"{k}_first"] = v[0]
                    else:
                        info[f"{k}_first"] = v[0]
                break

        # garde un preview petit
        small = {}
        for k, v in data.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                small[k] = v
            elif isinstance(v, list):
                small[k] = f"list[{len(v)}]"
            elif isinstance(v, dict):
                small[k] = f"dict[{len(v)}]"
            else:
                small[k] = type(v).__name__
        info["scalar_preview"] = small

    elif isinstance(data, list):
        info["len"] = len(data)
        if data:
            info["first_type"] = type(data[0]).__name__
            info["first"] = data[0]
            if isinstance(data[0], dict):
                info["first_keys"] = list(data[0].keys())[:80]

    return info


def inspect_script(path: Path):
    if not path.is_file():
        return {"exists": False}

    txt = path.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    interesting = []
    patterns = [
        "argparse",
        "add_argument",
        "configs",
        "batch_001E",
        "inventory",
        "subprocess",
        "python",
        "cli.py",
        "--config",
        "--run",
        "validation",
        "Path(",
        "json",
    ]

    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(p.lower() in low for p in patterns):
            interesting.append({
                "line": i,
                "text": line[:240],
            })

    return {
        "exists": True,
        "line_count": len(lines),
        "argparse_args": re.findall(r'add_argument\((.*?)\)', txt, flags=re.S),
        "interesting_lines": interesting[:220],
        "head": "\n".join(lines[:80]),
    }


def main():
    result = {
        "version": "003P",
        "root": str(ROOT),
        "scripts": {rel(p): inspect_script(p) for p in SCRIPT_PATHS},
        "jsons": {rel(p): load_json_preview(p) for p in JSON_PATHS},
        "batch_001E_config_files": [],
    }

    cfg_dir = ROOT / "configs" / "batch_001E"
    if cfg_dir.is_dir():
        for p in sorted(cfg_dir.glob("*.json")):
            result["batch_001E_config_files"].append({
                "path": rel(p),
                "preview": load_json_preview(p),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("003P wrote", OUT)

    print("\n=== scripts ===")
    for p, info in result["scripts"].items():
        print(p, "exists=", info.get("exists"), "lines=", info.get("line_count"))
        for item in info.get("interesting_lines", [])[:20]:
            print(f"  L{item['line']}: {item['text']}")

    print("\n=== json previews ===")
    for p, info in result["jsons"].items():
        print(p, "exists=", info.get("exists"), "type=", info.get("type"), "keys=", info.get("keys", info.get("first_keys", "-")))

    print("\n=== batch config files ===")
    for item in result["batch_001E_config_files"]:
        print(item["path"], "keys=", item["preview"].get("keys", "-"))

    print("\nOpen:")
    print("notepad runs\\batch_factory_probe_003P.json")


if __name__ == "__main__":
    main()
