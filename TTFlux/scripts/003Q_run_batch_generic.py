from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-file", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--summary-name", default="")
    args = ap.parse_args()

    root = Path.cwd()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from ttflux.cli import run_validation

    batch_file = Path(args.batch_file)
    if not batch_file.is_absolute():
        batch_file = root / batch_file

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    if not batch_file.is_file():
        raise SystemExit(f"Batch file introuvable: {batch_file}")

    run_dir.mkdir(parents=True, exist_ok=True)

    batch = json.loads(batch_file.read_text(encoding="utf-8-sig"))
    configs = batch.get("configs", [])

    results = []

    for idx, cfg_raw in enumerate(configs, start=1):
        cfg_path = Path(cfg_raw)
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path

        name = cfg_path.stem
        out_dir = run_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        print("")
        print("====================================")
        print(f"TTFlux generic batch item {idx}/{len(configs)}")
        print(name)
        print("====================================")

        item = {
            "idx": idx,
            "config": str(cfg_path),
            "out_dir": str(out_dir),
            "ok": False,
        }

        try:
            summary = run_validation(cfg_path, out_dir)

            item["ok"] = True
            item["selected_count"] = summary.get("selected_count")
            item["candidate_count"] = summary.get("candidate_count")
            item["summary"] = summary

            print("OK")
            print("selected_count=", item["selected_count"])
            print("candidate_count=", item["candidate_count"])

        except Exception as exc:
            item["error"] = repr(exc)
            item["traceback"] = traceback.format_exc()
            print("ERROR")
            print(repr(exc))

        results.append(item)

    batch_id = batch.get("batch_id", run_dir.name)
    summary_name = args.summary_name or f"batch_summary_{batch_id}.json"

    final = {
        "version": "003Q_RUNNER",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch_file": str(batch_file),
        "run_root": str(run_dir),
        "count": len(results),
        "ok_count": sum(1 for r in results if r["ok"]),
        "error_count": sum(1 for r in results if not r["ok"]),
        "results": results,
    }

    summary_path = run_dir / summary_name
    summary_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print("003Q_RUN_BATCH_DONE")
    print("summary=", summary_path)
    print("count=", final["count"])
    print("ok_count=", final["ok_count"])
    print("error_count=", final["error_count"])

    if final["error_count"]:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
