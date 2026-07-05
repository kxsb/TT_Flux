from __future__ import annotations

import json
import traceback
from pathlib import Path

from ttflux.cli import run_validation


def main():
    root = Path.home() / "Desktop" / "développement"
    root = root / "ping" / "TTFlux"

    batch_file = root / "configs" / "batch_001E"
    batch_file = batch_file / "batch_001E.json"

    run_root = root / "runs" / "batch_001E"
    run_root.mkdir(parents=True, exist_ok=True)

    batch = json.loads(
        batch_file.read_text(encoding="utf-8-sig")
    )

    results = []

    for idx, cfg in enumerate(batch["configs"], start=1):
        cfg_path = Path(cfg)
        name = cfg_path.stem

        out_dir = run_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        print("")
        print("====================================")
        print(f"TTFlux batch item {idx}")
        print(name)
        print("====================================")

        item = {
            "idx": idx,
            "config": str(cfg_path),
            "out_dir": str(out_dir),
            "ok": False,
        }

        try:
            summary = run_validation(
                cfg_path,
                out_dir,
            )

            item["ok"] = True
            item["selected_count"] = summary.get(
                "selected_count"
            )
            item["candidate_count"] = summary.get(
                "candidate_count"
            )
            print("OK")

        except Exception as exc:
            item["error"] = repr(exc)
            item["traceback"] = traceback.format_exc()
            print("ERROR")
            print(repr(exc))

        results.append(item)

    final = {
        "batch_file": str(batch_file),
        "run_root": str(run_root),
        "count": len(results),
        "ok_count": sum(1 for r in results if r["ok"]),
        "error_count": sum(1 for r in results if not r["ok"]),
        "results": results,
    }

    summary_path = run_root / "batch_summary_001E.json"

    summary_path.write_text(
        json.dumps(final, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("")
    print("TTFLUX_BATCH_001E_DONE")
    print("ok_count =", final["ok_count"])
    print("error_count =", final["error_count"])
    print("summary =", summary_path)


if __name__ == "__main__":
    main()
