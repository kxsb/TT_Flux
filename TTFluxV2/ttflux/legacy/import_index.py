from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from ttflux.core.paths import project_root, legacy_root
from ttflux.core.file_inspect import inspect_artifact


def load_legacy_seed_index() -> dict[str, Any]:
    path = project_root() / "legacy_imports" / "legacy_artifacts_index.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def classify_artifact(group: str, rel_path: str) -> str:
    p = rel_path.lower()
    if "table" in p or "scene_table" in p or "camera" in p:
        return "table_camera_scene"
    if "ball" in p or "ranker" in p or "candidate" in p or "track" in p or "relink" in p:
        return "ball_tracking_reference"
    if group:
        return group
    return "unknown"


def build_legacy_artifact_report(output_dir: Path | None = None) -> dict[str, Any]:
    root = project_root()
    legacy = legacy_root()
    seed = load_legacy_seed_index()

    if output_dir is None:
        output_dir = root / "runs" / "007B_legacy_artifacts_index"
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        "validated_table_chain",
        "partial_ball_chain_reference",
    ]

    records: list[dict[str, Any]] = []
    for group in groups:
        for order_idx, rel in enumerate(seed.get(group, [])):
            abs_path = legacy / rel
            info = inspect_artifact(abs_path)
            record = {
                "group": group,
                "order_idx": order_idx,
                "artifact_role": classify_artifact(group, rel),
                "relative_path": rel,
                **info,
            }
            records.append(record)

    summary = {
        "patch": "007B_TTFluxV2_import_legacy_artifacts_index",
        "created_at": time.time(),
        "project_root": str(root),
        "legacy_root": str(legacy),
        "legacy_exists": legacy.exists(),
        "records_total": len(records),
        "records_exists": sum(1 for r in records if r.get("exists")),
        "records_missing": sum(1 for r in records if not r.get("exists")),
        "table_records": sum(1 for r in records if r.get("group") == "validated_table_chain"),
        "ball_records": sum(1 for r in records if r.get("group") == "partial_ball_chain_reference"),
        "seed_key_diagnostic": seed.get("key_diagnostic", {}),
        "records": records,
    }

    json_path = output_dir / "legacy_artifacts_detailed_007B.json"
    csv_path = output_dir / "legacy_artifacts_detailed_007B.csv"
    summary_path = output_dir / "legacy_artifacts_summary_007B.json"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    compact_summary = {k: v for k, v in summary.items() if k != "records"}
    summary_path.write_text(json.dumps(compact_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "group",
        "order_idx",
        "artifact_role",
        "relative_path",
        "absolute_path",
        "exists",
        "kind",
        "suffix",
        "size_mb",
        "row_count_estimate",
        "columns",
        "json_type",
        "json_keys",
        "json_len",
        "error",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["columns"] = "|".join(row.get("columns") or [])
            row["json_keys"] = "|".join(row.get("json_keys") or [])
            writer.writerow(row)

    summary["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "summary_json": str(summary_path),
    }
    return summary


if __name__ == "__main__":
    report = build_legacy_artifact_report()
    print("OK 007B")
    print(f"records_total={report['records_total']}")
    print(f"records_exists={report['records_exists']}")
    print(f"records_missing={report['records_missing']}")
    print(f"json={report['outputs']['json']}")
    print(f"csv={report['outputs']['csv']}")
    print(f"summary_json={report['outputs']['summary_json']}")
