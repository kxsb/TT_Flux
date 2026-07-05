from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


VERSION = "003U"

TARGETS = {
    "003B0_scene_priors": [
        "scene_priors_003B0",
        "scene_object_schema_003B0",
    ],
    "003B1_table_bootstrap": [
        "table_bootstrap_003B1",
        "table_bootstrap_models_003B1",
    ],
    "003B1B_table_curated": [
        "table_models_curated_003B1B",
        "table_bootstrap_003B1B",
    ],
    "003B2_table_context": [
        "table_context_003B2",
        "point_distance_to_table_px_median",
        "table_context_summary_003B2",
    ],
    "003B3_contextual_audit": [
        "contextual_features_003B3",
        "contextual_candidate_rules_003B3",
        "contextual_audit_003B3",
    ],
    "003B4_contextual_arbiter": [
        "contextual_arbiter_003B4",
        "contextual_arbiter_summary_003B4",
    ],
    "003C_player_context": [
        "player_context_features_003C",
        "inside_player_motion_mask_ratio_003C",
        "candidate_on_body_risk_003C",
    ],
    "003D_multi_object_audit": [
        "multi_object_features_003D",
        "multi_object_candidate_rules_003D",
        "multi_object_audit_003D",
    ],
}


CANONICAL_NAMES = {
    "003B0_scene_priors": "003B0_scene_priors.py",
    "003B1_table_bootstrap": "003B1_table_bootstrap.py",
    "003B1B_table_curated": "003B1B_curate_table_models.py",
    "003B2_table_context": "003B2_table_context.py",
    "003B3_contextual_audit": "003B3_contextual_kinematic_table_audit.py",
    "003B4_contextual_arbiter": "003B4_contextual_arbiter_shadow.py",
    "003C_player_context": "003C_player_context.py",
    "003D_multi_object_audit": "003D_multi_object_audit.py",
}


def rel(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def score_file(path: Path) -> dict:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {
            "path": str(path),
            "read_error": repr(exc),
            "targets": {},
            "total_score": 0,
        }

    low = txt.lower()
    targets = {}
    total = 0

    for target, signatures in TARGETS.items():
        hits = []
        score = 0

        for sig in signatures:
            if sig.lower() in low:
                hits.append(sig)
                score += 1

        # bonus si le nom du fichier lui-même ressemble à la cible
        name_low = path.name.lower()
        if target[:4].lower() in name_low:
            score += 2
            hits.append("filename_prefix_bonus")

        if score:
            targets[target] = {
                "score": score,
                "hits": hits,
            }
            total += score

    return {
        "path": str(path),
        "name": path.name,
        "size": path.stat().st_size,
        "targets": targets,
        "total_score": total,
    }


def find_python_files(root: Path) -> list[Path]:
    skip_parts = {
        ".git",
        "__pycache__",
        ".venv",
        ".venv_torch",
        "node_modules",
    }

    files = []

    for p in root.rglob("*.py"):
        if any(part in skip_parts for part in p.parts):
            continue
        files.append(p)

    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--min-score", type=int, default=2)
    args = ap.parse_args()

    root = Path.cwd()
    out_json = root / "runs" / "missing_feature_scripts_recovery_003U.json"
    out_html = root / "runs" / "missing_feature_scripts_recovery_003U.html"
    restore_dir = root / "scripts" / "recovered_003U"

    py_files = find_python_files(root)

    scored = []
    for p in py_files:
        info = score_file(p)
        if info["total_score"] >= args.min_score:
            info["path_rel"] = rel(root, p)
            scored.append(info)

    by_target = {}

    for target in TARGETS:
        candidates = []

        for info in scored:
            t = info["targets"].get(target)
            if not t:
                continue

            candidates.append({
                **info,
                "target_score": t["score"],
                "target_hits": t["hits"],
            })

        candidates = sorted(
            candidates,
            key=lambda x: (x["target_score"], x["total_score"], x["size"]),
            reverse=True,
        )

        by_target[target] = candidates[:10]

    best = {}
    for target, candidates in by_target.items():
        if candidates:
            best[target] = candidates[0]

    restored = []

    if args.restore:
        restore_dir.mkdir(parents=True, exist_ok=True)

        for target, info in best.items():
            src = Path(info["path"])
            dst = restore_dir / CANONICAL_NAMES[target]

            shutil.copy2(src, dst)

            restored.append({
                "target": target,
                "src": str(src),
                "dst": str(dst),
            })

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "python_files_scanned": len(py_files),
        "candidate_files": len(scored),
        "targets_found": sorted(best.keys()),
        "targets_missing": sorted(set(TARGETS) - set(best)),
        "best_by_target": best,
        "restored": restored,
        "restore_dir": str(restore_dir),
        "policy": "recovery_probe_only_no_pipeline_change" if not args.restore else "restored_to_scripts_recovered_003U_only",
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for target in TARGETS:
        candidates = by_target.get(target, [])
        if not candidates:
            rows.append(
                f"<tr><td>{target}</td><td>MISSING</td><td></td><td></td></tr>"
            )
            continue

        top = candidates[0]
        rows.append(
            "<tr>"
            f"<td>{target}</td>"
            f"<td>FOUND</td>"
            f"<td>{top['target_score']}</td>"
            f"<td>{top['path_rel']}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003U recovery probe</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e}}
</style>
</head>
<body>
<h1>TTFlux · 003U missing feature scripts recovery</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
<section>
<h2>Cibles</h2>
<table>
<thead><tr><th>target</th><th>status</th><th>score</th><th>best path</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</section>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")

    print("003U status=OK")
    print("python_files_scanned=", len(py_files))
    print("candidate_files=", len(scored))
    print("targets_found=" + (",".join(summary["targets_found"]) or "-"))
    print("targets_missing=" + (",".join(summary["targets_missing"]) or "-"))

    for target in TARGETS:
        candidates = by_target.get(target, [])
        if candidates:
            top = candidates[0]
            print(f"{target}: FOUND score={top['target_score']} path={top['path_rel']}")
        else:
            print(f"{target}: MISSING")

    if restored:
        print("restored_to=", restore_dir)
        for r in restored:
            print("RESTORED", r["target"], "=>", r["dst"])

    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
