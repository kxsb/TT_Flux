from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from datetime import datetime


PATCH_ID = "006Z_LOCK_table_pipeline_audit_and_archive"

KEEP_TOKENS = [
    "005C1",
    "005C3",
    "005C5",
    "005C5C2",
    "005C6",
    "005C6B",
    "005C7",
    "005C7A",
    "005C8",
    "005C8B",
    "005C9",
    "005C9A",
    "005C9B",
    "005D1",
    "005D2",
    "005D3",
    "005D3B",
    "005D4",
]

CANONICAL_CHAIN = [
    {
        "id": "005C9A",
        "role": "Optimisation projection table officielle contre masque appris / lignes support",
        "expected": [
            "runs/rally_table_reference_fit_005C9A/table_reference_fit_005C9A.csv",
            "runs/rally_table_reference_fit_005C9A/table_reference_fit_summary_005C9A.json",
            "runs/rally_table_reference_fit_005C9A/table_reference_fit_report_005C9A.html",
        ],
    },
    {
        "id": "005C9B",
        "role": "Promotion objets table scène consolidés",
        "expected": [
            "runs/rally_scene_table_objects_005C9B/scene_table_objects_005C9B.csv",
        ],
    },
    {
        "id": "005D1",
        "role": "Gate table/caméra conservatif pour points balle existants",
        "expected": [
            "runs/rally_ball_table_gate_005D1/ball_points_table_gated_005D1.csv",
            "runs/rally_ball_table_gate_005D1/ball_table_gate_summary_005D1.json",
        ],
    },
    {
        "id": "005D2",
        "role": "Score table/caméra pour reranking futur",
        "expected": [
            "runs/rally_ball_table_score_005D2/ball_points_table_scored_005D2.csv",
            "runs/rally_ball_table_score_005D2/ball_points_table_safe_005D2.csv",
            "runs/rally_ball_table_score_005D2/ball_table_score_summary_005D2.json",
            "runs/rally_ball_table_score_005D2/ball_table_score_topdown_report_005D2.html",
        ],
    },
    {
        "id": "005D3B",
        "role": "Overlay de validation visuelle table-aware ball score",
        "expected": [
            "runs/rally_ball_table_score_overlay_005D3B/ball_table_score_overlay_summary_005D3B.json",
        ],
    },
    {
        "id": "005D4",
        "role": "Relink temporel conservatif table-aware + interpolation courts gaps",
        "expected": [
            "runs/rally_ball_table_relink_005D4/ball_points_table_relinked_005D4.csv",
            "runs/rally_ball_table_relink_005D4/ball_table_relink_summary_005D4.json",
        ],
    },
]

DEAD_ENDS = [
    {
        "id": "006C",
        "reason": "heuristique HSV naïve ; ne remplace pas 005C9A/005C9B",
    },
    {
        "id": "006C_FIX2",
        "reason": "sélection d'un seul template global ; ignore changements caméra",
    },
    {
        "id": "006C3",
        "reason": "bonne idée conceptuelle caméra/shot, mais matching template instable ; à reprendre plus tard à partir des objets 005C9B",
    },
]

COPY_SUFFIXES = {".py", ".json", ".csv", ".html", ".md", ".txt", ".jpg", ".jpeg", ".png", ".pkl"}
SKIP_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_COPY_MB = 50.0


def rel_str(path: Path, root: Path):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def file_size_mb(path: Path):
    try:
        return path.stat().st_size / 1024 / 1024
    except Exception:
        return 0.0


def has_keep_token(text: str):
    return any(tok in text for tok in KEEP_TOKENS)


def resolve_path(value: str, root: Path):
    if not value:
        return None

    s = str(value).strip().strip('"')
    if not s:
        return None

    p = Path(s)
    if p.exists():
        return p.resolve()

    p2 = root / s
    if p2.exists():
        return p2.resolve()

    p3 = root / s.replace("\\", "/")
    if p3.exists():
        return p3.resolve()

    return None


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def flatten_json_paths(obj):
    found = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            s = x.strip()
            if any(ext in s.lower() for ext in [".csv", ".json", ".html", ".jpg", ".png", ".pkl", ".mp4", ".txt"]):
                found.append(s)

    walk(obj)
    return found


def extract_json_summary(path: Path, root: Path):
    data = read_json(path)
    if data is None:
        return None

    # Certains JSON de runs sont des listes racine, pas des objets.
    # Pour le lock pipeline, on ne garde que les summaries structurés en dict.
    if not isinstance(data, dict):
        return None

    text = json.dumps(data, ensure_ascii=False)
    if not has_keep_token(text) and not has_keep_token(str(path)):
        return None

    version = data.get("version") or data.get("patch") or ""
    policy = data.get("policy") or data.get("note") or data.get("next") or ""
    created_at = data.get("created_at") or ""

    outputs = []
    inputs = []

    for key, val in data.items():
        kl = str(key).lower()
        if isinstance(val, str):
            if any(ext in val.lower() for ext in [".csv", ".json", ".html", ".jpg", ".png", ".pkl", ".mp4", ".txt"]):
                if any(k in kl for k in ["output", "out", "overlay", "report", "csv", "html", "safe", "scored", "relinked"]):
                    outputs.append(val)
                else:
                    inputs.append(val)

    for p in flatten_json_paths(data.get("outputs", {})):
        outputs.append(p)

    return {
        "path": rel_str(path, root),
        "version": version,
        "policy": policy,
        "created_at": created_at,
        "inputs": sorted(set(inputs)),
        "outputs": sorted(set(outputs)),
    }


def scan_scripts(root: Path):
    scripts = []

    scripts_dir = root / "scripts"
    if not scripts_dir.exists():
        return scripts

    for p in scripts_dir.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if has_keep_token(txt) or has_keep_token(str(p)):
            tokens = [tok for tok in KEEP_TOKENS if tok in txt or tok in str(p)]
            scripts.append({
                "path": rel_str(p, root),
                "tokens": "|".join(sorted(set(tokens))),
                "size_mb": round(file_size_mb(p), 3),
            })

    return sorted(scripts, key=lambda x: x["path"])


def scan_run_summaries(root: Path):
    summaries = []
    runs_dir = root / "runs"

    if not runs_dir.exists():
        return summaries

    for p in runs_dir.rglob("*.json"):
        item = extract_json_summary(p, root)
        if item:
            item["size_mb"] = round(file_size_mb(p), 3)
            summaries.append(item)

    return sorted(summaries, key=lambda x: x["path"])


def scan_run_artifacts(root: Path):
    artifacts = []
    runs_dir = root / "runs"

    if not runs_dir.exists():
        return artifacts

    for p in runs_dir.rglob("*"):
        if not p.is_file():
            continue

        s = rel_str(p, root)
        if not has_keep_token(s):
            continue

        artifacts.append({
            "path": s,
            "suffix": p.suffix.lower(),
            "size_mb": round(file_size_mb(p), 3),
        })

    return sorted(artifacts, key=lambda x: x["path"])


def copy_file_to_archive(src: Path, root: Path, archive_root: Path, copied: set[str]):
    if not src or not src.exists() or not src.is_file():
        return None

    suffix = src.suffix.lower()
    if suffix in SKIP_SUFFIXES:
        return None
    if suffix not in COPY_SUFFIXES:
        return None
    if file_size_mb(src) > MAX_COPY_MB:
        return None

    key = str(src.resolve())
    if key in copied:
        return None

    try:
        rel = src.resolve().relative_to(root.resolve())
        dst = archive_root / "locked_files" / rel
    except Exception:
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(src))
        dst = archive_root / "locked_files_external" / safe_name

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.add(key)

    return rel_str(dst, root)


def collect_copy_candidates(root: Path, summaries: list[dict], scripts: list[dict], artifacts: list[dict]):
    candidates = set()

    for item in CANONICAL_CHAIN:
        for exp in item["expected"]:
            p = resolve_path(exp, root)
            if p:
                candidates.add(p)

    for s in scripts:
        p = resolve_path(s["path"], root)
        if p:
            candidates.add(p)

    for sm in summaries:
        p = resolve_path(sm["path"], root)
        if p:
            candidates.add(p)

        for v in sm.get("inputs", []) + sm.get("outputs", []):
            p2 = resolve_path(v, root)
            if p2:
                candidates.add(p2)

    for a in artifacts:
        p = resolve_path(a["path"], root)
        if p:
            candidates.add(p)

    return sorted(candidates, key=lambda p: str(p))


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def expected_status(root: Path):
    rows = []

    for item in CANONICAL_CHAIN:
        for exp in item["expected"]:
            p = resolve_path(exp, root)
            rows.append({
                "stage": item["id"],
                "role": item["role"],
                "expected_path": exp,
                "exists": "1" if p else "0",
                "resolved_path": rel_str(p, root) if p else "",
                "size_mb": round(file_size_mb(p), 3) if p else "",
            })

    return rows


def write_markdown(root: Path, out_dir: Path, status_rows, scripts, summaries, copied_files):
    md_path = out_dir / "TABLE_PIPELINE_CANONICAL_006Z.md"

    lines = []
    lines.append("# TTFlux — Table pipeline canonical lock 006Z")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Décision")
    lines.append("")
    lines.append("La méthode table fonctionnelle à conserver est la chaîne `005C9A → 005C9B → 005D1 → 005D2 → 005D4`.")
    lines.append("")
    lines.append("Les scripts `006C*` ne doivent pas remplacer cette base. Ils sont classés comme diagnostics ou pistes de reprise, pas comme source canonique.")
    lines.append("")
    lines.append("## Chaîne canonique")
    lines.append("")
    for item in CANONICAL_CHAIN:
        lines.append(f"### {item['id']}")
        lines.append(item["role"])
        lines.append("")
        for exp in item["expected"]:
            ok = next((r for r in status_rows if r["expected_path"] == exp), None)
            marker = "OK" if ok and ok["exists"] == "1" else "MISSING"
            lines.append(f"- `{marker}` `{exp}`")
        lines.append("")
    lines.append("## À ne pas utiliser comme table canonique")
    lines.append("")
    for d in DEAD_ENDS:
        lines.append(f"- `{d['id']}` : {d['reason']}")
    lines.append("")
    lines.append("## Règle de reprise pour 006D")
    lines.append("")
    lines.append("1. Ne pas redétecter la table avec HSV simple.")
    lines.append("2. Repartir de `scene_table_objects_005C9B.csv` pour les objets table.")
    lines.append("3. Joindre les labels humains 006A/006B aux candidats balle et aux objets table par `review_id`, `clip_path`, `video_id`, plage de frames, puis frame locale.")
    lines.append("4. La table doit être dynamique par segment/caméra. Si aucun contexte table fiable n’existe pour un plan rapproché, inscrire `NO_TABLE_CONTEXT` plutôt que projeter un mauvais quad.")
    lines.append("5. Le score table/caméra doit être une feature de reranking, pas un filtre dur unique.")
    lines.append("")
    lines.append("## Scripts liés trouvés")
    lines.append("")
    for s in scripts:
        lines.append(f"- `{s['path']}` tokens={s['tokens']}")
    lines.append("")
    lines.append("## Summaries liés trouvés")
    lines.append("")
    for sm in summaries:
        version = sm.get("version", "")
        lines.append(f"- `{sm['path']}` version={version}")
    lines.append("")
    lines.append("## Fichiers copiés dans l’archive")
    lines.append("")
    for cf in copied_files:
        lines.append(f"- `{cf}`")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    root_md = root / "TABLE_PIPELINE_CANONICAL_006Z.md"
    root_md.write_text("\n".join(lines), encoding="utf-8")

    return md_path, root_md


def main():
    root = Path(".").resolve()
    out_dir = root / "runs" / "006Z_table_pipeline_lock"
    out_dir.mkdir(parents=True, exist_ok=True)

    scripts = scan_scripts(root)
    summaries = scan_run_summaries(root)
    artifacts = scan_run_artifacts(root)
    status_rows = expected_status(root)

    scripts_csv = out_dir / "006Z_scripts_manifest.csv"
    summaries_json = out_dir / "006Z_run_summaries_manifest.json"
    artifacts_csv = out_dir / "006Z_artifacts_manifest.csv"
    status_csv = out_dir / "006Z_expected_canonical_files.csv"

    write_csv(scripts_csv, scripts, ["path", "tokens", "size_mb"])
    summaries_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(artifacts_csv, artifacts, ["path", "suffix", "size_mb"])
    write_csv(status_csv, status_rows, ["stage", "role", "expected_path", "exists", "resolved_path", "size_mb"])

    copy_candidates = collect_copy_candidates(root, summaries, scripts, artifacts)

    copied = set()
    copied_files = []

    archive_root = out_dir
    for src in copy_candidates:
        copied_path = copy_file_to_archive(src, root, archive_root, copied)
        if copied_path:
            copied_files.append(copied_path)

    md_path, root_md = write_markdown(root, out_dir, status_rows, scripts, summaries, copied_files)

    summary = {
        "patch": PATCH_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "out_dir": str(out_dir),
        "canonical_chain": CANONICAL_CHAIN,
        "dead_ends": DEAD_ENDS,
        "scripts_count": len(scripts),
        "summaries_count": len(summaries),
        "artifacts_count": len(artifacts),
        "expected_status": status_rows,
        "copied_files_count": len(copied_files),
        "outputs": {
            "lock_markdown": str(md_path),
            "root_markdown": str(root_md),
            "scripts_csv": str(scripts_csv),
            "summaries_json": str(summaries_json),
            "artifacts_csv": str(artifacts_csv),
            "status_csv": str(status_csv),
            "locked_files_dir": str(out_dir / "locked_files"),
        },
        "next": "Reprendre 006D depuis cette chaîne canonique, surtout scene_table_objects_005C9B.csv, pas depuis les essais 006C*.",
    }

    summary_path = out_dir / "006Z_table_pipeline_lock_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print("=" * 72)
    print("")
    print("OK 006Z_LOCK")
    print(f"summary       = {summary_path}")
    print(f"lock_md       = {md_path}")
    print(f"root_md       = {root_md}")
    print(f"status_csv    = {status_csv}")
    print(f"scripts_csv   = {scripts_csv}")
    print(f"artifacts_csv = {artifacts_csv}")
    print(f"archive_dir   = {out_dir / 'locked_files'}")
    print("")
    print(f"scripts_count={len(scripts)}")
    print(f"summaries_count={len(summaries)}")
    print(f"artifacts_count={len(artifacts)}")
    print(f"copied_files_count={len(copied_files)}")
    print("")
    print("CANONICAL EXPECTED FILES")
    for r in status_rows:
        mark = "OK" if r["exists"] == "1" else "MISSING"
        print(f"{mark} {r['stage']} {r['expected_path']}")
    print("")
    print("Decision: 005C9A/005C9B/005D1/005D2/005D4 are locked as canonical table pipeline.")
    print("Do not use 006C* as canonical table context.")


if __name__ == "__main__":
    main()

