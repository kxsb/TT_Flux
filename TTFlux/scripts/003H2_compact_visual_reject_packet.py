from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003H2"

PRIORITY_IDS = ["R0014", "R0010", "R0009", "R0021", "R0020", "R0008"]
EXPECTED_IDS = set(PRIORITY_IDS)

MEDIA_EXTS = {".mp4", ".webm", ".mov", ".avi", ".png", ".jpg", ".jpeg", ".gif", ".html", ".csv"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif"}
OTHER_EXTS = {".html", ".csv"}

MAX_VIDEOS_PER_ID = 3
MAX_IMAGES_PER_ID = 5
MAX_OTHER_PER_ID = 2
MAX_TOTAL_PER_ID = 8

GOOD_KEYWORDS = [
    "overlay",
    "visual",
    "review",
    "segment",
    "selected",
    "arbiter",
    "packet",
    "contact",
    "sheet",
    "debug",
    "table",
    "player",
    "multi",
    "context",
]

BAD_KEYWORDS = [
    "raw_candidate",
    "candidate_frame",
    "frame_",
    "crop_",
    "mask_",
    "tmp",
    "cache",
    "debug_points",
]


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "file"


def exact_id_regex(rid: str) -> re.Pattern:
    return re.compile(rf"(?i)(^|[^a-z0-9]){re.escape(rid)}([^a-z0-9]|$)")


def looks_like_media_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip().strip('"').strip("'")
    if not s or len(s) > 500:
        return False
    low = s.lower()
    return any(ext in low for ext in MEDIA_EXTS) and ("\\" in s or "/" in s or low.endswith(tuple(MEDIA_EXTS)))


def resolve_path(value: str, project_root: Path, run_dir: Path) -> Path | None:
    raw = str(value).strip().strip('"').strip("'")
    if not raw:
        return None

    raw = raw.replace("\\", os.sep)
    p = Path(raw)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(project_root / p)
        candidates.append(run_dir / p)

    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            pass

    return None


def media_columns(df: pd.DataFrame) -> list[str]:
    key_parts = [
        "path", "file", "video", "overlay", "html", "png", "jpg",
        "mp4", "csv", "contact", "sheet", "segment", "render",
    ]

    cols = []
    for c in df.columns:
        low = str(c).lower()
        if not any(k in low for k in key_parts):
            continue
        vals = df[c].dropna().astype(str).head(50).tolist()
        if any(looks_like_media_value(v) for v in vals):
            cols.append(c)

    return cols


def score_file(path: Path, rid: str, source: str) -> int:
    name = path.name.lower()
    suffix = path.suffix.lower()

    score = 0

    if source == "csv_column":
        score += 900

    if suffix in VIDEO_EXTS:
        score += 500
    elif suffix in IMAGE_EXTS:
        score += 360
    elif suffix == ".html":
        score += 180
    elif suffix == ".csv":
        score += 80

    for kw in GOOD_KEYWORDS:
        if kw in name:
            score += 35

    for kw in BAD_KEYWORDS:
        if kw in name:
            score -= 80

    if rid.lower() in name:
        score += 120

    # Favorise les fichiers synthétiques plutôt que les centaines de frames.
    if re.search(r"frame[_-]?\d{3,}", name):
        score -= 100
    if re.search(r"\d{5,}", name):
        score -= 25

    try:
        size = path.stat().st_size
        if size > 0:
            score += min(int(size / 500_000), 40)
    except Exception:
        pass

    return score


def add_candidate(candidates: list[dict], seen: set[str], path: Path, rid: str, source: str):
    try:
        path = path.resolve()
    except Exception:
        return

    if not path.is_file():
        return
    if path.suffix.lower() not in MEDIA_EXTS:
        return

    key = str(path).lower()
    if key in seen:
        return

    seen.add(key)
    candidates.append({
        "review_id": rid,
        "source": source,
        "path": path,
        "suffix": path.suffix.lower(),
        "score": score_file(path, rid, source),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    })


def discover_candidates_for_id(
    rid: str,
    row: pd.Series,
    media_cols: list[str],
    project_root: Path,
    run_dir: Path,
) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()

    # 1) Chemins déjà présents dans les CSV : source la plus fiable.
    for c in media_cols:
        val = row.get(c)
        if not looks_like_media_value(val):
            continue
        p = resolve_path(str(val), project_root, run_dir)
        if p:
            add_candidate(candidates, seen, p, rid, "csv_column")

    # 2) Découverte par nom de fichier, mais avec regex exacte sur RID.
    rx = exact_id_regex(rid)
    for p in run_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in MEDIA_EXTS:
            continue
        if not rx.search(p.name):
            continue
        add_candidate(candidates, seen, p, rid, "name_exact")

    # 3) Fallback plus large seulement si rien trouvé.
    if not candidates:
        low = rid.lower()
        for p in run_dir.rglob(f"*{rid}*"):
            if p.is_file() and p.suffix.lower() in MEDIA_EXTS and low in p.name.lower():
                add_candidate(candidates, seen, p, rid, "name_fallback")

    candidates.sort(key=lambda d: (d["score"], d["size_bytes"]), reverse=True)
    return candidates


def pick_compact(candidates: list[dict]) -> list[dict]:
    videos = [c for c in candidates if c["suffix"] in VIDEO_EXTS][:MAX_VIDEOS_PER_ID]
    images = [c for c in candidates if c["suffix"] in IMAGE_EXTS][:MAX_IMAGES_PER_ID]
    other = [c for c in candidates if c["suffix"] in OTHER_EXTS][:MAX_OTHER_PER_ID]

    picked = videos + images + other
    picked.sort(key=lambda d: (d["score"], d["size_bytes"]), reverse=True)

    return picked[:MAX_TOTAL_PER_ID]


def copy_assets(picked_map: dict[str, list[dict]], out_asset_dir: Path) -> dict[str, list[dict]]:
    if out_asset_dir.exists():
        shutil.rmtree(out_asset_dir)
    out_asset_dir.mkdir(parents=True, exist_ok=True)

    copied_map: dict[str, list[dict]] = {}

    for rid, items in picked_map.items():
        rid_dir = out_asset_dir / rid
        rid_dir.mkdir(parents=True, exist_ok=True)
        copied_map[rid] = []

        for i, item in enumerate(items, start=1):
            src = item["path"]
            dst_name = f"{i:02d}_{safe_name(src.name)}"
            dst = rid_dir / dst_name

            try:
                shutil.copy2(src, dst)
                copied = dict(item)
                copied["copied_path"] = dst
                copied_map[rid].append(copied)
            except Exception as e:
                copied = dict(item)
                copied["copy_error"] = str(e)
                copied_map[rid].append(copied)

    return copied_map


def rel(path: Path, start: Path) -> str:
    try:
        return path.resolve().relative_to(start.resolve()).as_posix()
    except Exception:
        return os.path.relpath(path.resolve(), start.resolve()).replace("\\", "/")


def render_media(items: list[dict], html_dir: Path) -> str:
    if not items:
        return "<p class='muted'>Aucun média compact sélectionné.</p>"

    blocks = []
    for it in items:
        p = it.get("copied_path")
        if not p:
            msg = it.get("copy_error", "copie impossible")
            blocks.append(f"<p class='error'>{esc(msg)} · {esc(it.get('path'))}</p>")
            continue

        p = Path(p)
        href = rel(p, html_dir)
        label = (
            f"{it.get('source')} · score={it.get('score')} · "
            f"{p.name}"
        )

        suffix = p.suffix.lower()
        if suffix in VIDEO_EXTS:
            blocks.append(
                f"<div class='media'>"
                f"<div class='label'>{esc(label)}</div>"
                f"<video controls preload='metadata' src='{esc(href)}'></video>"
                f"<a href='{esc(href)}'>ouvrir</a>"
                f"</div>"
            )
        elif suffix in IMAGE_EXTS:
            blocks.append(
                f"<div class='media'>"
                f"<div class='label'>{esc(label)}</div>"
                f"<a href='{esc(href)}'><img src='{esc(href)}'></a>"
                f"</div>"
            )
        else:
            blocks.append(
                f"<div class='media'>"
                f"<div class='label'>{esc(label)}</div>"
                f"<a href='{esc(href)}'>{esc(p.name)}</a>"
                f"</div>"
            )

    return "\n".join(blocks)


def render_table(row: pd.Series) -> str:
    preferred = [
        "review_id_003G",
        "target_class_003G",
        "would_reject_shadow_003G",
        "review_bucket_003G",
        "new_reject_suggested_vs_003B4_003F",
        "contextual_hit_003B4_resolved",
        "multi_object_shadow_hit_003E",
        "arbiter_reason_003F",
        "inside_player_motion_mask_ratio_003C",
        "near_player_motion_mask_ratio_003C",
        "candidate_on_body_risk_003C",
        "point_distance_to_table_px_median",
        "point_inside_table_quad_ratio",
        "table_context_score_003B2",
        "max_accel",
        "v_max",
        "density_points_per_frame",
        "duration_frames",
        "final_policy_003G",
    ]

    rows = []
    for c in preferred:
        if c not in row.index:
            continue
        v = row.get(c)
        if pd.isna(v):
            continue
        rows.append(f"<tr><th>{esc(c)}</th><td>{esc(v)}</td></tr>")

    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def write_html(path: Path, rejects: pd.DataFrame, summary: dict, copied_map: dict[str, list[dict]]) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section,.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card.priority{border-color:#74d99f;box-shadow:inset 4px 0 0 rgba(116,217,159,.95)}
h1,h2,h3{margin-top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{width:260px;background:#20242e}
.grid{display:grid;grid-template-columns:420px 1fr;gap:16px}
@media(max-width:950px){.grid{grid-template-columns:1fr}}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.new{border-color:#74d99f;background:rgba(116,217,159,.08)}
.badge.reject{border-color:rgba(255,80,80,.8);background:rgba(255,80,80,.08)}
.muted{color:#aab2c5}
.error{color:#ff9999}
.media{margin:0 0 14px 0}
.label{font-size:12px;color:#aab2c5;margin-bottom:5px}
video{display:block;width:760px;max-width:100%;max-height:520px;background:#05060a;border:1px solid #2b303b;border-radius:10px}
img{display:block;width:760px;max-width:100%;max-height:520px;object-fit:contain;background:#05060a;border:1px solid #2b303b;border-radius:10px}
"""

    cards = []
    for _, row in rejects.iterrows():
        rid = str(row["review_id_003G"])
        target = str(row["target_class_003G"])
        is_new = boolish(row.get("new_reject_suggested_vs_003B4_003F", False))

        cls = "card priority" if rid == "R0014" else "card"
        badges = [
            "<span class='badge reject'>would reject</span>",
            f"<span class='badge'>{esc(target)}</span>",
        ]
        if is_new:
            badges.append("<span class='badge new'>new vs 003B4</span>")

        cards.append(f"""
<div class="{cls}">
<h2>{esc(rid)}</h2>
<div>{''.join(badges)}</div>
<div class="grid">
  <div>
    <h3>Signal arbiter</h3>
    {render_table(row)}
  </div>
  <div>
    <h3>Médias compacts sélectionnés</h3>
    {render_media(copied_map.get(rid, []), path.parent)}
  </div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003H2 compact visual reject packet</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003H2 compact visual reject packet</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

{''.join(cards)}

</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_001E")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    project_root = Path.cwd()
    run_dir = Path(args.run)

    input_csv = run_dir / "shadow_filter_simulation_003G.csv"
    input_json = run_dir / "shadow_filter_simulation_summary_003G.json"
    input_h_json = run_dir / "visual_reject_packet_summary_003H.json"

    if not input_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {input_csv}")
    if not input_json.is_file():
        raise SystemExit(f"Fichier introuvable: {input_json}")

    summary_003g = json.loads(input_json.read_text(encoding="utf-8"))
    if summary_003g.get("status") != "OK":
        raise SystemExit(f"003G non OK: {summary_003g.get('status')}")

    if input_h_json.is_file():
        summary_003h = json.loads(input_h_json.read_text(encoding="utf-8"))
        if summary_003h.get("status") not in {"OK", "WARN"}:
            raise SystemExit(f"003H invalide: {summary_003h.get('status')}")

    df = pd.read_csv(input_csv)
    df["review_id_003G"] = df["review_id_003G"].astype(str)
    df["would_reject_shadow_003G"] = df["would_reject_shadow_003G"].map(boolish)

    rejects = df[df["would_reject_shadow_003G"]].copy()

    priority = {rid: i for i, rid in enumerate(PRIORITY_IDS)}
    rejects["_priority"] = rejects["review_id_003G"].map(lambda x: priority.get(str(x), 999))
    rejects = rejects.sort_values(["_priority", "review_id_003G"]).drop(columns=["_priority"])

    ids = rejects["review_id_003G"].astype(str).tolist()
    ids_set = set(ids)

    media_cols = media_columns(df)

    candidate_counts = {}
    selected_counts = {}
    selected_map = {}

    for _, row in rejects.iterrows():
        rid = str(row["review_id_003G"])
        candidates = discover_candidates_for_id(rid, row, media_cols, project_root, run_dir)
        picked = pick_compact(candidates)

        candidate_counts[rid] = len(candidates)
        selected_counts[rid] = len(picked)
        selected_map[rid] = picked

    asset_dir = run_dir / "visual_reject_packet_003H2_assets"
    copied_map = copy_assets(selected_map, asset_dir)

    missing_expected = sorted(EXPECTED_IDS - ids_set)
    extra_expected = sorted(ids_set - EXPECTED_IDS)
    no_media = sorted([rid for rid, n in selected_counts.items() if n == 0])

    warnings = []
    if missing_expected:
        warnings.append("missing expected reject ids: " + ",".join(missing_expected))
    if extra_expected:
        warnings.append("unexpected reject ids: " + ",".join(extra_expected))
    if no_media:
        warnings.append("no selected compact media for: " + ",".join(no_media))

    status = "OK" if not missing_expected and not extra_expected and not no_media else "WARN"

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "source_csv": str(input_csv),
        "source_json": str(input_json),
        "policy": "compact_visual_review_packet_only_no_file_delete_no_final_change",
        "reject_ids_ordered": ids,
        "priority_review_id": "R0014",
        "media_columns_detected": media_cols,
        "candidate_media_counts_by_id": candidate_counts,
        "selected_media_counts_by_id": selected_counts,
        "max_videos_per_id": MAX_VIDEOS_PER_ID,
        "max_images_per_id": MAX_IMAGES_PER_ID,
        "max_other_per_id": MAX_OTHER_PER_ID,
        "max_total_per_id": MAX_TOTAL_PER_ID,
        "asset_dir": str(asset_dir),
        "expected_missing": missing_expected,
        "expected_extra": extra_expected,
        "warnings": warnings,
        "next_decision": (
            "Ouvrir visual_reject_packet_003H2.html. "
            "Verifier d'abord R0014. Si R0014 est visuellement un mauvais segment, "
            "preparer 003I pour integrer la regle en live avec garde-fous."
        ),
    }

    out_csv = run_dir / "visual_reject_packet_003H2.csv"
    out_json = run_dir / "visual_reject_packet_summary_003H2.json"
    out_html = run_dir / "visual_reject_packet_003H2.html"

    reject_export_cols = [
        c for c in rejects.columns
        if c.startswith("review_id_")
        or c.startswith("target_class_")
        or c in {
            "would_reject_shadow_003G",
            "review_bucket_003G",
            "arbiter_reason_003F",
            "new_reject_suggested_vs_003B4_003F",
            "contextual_hit_003B4_resolved",
            "multi_object_shadow_hit_003E",
            "inside_player_motion_mask_ratio_003C",
            "near_player_motion_mask_ratio_003C",
            "candidate_on_body_risk_003C",
            "point_distance_to_table_px_median",
            "point_inside_table_quad_ratio",
            "table_context_score_003B2",
            "max_accel",
            "v_max",
            "density_points_per_frame",
            "duration_frames",
            "final_policy_003G",
        }
    ]

    rejects[reject_export_cols].to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, rejects, summary, copied_map)

    print(f"003H2 status={status}")
    print(f"reject_ids={','.join(ids)}")
    print(f"priority=R0014")
    print("candidate_media_counts_by_id=" + json.dumps(candidate_counts, ensure_ascii=False))
    print("selected_media_counts_by_id=" + json.dumps(selected_counts, ensure_ascii=False))
    for w in warnings:
        print("warning:", w)
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)
    print("assets", asset_dir)

    if args.strict and status != "OK":
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
