from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003H"

EXPECTED_REJECT_IDS = {"R0010", "R0009", "R0020", "R0014", "R0008", "R0021"}
PRIORITY_IDS = ["R0014", "R0010", "R0009", "R0021", "R0020", "R0008"]

FOCUS_NOTE = {
    "R0014": "NEW_VS_003B4 — priorite controle visuel",
    "R0010": "human_reject deja capture par 003B4 + 003E",
    "R0009": "human_reject deja capture par 003B4 + 003E",
    "R0021": "human_reject deja capture par 003B4 + 003E",
    "R0020": "auto_reject deja capture par 003B4 + 003E",
    "R0008": "auto_reject deja capture par 003B4 + 003E",
}

CORE_COLS = [
    "review_id_003G",
    "target_class_003G",
    "would_reject_shadow_003G",
    "review_bucket_003G",
    "arbiter_reason_003F",
    "final_policy_003G",
]

METRIC_COLS = [
    "inside_player_motion_mask_ratio_003C",
    "near_player_motion_mask_ratio_003C",
    "candidate_on_body_risk_003C",
    "occlusion_entry_exit_count_003C",
    "point_distance_to_table_px_median",
    "point_inside_table_quad_ratio",
    "table_context_score_003B2",
    "max_accel",
    "p90_accel",
    "v_max",
    "median_speed",
    "speed_cv",
    "density_points_per_frame",
    "duration_frames",
    "tortuosity",
    "sharp_turn_ratio",
    "jump_ratio",
    "stop_ratio",
    "endpoint_gap_ratio",
    "has_table_model",
]

MEDIA_EXTS = {".mp4", ".webm", ".mov", ".avi", ".png", ".jpg", ".jpeg", ".gif", ".html", ".csv"}
VIDEO_EXTS = {".mp4", ".webm", ".mov"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif"}


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def safe_relpath(path: Path, start: Path) -> str:
    try:
        return path.resolve().relative_to(start.resolve()).as_posix()
    except Exception:
        try:
            return os.path.relpath(path.resolve(), start.resolve()).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")


def resolve_path(value: str, project_root: Path, run_dir: Path) -> Path | None:
    if not value or not isinstance(value, str):
        return None

    raw = value.strip().strip('"').strip("'")
    if not raw:
        return None

    # Evite les gros blobs ou textes de regle.
    if len(raw) > 500:
        return None

    # Nettoyage minimal si cellule contient un chemin Windows avec backslashes.
    raw_norm = raw.replace("\\", os.sep)

    candidates = []

    p = Path(raw_norm)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(project_root / p)
        candidates.append(run_dir / p)

    for c in candidates:
        try:
            if c.is_file():
                return c
        except Exception:
            pass

    return None


def looks_like_path_value(value: object) -> bool:
    if not isinstance(value, str):
        return False

    s = value.strip().strip('"').strip("'")
    if not s or len(s) > 500:
        return False

    low = s.lower()
    if any(low.endswith(ext) for ext in MEDIA_EXTS):
        return True

    if ("\\" in s or "/" in s) and any(ext in low for ext in MEDIA_EXTS):
        return True

    return False


def candidate_media_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    key_parts = [
        "path", "file", "video", "overlay", "html", "png", "jpg",
        "mp4", "csv", "contact", "sheet", "segment", "render",
    ]

    for c in df.columns:
        low = str(c).lower()
        if any(k in low for k in key_parts):
            # On garde seulement si au moins une cellule ressemble a un chemin.
            vals = df[c].dropna().astype(str).head(20).tolist()
            if any(looks_like_path_value(v) for v in vals):
                cols.append(c)

    return cols


def discover_files_by_id(run_dir: Path, review_id: str) -> list[Path]:
    hits = []
    patterns = [
        f"*{review_id}*",
        f"*{review_id.lower()}*",
        f"*{review_id.upper()}*",
    ]

    seen = set()
    for pat in patterns:
        for p in run_dir.rglob(pat):
            if not p.is_file():
                continue
            if p in seen:
                continue
            if p.suffix.lower() in MEDIA_EXTS:
                hits.append(p)
                seen.add(p)

    return sorted(hits)


def collect_media_for_row(row: pd.Series, media_cols: list[str], project_root: Path, run_dir: Path) -> list[dict]:
    review_id = str(row.get("review_id_003G", "")).strip()
    items = []
    seen = set()

    for c in media_cols:
        val = row.get(c)
        if not looks_like_path_value(val):
            continue

        p = resolve_path(str(val), project_root, run_dir)
        if not p:
            continue

        key = str(p.resolve()).lower()
        if key in seen:
            continue

        seen.add(key)
        items.append({
            "source": c,
            "path": p,
            "suffix": p.suffix.lower(),
        })

    for p in discover_files_by_id(run_dir, review_id):
        key = str(p.resolve()).lower()
        if key in seen:
            continue

        seen.add(key)
        items.append({
            "source": "discovered_by_review_id",
            "path": p,
            "suffix": p.suffix.lower(),
        })

    return items


def media_html(items: list[dict], report_dir: Path) -> str:
    if not items:
        return "<p class='muted'>Aucun media detecte automatiquement pour cette ligne. Utiliser les chemins du manifest precedent si besoin.</p>"

    parts = []
    for it in items:
        p = it["path"]
        suffix = it["suffix"]
        rel = safe_relpath(p, report_dir)
        label = esc(f"{it['source']} · {p.name}")

        if suffix in VIDEO_EXTS:
            parts.append(
                f"<div class='media-block'>"
                f"<div class='media-label'>{label}</div>"
                f"<video controls preload='metadata' src='{esc(rel)}'></video>"
                f"<div><a href='{esc(rel)}'>ouvrir video</a></div>"
                f"</div>"
            )
        elif suffix in IMAGE_EXTS:
            parts.append(
                f"<div class='media-block'>"
                f"<div class='media-label'>{label}</div>"
                f"<a href='{esc(rel)}'><img src='{esc(rel)}' alt='{label}'></a>"
                f"</div>"
            )
        else:
            parts.append(
                f"<div class='media-block'>"
                f"<div class='media-label'>{label}</div>"
                f"<a href='{esc(rel)}'>{esc(str(p))}</a>"
                f"</div>"
            )

    return "\n".join(parts)


def metric_table(row: pd.Series) -> str:
    rows = []
    for c in METRIC_COLS:
        if c not in row.index:
            continue
        v = row.get(c)
        if pd.isna(v):
            continue
        rows.append(f"<tr><th>{esc(c)}</th><td>{esc(v)}</td></tr>")

    if not rows:
        return "<p class='muted'>Aucune metrique detaillee detectee dans le CSV 003G.</p>"

    return "<table class='metric-table'><tbody>" + "".join(rows) + "</tbody></table>"


def write_html(path: Path, rejects: pd.DataFrame, summary: dict, media_map: dict[str, list[dict]]) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:18px}
.card.priority{border-color:#74d99f;box-shadow:inset 4px 0 0 rgba(116,217,159,.95)}
h1,h2,h3{margin-top:0}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{background:#20242e}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.reject{border-color:rgba(255,80,80,.8);background:rgba(255,80,80,.08)}
.badge.new{border-color:rgba(116,217,159,.9);background:rgba(116,217,159,.08)}
.badge.target{border-color:rgba(150,170,255,.7);background:rgba(150,170,255,.06)}
.muted{color:#aab2c5}
video{display:block;max-width:100%;width:760px;max-height:520px;background:#05060a;border:1px solid #2b303b;border-radius:10px}
img{display:block;max-width:100%;width:760px;max-height:520px;object-fit:contain;background:#05060a;border:1px solid #2b303b;border-radius:10px}
.media-block{margin:10px 0 14px 0}
.media-label{font-size:12px;color:#aab2c5;margin-bottom:6px}
.grid{display:grid;grid-template-columns:minmax(280px, 420px) 1fr;gap:16px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.metric-table th{width:260px}
"""

    cards = []
    for _, r in rejects.iterrows():
        rid = str(r["review_id_003G"])
        target = str(r["target_class_003G"])
        is_new = boolish(r.get("new_reject_suggested_vs_003B4_003F", False))
        cls = "card priority" if rid == "R0014" else "card"

        badges = [
            "<span class='badge reject'>would reject</span>",
            f"<span class='badge target'>{esc(target)}</span>",
        ]
        if is_new:
            badges.append("<span class='badge new'>new vs 003B4</span>")

        note = FOCUS_NOTE.get(rid, "")

        core_rows = []
        for c in CORE_COLS:
            if c in r.index:
                core_rows.append(f"<tr><th>{esc(c)}</th><td>{esc(r.get(c))}</td></tr>")
        if "new_reject_suggested_vs_003B4_003F" in r.index:
            core_rows.append(
                "<tr><th>new_reject_suggested_vs_003B4_003F</th>"
                f"<td>{esc(r.get('new_reject_suggested_vs_003B4_003F'))}</td></tr>"
            )

        cards.append(f"""
<div class="{cls}">
<h2>{esc(rid)} · {esc(note)}</h2>
<div>{''.join(badges)}</div>
<div class="grid">
  <div>
    <h3>Decision shadow</h3>
    <table><tbody>{''.join(core_rows)}</tbody></table>
    <h3>Metriques utiles</h3>
    {metric_table(r)}
  </div>
  <div>
    <h3>Medias / liens detectes</h3>
    {media_html(media_map.get(rid, []), path.parent)}
  </div>
</div>
</div>
""")

    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003H visual reject packet</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003H visual reject packet</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

{''.join(cards)}

</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_001E")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    project_root = Path.cwd()
    run_dir = Path(args.run)

    input_csv = run_dir / "shadow_filter_simulation_003G.csv"
    input_json = run_dir / "shadow_filter_simulation_summary_003G.json"

    if not input_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {input_csv}")
    if not input_json.is_file():
        raise SystemExit(f"Fichier introuvable: {input_json}")

    summary_003g = json.loads(input_json.read_text(encoding="utf-8"))
    if summary_003g.get("status") != "OK":
        raise SystemExit(f"003G non OK: {summary_003g.get('status')}")

    df = pd.read_csv(input_csv)

    required = ["review_id_003G", "target_class_003G", "would_reject_shadow_003G"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Colonnes manquantes dans 003G CSV: " + ", ".join(missing))

    df["would_reject_shadow_003G"] = df["would_reject_shadow_003G"].map(boolish)
    df["review_id_003G"] = df["review_id_003G"].astype(str)

    rejects = df[df["would_reject_shadow_003G"]].copy()

    priority = {rid: i for i, rid in enumerate(PRIORITY_IDS)}
    rejects["_priority_003H"] = rejects["review_id_003G"].map(lambda x: priority.get(str(x), 999))
    rejects = rejects.sort_values(["_priority_003H", "review_id_003G"]).drop(columns=["_priority_003H"])

    reject_ids = rejects["review_id_003G"].astype(str).tolist()
    reject_set = set(reject_ids)

    media_cols = candidate_media_columns(df)
    media_map = {}
    for _, r in rejects.iterrows():
        rid = str(r["review_id_003G"])
        media_map[rid] = collect_media_for_row(r, media_cols, project_root, run_dir)

    media_counts = {rid: len(items) for rid, items in media_map.items()}

    expected_missing = sorted(EXPECTED_REJECT_IDS - reject_set)
    expected_extra = sorted(reject_set - EXPECTED_REJECT_IDS)

    warnings = []
    if expected_missing:
        warnings.append("missing expected reject ids: " + ",".join(expected_missing))
    if expected_extra:
        warnings.append("unexpected reject ids: " + ",".join(expected_extra))
    if "R0014" not in reject_set:
        warnings.append("R0014 absent du packet visuel")
    if not media_cols:
        warnings.append("aucune colonne media/path detectee automatiquement")
    if all(v == 0 for v in media_counts.values()):
        warnings.append("aucun fichier media resolu automatiquement pour les rejets")

    status = "OK" if not expected_missing and not expected_extra and "R0014" in reject_set else "WARN"

    by_target = (
        rejects.groupby("target_class_003G")["review_id_003G"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    packet_cols = [c for c in CORE_COLS if c in rejects.columns]
    for c in ["new_reject_suggested_vs_003B4_003F"] + METRIC_COLS:
        if c in rejects.columns and c not in packet_cols:
            packet_cols.append(c)

    packet = rejects[packet_cols].copy()

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "source_csv": str(input_csv),
        "source_json": str(input_json),
        "policy": "visual_review_packet_only_no_file_delete_no_final_change",
        "reject_packet_total": int(len(rejects)),
        "reject_packet_ids_ordered": reject_ids,
        "priority_review_id": "R0014",
        "by_target": by_target,
        "media_columns_detected": media_cols,
        "media_counts_by_id": media_counts,
        "expected_missing": expected_missing,
        "expected_extra": expected_extra,
        "warnings": warnings,
        "next_decision": (
            "Inspecter visuellement R0014 en premier. "
            "Si R0014 est confirme comme mauvais segment, la regle 003E peut passer en candidat "
            "d'integration live avec garde-fous. Sinon, conserver 003E en shadow."
        ),
    }

    out_csv = run_dir / "visual_reject_packet_003H.csv"
    out_json = run_dir / "visual_reject_packet_summary_003H.json"
    out_html = run_dir / "visual_reject_packet_003H.html"

    packet.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, rejects, summary, media_map)

    print(f"003H status={status}")
    print(f"reject_packet={','.join(reject_ids)}")
    print(f"priority=R0014")
    print("media_counts_by_id=" + json.dumps(media_counts, ensure_ascii=False))
    for w in warnings:
        print("warning:", w)
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    if args.strict and status != "OK":
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
