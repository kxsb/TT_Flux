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


VERSION = "003I"

TABLE_METRICS = [
    "has_table_model",
    "table_context_score_003B2",
    "point_inside_table_quad_ratio",
    "point_distance_to_table_px_median",
    "point_distance_to_table_px_p10",
    "point_distance_to_table_px_p90",
    "point_above_table_ratio",
    "point_below_table_ratio",
    "point_left_of_table_ratio",
    "point_right_of_table_ratio",
    "table_quad_area_px",
    "table_quad_width_px",
    "table_quad_height_px",
]

ARB_COLS = [
    "review_id_003G",
    "target_class_003G",
    "would_reject_shadow_003G",
    "review_bucket_003G",
    "contextual_hit_003B4_resolved",
    "multi_object_shadow_hit_003E",
    "new_reject_suggested_vs_003B4_003F",
    "arbiter_reason_003F",
]

KIN_PLAYER_COLS = [
    "inside_player_motion_mask_ratio_003C",
    "near_player_motion_mask_ratio_003C",
    "candidate_on_body_risk_003C",
    "max_accel",
    "v_max",
    "density_points_per_frame",
    "duration_frames",
]

MEDIA_PRIORITY = [
    "table_mp4",
    "table_csv",
    "mp4",
    "csv",
    "player_context_overlay_003C",
    "track_csv_resolved_003C",
    "track_csv_resolved_003B2",
    "track_csv_resolved_002G2",
]

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif"}
LINK_EXTS = {".csv", ".html", ".json", ".txt"}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def num(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "file"


def resolve_path(value, project_root: Path, run_dir: Path) -> Path | None:
    if not isinstance(value, str):
        return None

    raw = value.strip().strip('"').strip("'")
    if not raw or len(raw) > 600:
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


def rel(path: Path, start: Path) -> str:
    try:
        return path.resolve().relative_to(start.resolve()).as_posix()
    except Exception:
        return os.path.relpath(path.resolve(), start.resolve()).replace("\\", "/")


def table_status(row: pd.Series) -> tuple[str, list[str]]:
    reasons = []

    has_model = num(row.get("has_table_model"), 1.0)
    score = num(row.get("table_context_score_003B2"), None)
    inside = num(row.get("point_inside_table_quad_ratio"), None)
    dist = num(row.get("point_distance_to_table_px_median"), None)

    if has_model is not None and has_model <= 0:
        reasons.append("missing_table_model")

    if inside is not None and inside <= 0.02:
        reasons.append("inside_ratio_near_zero")
    elif inside is not None and inside <= 0.10:
        reasons.append("inside_ratio_weak")

    if score is not None and score <= 0.30:
        reasons.append("table_score_low")
    elif score is not None and score <= 0.50:
        reasons.append("table_score_mid")

    if dist is not None and dist >= 180:
        reasons.append("median_distance_high")
    elif dist is not None and dist >= 100:
        reasons.append("median_distance_mid")

    if "missing_table_model" in reasons:
        return "MISSING_MODEL", reasons

    if (
        "inside_ratio_near_zero" in reasons
        or "table_score_low" in reasons
        or "median_distance_high" in reasons
    ):
        return "BAD_OR_UNRELIABLE", reasons

    if (
        "inside_ratio_weak" in reasons
        or "table_score_mid" in reasons
        or "median_distance_mid" in reasons
    ):
        return "WEAK", reasons

    return "OK_OR_NOT_ENOUGH_EVIDENCE", reasons


def collect_media(row: pd.Series, project_root: Path, run_dir: Path, asset_dir: Path) -> list[dict]:
    rid = str(row.get("review_id_003G", row.get("review_id_003E", row.get("review_id", "unknown"))))
    rid_dir = asset_dir / safe_name(rid)
    rid_dir.mkdir(parents=True, exist_ok=True)

    items = []
    seen = set()

    for col in MEDIA_PRIORITY:
        if col not in row.index:
            continue

        p = resolve_path(row.get(col), project_root, run_dir)
        if not p:
            continue

        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)

        suffix = p.suffix.lower()
        dst = rid_dir / f"{len(items)+1:02d}_{safe_name(col)}_{safe_name(p.name)}"

        try:
            shutil.copy2(p, dst)
            items.append({
                "column": col,
                "src": str(p),
                "dst": str(dst),
                "suffix": suffix,
                "copied": True,
            })
        except Exception as e:
            items.append({
                "column": col,
                "src": str(p),
                "dst": "",
                "suffix": suffix,
                "copied": False,
                "error": str(e),
            })

    return items


def render_media(items: list[dict], html_dir: Path) -> str:
    if not items:
        return "<p class='muted'>Aucun média table/segment résolu.</p>"

    blocks = []
    for it in items:
        label = f"{it.get('column')} · {Path(it.get('src', '')).name}"

        if not it.get("copied"):
            blocks.append(
                f"<p class='error'>{esc(label)} — copie impossible : {esc(it.get('error', ''))}</p>"
            )
            continue

        p = Path(it["dst"])
        href = rel(p, html_dir)
        suffix = it["suffix"]

        if suffix in VIDEO_EXTS:
            blocks.append(
                f"<div class='media'>"
                f"<div class='label'>{esc(label)}</div>"
                f"<video controls preload='metadata' src='{esc(href)}'></video>"
                f"<div><a href='{esc(href)}'>ouvrir vidéo</a></div>"
                f"</div>"
            )
        elif suffix in IMAGE_EXTS:
            blocks.append(
                f"<div class='media'>"
                f"<div class='label'>{esc(label)}</div>"
                f"<a href='{esc(href)}'><img src='{esc(href)}'></a>"
                f"</div>"
            )
        elif suffix in LINK_EXTS:
            blocks.append(
                f"<div class='media'>"
                f"<div class='label'>{esc(label)}</div>"
                f"<a href='{esc(href)}'>{esc(p.name)}</a>"
                f"</div>"
            )

    return "\n".join(blocks)


def render_metrics(row: pd.Series) -> str:
    rows = []

    for group_name, cols in [
        ("Arbiter", ARB_COLS),
        ("Table", TABLE_METRICS),
        ("Player / cinématique", KIN_PLAYER_COLS),
    ]:
        added = False
        for c in cols:
            if c not in row.index:
                continue

            v = row.get(c)
            if pd.isna(v):
                continue

            if not added:
                rows.append(f"<tr class='group'><th colspan='2'>{esc(group_name)}</th></tr>")
                added = True

            rows.append(f"<tr><th>{esc(c)}</th><td>{esc(v)}</td></tr>")

    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def write_html(path: Path, rows: pd.DataFrame, summary: dict, media_map: dict[str, list[dict]]) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section,.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card.bad{border-color:rgba(255,80,80,.75);box-shadow:inset 4px 0 0 rgba(255,80,80,.85)}
.card.weak{border-color:rgba(255,200,80,.65);box-shadow:inset 4px 0 0 rgba(255,200,80,.75)}
.card.ok{border-color:rgba(116,217,159,.55)}
h1,h2,h3{margin-top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
.grid{display:grid;grid-template-columns:430px 1fr;gap:16px}
@media(max-width:950px){.grid{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{width:260px;background:#20242e}
tr.group th{background:#11141b;color:#dce4ff}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.bad{border-color:rgba(255,80,80,.8);background:rgba(255,80,80,.08)}
.badge.weak{border-color:rgba(255,200,80,.8);background:rgba(255,200,80,.08)}
.badge.ok{border-color:rgba(116,217,159,.8);background:rgba(116,217,159,.08)}
.badge.reject{border-color:rgba(255,80,80,.8)}
.muted{color:#aab2c5}
.error{color:#ff9999}
.media{margin:0 0 14px 0}
.label{font-size:12px;color:#aab2c5;margin-bottom:5px}
video{display:block;width:760px;max-width:100%;max-height:520px;background:#05060a;border:1px solid #2b303b;border-radius:10px}
img{display:block;width:760px;max-width:100%;max-height:520px;object-fit:contain;background:#05060a;border:1px solid #2b303b;border-radius:10px}
"""

    cards = []

    for _, row in rows.iterrows():
        rid = str(row["review_id_003G"])
        status = str(row["table_reality_status_003I"])
        reasons = str(row["table_reality_reasons_003I"])
        reject = boolish(row.get("would_reject_shadow_003G", False))

        if status in {"BAD_OR_UNRELIABLE", "MISSING_MODEL"}:
            cls = "card bad"
            badge_cls = "badge bad"
        elif status == "WEAK":
            cls = "card weak"
            badge_cls = "badge weak"
        else:
            cls = "card ok"
            badge_cls = "badge ok"

        badges = [
            f"<span class='{badge_cls}'>{esc(status)}</span>",
            f"<span class='badge'>{esc(row.get('target_class_003G', ''))}</span>",
        ]
        if reject:
            badges.append("<span class='badge reject'>would reject</span>")

        cards.append(f"""
<div class="{cls}">
<h2>{esc(rid)} · {esc(status)}</h2>
<div>{''.join(badges)}</div>
<p class="muted">{esc(reasons)}</p>
<div class="grid">
  <div>
    <h3>Mesures</h3>
    {render_metrics(row)}
  </div>
  <div>
    <h3>Médias table / segment</h3>
    {render_media(media_map.get(rid, []), path.parent)}
  </div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003I table reality audit</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003I table reality audit</h1>

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

    if not input_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {input_csv}")
    if not input_json.is_file():
        raise SystemExit(f"Fichier introuvable: {input_json}")

    g_summary = json.loads(input_json.read_text(encoding="utf-8"))
    if g_summary.get("status") != "OK":
        raise SystemExit(f"003G non OK: {g_summary.get('status')}")

    df = pd.read_csv(input_csv)
    if "review_id_003G" not in df.columns:
        raise SystemExit("Colonne review_id_003G manquante.")

    df["review_id_003G"] = df["review_id_003G"].astype(str)

    statuses = []
    reasons_all = []

    for _, row in df.iterrows():
        st, reasons = table_status(row)
        statuses.append(st)
        reasons_all.append(",".join(reasons))

    df["table_reality_status_003I"] = statuses
    df["table_reality_reasons_003I"] = reasons_all

    # Tri : d'abord les cas rejetés, puis les tables mauvaises/faibles.
    status_priority = {
        "BAD_OR_UNRELIABLE": 0,
        "MISSING_MODEL": 1,
        "WEAK": 2,
        "OK_OR_NOT_ENOUGH_EVIDENCE": 3,
    }

    df["_status_priority"] = df["table_reality_status_003I"].map(lambda x: status_priority.get(x, 9))
    df["_reject_priority"] = df.get("would_reject_shadow_003G", False).map(lambda x: 0 if boolish(x) else 1)

    df = df.sort_values(["_reject_priority", "_status_priority", "review_id_003G"]).drop(
        columns=["_status_priority", "_reject_priority"]
    )

    asset_dir = run_dir / "table_reality_audit_003I_assets"
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    media_map = {}
    for _, row in df.iterrows():
        rid = str(row["review_id_003G"])
        media_map[rid] = collect_media(row, project_root, run_dir, asset_dir)

    by_status = (
        df.groupby("table_reality_status_003I")["review_id_003G"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    reject_df = df[df.get("would_reject_shadow_003G", False).map(boolish)]
    reject_by_status = (
        reject_df.groupby("table_reality_status_003I")["review_id_003G"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    bad_ids = df[df["table_reality_status_003I"].isin(["BAD_OR_UNRELIABLE", "MISSING_MODEL"])]["review_id_003G"].astype(str).tolist()
    weak_ids = df[df["table_reality_status_003I"].eq("WEAK")]["review_id_003G"].astype(str).tolist()

    warnings = []
    if bad_ids:
        warnings.append("table bad/unreliable ids: " + ",".join(bad_ids))
    if weak_ids:
        warnings.append("table weak ids: " + ",".join(weak_ids))

    # Ici WARN est attendu : on audite précisément parce que la table est douteuse.
    status = "WARN" if warnings else "OK"

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "source_csv": str(input_csv),
        "policy": "table_reality_audit_only_no_arbiter_change",
        "rows": int(len(df)),
        "by_table_reality_status": by_status,
        "reject_suggested_by_table_reality_status": reject_by_status,
        "bad_or_missing_table_ids": bad_ids,
        "weak_table_ids": weak_ids,
        "asset_dir": str(asset_dir),
        "warnings": warnings,
        "interpretation": (
            "Si beaucoup de would_reject reposent sur BAD_OR_UNRELIABLE ou WEAK, "
            "la règle 003E ne doit pas être promue live. "
            "Il faut corriger le modèle table avant intégration."
        ),
    }

    out_csv = run_dir / "table_reality_audit_003I.csv"
    out_json = run_dir / "table_reality_audit_summary_003I.json"
    out_html = run_dir / "table_reality_audit_003I.html"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, df, summary, media_map)

    print(f"003I status={status}")
    print("by_table_reality_status=" + json.dumps(by_status, ensure_ascii=False))
    print("reject_suggested_by_table_reality_status=" + json.dumps(reject_by_status, ensure_ascii=False))
    if bad_ids:
        print("bad_or_missing_table_ids=" + ",".join(bad_ids))
    if weak_ids:
        print("weak_table_ids=" + ",".join(weak_ids))
    for w in warnings:
        print("warning:", w)
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)
    print("assets", asset_dir)

    # En strict, WARN est acceptable ici, car c'est un audit de diagnostic table.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
