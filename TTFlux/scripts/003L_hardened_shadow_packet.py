from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "003L"

RULE = {
    "name": "003L_hardened_v1_player060_dist210_accel20",
    "player_col": "inside_player_motion_mask_ratio_003C",
    "player_min": 0.60,
    "table_distance_col": "point_distance_to_table_px_median",
    "table_distance_max": 210.0,
    "accel_col": "max_accel",
    "accel_min": 20.0,
}

CONFIRMED_REJECT_IDS = {"R0008", "R0009", "R0010", "R0014", "R0020", "R0021"}
PROTECTED_SENTINEL_IDS = {"R0003", "R0006", "R0015", "R0018", "R0019", "R0022"}

MEDIA_COLS = [
    "mp4",
    "player_context_overlay_003C",
    "table_mp4",
    "csv",
    "table_csv",
    "track_csv_resolved_003C",
    "track_csv_resolved_003B2",
    "track_csv_resolved_002G2",
    "overlay_png_rel_003J3",
    "overlay_png_003J3",
]


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def num(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def resolve_path(value, project_root: Path, run_dir: Path) -> Path | None:
    if not isinstance(value, str):
        return None

    raw = value.strip().strip('"').strip("'")
    if not raw:
        return None

    raw = raw.replace("\\", os.sep)
    p = Path(raw)

    candidates = [p] if p.is_absolute() else [project_root / p, run_dir / p]

    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            pass

    return None


def normalize_review_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["review_id", "review_id_003G", "review_id_003F", "review_id_003E"]:
        if c in df.columns:
            df["review_id"] = df[c].astype(str)
            return df
    raise SystemExit("Aucune colonne review_id trouvée.")


def merge_on_review_id(base: pd.DataFrame, other: pd.DataFrame, suffix: str) -> pd.DataFrame:
    other = normalize_review_id(other)

    rename = {}
    for c in other.columns:
        if c == "review_id":
            continue
        if c in base.columns:
            rename[c] = f"{c}_{suffix}"

    other = other.rename(columns=rename)
    return base.merge(other, on="review_id", how="left")


def eval_rule(row: pd.Series) -> dict:
    player = num(row.get(RULE["player_col"]))
    dist = num(row.get(RULE["table_distance_col"]))
    accel = num(row.get(RULE["accel_col"]))

    player_pass = np.isfinite(player) and player >= RULE["player_min"]
    dist_pass = np.isfinite(dist) and dist <= RULE["table_distance_max"]
    accel_pass = np.isfinite(accel) and accel >= RULE["accel_min"]

    hit = bool(player_pass and dist_pass and accel_pass)

    failed = []
    if not player_pass:
        failed.append("player")
    if not dist_pass:
        failed.append("table_distance")
    if not accel_pass:
        failed.append("accel")

    return {
        "003L_hardened_shadow_hit": bool(hit),
        "003L_player_value": player,
        "003L_player_pass": bool(player_pass),
        "003L_player_margin": round(float(player - RULE["player_min"]), 6) if np.isfinite(player) else np.nan,
        "003L_table_distance_value": dist,
        "003L_table_distance_pass": bool(dist_pass),
        "003L_table_distance_margin": round(float(RULE["table_distance_max"] - dist), 3) if np.isfinite(dist) else np.nan,
        "003L_accel_value": accel,
        "003L_accel_pass": bool(accel_pass),
        "003L_accel_margin": round(float(accel - RULE["accel_min"]), 3) if np.isfinite(accel) else np.nan,
        "003L_failed_clauses": ",".join(failed),
    }


def classify(row: pd.Series) -> tuple[str, str]:
    rid = str(row["review_id"])
    hit = boolish(row.get("003L_hardened_shadow_hit"))

    if rid in CONFIRMED_REJECT_IDS:
        if hit:
            return "CONFIRMED_REJECT_CAUGHT", "confirmed_reject_and_rule_hit"
        return "CONFIRMED_REJECT_MISSED", "confirmed_reject_but_rule_miss"

    if rid in PROTECTED_SENTINEL_IDS:
        if hit:
            return "PROTECTED_SENTINEL_CAUGHT_DANGER", "protected_sentinel_rule_hit"
        return "PROTECTED_SENTINEL_SAFE", "protected_sentinel_not_caught"

    if hit:
        return "OTHER_CAUGHT_REVIEW", "unprotected_or_unlabeled_rule_hit"

    return "OTHER_NOT_CAUGHT", "rule_miss"


def copy_media(row: pd.Series, project_root: Path, run_dir: Path, asset_dir: Path) -> list[dict]:
    rid = str(row["review_id"])
    dst_dir = asset_dir / rid
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    seen = set()

    for col in MEDIA_COLS:
        if col not in row.index:
            continue

        p = resolve_path(row.get(col), project_root, run_dir)
        if not p:
            continue

        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)

        safe_name = f"{len(copied)+1:02d}_{col}_{p.name}"
        dst = dst_dir / safe_name

        try:
            shutil.copy2(p, dst)
        except Exception:
            continue

        rel = os.path.relpath(dst, run_dir).replace("\\", "/")
        copied.append({
            "col": col,
            "src": str(p),
            "rel": rel,
            "name": safe_name,
            "suffix": dst.suffix.lower(),
        })

        if len(copied) >= 10:
            break

    return copied


def write_html(path: Path, df: pd.DataFrame, summary: dict, media_by_id: dict[str, list[dict]]) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section,.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card.good{border-color:rgba(116,217,159,.75);box-shadow:inset 4px 0 0 rgba(116,217,159,.85)}
.card.warn{border-color:rgba(255,200,80,.75);box-shadow:inset 4px 0 0 rgba(255,200,80,.85)}
.card.bad{border-color:rgba(255,80,80,.75);box-shadow:inset 4px 0 0 rgba(255,80,80,.85)}
.card.other{border-color:#2b303b}
h1,h2,h3{margin-top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{width:280px;background:#20242e}
.grid{display:grid;grid-template-columns:430px 1fr;gap:16px}
@media(max-width:950px){.grid{grid-template-columns:1fr}}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.good{border-color:rgba(116,217,159,.8);background:rgba(116,217,159,.08)}
.badge.warn{border-color:rgba(255,200,80,.8);background:rgba(255,200,80,.08)}
.badge.bad{border-color:rgba(255,80,80,.8);background:rgba(255,80,80,.08)}
video,img{display:block;width:900px;max-width:100%;max-height:560px;object-fit:contain;border:1px solid #2b303b;border-radius:10px;background:#05060a;margin-bottom:10px}
a{color:#b9cdfa}
.muted{color:#aab2c5}
"""

    priority = {
        "CONFIRMED_REJECT_CAUGHT": 0,
        "CONFIRMED_REJECT_MISSED": 1,
        "PROTECTED_SENTINEL_CAUGHT_DANGER": 2,
        "PROTECTED_SENTINEL_SAFE": 3,
        "OTHER_CAUGHT_REVIEW": 4,
        "OTHER_NOT_CAUGHT": 5,
    }

    work = df.copy()
    work["_prio"] = work["003L_audit_class"].map(lambda x: priority.get(str(x), 99))
    work = work.sort_values(["_prio", "review_id"]).drop(columns=["_prio"])

    cols = [
        "review_id",
        "target_class_003G",
        "003L_audit_class",
        "003L_audit_reason",
        "003L_hardened_shadow_hit",
        "would_reject_shadow_003G",
        "multi_object_shadow_hit_003E",
        "manual_outside_table_003K",
        "manual_overlap_table_003K",
        "inside_ratio_003J3",
        "distance_median_003J3",
        "003L_player_value",
        "003L_player_margin",
        "003L_table_distance_value",
        "003L_table_distance_margin",
        "003L_accel_value",
        "003L_accel_margin",
        "003L_failed_clauses",
    ]

    cards = []

    for _, r in work.iterrows():
        rid = str(r["review_id"])
        cls_name = str(r.get("003L_audit_class", ""))

        if cls_name == "CONFIRMED_REJECT_CAUGHT":
            cls = "card good"
            badge = "badge good"
        elif cls_name in {"CONFIRMED_REJECT_MISSED", "PROTECTED_SENTINEL_CAUGHT_DANGER"}:
            cls = "card bad"
            badge = "badge bad"
        elif cls_name == "PROTECTED_SENTINEL_SAFE":
            cls = "card warn"
            badge = "badge warn"
        else:
            cls = "card other"
            badge = "badge"

        trs = []
        for c in cols:
            if c in r.index:
                trs.append(f"<tr><th>{esc(c)}</th><td>{esc(r.get(c, ''))}</td></tr>")

        media_html = []
        for m in media_by_id.get(rid, []):
            label = f"{esc(m['col'])} · {esc(m['name'])}"
            rel = esc(m["rel"])
            if m["suffix"] in {".mp4", ".webm", ".mov"}:
                media_html.append(f"<div><p class='muted'>{label}</p><video controls preload='metadata' src='{rel}'></video><a href='{rel}'>ouvrir</a></div>")
            elif m["suffix"] in {".png", ".jpg", ".jpeg", ".webp"}:
                media_html.append(f"<div><p class='muted'>{label}</p><a href='{rel}'><img src='{rel}'></a></div>")
            else:
                media_html.append(f"<div><p class='muted'>{label}</p><a href='{rel}'>{esc(m['name'])}</a></div>")

        cards.append(f"""
<div class="{cls}">
<h2>{esc(rid)}</h2>
<div><span class="{badge}">{esc(cls_name)}</span></div>
<div class="grid">
  <div>
    <h3>Audit 003L</h3>
    <table><tbody>{''.join(trs)}</tbody></table>
  </div>
  <div>
    <h3>Médias</h3>
    {''.join(media_html) if media_html else "<p class='muted'>Aucun média copié.</p>"}
  </div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003L hardened shadow packet</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003L hardened shadow packet</h1>

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

    g_csv = run_dir / "shadow_filter_simulation_003G.csv"
    e_csv = run_dir / "multi_object_arbiter_shadow_003E.csv"
    k_csv = run_dir / "rule_safety_audit_003K.csv"
    j3_csv = run_dir / "manual_table_track_sanity_003J3.csv"

    if not g_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {g_csv}")

    df = normalize_review_id(pd.read_csv(g_csv))

    for path, suffix in [(e_csv, "003E"), (k_csv, "003K"), (j3_csv, "003J3")]:
        if path.is_file():
            df = merge_on_review_id(df, pd.read_csv(path), suffix)

    evals = [eval_rule(row) for _, row in df.iterrows()]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(evals).reset_index(drop=True)], axis=1)

    classes = []
    reasons = []
    for _, row in df.iterrows():
        c, r = classify(row)
        classes.append(c)
        reasons.append(r)

    df["003L_audit_class"] = classes
    df["003L_audit_reason"] = reasons

    caught = df[df["003L_hardened_shadow_hit"].map(boolish)]["review_id"].astype(str).tolist()
    caught_confirmed = sorted([x for x in caught if x in CONFIRMED_REJECT_IDS])
    caught_protected = sorted([x for x in caught if x in PROTECTED_SENTINEL_IDS])
    missed_confirmed = sorted(list(CONFIRMED_REJECT_IDS - set(caught_confirmed)))

    by_class = (
        df.groupby("003L_audit_class")["review_id"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    warnings = []
    if missed_confirmed:
        warnings.append("missed confirmed rejects: " + ",".join(missed_confirmed))
    if caught_protected:
        warnings.append("caught protected sentinels: " + ",".join(caught_protected))

    status = "OK_SHADOW_CANDIDATE" if not warnings else "WARN_REVIEW_REQUIRED"

    asset_dir = run_dir / "hardened_shadow_packet_003L_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    # Copie médias pour cas touchés + sentinelles.
    interesting_ids = set(caught) | CONFIRMED_REJECT_IDS | PROTECTED_SENTINEL_IDS
    media_by_id = {}

    for _, row in df.iterrows():
        rid = str(row["review_id"])
        if rid in interesting_ids:
            media_by_id[rid] = copy_media(row, project_root, run_dir, asset_dir)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "policy": "hardened_shadow_only_no_live_filter_no_delete",
        "rule": RULE,
        "rows": int(len(df)),
        "caught_total": int(len(caught)),
        "caught_ids": caught,
        "caught_confirmed_reject_ids": caught_confirmed,
        "missed_confirmed_reject_ids": missed_confirmed,
        "caught_protected_sentinel_ids": caught_protected,
        "audit_class_by_id": by_class,
        "warnings": warnings,
        "recommendation": (
            "If visual packet confirms these 6 and no protected sentinel is caught, "
            "003L can become the preferred shadow rule. Still do not promote live before larger batch."
        ),
        "asset_dir": str(asset_dir),
    }

    out_csv = run_dir / "hardened_shadow_packet_003L.csv"
    out_json = run_dir / "hardened_shadow_packet_summary_003L.json"
    out_html = run_dir / "hardened_shadow_packet_003L.html"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, df, summary, media_by_id)

    print(f"003L status={status}")
    print("caught_ids=" + (",".join(caught) or "-"))
    print("caught_confirmed=" + (",".join(caught_confirmed) or "-"))
    print("missed_confirmed=" + (",".join(missed_confirmed) or "-"))
    print("caught_protected=" + (",".join(caught_protected) or "-"))
    print("warnings=" + (" | ".join(warnings) if warnings else "-"))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    if args.strict and warnings:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
