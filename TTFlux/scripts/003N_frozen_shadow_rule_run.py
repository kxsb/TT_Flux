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


VERSION = "003N"

DEFAULT_RULE_JSON = Path("runs/batch_001E/shadow_rules/003L_hardened_v1_player060_dist210_accel20.json")

SOURCE_CANDIDATES = [
    "hardened_shadow_packet_003L.csv",
    "shadow_rule_freeze_regression_003M.csv",
    "shadow_filter_simulation_003G.csv",
    "multi_object_arbiter_shadow_003E.csv",
    "multi_object_audit_003D/multi_object_features_003D.csv",
]

MEDIA_COLS = [
    "mp4",
    "player_context_overlay_003C",
    "table_mp4",
    "csv",
    "table_csv",
    "track_csv_resolved_003C",
    "track_csv_resolved_003B2",
    "track_csv_resolved_002G2",
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
    for c in ["review_id", "review_id_003G", "review_id_003F", "review_id_003E", "id"]:
        if c in df.columns:
            df["review_id"] = df[c].astype(str)
            return df
    df["review_id"] = [f"ROW{i+1:04d}" for i in range(len(df))]
    return df


def find_source_csv(run_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = run_dir / p
        if not p.is_file():
            raise SystemExit(f"CSV source introuvable: {p}")
        return p

    for rel in SOURCE_CANDIDATES:
        p = run_dir / rel
        if p.is_file():
            return p

    raise SystemExit(
        "Aucun CSV source trouvé dans le run. Attendus possibles:\n"
        + "\n".join(" - " + str(run_dir / rel) for rel in SOURCE_CANDIDATES)
    )


def load_rule(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Rule JSON introuvable: {path}")

    rule = json.loads(path.read_text(encoding="utf-8"))

    if "clauses" not in rule:
        raise SystemExit("Rule JSON invalide: champ 'clauses' manquant.")

    return rule


def eval_rule(row: pd.Series, rule: dict) -> dict:
    failed = []
    values = {}
    margins = {}

    for clause in rule["clauses"]:
        name = str(clause["name"])
        col = str(clause["column"])
        op = str(clause["op"])
        threshold = float(clause["threshold"])

        value = num(row.get(col))
        values[name] = value

        if not np.isfinite(value):
            ok = False
            margin = np.nan
        elif op == ">=":
            ok = value >= threshold
            margin = value - threshold
        elif op == "<=":
            ok = value <= threshold
            margin = threshold - value
        else:
            raise SystemExit(f"Opérateur non supporté dans règle: {op}")

        margins[name] = margin

        if not ok:
            failed.append(name)

    hit = len(failed) == 0

    out = {
        "003N_shadow_hit": bool(hit),
        "003N_failed_clauses": ",".join(failed),
    }

    for name, value in values.items():
        out[f"003N_{name}_value"] = value

    for name, margin in margins.items():
        out[f"003N_{name}_margin"] = (
            round(float(margin), 6) if np.isfinite(margin) else np.nan
        )

    return out


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

        dst = dst_dir / f"{len(copied)+1:02d}_{col}_{p.name}"

        try:
            shutil.copy2(p, dst)
        except Exception:
            continue

        copied.append({
            "col": col,
            "src": str(p),
            "rel": os.path.relpath(dst, run_dir).replace("\\", "/"),
            "name": dst.name,
            "suffix": dst.suffix.lower(),
        })

        if len(copied) >= 8:
            break

    return copied


def write_html(path: Path, df: pd.DataFrame, summary: dict, media_by_id: dict[str, list[dict]]) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section,.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card.hit{border-color:rgba(255,200,80,.75);box-shadow:inset 4px 0 0 rgba(255,200,80,.85)}
.card.miss{border-color:#2b303b}
h1,h2,h3{margin-top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{width:270px;background:#20242e}
.grid{display:grid;grid-template-columns:430px 1fr;gap:16px}
@media(max-width:950px){.grid{grid-template-columns:1fr}}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.hit{border-color:rgba(255,200,80,.8);background:rgba(255,200,80,.08)}
.badge.miss{border-color:#3a4050}
video,img{display:block;width:900px;max-width:100%;max-height:560px;object-fit:contain;border:1px solid #2b303b;border-radius:10px;background:#05060a;margin-bottom:10px}
a{color:#b9cdfa}
.muted{color:#aab2c5}
"""

    show_cols = [
        "review_id",
        "target_class_003G",
        "003N_shadow_hit",
        "003N_failed_clauses",
        "003N_player_motion_inside_value",
        "003N_player_motion_inside_margin",
        "003N_old_table_distance_value",
        "003N_old_table_distance_margin",
        "003N_max_accel_value",
        "003N_max_accel_margin",
        "mp4",
        "csv",
    ]

    work = df.copy()
    work["_p"] = work["003N_shadow_hit"].map(lambda x: 0 if boolish(x) else 1)
    work = work.sort_values(["_p", "review_id"]).drop(columns=["_p"])

    cards = []

    for _, row in work.iterrows():
        rid = str(row["review_id"])
        hit = boolish(row.get("003N_shadow_hit"))

        cls = "card hit" if hit else "card miss"
        badge = "badge hit" if hit else "badge miss"
        label = "SHADOW_HIT" if hit else "not caught"

        trs = []
        for c in show_cols:
            if c in row.index:
                trs.append(f"<tr><th>{esc(c)}</th><td>{esc(row.get(c, ''))}</td></tr>")

        media_html = []
        for m in media_by_id.get(rid, []):
            rel = esc(m["rel"])
            label_m = f"{esc(m['col'])} · {esc(m['name'])}"

            if m["suffix"] in {".mp4", ".webm", ".mov"}:
                media_html.append(
                    f"<p class='muted'>{label_m}</p>"
                    f"<video controls preload='metadata' src='{rel}'></video>"
                    f"<a href='{rel}'>ouvrir</a>"
                )
            elif m["suffix"] in {".png", ".jpg", ".jpeg", ".webp"}:
                media_html.append(
                    f"<p class='muted'>{label_m}</p>"
                    f"<a href='{rel}'><img src='{rel}'></a>"
                )
            else:
                media_html.append(f"<p><a href='{rel}'>{label_m}</a></p>")

        cards.append(f"""
<div class="{cls}">
<h2>{esc(rid)}</h2>
<div><span class="{badge}">{esc(label)}</span></div>
<div class="grid">
  <div>
    <h3>Données</h3>
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
<title>TTFlux 003N frozen shadow rule run</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003N frozen shadow rule run</h1>

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
    ap.add_argument("--run", required=True, help="run dir, ex: runs/batch_001E ou runs/batch_002A")
    ap.add_argument("--source", default="", help="CSV source relatif au run ou absolu")
    ap.add_argument("--rule-json", default=str(DEFAULT_RULE_JSON))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    project_root = Path.cwd()
    run_dir = Path(args.run)

    rule_json = Path(args.rule_json)
    if not rule_json.is_absolute():
        rule_json = project_root / rule_json

    rule = load_rule(rule_json)
    source_csv = find_source_csv(run_dir, args.source or None)

    df = normalize_review_id(pd.read_csv(source_csv))

    required_cols = [str(c["column"]) for c in rule["clauses"]]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise SystemExit("Colonnes manquantes pour appliquer la règle: " + ",".join(missing))

    evals = [eval_rule(row, rule) for _, row in df.iterrows()]
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(evals).reset_index(drop=True)], axis=1)

    hits = out[out["003N_shadow_hit"].map(boolish)].copy()
    hit_ids = hits["review_id"].astype(str).tolist()

    asset_dir = run_dir / "frozen_shadow_rule_003N_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    media_by_id = {}
    for _, row in hits.iterrows():
        rid = str(row["review_id"])
        media_by_id[rid] = copy_media(row, project_root, run_dir, asset_dir)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "frozen_shadow_rule_run_only_no_live_filter_no_delete",
        "run_dir": str(run_dir),
        "source_csv": str(source_csv),
        "rule_json": str(rule_json),
        "rule_id": rule.get("id", rule.get("name", "unknown_rule")),
        "rule_expression": rule.get("expression", ""),
        "rows": int(len(out)),
        "shadow_hit_total": int(len(hit_ids)),
        "shadow_hit_ids": hit_ids,
        "asset_dir": str(asset_dir),
        "next_step": (
            "Inspect hit videos. If running on a larger batch, manually label hits as reject/keep/partial "
            "before considering any live integration."
        ),
    }

    out_csv = run_dir / "frozen_shadow_rule_run_003N.csv"
    out_json = run_dir / "frozen_shadow_rule_run_summary_003N.json"
    out_html = run_dir / "frozen_shadow_rule_run_003N.html"

    out.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, out, summary, media_by_id)

    print("003N status=OK")
    print("run_dir=", run_dir)
    print("source_csv=", source_csv)
    print("rule_json=", rule_json)
    print("shadow_hit_total=", len(hit_ids))
    print("shadow_hit_ids=" + (",".join(hit_ids) or "-"))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
