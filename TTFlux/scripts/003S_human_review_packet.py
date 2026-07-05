from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003S"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def resolve_path(value, root: Path, run_dir: Path) -> Path | None:
    if value is None:
        return None

    raw = str(value).strip().strip('"').strip("'")
    if not raw or raw.lower() == "nan":
        return None

    raw = raw.replace("\\", os.sep)
    p = Path(raw)

    candidates = [p] if p.is_absolute() else [root / p, run_dir / p]

    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            pass

    return None


def copy_media(row, root: Path, run_dir: Path, asset_dir: Path) -> list[dict]:
    rid = str(row["review_id"])
    dst_dir = asset_dir / rid
    dst_dir.mkdir(parents=True, exist_ok=True)

    candidates = []

    for col in ["mp4", "csv"]:
        if col in row.index:
            p = resolve_path(row.get(col), root, run_dir)
            if p:
                candidates.append((col, p))

    # Reprend aussi les médias déjà copiés par 003N.
    old_asset = run_dir / "frozen_shadow_rule_003N_assets" / rid
    if old_asset.is_dir():
        for p in sorted(old_asset.iterdir()):
            if p.is_file():
                kind = "mp4" if p.suffix.lower() == ".mp4" else "csv" if p.suffix.lower() == ".csv" else "asset"
                candidates.append((kind, p))

    copied = []
    seen = set()

    for kind, src in candidates:
        key = str(src).lower()
        if key in seen:
            continue
        seen.add(key)

        dst = dst_dir / f"{len(copied)+1:02d}_{kind}_{src.name}"

        try:
            shutil.copy2(src, dst)
        except Exception:
            continue

        copied.append({
            "kind": kind,
            "src": str(src),
            "rel": os.path.relpath(dst, run_dir).replace("\\", "/"),
            "name": dst.name,
            "suffix": dst.suffix.lower(),
        })

    return copied


def write_html(path: Path, summary: dict, rows: list[dict], media_by_id: dict[str, list[dict]]) -> None:
    cards = []

    for r in rows:
        rid = str(r["review_id"])

        table_rows = []
        for c in [
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
            "003R_status",
            "003R_feature_note",
            "mp4",
            "csv",
        ]:
            if c in r:
                table_rows.append(f"<tr><th>{esc(c)}</th><td>{esc(r.get(c, ''))}</td></tr>")

        media_html = []

        for m in media_by_id.get(rid, []):
            rel = esc(m["rel"])
            label = f"{esc(m['kind'])} · {esc(m['name'])}"

            if m["suffix"] in {".mp4", ".webm", ".mov"}:
                media_html.append(
                    f"<p class='muted'>{label}</p>"
                    f"<video controls preload='metadata' src='{rel}'></video>"
                    f"<p><a href='{rel}'>ouvrir vidéo</a></p>"
                )
            elif m["suffix"] in {".csv"}:
                media_html.append(f"<p><a href='{rel}'>{label}</a></p>")

        cards.append(f"""
<div class="card hit">
<h2>{esc(rid)}</h2>
<div class="grid">
  <div>
    <h3>Données</h3>
    <table><tbody>{''.join(table_rows)}</tbody></table>
  </div>
  <div>
    <h3>Médias</h3>
    {''.join(media_html) if media_html else "<p class='muted'>Aucun média trouvé.</p>"}
    <p class="labelbox"><b>À annoter dans le CSV :</b> reject / keep / partial / unsure</p>
  </div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003S human review packet</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section,.card{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
.card.hit{{border-color:rgba(255,200,80,.75);box-shadow:inset 4px 0 0 rgba(255,200,80,.85)}}
h1,h2,h3{{margin-top:0}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{width:280px;background:#20242e}}
.grid{{display:grid;grid-template-columns:430px 1fr;gap:16px}}
@media(max-width:950px){{.grid{{grid-template-columns:1fr}}}}
video{{display:block;width:900px;max-width:100%;max-height:560px;object-fit:contain;border:1px solid #2b303b;border-radius:10px;background:#05060a;margin-bottom:10px}}
a{{color:#b9cdfa}}
.muted{{color:#aab2c5}}
.labelbox{{background:#101218;border:1px solid #3a4050;border-radius:10px;padding:10px}}
</style>
</head>
<body>
<h1>TTFlux · 003S human review packet</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

<section>
<h2>Consigne annotation</h2>
<pre>Ouvre les vidéos touchées.
Remplis human_label_003S dans le CSV avec :
- reject  = clairement faux tracking / bruit / corps / hors balle
- keep    = tracking exploitable
- partial = exploitable partiellement
- unsure  = doute</pre>
</section>

{''.join(cards)}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", default="frozen_shadow_rule_run_003N.csv")
    args = ap.parse_args()

    root = Path.cwd()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    source = Path(args.source)
    if not source.is_absolute():
        source = run_dir / source

    if not source.is_file():
        raise SystemExit(f"Source introuvable: {source}")

    df = pd.read_csv(source)

    if "003N_shadow_hit" not in df.columns:
        raise SystemExit("Colonne 003N_shadow_hit absente.")

    hits = df[df["003N_shadow_hit"].map(boolish)].copy()

    if "review_id" not in hits.columns:
        hits["review_id"] = [f"R{i+1:04d}" for i in range(len(hits))]

    asset_dir = run_dir / "human_review_packet_003S_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    rows = hits.to_dict("records")

    media_by_id = {}
    for r in rows:
        rid = str(r["review_id"])
        media_by_id[rid] = copy_media(pd.Series(r), root, run_dir, asset_dir)

    template_rows = []

    for r in rows:
        rid = str(r["review_id"])
        media = media_by_id.get(rid, [])
        first_video = ""
        for m in media:
            if m["suffix"] in {".mp4", ".webm", ".mov"}:
                first_video = m["rel"]
                break

        template_rows.append({
            "review_id": rid,
            "human_label_003S": "",
            "comment_003S": "",
            "video_rel": first_video,
            "target_class_003G": r.get("target_class_003G", ""),
            "003N_player_motion_inside_value": r.get("003N_player_motion_inside_value", ""),
            "003N_old_table_distance_value": r.get("003N_old_table_distance_value", ""),
            "003N_max_accel_value": r.get("003N_max_accel_value", ""),
            "mp4": r.get("mp4", ""),
            "csv": r.get("csv", ""),
        })

    out_template = run_dir / "human_review_003S_template.csv"
    out_summary = run_dir / "human_review_packet_summary_003S.json"
    out_html = run_dir / "human_review_packet_003S.html"

    with out_template.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(template_rows[0].keys()))
        writer.writeheader()
        writer.writerows(template_rows)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "human_review_only_no_live_filter_no_delete",
        "run_dir": str(run_dir),
        "source": str(source),
        "rows_source": int(len(df)),
        "hit_total": int(len(hits)),
        "hit_ids": [str(x) for x in hits["review_id"].tolist()],
        "label_template": str(out_template),
        "html": str(out_html),
        "asset_dir": str(asset_dir),
        "warning": "batch_002A features came from 003R proxy bridge, not historical 003D pipeline.",
    }

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, summary, rows, media_by_id)

    print("003S status=OK")
    print("source=", source)
    print("hit_total=", len(hits))
    print("hit_ids=" + ",".join(str(x) for x in hits["review_id"].tolist()))
    print("template=", out_template)
    print("html=", out_html)
    print("summary=", out_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
