from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003Y3"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def clean_label(x) -> str:
    s = str(x or "").strip().lower()
    return s if s in {"reject", "keep", "partial", "unsure"} else ""


def find_col(df: pd.DataFrame, target: str) -> str | None:
    low = {c.lower(): c for c in df.columns}
    if target.lower() in low:
        return low[target.lower()]

    parts = target.lower().split("_")
    for c in df.columns:
        cl = c.lower()
        if all(p in cl for p in parts if p):
            return c

    return None


def read_optional(path: Path, prefix: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "review_id" not in df.columns:
        return pd.DataFrame()

    df["review_id"] = df["review_id"].astype(str)

    rename = {}
    for c in df.columns:
        if c != "review_id":
            rename[c] = f"{prefix}__{c}"

    return df.rename(columns=rename)


def resolve_path(raw, root: Path, run_dir: Path) -> Path | None:
    if raw is None:
        return None

    s = str(raw).strip().strip('"').strip("'")
    if not s or s.lower() == "nan":
        return None

    p = Path(s.replace("\\", os.sep))
    candidates = [p] if p.is_absolute() else [root / p, run_dir / p]

    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            pass

    return None


def copy_assets(row: pd.Series, root: Path, run_dir: Path, asset_dir: Path) -> list[dict]:
    rid = str(row.get("review_id", "")).strip()
    dst_dir = asset_dir / rid
    dst_dir.mkdir(parents=True, exist_ok=True)

    candidates = []

    for col in ["manifest__mp4", "manifest__video", "manifest__video_path", "true003E__mp4", "mp4"]:
        if col in row.index:
            p = resolve_path(row.get(col), root, run_dir)
            if p:
                candidates.append(p)

    copied = []
    seen = set()

    for src in candidates:
        key = str(src).lower()
        if key in seen:
            continue
        seen.add(key)

        dst = dst_dir / f"{len(copied)+1:02d}_{src.name}"
        try:
            shutil.copy2(src, dst)
        except Exception:
            continue

        copied.append({
            "src": str(src),
            "rel": os.path.relpath(dst, run_dir).replace("\\", "/"),
            "suffix": dst.suffix.lower(),
            "name": dst.name,
        })

    return copied


def write_html(path: Path, df: pd.DataFrame, summary: dict, asset_map: dict[str, list[dict]]) -> None:
    cards = []

    for _, r in df.iterrows():
        rid = str(r["review_id"])
        hit = bool(r.get("003Y3_shadow_hit", False))
        label = clean_label(r.get("human_label_003S", ""))

        if hit and label == "reject":
            cls = "true"
        elif hit and label in {"keep", "partial"}:
            cls = "danger"
        elif hit and label == "":
            cls = "unknown"
        elif hit:
            cls = "warn"
        else:
            cls = "safe"

        media = []
        for m in asset_map.get(rid, []):
            rel = esc(m["rel"])
            if m["suffix"] in {".mp4", ".webm", ".mov"}:
                media.append(
                    f"<p>{esc(m['name'])}</p>"
                    f"<video controls preload='metadata' src='{rel}'></video>"
                    f"<p><a href='{rel}'>ouvrir</a></p>"
                )

        trs = []
        for c in [
            "review_id",
            "human_label_003S",
            "comment_003S",
            "003Y3_shadow_hit",
            "003Y3_feature_value",
            "003Y3_feature_col_used",
            "manifest__mp4",
            "manifest__video_path",
        ]:
            if c in r.index:
                trs.append(f"<tr><th>{esc(c)}</th><td>{esc(r.get(c, ''))}</td></tr>")

        cards.append(f"""
<div class="card {cls}">
<h2>{esc(rid)} · hit={esc(hit)} · {esc(label or 'unlabeled')}</h2>
<div class="grid">
  <div><table><tbody>{''.join(trs)}</tbody></table></div>
  <div>{''.join(media) if media else '<p class="muted">Aucune vidéo copiée.</p>'}</div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003Y3 appearance rule run</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section,.card{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
.card.true{{border-color:rgba(116,217,159,.8);box-shadow:inset 4px 0 0 rgba(116,217,159,.9)}}
.card.danger{{border-color:rgba(255,80,80,.85);box-shadow:inset 4px 0 0 rgba(255,80,80,.95)}}
.card.unknown{{border-color:rgba(255,200,80,.85);box-shadow:inset 4px 0 0 rgba(255,200,80,.95)}}
.card.warn{{border-color:rgba(157,180,255,.8);box-shadow:inset 4px 0 0 rgba(157,180,255,.9)}}
.card.safe{{opacity:.55}}
h1,h2{{margin-top:0}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{width:260px;background:#20242e}}
.grid{{display:grid;grid-template-columns:460px 1fr;gap:16px}}
@media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}
video{{display:block;width:900px;max-width:100%;max-height:560px;object-fit:contain;border:1px solid #2b303b;border-radius:10px;background:#05060a}}
a{{color:#b9cdfa}}
.muted{{color:#aab2c5}}
</style>
</head>
<body>
<h1>TTFlux · 003Y3 appearance/logo/shoes shadow run</h1>
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
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--rule-json",
        default="runs/batch_002A/shadow_rules/003Y_center_blob_fill_med_ge_082379.json",
    )
    args = ap.parse_args()

    root = Path.cwd()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    rule_json = Path(args.rule_json)
    if not rule_json.is_absolute():
        rule_json = root / rule_json

    if not rule_json.is_file():
        raise SystemExit(f"Rule JSON introuvable: {rule_json}")

    rule = json.loads(rule_json.read_text(encoding="utf-8-sig"))

    center_path = run_dir / rule.get("feature_source", "center_blob_features_001T2.csv")
    if not center_path.is_file():
        raise SystemExit(f"Center features introuvables: {center_path}")

    df = pd.read_csv(center_path)

    if "review_id" not in df.columns:
        raise SystemExit("review_id absent de center features.")

    df["review_id"] = df["review_id"].astype(str)

    feature_col = find_col(df, rule["feature_col"])
    if not feature_col:
        raise SystemExit(f"Colonne feature introuvable: {rule['feature_col']}")

    threshold = float(rule["threshold"])

    df["003Y3_feature_col_used"] = feature_col
    df["003Y3_feature_value"] = pd.to_numeric(df[feature_col], errors="coerce")
    df["003Y3_shadow_hit"] = df["003Y3_feature_value"] >= threshold

    labels_path = run_dir / "human_review_003S_labels.csv"
    if labels_path.is_file():
        lab = pd.read_csv(labels_path)
        lab["review_id"] = lab["review_id"].astype(str)
        lab["human_label_003S"] = lab["human_label_003S"].map(clean_label)
        df = df.merge(
            lab[["review_id", "human_label_003S", "comment_003S", "updated_at"]],
            on="review_id",
            how="left",
        )
    else:
        df["human_label_003S"] = ""
        df["comment_003S"] = ""

    manifest = read_optional(run_dir / "review_manifest_001T2_trajectory.csv", "manifest")
    true003e = read_optional(run_dir / "multi_object_arbiter_shadow_003E.csv", "true003E")

    for sub in [manifest, true003e]:
        if not sub.empty:
            df = df.merge(sub, on="review_id", how="left")

    hits = df[df["003Y3_shadow_hit"].astype(bool)].copy()

    asset_dir = run_dir / "appearance_shadow_rule_003Y3_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    asset_map = {}
    for _, r in hits.iterrows():
        rid = str(r["review_id"])
        asset_map[rid] = copy_assets(r, root, run_dir, asset_dir)

    hit_labels = hits["human_label_003S"].map(clean_label).value_counts(dropna=False).to_dict()

    dangerous = hits[hits["human_label_003S"].map(clean_label).isin(["keep", "partial"])]
    unknown = hits[hits["human_label_003S"].map(clean_label).eq("")]

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "frozen_appearance_shadow_rule_run_only_no_live_filter_no_delete",
        "run_dir": str(run_dir),
        "rule_json": str(rule_json),
        "rule_id": rule.get("rule_id"),
        "expression": rule.get("expression"),
        "feature_col_used": feature_col,
        "threshold": threshold,
        "rows": int(len(df)),
        "shadow_hit_total": int(len(hits)),
        "shadow_hit_ids": hits["review_id"].astype(str).tolist(),
        "hit_label_counts": hit_labels,
        "dangerous_hit_total": int(len(dangerous)),
        "dangerous_hit_ids": dangerous["review_id"].astype(str).tolist(),
        "unknown_hit_total": int(len(unknown)),
        "unknown_hit_ids": unknown["review_id"].astype(str).tolist(),
        "asset_dir": str(asset_dir),
    }

    out_csv = run_dir / "appearance_shadow_rule_run_003Y3.csv"
    out_json = run_dir / "appearance_shadow_rule_run_summary_003Y3.json"
    out_html = run_dir / "appearance_shadow_rule_run_003Y3.html"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, df, summary, asset_map)

    print("003Y3 status=OK")
    print("run_dir=", run_dir)
    print("shadow_hit_total=", len(hits))
    print("shadow_hit_ids=" + (",".join(summary["shadow_hit_ids"]) or "-"))
    print("hit_label_counts=", hit_labels)
    print("dangerous_hit_total=", summary["dangerous_hit_total"])
    print("dangerous_hit_ids=" + (",".join(summary["dangerous_hit_ids"]) or "-"))
    print("unknown_hit_total=", summary["unknown_hit_total"])
    print("unknown_hit_ids=" + (",".join(summary["unknown_hit_ids"]) or "-"))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
