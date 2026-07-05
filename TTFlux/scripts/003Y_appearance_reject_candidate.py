from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "003Y"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def clean_label(x) -> str:
    s = str(x or "").strip().lower()
    return s if s in {"reject", "keep", "partial", "unsure"} else ""


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def num(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def find_col(df: pd.DataFrame, exact: list[str], contains: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}

    for e in exact:
        if e.lower() in low:
            return low[e.lower()]

    for c in df.columns:
        cl = c.lower()
        if all(s.lower() in cl for s in contains):
            return c

    return None


def read_csv(path: Path, prefix: str) -> pd.DataFrame:
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


def copy_media_for_row(row: pd.Series, root: Path, run_dir: Path, asset_dir: Path) -> list[dict]:
    rid = str(row.get("review_id", "")).strip()
    dst_dir = asset_dir / rid
    dst_dir.mkdir(parents=True, exist_ok=True)

    candidates = []

    for col in ["manifest__mp4", "true003E__mp4", "proxy003N__mp4", "mp4"]:
        if col in row.index:
            p = resolve_path(row.get(col), root, run_dir)
            if p:
                candidates.append(("mp4", p))

    # assets H264 déjà convertis, si présents
    h264 = run_dir / "human_review_packet_003S2_h264" / f"{rid}_review_h264.mp4"
    if h264.is_file():
        candidates.append(("h264", h264))

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
            "suffix": dst.suffix.lower(),
            "name": dst.name,
        })

    return copied


def write_html(path: Path, df: pd.DataFrame, summary: dict, asset_map: dict[str, list[dict]]) -> None:
    cards = []

    for _, r in df.iterrows():
        rid = str(r["review_id"])
        label = clean_label(r.get("human_label_003S", ""))
        hit = boolish(r.get("003Y_candidate_hit"))

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

        trs = []
        for c in [
            "review_id",
            "human_label_003S",
            "comment_003S",
            "003Y_candidate_hit",
            "003Y_center_fill_value",
            "003Y_micro_distance_value",
            "003Y_center_fill_pass",
            "003Y_micro_distance_pass",
            "center_col_used",
            "micro_col_used",
            "manifest__mp4",
        ]:
            if c in r.index:
                trs.append(f"<tr><th>{esc(c)}</th><td>{esc(r.get(c, ''))}</td></tr>")

        media_html = []
        for m in asset_map.get(rid, []):
            rel = esc(m["rel"])
            if m["suffix"] in {".mp4", ".webm", ".mov"}:
                media_html.append(
                    f"<p>{esc(m['kind'])} · {esc(m['name'])}</p>"
                    f"<video controls preload='metadata' src='{rel}'></video>"
                    f"<p><a href='{rel}'>ouvrir</a></p>"
                )

        cards.append(f"""
<div class="card {cls}">
<h2>{esc(rid)} · {esc(label or 'unlabeled')} · hit={esc(hit)}</h2>
<div class="grid">
  <div><table><tbody>{''.join(trs)}</tbody></table></div>
  <div>{''.join(media_html) if media_html else '<p class="muted">Aucune vidéo copiée.</p>'}</div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003Y appearance reject candidate</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section,.card{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
.card.true{{border-color:rgba(116,217,159,.8);box-shadow:inset 4px 0 0 rgba(116,217,159,.9)}}
.card.danger{{border-color:rgba(255,80,80,.85);box-shadow:inset 4px 0 0 rgba(255,80,80,.95)}}
.card.unknown{{border-color:rgba(255,200,80,.85);box-shadow:inset 4px 0 0 rgba(255,200,80,.95)}}
.card.warn{{border-color:rgba(157,180,255,.8);box-shadow:inset 4px 0 0 rgba(157,180,255,.9)}}
.card.safe{{opacity:.72}}
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
<h1>TTFlux · 003Y appearance/logo/shoes reject candidate</h1>
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
    ap.add_argument("--run", default="runs/batch_002A")
    ap.add_argument("--center-threshold", type=float, default=0.82379)
    ap.add_argument("--micro-threshold", type=float, default=501.8775)
    ap.add_argument("--mode", choices=["center", "micro", "both", "either"], default="center")
    args = ap.parse_args()

    root = Path.cwd()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    labels_path = run_dir / "human_review_003S_labels.csv"
    manifest_path = run_dir / "review_manifest_001T2_trajectory.csv"
    center_path = run_dir / "center_blob_features_001T2.csv"
    micro_path = run_dir / "micro_blob_features_001T2.csv"
    true003e_path = run_dir / "multi_object_arbiter_shadow_003E.csv"

    center = read_csv(center_path, "center")
    micro = read_csv(micro_path, "micro")
    manifest = read_csv(manifest_path, "manifest")
    true003e = read_csv(true003e_path, "true003E")

    if center.empty:
        raise SystemExit(f"Center features introuvables ou invalides: {center_path}")

    df = center.copy()

    for sub in [micro, manifest, true003e]:
        if not sub.empty:
            df = df.merge(sub, on="review_id", how="left")

    if labels_path.is_file():
        labels = pd.read_csv(labels_path)
        labels["review_id"] = labels["review_id"].astype(str)
        labels["human_label_003S"] = labels["human_label_003S"].map(clean_label)
        df = df.merge(
            labels[["review_id", "human_label_003S", "comment_003S", "updated_at"]],
            on="review_id",
            how="left",
        )
    else:
        df["human_label_003S"] = ""
        df["comment_003S"] = ""

    center_col = find_col(
        df,
        exact=["center__center_blob_fill_med"],
        contains=["center", "fill", "med"],
    )

    micro_col = find_col(
        df,
        exact=["micro__micro_distance_med_001O"],
        contains=["micro", "distance", "med"],
    )

    if not center_col:
        raise SystemExit("Colonne center fill median introuvable.")

    df["003Y_center_fill_value"] = pd.to_numeric(df[center_col], errors="coerce")

    if micro_col:
        df["003Y_micro_distance_value"] = pd.to_numeric(df[micro_col], errors="coerce")
    else:
        df["003Y_micro_distance_value"] = pd.NA

    df["003Y_center_fill_pass"] = df["003Y_center_fill_value"] >= args.center_threshold
    df["003Y_micro_distance_pass"] = df["003Y_micro_distance_value"] >= args.micro_threshold

    if args.mode == "center":
        hit = df["003Y_center_fill_pass"]
    elif args.mode == "micro":
        hit = df["003Y_micro_distance_pass"]
    elif args.mode == "both":
        hit = df["003Y_center_fill_pass"] & df["003Y_micro_distance_pass"]
    else:
        hit = df["003Y_center_fill_pass"] | df["003Y_micro_distance_pass"]

    df["003Y_candidate_hit"] = hit.astype(bool)
    df["center_col_used"] = center_col
    df["micro_col_used"] = micro_col or ""

    hits = df[df["003Y_candidate_hit"]].copy()
    labels = df["human_label_003S"].map(clean_label)

    hit_labels = hits["human_label_003S"].map(clean_label).value_counts(dropna=False).to_dict()

    dangerous = hits[hits["human_label_003S"].map(clean_label).isin(["keep", "partial"])]
    unknown = hits[hits["human_label_003S"].map(clean_label).eq("")]

    asset_dir = run_dir / "appearance_reject_candidate_003Y_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    asset_map = {}
    for _, row in hits.iterrows():
        rid = str(row["review_id"])
        asset_map[rid] = copy_media_for_row(row, root, run_dir, asset_dir)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "candidate_shadow_rule_only_no_live_filter_no_delete",
        "run_dir": str(run_dir),
        "mode": args.mode,
        "rule": {
            "center_col": center_col,
            "center_threshold": args.center_threshold,
            "micro_col": micro_col,
            "micro_threshold": args.micro_threshold,
            "expression": {
                "center": f"{center_col} >= {args.center_threshold}",
                "micro": f"{micro_col} >= {args.micro_threshold}" if micro_col else "",
                "mode": args.mode,
            },
        },
        "rows": int(len(df)),
        "hit_total": int(len(hits)),
        "hit_ids": hits["review_id"].astype(str).tolist(),
        "hit_label_counts": hit_labels,
        "dangerous_hit_total": int(len(dangerous)),
        "dangerous_hit_ids": dangerous["review_id"].astype(str).tolist(),
        "unknown_hit_total": int(len(unknown)),
        "unknown_hit_ids": unknown["review_id"].astype(str).tolist(),
        "reject_hit_ids": hits[hits["human_label_003S"].map(clean_label).eq("reject")]["review_id"].astype(str).tolist(),
        "warning": "Only candidate. Needs cross-batch validation and review of unknown hits before freeze.",
    }

    out_csv = run_dir / "appearance_reject_candidate_003Y.csv"
    out_json = run_dir / "appearance_reject_candidate_summary_003Y.json"
    out_html = run_dir / "appearance_reject_candidate_003Y.html"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, df, summary, asset_map)

    print("003Y status=OK")
    print("mode=", args.mode)
    print("rows=", len(df))
    print("hit_total=", len(hits))
    print("hit_ids=" + (",".join(summary["hit_ids"]) or "-"))
    print("hit_label_counts=", hit_labels)
    print("reject_hit_ids=" + (",".join(summary["reject_hit_ids"]) or "-"))
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
