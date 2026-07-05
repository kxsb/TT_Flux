from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "004U"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def write_html(path: Path, rows: pd.DataFrame, summary: dict):
    trs = []

    for _, r in rows.iterrows():
        mp4 = str(r.get("mp4", ""))
        video = f"<video controls preload='metadata' width='360' src='../../{esc(mp4)}'></video>" if mp4 else ""

        trs.append(
            "<tr>"
            f"<td>{esc(r.get('gold_rank_004I'))}</td>"
            f"<td>{esc(r.get('review_id'))}</td>"
            f"<td>{esc(r.get('video_id'))}</td>"
            f"<td>{esc(r.get('duration_sec'))}</td>"
            f"<td>{esc(r.get('activity_score'))}</td>"
            f"<td>{video}</td>"
            f"<td>{esc(r.get('mp4'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004U rally goldset packet</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:78vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
video{{border-radius:8px;background:#000}}
</style>
</head>
<body>
<h1>TTFlux · 004U rally goldset packet</h1>
<section><h2>Résumé</h2><pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre></section>
<section>
<h2>Rallys à annoter</h2>
<div class="wrap">
<table>
<thead><tr><th>rank</th><th>review</th><th>video</th><th>duration</th><th>activity</th><th>preview</th><th>mp4</th></tr></thead>
<tbody>{''.join(trs)}</tbody>
</table>
</div>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/rally_dataset_004T/rally_manifest_004T.csv")
    ap.add_argument("--out-dir", default="runs/rally_goldset_004U")
    ap.add_argument("--count", type=int, default=45)
    ap.add_argument("--per-video", type=int, default=6)
    args = ap.parse_args()

    root = Path.cwd()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = root / manifest

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest).fillna("")

    for c in ["activity_score", "duration_sec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    picks = []
    for video_id, g in df.groupby("video_id", dropna=False):
        g = g.sort_values("activity_score", ascending=False)
        n = min(args.per_video, len(g))
        picks.append(g.head(n))

    out = pd.concat(picks, ignore_index=False).drop_duplicates("review_id")

    if len(out) < args.count:
        rest = df[~df["review_id"].isin(out["review_id"])]
        rest = rest.sort_values("activity_score", ascending=False)
        out = pd.concat([out, rest.head(args.count - len(out))], ignore_index=False)

    out = out.head(args.count).copy()
    out = out.reset_index(drop=True)

    # Colonnes attendues par 004J.
    out["gold_rank_004I"] = range(1, len(out) + 1)
    out["annotation_status_004I"] = ""
    out["annotation_notes_004I"] = ""

    # 004J utilise first_frame / last_frame pour convertir local_frame -> source_frame.
    # Ici source_frame = frame locale dans le rally clip. C’est ce qu’on veut.
    out["first_frame"] = 0
    out["last_frame"] = pd.to_numeric(out["frame_count"], errors="coerce").fillna(0).astype(int) - 1

    by_video = out["video_id"].astype(str).value_counts().to_dict()

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(manifest),
        "source_rallys": int(len(df)),
        "packet_rallys": int(len(out)),
        "by_video": by_video,
        "instruction": {
            "goal": "Annoter la balle réelle sur des séquences cohérentes de rally.",
            "recommended_clicks_per_rally": "3 à 8 clics, seulement quand la balle est nette.",
            "avoid": "Ne pas cliquer toutes les frames. Ne pas cliquer si balle invisible/floue."
        }
    }

    out_csv = out_dir / "rally_goldset_packet_004U.csv"
    out_json = out_dir / "rally_goldset_packet_summary_004U.json"
    out_html = out_dir / "rally_goldset_packet_004U.html"

    out.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, out, summary)

    print("004U status=OK")
    print("source_rallys=", len(df))
    print("packet_rallys=", len(out))
    print("by_video=", json.dumps(by_video, ensure_ascii=False))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
