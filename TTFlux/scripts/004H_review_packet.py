from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "004H"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def pick_balanced(df: pd.DataFrame, per_video: int, total: int) -> pd.DataFrame:
    df = df.copy()

    for c in ["travel", "density", "n_points", "x_range", "y_range"]:
        if c in df.columns:
            df[c] = to_num(df[c]).fillna(0)

    if "video_id" not in df.columns:
        # clip_id contient souvent le video_id à la fin.
        df["video_id"] = df["clip_id"].astype(str).str.split("_").str[-1]

    buckets = []

    for video_id, g in df.groupby("video_id", dropna=False):
        picks = []

        # échantillons extrêmes + moyens
        picks.append(g.sort_values("travel", ascending=False).head(max(1, per_video // 4)))
        picks.append(g.sort_values("travel", ascending=True).head(max(1, per_video // 4)))
        picks.append(g.sort_values("density", ascending=False).head(max(1, per_video // 4)))
        picks.append(g.sort_values("n_points", ascending=False).head(max(1, per_video // 4)))

        mid = g.sample(min(len(g), max(1, per_video // 3)), random_state=2026)
        picks.append(mid)

        sub = pd.concat(picks, ignore_index=False).drop_duplicates("review_id").head(per_video)
        buckets.append(sub)

    out = pd.concat(buckets, ignore_index=False).drop_duplicates("review_id")

    if len(out) < total:
        missing = total - len(out)
        rest = df[~df["review_id"].isin(out["review_id"])]
        if len(rest):
            out = pd.concat([out, rest.sample(min(missing, len(rest)), random_state=2027)], ignore_index=False)

    out = out.head(total).copy()
    out.insert(0, "review_rank_004H", range(1, len(out) + 1))

    return out


def write_html(path: Path, run_dir: Path, rows: pd.DataFrame, summary: dict):
    trs = []

    for _, r in rows.iterrows():
        mp4 = str(r.get("mp4", ""))
        video = f"<video controls preload='metadata' width='300' src='{esc(mp4)}'></video>" if mp4 else ""

        trs.append(
            "<tr>"
            f"<td>{esc(r.get('review_rank_004H'))}</td>"
            f"<td>{esc(r.get('review_id'))}</td>"
            f"<td>{esc(r.get('video_id'))}</td>"
            f"<td>{esc(r.get('clip_id'))}</td>"
            f"<td>{esc(r.get('segment_idx'))}</td>"
            f"<td>{esc(r.get('first_frame'))}</td>"
            f"<td>{esc(r.get('last_frame'))}</td>"
            f"<td>{esc(r.get('n_points'))}</td>"
            f"<td>{esc(r.get('density'))}</td>"
            f"<td>{esc(r.get('travel'))}</td>"
            f"<td>{esc(r.get('guess'))}</td>"
            f"<td>{video}</td>"
            f"<td>{esc(r.get('flags'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004H review packet</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:80vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
video{{border-radius:8px;background:#000}}
</style>
</head>
<body>
<h1>TTFlux · 004H review packet</h1>
<section><h2>Résumé</h2><pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre></section>
<section>
<h2>Échantillon stratifié</h2>
<div class="wrap">
<table>
<thead>
<tr>
<th>rank</th><th>review</th><th>video</th><th>clip</th><th>seg</th><th>first</th><th>last</th><th>n</th><th>density</th><th>travel</th><th>guess</th><th>preview</th><th>flags</th>
</tr>
</thead>
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
    ap.add_argument("--run-dir", default="runs/batch_004F_full240")
    ap.add_argument("--total", type=int, default=120)
    ap.add_argument("--per-video", type=int, default=16)
    ap.add_argument("--out-dir", default="runs/review_packet_004H")
    args = ap.parse_args()

    root = Path.cwd()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = run_dir / "operational_manifest_001T2.csv"
    if not manifest.is_file():
        raise SystemExit(f"Manifest absent: {manifest}")

    df = pd.read_csv(manifest)
    if df.empty:
        raise SystemExit("Manifest vide")

    rows = pick_balanced(df, per_video=args.per_video, total=args.total)

    by_video = rows["video_id"].fillna("").astype(str).value_counts().to_dict()
    by_guess = rows["guess"].fillna("").astype(str).value_counts().to_dict()

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "human_review_packet_only_no_delete_no_filter",
        "source_manifest": str(manifest),
        "source_segments": int(len(df)),
        "packet_segments": int(len(rows)),
        "by_video": by_video,
        "by_guess": by_guess,
        "instruction": "Inspect visually. Labels to use later: keep / partial / reject / unsure.",
    }

    out_csv = out_dir / "review_packet_004H.csv"
    out_json = out_dir / "review_packet_summary_004H.json"
    out_html = out_dir / "review_packet_004H.html"

    rows.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, run_dir, rows, summary)

    print("004H status=OK")
    print("source_segments=", len(df))
    print("packet_segments=", len(rows))
    print("by_video=", json.dumps(by_video, ensure_ascii=False))
    print("by_guess=", json.dumps(by_guess, ensure_ascii=False))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
