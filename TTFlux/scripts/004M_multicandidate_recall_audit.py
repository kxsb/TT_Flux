from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "004M"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def read_frame_pair(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    prev = None
    cur = None

    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx - 1))
        ok, prev = cap.read()
        if not ok:
            prev = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, cur = cap.read()
    if not ok:
        cur = None

    cap.release()
    return prev, cur


def local_patch_stats(gray, hsv, motion, x, y, radius=6):
    h, w = gray.shape[:2]
    x0 = max(0, int(round(x)) - radius)
    x1 = min(w, int(round(x)) + radius + 1)
    y0 = max(0, int(round(y)) - radius)
    y1 = min(h, int(round(y)) + radius + 1)

    patch_g = gray[y0:y1, x0:x1]
    patch_hsv = hsv[y0:y1, x0:x1]
    patch_m = motion[y0:y1, x0:x1] if motion is not None else None

    if patch_g.size == 0:
        return {}

    hh = patch_hsv[:, :, 0]
    ss = patch_hsv[:, :, 1]
    vv = patch_hsv[:, :, 2]

    return {
        "patch_gray_mean": float(np.mean(patch_g)),
        "patch_gray_std": float(np.std(patch_g)),
        "patch_s_mean": float(np.mean(ss)),
        "patch_v_mean": float(np.mean(vv)),
        "patch_motion_mean": float(np.mean(patch_m)) if patch_m is not None and patch_m.size else 0.0,
        "patch_motion_max": float(np.max(patch_m)) if patch_m is not None and patch_m.size else 0.0,
    }


def candidate_components(mask, gray, hsv, motion, pass_name: str, min_area: int, max_area: int):
    out = []

    mask = cv2.medianBlur(mask, 3)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]

        if area < min_area or area > max_area:
            continue
        if bw < 1 or bh < 1 or bw > 34 or bh > 34:
            continue

        aspect = bw / max(1, bh)
        if aspect < 0.20 or aspect > 5.0:
            continue

        cx, cy = cents[i]

        st = local_patch_stats(gray, hsv, motion, cx, cy, radius=6)

        fill = area / max(1, bw * bh)
        compact = 1.0 - min(1.0, abs(aspect - 1.0))
        motion_mean = st.get("patch_motion_mean", 0.0)
        v_mean = st.get("patch_v_mean", 0.0)
        s_mean = st.get("patch_s_mean", 0.0)
        gray_std = st.get("patch_gray_std", 0.0)

        # Score permissif, pas décision finale.
        # On favorise petit objet clair/mobile, mais on garde beaucoup de candidats.
        score = (
            motion_mean * 2.2
            + v_mean * 0.13
            + gray_std * 0.45
            + fill * 7.0
            + compact * 4.0
            - max(0.0, area - 45) * 0.20
            - max(0.0, s_mean - 170) * 0.03
        )

        row = {
            "cand_x": float(cx),
            "cand_y": float(cy),
            "area": int(area),
            "bbox_w": int(bw),
            "bbox_h": int(bh),
            "aspect": round(float(aspect), 4),
            "fill": round(float(fill), 4),
            "compact": round(float(compact), 4),
            "pass_name": pass_name,
            "candidate_score_004M": round(float(score), 6),
        }
        row.update({k: round(float(v), 6) for k, v in st.items()})
        out.append(row)

    return out


def detect_multi_candidates(prev_frame, frame):
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)

    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        motion = cv2.absdiff(prev_gray, gray_blur)
    else:
        motion = np.zeros_like(gray_blur)

    roi = np.zeros_like(gray_blur, dtype=np.uint8)
    roi[int(h * 0.04):int(h * 0.98), int(w * 0.02):int(w * 0.98)] = 255

    # Plusieurs masques, volontairement redondants.
    white_mask = ((vv >= 135) & (ss <= 155)).astype(np.uint8) * 255
    yellow_mask = ((vv >= 120) & (hh >= 10) & (hh <= 48) & (ss >= 25) & (ss <= 220)).astype(np.uint8) * 255
    motion_mask = ((motion >= 6) & (vv >= 90)).astype(np.uint8) * 255
    bright_motion_mask = ((motion >= 4) & (vv >= 125) & (ss <= 210)).astype(np.uint8) * 255
    edge = cv2.Canny(gray_blur, 50, 130)
    edge_bright = ((edge > 0) & (vv >= 115)).astype(np.uint8) * 255

    masks = [
        ("white", cv2.bitwise_and(white_mask, roi), 2, 140),
        ("yellow", cv2.bitwise_and(yellow_mask, roi), 2, 140),
        ("motion", cv2.bitwise_and(motion_mask, roi), 2, 160),
        ("bright_motion", cv2.bitwise_and(bright_motion_mask, roi), 2, 160),
        ("edge_bright", cv2.bitwise_and(edge_bright, roi), 2, 120),
    ]

    candidates = []
    for pass_name, mask, min_area, max_area in masks:
        candidates.extend(candidate_components(mask, gray_blur, hsv, motion, pass_name, min_area, max_area))

    if not candidates:
        return []

    df = pd.DataFrame(candidates)

    # Déduplication spatiale grossière : garder le meilleur candidat dans rayon ~5 px.
    df = df.sort_values("candidate_score_004M", ascending=False).reset_index(drop=True)

    kept = []
    kept_xy = []

    for _, r in df.iterrows():
        x = float(r["cand_x"])
        y = float(r["cand_y"])

        duplicate = False
        for ox, oy in kept_xy:
            if math.hypot(x - ox, y - oy) <= 5.0:
                duplicate = True
                break

        if duplicate:
            continue

        kept.append(r.to_dict())
        kept_xy.append((x, y))

        if len(kept) >= 80:
            break

    return kept


def hit_at_k(cands, click_x, click_y, k, threshold):
    if not cands:
        return False

    sub = cands[:k]
    for c in sub:
        d = math.hypot(float(c["cand_x"]) - click_x, float(c["cand_y"]) - click_y)
        if d <= threshold:
            return True

    return False


def best_distance(cands, click_x, click_y):
    if not cands:
        return None, None

    best_i = None
    best_d = 1e18

    for i, c in enumerate(cands, start=1):
        d = math.hypot(float(c["cand_x"]) - click_x, float(c["cand_y"]) - click_y)
        if d < best_d:
            best_d = d
            best_i = i

    return best_i, best_d


def write_html(path: Path, summary: dict, per_review: pd.DataFrame):
    trs = []

    for _, r in per_review.iterrows():
        cls = "good" if float(r.get("recall50_top20", 0) or 0) >= 0.70 else "mid" if float(r.get("recall50_top20", 0) or 0) >= 0.35 else "bad"
        trs.append(
            f"<tr class='{cls}'>"
            f"<td>{esc(r.get('review_id'))}</td>"
            f"<td>{esc(r.get('video_id'))}</td>"
            f"<td>{esc(r.get('clip_id'))}</td>"
            f"<td>{esc(r.get('clicks'))}</td>"
            f"<td>{esc(r.get('cand_count_med'))}</td>"
            f"<td>{esc(r.get('best_dist_med'))}</td>"
            f"<td>{esc(r.get('best_rank_med'))}</td>"
            f"<td>{esc(r.get('recall20_top20'))}</td>"
            f"<td>{esc(r.get('recall50_top20'))}</td>"
            f"<td>{esc(r.get('recall50_top50'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004M multi-candidate recall</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:75vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
tr.good td{{background:rgba(116,217,159,.08)}}
tr.mid td{{background:rgba(255,200,80,.08)}}
tr.bad td{{background:rgba(255,80,80,.10)}}
</style>
</head>
<body>
<h1>TTFlux · 004M multi-candidate recall</h1>
<section><h2>Résumé</h2><pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre></section>
<section>
<h2>Par segment annoté</h2>
<div class="wrap">
<table>
<thead>
<tr>
<th>review</th><th>video</th><th>clip</th><th>clicks</th><th>cand med</th><th>best dist med</th><th>best rank med</th><th>≤20 top20</th><th>≤50 top20</th><th>≤50 top50</th>
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
    ap.add_argument("--clicks", default="runs/ball_goldset_004J/ball_clicks_004J.csv")
    ap.add_argument("--manifest", default="runs/batch_004F_full240/operational_manifest_001T2.csv")
    ap.add_argument("--run-dir", default="runs/batch_004F_full240")
    ap.add_argument("--out-dir", default="runs/multicandidate_audit_004M")
    ap.add_argument("--limit-clicks", type=int, default=0)
    args = ap.parse_args()

    root = Path.cwd()

    clicks_path = Path(args.clicks)
    manifest_path = Path(args.manifest)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)

    if not clicks_path.is_absolute():
        clicks_path = root / clicks_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    clicks = pd.read_csv(clicks_path).fillna("")
    clicks = clicks[clicks["visibility"].astype(str).eq("ball")].copy()

    for c in ["x", "y", "local_frame"]:
        clicks[c] = to_num(clicks[c])

    clicks = clicks.dropna(subset=["x", "y", "local_frame"])
    clicks["local_frame"] = clicks["local_frame"].astype(int)

    if args.limit_clicks and args.limit_clicks > 0:
        clicks = clicks.head(args.limit_clicks).copy()

    manifest = pd.read_csv(manifest_path).fillna("")
    manifest_by_review = {str(r["review_id"]): r for _, r in manifest.iterrows()}

    all_candidates = []
    click_scores = []

    cache = {}

    print("004M ball_clicks=", len(clicks))

    for idx, (_, click) in enumerate(clicks.iterrows(), start=1):
        review_id = str(click["review_id"])
        if review_id not in manifest_by_review:
            continue

        mrow = manifest_by_review[review_id]
        mp4_rel = str(mrow.get("mp4", ""))
        video_path = run_dir / mp4_rel

        local_frame = int(click["local_frame"])
        click_x = float(click["x"])
        click_y = float(click["y"])

        cache_key = (str(video_path), local_frame)
        if cache_key in cache:
            cands = cache[cache_key]
        else:
            prev, frame = read_frame_pair(video_path, local_frame)
            if frame is None:
                cands = []
            else:
                cands = detect_multi_candidates(prev, frame)
            cache[cache_key] = cands

        ranked = []
        for rank, c in enumerate(cands, start=1):
            d = math.hypot(float(c["cand_x"]) - click_x, float(c["cand_y"]) - click_y)
            row = {
                "review_id": review_id,
                "video_id": str(click.get("video_id", "")),
                "clip_id": str(click.get("clip_id", "")),
                "segment_idx": str(click.get("segment_idx", "")),
                "local_frame": local_frame,
                "click_x": round(click_x, 3),
                "click_y": round(click_y, 3),
                "rank_004M": rank,
                "dist_to_click": round(float(d), 3),
                "label_pos_10": int(d <= 10),
                "label_pos_20": int(d <= 20),
                "label_pos_50": int(d <= 50),
                **c,
            }
            ranked.append(row)

        all_candidates.extend(ranked)

        best_rank, best_dist = best_distance(cands, click_x, click_y)

        click_score = {
            "review_id": review_id,
            "video_id": str(click.get("video_id", "")),
            "clip_id": str(click.get("clip_id", "")),
            "segment_idx": str(click.get("segment_idx", "")),
            "local_frame": local_frame,
            "click_x": round(click_x, 3),
            "click_y": round(click_y, 3),
            "candidate_count": len(cands),
            "best_rank": best_rank if best_rank is not None else "",
            "best_dist": round(float(best_dist), 3) if best_dist is not None else "",
            "hit10_top5": int(hit_at_k(cands, click_x, click_y, 5, 10)),
            "hit20_top5": int(hit_at_k(cands, click_x, click_y, 5, 20)),
            "hit50_top5": int(hit_at_k(cands, click_x, click_y, 5, 50)),
            "hit10_top20": int(hit_at_k(cands, click_x, click_y, 20, 10)),
            "hit20_top20": int(hit_at_k(cands, click_x, click_y, 20, 20)),
            "hit50_top20": int(hit_at_k(cands, click_x, click_y, 20, 50)),
            "hit10_top50": int(hit_at_k(cands, click_x, click_y, 50, 10)),
            "hit20_top50": int(hit_at_k(cands, click_x, click_y, 50, 20)),
            "hit50_top50": int(hit_at_k(cands, click_x, click_y, 50, 50)),
        }

        click_scores.append(click_score)

        if idx % 100 == 0 or idx == len(clicks):
            print(f"  processed {idx}/{len(clicks)} candidates={len(all_candidates)}")

    cand_df = pd.DataFrame(all_candidates)
    click_df = pd.DataFrame(click_scores)

    def rate(col):
        if click_df.empty or col not in click_df.columns:
            return 0.0
        return round(float(pd.to_numeric(click_df[col], errors="coerce").fillna(0).mean()), 4)

    def med(col):
        if click_df.empty or col not in click_df.columns:
            return None
        vals = pd.to_numeric(click_df[col], errors="coerce").dropna()
        return round(float(vals.median()), 3) if len(vals) else None

    per_review_rows = []
    if not click_df.empty:
        for review_id, g in click_df.groupby("review_id", dropna=False):
            first = g.iloc[0]
            per_review_rows.append({
                "review_id": review_id,
                "video_id": first.get("video_id", ""),
                "clip_id": first.get("clip_id", ""),
                "segment_idx": first.get("segment_idx", ""),
                "clicks": int(len(g)),
                "cand_count_med": round(float(pd.to_numeric(g["candidate_count"], errors="coerce").median()), 3),
                "best_dist_med": round(float(pd.to_numeric(g["best_dist"], errors="coerce").median()), 3),
                "best_rank_med": round(float(pd.to_numeric(g["best_rank"], errors="coerce").median()), 3),
                "recall20_top20": round(float(pd.to_numeric(g["hit20_top20"], errors="coerce").mean()), 4),
                "recall50_top20": round(float(pd.to_numeric(g["hit50_top20"], errors="coerce").mean()), 4),
                "recall50_top50": round(float(pd.to_numeric(g["hit50_top50"], errors="coerce").mean()), 4),
            })

    per_review = pd.DataFrame(per_review_rows)

    if not per_review.empty:
        per_review = per_review.sort_values(["recall50_top20", "best_dist_med"], ascending=[False, True])

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "multi_candidate_recall_audit_against_manual_ball_clicks",
        "ball_clicks": int(len(click_df)),
        "candidate_rows": int(len(cand_df)),
        "reviews": int(click_df["review_id"].nunique() if not click_df.empty else 0),
        "global": {
            "candidate_count_med": med("candidate_count"),
            "best_dist_med": med("best_dist"),
            "best_rank_med": med("best_rank"),
            "recall10_top5": rate("hit10_top5"),
            "recall20_top5": rate("hit20_top5"),
            "recall50_top5": rate("hit50_top5"),
            "recall10_top20": rate("hit10_top20"),
            "recall20_top20": rate("hit20_top20"),
            "recall50_top20": rate("hit50_top20"),
            "recall10_top50": rate("hit10_top50"),
            "recall20_top50": rate("hit20_top50"),
            "recall50_top50": rate("hit50_top50"),
        },
        "interpretation": {
            "high_recall50_top20": "candidate reservoir contains the ball often enough; build a ranker/refiner.",
            "low_recall50_top20": "candidate detector still misses the ball; add other candidate generation modes."
        }
    }

    out_candidates = out_dir / "multicandidates_004M.csv"
    out_clicks = out_dir / "click_candidate_recall_004M.csv"
    out_reviews = out_dir / "review_recall_004M.csv"
    out_json = out_dir / "multicandidate_audit_summary_004M.json"
    out_html = out_dir / "multicandidate_audit_004M.html"

    cand_df.to_csv(out_candidates, index=False, encoding="utf-8")
    click_df.to_csv(out_clicks, index=False, encoding="utf-8")
    per_review.to_csv(out_reviews, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, summary, per_review)

    print("004M status=OK")
    print("ball_clicks=", summary["ball_clicks"])
    print("candidate_rows=", summary["candidate_rows"])
    print("reviews=", summary["reviews"])
    print("global=", json.dumps(summary["global"], ensure_ascii=False))
    print("wrote", out_candidates)
    print("wrote", out_clicks)
    print("wrote", out_reviews)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
