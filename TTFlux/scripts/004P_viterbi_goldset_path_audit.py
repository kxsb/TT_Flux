from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "004P"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def make_click_key(df: pd.DataFrame) -> pd.Series:
    if "click_key" in df.columns:
        return df["click_key"].astype(str)

    return (
        df["review_id"].astype(str)
        + "|"
        + pd.to_numeric(df["local_frame"], errors="coerce").fillna(-1).astype(int).astype(str)
        + "|"
        + pd.to_numeric(df["click_x"], errors="coerce").round(2).astype(str)
        + "|"
        + pd.to_numeric(df["click_y"], errors="coerce").round(2).astype(str)
    )


def candidate_score_to_logit(p):
    p = float(p)
    p = min(0.999, max(0.001, p))
    return math.log(p / (1.0 - p))


def build_frame_candidates(g: pd.DataFrame, score_col: str, topk: int) -> list[dict]:
    out = []

    gg = g.copy()
    gg[score_col] = to_num(gg[score_col]).fillna(-1e9)
    gg = gg.sort_values(score_col, ascending=False).head(topk)

    for _, r in gg.iterrows():
        score = float(r[score_col])
        if score_col == "model_score_004N":
            unary = candidate_score_to_logit(score)
        else:
            unary = score / 50.0

        out.append({
            "cand_x": float(r["cand_x"]),
            "cand_y": float(r["cand_y"]),
            "score": score,
            "unary": unary,
            "dist_to_click": float(r["dist_to_click"]),
            "rank_input": int(r.get("rank_004M", 999999) or 999999),
            "pass_name": str(r.get("pass_name", "")),
        })

    return out


def viterbi_path(frames: list[dict], smooth_lambda: float, speed_cap: float, accel_lambda: float = 0.0):
    # First-order Viterbi; acceleration penalty kept optional/simple.
    if not frames:
        return []

    dp = []
    back = []

    first = frames[0]["candidates"]
    dp.append(np.array([c["unary"] for c in first], dtype=np.float64))
    back.append(np.full(len(first), -1, dtype=np.int32))

    for t in range(1, len(frames)):
        prev_c = frames[t - 1]["candidates"]
        cur_c = frames[t]["candidates"]
        prev_dp = dp[-1]

        dt = max(1, int(frames[t]["local_frame"] - frames[t - 1]["local_frame"]))

        cur_dp = np.full(len(cur_c), -1e18, dtype=np.float64)
        cur_back = np.full(len(cur_c), -1, dtype=np.int32)

        for j, c in enumerate(cur_c):
            cx, cy = c["cand_x"], c["cand_y"]

            best_score = -1e18
            best_i = -1

            for i, p in enumerate(prev_c):
                px, py = p["cand_x"], p["cand_y"]
                d = math.hypot(cx - px, cy - py)
                speed = d / dt

                # pénalité douce : on autorise les grosses vitesses, mais elles coûtent.
                transition_penalty = smooth_lambda * min(speed, speed_cap)

                # gros saut au-delà du cap : coût supplémentaire.
                if speed > speed_cap:
                    transition_penalty += smooth_lambda * 2.5 * (speed - speed_cap)

                s = prev_dp[i] + c["unary"] - transition_penalty

                if s > best_score:
                    best_score = s
                    best_i = i

            cur_dp[j] = best_score
            cur_back[j] = best_i

        dp.append(cur_dp)
        back.append(cur_back)

    idx = int(np.argmax(dp[-1]))
    path_idx = [idx]

    for t in range(len(frames) - 1, 0, -1):
        idx = int(back[t][idx])
        path_idx.append(idx)

    path_idx.reverse()

    path = []
    for f, i in zip(frames, path_idx):
        c = f["candidates"][i]
        path.append({
            **f,
            "chosen_x": c["cand_x"],
            "chosen_y": c["cand_y"],
            "chosen_score": c["score"],
            "chosen_unary": c["unary"],
            "chosen_input_rank": c["rank_input"],
            "chosen_pass_name": c["pass_name"],
            "chosen_dist": c["dist_to_click"],
        })

    return path


def eval_path(path: list[dict], prefix: str) -> dict:
    if not path:
        return {
            f"{prefix}_frames": 0,
            f"{prefix}_hit10": 0.0,
            f"{prefix}_hit20": 0.0,
            f"{prefix}_hit50": 0.0,
            f"{prefix}_dist_med": None,
        }

    d = np.array([float(x["chosen_dist"]) for x in path], dtype=np.float64)

    return {
        f"{prefix}_frames": int(len(path)),
        f"{prefix}_hit10": round(float(np.mean(d <= 10)), 4),
        f"{prefix}_hit20": round(float(np.mean(d <= 20)), 4),
        f"{prefix}_hit50": round(float(np.mean(d <= 50)), 4),
        f"{prefix}_dist_med": round(float(np.median(d)), 3),
    }


def top1_path(frames: list[dict]) -> list[dict]:
    out = []
    for f in frames:
        if not f["candidates"]:
            continue
        c = f["candidates"][0]
        out.append({
            **f,
            "chosen_x": c["cand_x"],
            "chosen_y": c["cand_y"],
            "chosen_score": c["score"],
            "chosen_unary": c["unary"],
            "chosen_input_rank": c["rank_input"],
            "chosen_pass_name": c["pass_name"],
            "chosen_dist": c["dist_to_click"],
        })
    return out


def aggregate(rows: list[dict], prefix: str) -> dict:
    if not rows:
        return {}

    df = pd.DataFrame(rows)

    def rate(col):
        return round(float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean()), 4)

    def med(col):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        return round(float(vals.median()), 3) if len(vals) else None

    return {
        "reviews": int(df["review_id"].nunique()),
        "frames": int(df["frames"].sum()),
        "hit10": rate(f"{prefix}_hit10"),
        "hit20": rate(f"{prefix}_hit20"),
        "hit50": rate(f"{prefix}_hit50"),
        "dist_med": med(f"{prefix}_dist_med"),
    }


def write_html(path: Path, summary: dict, rows: list[dict]):
    trs = []

    for r in rows:
        cls = "good" if float(r.get("viterbi_hit50", 0) or 0) >= 0.6 else "mid" if float(r.get("viterbi_hit50", 0) or 0) >= 0.3 else "bad"
        trs.append(
            f"<tr class='{cls}'>"
            f"<td>{esc(r.get('review_id'))}</td>"
            f"<td>{esc(r.get('frames'))}</td>"
            f"<td>{esc(r.get('top1_hit50'))}</td>"
            f"<td>{esc(r.get('viterbi_hit50'))}</td>"
            f"<td>{esc(r.get('top1_hit20'))}</td>"
            f"<td>{esc(r.get('viterbi_hit20'))}</td>"
            f"<td>{esc(r.get('top1_dist_med'))}</td>"
            f"<td>{esc(r.get('viterbi_dist_med'))}</td>"
            f"<td>{esc(r.get('topk'))}</td>"
            f"<td>{esc(r.get('smooth_lambda'))}</td>"
            f"<td>{esc(r.get('speed_cap'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004P Viterbi goldset path</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e}}
tr.good td{{background:rgba(116,217,159,.08)}}
tr.mid td{{background:rgba(255,200,80,.08)}}
tr.bad td{{background:rgba(255,80,80,.10)}}
</style>
</head>
<body>
<h1>TTFlux · 004P Viterbi goldset path</h1>
<section><h2>Résumé</h2><pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre></section>
<section>
<h2>Meilleur réglage par review</h2>
<table>
<thead>
<tr>
<th>review</th><th>frames</th>
<th>top1 ≤50</th><th>viterbi ≤50</th>
<th>top1 ≤20</th><th>viterbi ≤20</th>
<th>top1 med</th><th>viterbi med</th>
<th>topK</th><th>smooth</th><th>speed cap</th>
</tr>
</thead>
<tbody>{''.join(trs)}</tbody>
</table>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-scores", default="runs/ball_ranker_004N/candidate_ranker_cv_scores_004N.csv")
    ap.add_argument("--out-dir", default="runs/viterbi_goldset_004P")
    ap.add_argument("--score-col", default="model_score_004N")
    args = ap.parse_args()

    root = Path.cwd()

    cv_path = Path(args.cv_scores)
    if not cv_path.is_absolute():
        cv_path = root / cv_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cv_path).fillna("")

    for c in ["local_frame", "click_x", "click_y", "cand_x", "cand_y", "dist_to_click", args.score_col]:
        if c not in df.columns:
            raise SystemExit(f"Colonne absente: {c}")

    df["local_frame"] = to_num(df["local_frame"]).astype("Int64")
    df["click_x"] = to_num(df["click_x"])
    df["click_y"] = to_num(df["click_y"])
    df["cand_x"] = to_num(df["cand_x"])
    df["cand_y"] = to_num(df["cand_y"])
    df["dist_to_click"] = to_num(df["dist_to_click"])
    df[args.score_col] = to_num(df[args.score_col])

    df = df.dropna(subset=["local_frame", "click_x", "click_y", "cand_x", "cand_y", "dist_to_click", args.score_col]).copy()
    df["local_frame"] = df["local_frame"].astype(int)
    df["click_key_004P"] = make_click_key(df)

    configs = []
    for topk in [10, 20, 30, 50]:
        for smooth in [0.0, 0.01, 0.02, 0.04, 0.08, 0.12]:
            for cap in [25, 45, 70, 110]:
                configs.append((topk, smooth, cap))

    review_best = []
    all_path_rows = []

    print("004P rows=", len(df))
    print("004P reviews=", df["review_id"].nunique())
    print("004P configs=", len(configs))

    for review_id, rg in df.groupby("review_id", dropna=False):
        frame_groups = []

        # Une frame annotée = un click_key, pas juste local_frame.
        for key, kg in rg.groupby("click_key_004P", dropna=False):
            first = kg.iloc[0]
            cands = build_frame_candidates(kg, args.score_col, topk=80)
            if not cands:
                continue

            frame_groups.append({
                "review_id": str(review_id),
                "click_key": str(key),
                "local_frame": int(first["local_frame"]),
                "click_x": float(first["click_x"]),
                "click_y": float(first["click_y"]),
                "candidates": cands,
            })

        frame_groups = sorted(frame_groups, key=lambda x: x["local_frame"])

        if len(frame_groups) < 2:
            continue

        # top1 modèle direct pour comparaison.
        top1 = top1_path([
            {**f, "candidates": sorted(f["candidates"], key=lambda c: c["score"], reverse=True)}
            for f in frame_groups
        ])
        top1_eval = eval_path(top1, "top1")

        best = None
        best_path = None

        for topk, smooth, cap in configs:
            frames = []
            for f in frame_groups:
                cands = sorted(f["candidates"], key=lambda c: c["score"], reverse=True)[:topk]
                frames.append({**f, "candidates": cands})

            path = viterbi_path(frames, smooth_lambda=smooth, speed_cap=cap)
            ev = eval_path(path, "viterbi")

            score = (
                ev["viterbi_hit50"] * 100
                + ev["viterbi_hit20"] * 35
                - (ev["viterbi_dist_med"] or 999) * 0.03
            )

            if best is None or score > best["score_internal"]:
                best = {
                    "review_id": str(review_id),
                    "frames": int(len(path)),
                    "topk": topk,
                    "smooth_lambda": smooth,
                    "speed_cap": cap,
                    "score_internal": round(float(score), 6),
                    **top1_eval,
                    **ev,
                }
                best_path = path

        if best is not None:
            # Flatten names for report.
            best["top1_hit10"] = best.pop("top1_hit10")
            best["top1_hit20"] = best.pop("top1_hit20")
            best["top1_hit50"] = best.pop("top1_hit50")
            best["top1_dist_med"] = best.pop("top1_dist_med")

            best["viterbi_hit10"] = best.pop("viterbi_hit10")
            best["viterbi_hit20"] = best.pop("viterbi_hit20")
            best["viterbi_hit50"] = best.pop("viterbi_hit50")
            best["viterbi_dist_med"] = best.pop("viterbi_dist_med")

            review_best.append(best)

            for p in best_path or []:
                all_path_rows.append({
                    "review_id": str(review_id),
                    "local_frame": p["local_frame"],
                    "click_x": round(p["click_x"], 3),
                    "click_y": round(p["click_y"], 3),
                    "chosen_x": round(p["chosen_x"], 3),
                    "chosen_y": round(p["chosen_y"], 3),
                    "chosen_dist": round(p["chosen_dist"], 3),
                    "chosen_score": round(p["chosen_score"], 6),
                    "chosen_input_rank": p["chosen_input_rank"],
                    "chosen_pass_name": p["chosen_pass_name"],
                    "topk": best["topk"],
                    "smooth_lambda": best["smooth_lambda"],
                    "speed_cap": best["speed_cap"],
                })

    review_df = pd.DataFrame(review_best)
    path_df = pd.DataFrame(all_path_rows)

    if not review_df.empty:
        review_df = review_df.sort_values(["viterbi_hit50", "viterbi_hit20"], ascending=[False, False])

    global_top1 = {
        "reviews": int(len(review_df)),
        "frames": int(review_df["frames"].sum()) if not review_df.empty else 0,
        "hit10": round(float(review_df["top1_hit10"].mean()), 4) if not review_df.empty else 0,
        "hit20": round(float(review_df["top1_hit20"].mean()), 4) if not review_df.empty else 0,
        "hit50": round(float(review_df["top1_hit50"].mean()), 4) if not review_df.empty else 0,
        "dist_med": round(float(review_df["top1_dist_med"].median()), 3) if not review_df.empty else None,
    }

    global_viterbi = {
        "reviews": int(len(review_df)),
        "frames": int(review_df["frames"].sum()) if not review_df.empty else 0,
        "hit10": round(float(review_df["viterbi_hit10"].mean()), 4) if not review_df.empty else 0,
        "hit20": round(float(review_df["viterbi_hit20"].mean()), 4) if not review_df.empty else 0,
        "hit50": round(float(review_df["viterbi_hit50"].mean()), 4) if not review_df.empty else 0,
        "dist_med": round(float(review_df["viterbi_dist_med"].median()), 3) if not review_df.empty else None,
    }

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "goldset_viterbi_path_audit_not_live",
        "cv_scores": str(cv_path),
        "score_col": args.score_col,
        "reviews": int(len(review_df)),
        "path_rows": int(len(path_df)),
        "global_top1_model": global_top1,
        "global_viterbi_best_per_review": global_viterbi,
        "warning": "Best config selected per review; useful diagnostic, optimistic. If positive, freeze one global config next.",
    }

    out_reviews = out_dir / "viterbi_review_scores_004P.csv"
    out_path = out_dir / "viterbi_path_points_004P.csv"
    out_json = out_dir / "viterbi_summary_004P.json"
    out_html = out_dir / "viterbi_004P.html"

    review_df.to_csv(out_reviews, index=False, encoding="utf-8")
    path_df.to_csv(out_path, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, summary, review_best)

    print("004P status=OK")
    print("reviews=", summary["reviews"])
    print("path_rows=", summary["path_rows"])
    print("global_top1_model=", json.dumps(global_top1, ensure_ascii=False))
    print("global_viterbi_best_per_review=", json.dumps(global_viterbi, ensure_ascii=False))
    print("wrote", out_reviews)
    print("wrote", out_path)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
