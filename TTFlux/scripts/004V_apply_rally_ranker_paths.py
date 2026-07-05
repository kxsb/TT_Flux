from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd


VERSION = "004V"


NUMERIC_FEATURES = [
    "cand_x",
    "cand_y",
    "area",
    "bbox_w",
    "bbox_h",
    "aspect",
    "fill",
    "compact",
    "candidate_score_004M",
    "patch_gray_mean",
    "patch_gray_std",
    "patch_s_mean",
    "patch_v_mean",
    "patch_motion_mean",
    "patch_motion_max",
]


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def imwrite_unicode(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def build_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)

    for c in NUMERIC_FEATURES:
        if c in df.columns:
            x[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            x[c] = 0.0

    if "pass_name" in df.columns:
        dummies = pd.get_dummies(df["pass_name"].fillna("").astype(str), prefix="pass")
        x = pd.concat([x, dummies], axis=1)

    for c in feature_cols:
        if c not in x.columns:
            x[c] = 0.0

    return x[feature_cols]


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


def score_candidates(cands: list[dict], clf, feature_cols: list[str]) -> list[dict]:
    if not cands:
        return []

    df = pd.DataFrame(cands)
    x = build_features(df, feature_cols)
    scores = clf.predict_proba(x)[:, 1]

    out = []
    for c, s in zip(cands, scores):
        cc = dict(c)
        cc["model_score_004N2"] = float(s)
        out.append(cc)

    out = sorted(out, key=lambda r: float(r["model_score_004N2"]), reverse=True)
    return out


def logit(p):
    p = min(0.999, max(0.001, float(p)))
    return math.log(p / (1.0 - p))


def viterbi(frames: list[dict], topk: int, smooth_lambda: float, speed_cap: float):
    frames2 = []

    for f in frames:
        cands = sorted(f["candidates"], key=lambda c: c["model_score_004N2"], reverse=True)[:topk]
        if cands:
            frames2.append({**f, "candidates": cands})

    if not frames2:
        return []

    dp = []
    back = []

    first = frames2[0]["candidates"]
    dp.append(np.array([logit(c["model_score_004N2"]) for c in first], dtype=np.float64))
    back.append(np.full(len(first), -1, dtype=np.int32))

    for t in range(1, len(frames2)):
        prev_c = frames2[t - 1]["candidates"]
        cur_c = frames2[t]["candidates"]
        prev_dp = dp[-1]

        dt = max(1, int(frames2[t]["frame"] - frames2[t - 1]["frame"]))

        cur_dp = np.full(len(cur_c), -1e18, dtype=np.float64)
        cur_back = np.full(len(cur_c), -1, dtype=np.int32)

        for j, c in enumerate(cur_c):
            cx = float(c["cand_x"])
            cy = float(c["cand_y"])
            unary = logit(c["model_score_004N2"])

            best_score = -1e18
            best_i = -1

            for i, p in enumerate(prev_c):
                px = float(p["cand_x"])
                py = float(p["cand_y"])
                d = math.hypot(cx - px, cy - py)
                speed = d / dt

                penalty = smooth_lambda * min(speed, speed_cap)
                if speed > speed_cap:
                    penalty += smooth_lambda * 2.5 * (speed - speed_cap)

                s = prev_dp[i] + unary - penalty

                if s > best_score:
                    best_score = s
                    best_i = i

            cur_dp[j] = best_score
            cur_back[j] = best_i

        dp.append(cur_dp)
        back.append(cur_back)

    idx = int(np.argmax(dp[-1]))
    path_idx = [idx]

    for t in range(len(frames2) - 1, 0, -1):
        idx = int(back[t][idx])
        path_idx.append(idx)

    path_idx.reverse()

    out = []
    for f, idx in zip(frames2, path_idx):
        c = f["candidates"][idx]
        out.append({
            "frame": int(f["frame"]),
            "x": float(c["cand_x"]),
            "y": float(c["cand_y"]),
            "score": float(c["model_score_004N2"]),
            "pass_name": str(c.get("pass_name", "")),
            "area": int(c.get("area", 0)),
            "bbox_w": int(c.get("bbox_w", 0)),
            "bbox_h": int(c.get("bbox_h", 0)),
        })

    return out


def draw_cross(img, x, y, color, label):
    x = int(round(float(x)))
    y = int(round(float(y)))

    cv2.circle(img, (x, y), 7, color, 2, cv2.LINE_AA)
    cv2.line(img, (x - 12, y), (x + 12, y), color, 2, cv2.LINE_AA)
    cv2.line(img, (x, y - 12), (x, y + 12), color, 2, cv2.LINE_AA)
    cv2.putText(img, label, (x + 9, max(20, y - 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def process_rally(
    row: dict,
    model_bundle: dict,
    out_dir: Path,
    topk: int,
    smooth_lambda: float,
    speed_cap: float,
    max_frames: int,
    make_video: bool,
):
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    video_path = Path(str(row.get("clip_path", "")))
    if not video_path.is_absolute():
        video_path = Path.cwd() / video_path

    review_id = str(row.get("review_id", ""))
    rally_id = str(row.get("rally_id", review_id))
    video_id = str(row.get("video_id", ""))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, [], "open_failed"

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if max_frames > 0:
        frame_limit = min(frame_count, max_frames)
    else:
        frame_limit = frame_count

    frames = []
    top1_rows = []

    prev = None
    fidx = 0

    while fidx < frame_limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        cands = detect_multi_candidates(prev, frame)
        scored = score_candidates(cands, model, feature_cols)

        if scored:
            best = scored[0]
            top1_rows.append({
                "review_id": review_id,
                "rally_id": rally_id,
                "video_id": video_id,
                "frame": fidx,
                "x": round(float(best["cand_x"]), 3),
                "y": round(float(best["cand_y"]), 3),
                "score": round(float(best["model_score_004N2"]), 6),
                "pass_name": str(best.get("pass_name", "")),
                "candidate_count": len(scored),
            })

        frames.append({
            "frame": fidx,
            "candidates": scored,
        })

        prev = frame
        fidx += 1

        if fidx % 250 == 0:
            print(f"    {review_id} frame {fidx}/{frame_limit}")

    cap.release()

    path = viterbi(frames, topk=topk, smooth_lambda=smooth_lambda, speed_cap=speed_cap)

    path_by_frame = {int(p["frame"]): p for p in path}
    top1_by_frame = {int(p["frame"]): p for p in top1_rows}

    out_video_rel = ""

    if make_video:
        cap = cv2.VideoCapture(str(video_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        overlay_path = out_dir / "overlays" / f"{review_id}_{rally_id}_004V_overlay.mp4"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (w, h))

        trail = []

        fidx = 0
        while fidx < frame_limit:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t = top1_by_frame.get(fidx)
            p = path_by_frame.get(fidx)

            if t is not None:
                draw_cross(frame, t["x"], t["y"], (0, 120, 255), f"top1 {t['score']:.2f}")

            if p is not None:
                trail.append((float(p["x"]), float(p["y"])))
                if len(trail) > 24:
                    trail = trail[-24:]

                for a, b in zip(trail[:-1], trail[1:]):
                    cv2.line(
                        frame,
                        (int(round(a[0])), int(round(a[1]))),
                        (int(round(b[0])), int(round(b[1]))),
                        (255, 80, 0),
                        2,
                        cv2.LINE_AA,
                    )

                draw_cross(frame, p["x"], p["y"], (255, 80, 0), f"path {p['score']:.2f}")

            cv2.putText(
                frame,
                f"{review_id} {video_id} f={fidx} orange=top1 blue=path",
                (22, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            fidx += 1

        cap.release()
        writer.release()

        out_video_rel = str(overlay_path)

    path_rows = []
    for p in path:
        path_rows.append({
            "review_id": review_id,
            "rally_id": rally_id,
            "video_id": video_id,
            "frame": int(p["frame"]),
            "time_sec": round(int(p["frame"]) / fps, 4),
            "x": round(float(p["x"]), 3),
            "y": round(float(p["y"]), 3),
            "score": round(float(p["score"]), 6),
            "pass_name": p["pass_name"],
            "area": p["area"],
            "bbox_w": p["bbox_w"],
            "bbox_h": p["bbox_h"],
        })

    speeds = []
    for a, b in zip(path_rows[:-1], path_rows[1:]):
        dt = max(1e-6, float(b["time_sec"]) - float(a["time_sec"]))
        d = math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))
        speeds.append(d / dt)

    scores = [float(p["score"]) for p in path_rows]

    summary = {
        "review_id": review_id,
        "rally_id": rally_id,
        "video_id": video_id,
        "clip_path": str(video_path),
        "frames_video": frame_count,
        "frames_processed": frame_limit,
        "path_points": len(path_rows),
        "top1_points": len(top1_rows),
        "fps": round(fps, 3),
        "duration_processed_sec": round(frame_limit / fps, 3) if fps else "",
        "path_score_med": round(float(np.median(scores)), 6) if scores else "",
        "path_score_p10": round(float(np.percentile(scores, 10)), 6) if scores else "",
        "speed_px_sec_med": round(float(np.median(speeds)), 3) if speeds else "",
        "speed_px_sec_p90": round(float(np.percentile(speeds, 90)), 3) if speeds else "",
        "overlay": out_video_rel,
    }

    return summary, path_rows, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/rally_dataset_004T/rally_manifest_004T.csv")
    ap.add_argument("--model", default="runs/rally_ball_ranker_004N2/ball_ranker_004N.joblib")
    ap.add_argument("--out-dir", default="runs/rally_apply_ranker_004V")
    ap.add_argument("--max-rallies", type=int, default=20)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--smooth-lambda", type=float, default=0.025)
    ap.add_argument("--speed-cap", type=float, default=75.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--make-video", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()

    manifest_path = Path(args.manifest)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)

    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not model_path.is_absolute():
        model_path = root / model_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)

    df = pd.read_csv(manifest_path).fillna("")
    df["activity_score_num"] = pd.to_numeric(df.get("activity_score", 0), errors="coerce").fillna(0.0)

    # Échantillon équilibré : jusqu’à 3 meilleurs rallys par vidéo, puis compléter par score.
    picks = []
    for video_id, g in df.groupby("video_id", dropna=False):
        picks.append(g.sort_values("activity_score_num", ascending=False).head(3))

    sel = pd.concat(picks, ignore_index=False).drop_duplicates("review_id")
    if len(sel) < args.max_rallies:
        rest = df[~df["review_id"].isin(sel["review_id"])]
        sel = pd.concat([sel, rest.sort_values("activity_score_num", ascending=False).head(args.max_rallies - len(sel))])

    sel = sel.sort_values("activity_score_num", ascending=False).head(args.max_rallies).copy()

    print("004V status=RUN")
    print("rallies_selected=", len(sel))
    print("model=", model_path)

    all_path_rows = []
    summary_rows = []

    for i, (_, r) in enumerate(sel.iterrows(), start=1):
        print(f"  rally {i}/{len(sel)} {r.get('review_id')} {r.get('video_id')}")

        summary, path_rows, err = process_rally(
            row=r.to_dict(),
            model_bundle=bundle,
            out_dir=out_dir,
            topk=args.topk,
            smooth_lambda=args.smooth_lambda,
            speed_cap=args.speed_cap,
            max_frames=args.max_frames,
            make_video=args.make_video,
        )

        if summary is None:
            summary_rows.append({
                "review_id": str(r.get("review_id", "")),
                "rally_id": str(r.get("rally_id", "")),
                "video_id": str(r.get("video_id", "")),
                "error": err,
            })
            continue

        summary["error"] = ""
        summary_rows.append(summary)
        all_path_rows.extend(path_rows)

    out_path = out_dir / "rally_ranker_paths_004V.csv"
    out_reviews = out_dir / "rally_ranker_review_summary_004V.csv"
    out_json = out_dir / "rally_ranker_apply_summary_004V.json"

    pd.DataFrame(all_path_rows).to_csv(out_path, index=False, encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(out_reviews, index=False, encoding="utf-8")

    by_video = {}
    for r in summary_rows:
        vid = str(r.get("video_id", ""))
        by_video[vid] = by_video.get(vid, 0) + 1

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "apply_rally_ranker_004N2_to_long_rally_clips_visual_audit",
        "manifest": str(manifest_path),
        "model": str(model_path),
        "out_dir": str(out_dir),
        "rallies": len(summary_rows),
        "path_rows": len(all_path_rows),
        "topk": args.topk,
        "smooth_lambda": args.smooth_lambda,
        "speed_cap": args.speed_cap,
        "max_frames": args.max_frames,
        "make_video": bool(args.make_video),
        "by_video": by_video,
        "next": "Review overlays. If path follows ball enough, run on all 160 rallies and create track-quality review packet.",
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004V status=OK")
    print("rallies=", summary["rallies"])
    print("path_rows=", summary["path_rows"])
    print("by_video=", json.dumps(by_video, ensure_ascii=False))
    print("wrote", out_path)
    print("wrote", out_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
