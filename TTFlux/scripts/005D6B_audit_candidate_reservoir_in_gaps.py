from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "005D6B"

RLY_RE = re.compile(r"(RLY\d{4})", re.IGNORECASE)


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def norm(s: str) -> str:
    return str(s).strip().lower()


def extract_review_id_from_value(v):
    m = RLY_RE.search(str(v))
    if not m:
        return ""
    return m.group(1).upper()


def find_col(cols, candidates):
    low = {c: norm(c) for c in cols}

    for cand in candidates:
        cand = norm(cand)
        for c in cols:
            if low[c] == cand:
                return c

    for cand in candidates:
        cand = norm(cand)
        for c in cols:
            if cand in low[c]:
                return c

    return None


def read_header(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            return next(reader)
    except Exception:
        return []


def inventory_candidate_csvs(search_root: Path, max_files: int):
    rows = []

    patterns = [
        "candidate",
        "candidates",
        "topk",
        "ranker",
        "viterbi",
        "paths",
        "path",
        "tracking",
    ]

    all_csvs = []

    for p in search_root.rglob("*.csv"):
        name = p.name.lower()
        if any(k in name for k in patterns):
            all_csvs.append(p)

    all_csvs = sorted(all_csvs, key=lambda p: p.stat().st_size if p.is_file() else 0, reverse=True)

    for p in all_csvs[:max_files]:
        cols = read_header(p)
        if not cols:
            continue

        review_col = find_col(cols, ["review_id", "rally_id", "clip_id", "sequence_key", "video_id"])
        frame_col = find_col(cols, ["frame_num", "frame", "frame_idx", "local_frame"])
        x_col = find_col(cols, ["x_num", "x", "cx", "ball_x", "candidate_x"])
        y_col = find_col(cols, ["y_num", "y", "cy", "ball_y", "candidate_y"])
        score_col = find_col(cols, ["score", "rank_score", "prob", "confidence", "conf", "ranker_score", "candidate_score"])
        rank_col = find_col(cols, ["rank", "candidate_rank", "topk_rank"])

        name = p.name.lower()

        score = 0
        if "candidate" in name:
            score += 6
        if "topk" in name:
            score += 5
        if "ranker" in name:
            score += 4
        if "paths" in name or "path" in name:
            score += 1
        if review_col:
            score += 3
        if frame_col:
            score += 3
        if x_col and y_col:
            score += 4
        if score_col:
            score += 2
        if rank_col:
            score += 1

        usable = bool(frame_col and x_col and y_col and review_col)

        rows.append({
            "path": str(p),
            "filename": p.name,
            "size_mb": round(float(p.stat().st_size / (1024 * 1024)), 3),
            "usable_for_gap_audit": int(usable),
            "inventory_score": int(score),
            "review_col": review_col or "",
            "frame_col": frame_col or "",
            "x_col": x_col or "",
            "y_col": y_col or "",
            "score_col": score_col or "",
            "rank_col": rank_col or "",
            "columns": "|".join(cols[:80]),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["usable_for_gap_audit", "inventory_score", "size_mb"], ascending=[False, False, False])

    return df


def canonical_review_series(df: pd.DataFrame, review_col: str):
    vals = df[review_col].astype(str)

    extracted = vals.map(extract_review_id_from_value)

    if extracted.ne("").any():
        return extracted

    return vals


def load_candidate_counts_for_gaps(candidate_path: Path, meta: dict, gaps: pd.DataFrame, chunk_size: int):
    review_col = meta["review_col"]
    frame_col = meta["frame_col"]
    score_col = meta.get("score_col", "")

    gap_index = {}

    for _, g in gaps.iterrows():
        review_id = str(g["review_id"])
        start = int(g["gap_start"])
        end = int(g["gap_end"])

        gap_index.setdefault(review_id, []).append((start, end, int(g["_gap_uid"])))

    result = {
        int(uid): {
            "candidate_count": 0,
            "candidate_frames": set(),
            "best_score": None,
        }
        for uid in gaps["_gap_uid"].tolist()
    }

    usecols = [review_col, frame_col]

    if score_col:
        usecols.append(score_col)

    # éviter doublons si review_col == frame_col, etc.
    usecols = list(dict.fromkeys(usecols))

    try:
        chunks = pd.read_csv(candidate_path, usecols=usecols, chunksize=chunk_size, low_memory=False)
    except ValueError:
        chunks = pd.read_csv(candidate_path, chunksize=chunk_size, low_memory=False)

    for chunk in chunks:
        if review_col not in chunk.columns or frame_col not in chunk.columns:
            continue

        chunk = chunk.copy()
        chunk["_review_id"] = canonical_review_series(chunk, review_col)
        chunk["_frame"] = to_num(chunk[frame_col]).fillna(-999999).astype(int)

        needed_reviews = set(gap_index.keys())
        chunk = chunk[chunk["_review_id"].isin(needed_reviews)]

        if chunk.empty:
            continue

        if score_col and score_col in chunk.columns:
            chunk["_score"] = to_num(chunk[score_col])
        else:
            chunk["_score"] = pd.NA

        for review_id, gg in chunk.groupby("_review_id", dropna=False):
            intervals = gap_index.get(str(review_id), [])
            if not intervals:
                continue

            frames = gg["_frame"].to_numpy()

            for start, end, uid in intervals:
                mask = (frames >= start) & (frames <= end)
                if not mask.any():
                    continue

                sub = gg.loc[mask]
                result[uid]["candidate_count"] += int(len(sub))
                result[uid]["candidate_frames"].update([int(x) for x in sub["_frame"].tolist()])

                if "_score" in sub.columns:
                    best = to_num(sub["_score"]).dropna()
                    if len(best):
                        v = float(best.max())
                        if result[uid]["best_score"] is None or v > result[uid]["best_score"]:
                            result[uid]["best_score"] = v

    rows = []

    for uid, r in result.items():
        rows.append({
            "_gap_uid": int(uid),
            "candidate_file": str(candidate_path),
            "candidate_count": int(r["candidate_count"]),
            "candidate_frame_count": int(len(r["candidate_frames"])),
            "candidate_best_score": "" if r["best_score"] is None else round(float(r["best_score"]), 6),
        })

    return pd.DataFrame(rows)


def classify_candidate_audit(row):
    triage = str(row.get("gap_triage_kind_005D6A", ""))
    cand = int(row.get("candidate_count_total_005D6B", 0))

    if triage == "RECOVERABLE_FILTERED_SOURCE":
        return "RECOVER_BY_RELAXING_SCORE_GATE"

    if cand > 0:
        return "CANDIDATES_EXIST_RERANK_OR_VITERBI"

    if triage in {"NON_METRIC_OR_BROADCAST_VIEW", "LIKELY_CAMERA_OR_NON_TABLE_VIEW"}:
        return "BROADCAST_OR_CAMERA_STATE_NOT_TRACKING_FAILURE"

    if triage in {"LIVE_TABLE_NO_SOURCE_POINT", "TABLE_VISIBLE_BUT_NO_METRIC_SOURCE"}:
        return "CANDIDATE_RESERVOIR_EMPTY_OR_NOT_FOUND"

    return "UNKNOWN"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triage", default="runs/rally_ball_gap_triage_005D6A/ball_tracking_gap_triage_005D6A.csv")
    ap.add_argument("--search-root", default="runs")
    ap.add_argument("--out-dir", default="runs/rally_candidate_gap_audit_005D6B")
    ap.add_argument("--max-files", type=int, default=80)
    ap.add_argument("--top-files", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=400000)
    ap.add_argument("--candidate-csv", action="append", default=[])
    args = ap.parse_args()

    root = Path.cwd()

    triage_path = Path(args.triage)
    search_root = Path(args.search_root)
    out_dir = Path(args.out_dir)

    if not triage_path.is_absolute():
        triage_path = root / triage_path
    if not search_root.is_absolute():
        search_root = root / search_root
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    triage = pd.read_csv(triage_path).fillna("")
    triage["_gap_uid"] = range(len(triage))

    target_kinds = {
        "LIVE_TABLE_NO_SOURCE_POINT",
        "TABLE_VISIBLE_BUT_NO_METRIC_SOURCE",
        "RECOVERABLE_FILTERED_SOURCE",
    }

    gaps = triage[triage["gap_triage_kind_005D6A"].astype(str).isin(target_kinds)].copy()

    inventory = inventory_candidate_csvs(search_root, max_files=args.max_files)

    out_inventory = out_dir / "candidate_csv_inventory_005D6B.csv"
    inventory.to_csv(out_inventory, index=False, encoding="utf-8")

    selected = []

    if args.candidate_csv:
        for p in args.candidate_csv:
            cp = Path(p)
            if not cp.is_absolute():
                cp = root / cp

            cols = read_header(cp)
            selected.append({
                "path": str(cp),
                "filename": cp.name,
                "review_col": find_col(cols, ["review_id", "rally_id", "clip_id", "sequence_key", "video_id"]) or "",
                "frame_col": find_col(cols, ["frame_num", "frame", "frame_idx", "local_frame"]) or "",
                "x_col": find_col(cols, ["x_num", "x", "cx", "ball_x", "candidate_x"]) or "",
                "y_col": find_col(cols, ["y_num", "y", "cy", "ball_y", "candidate_y"]) or "",
                "score_col": find_col(cols, ["score", "rank_score", "prob", "confidence", "conf", "ranker_score", "candidate_score"]) or "",
            })
    else:
        if not inventory.empty:
            usable = inventory[inventory["usable_for_gap_audit"].astype(int).eq(1)].copy()
            selected = usable.head(args.top_files).to_dict("records")

    per_file_rows = []

    for meta in selected:
        cpath = Path(str(meta["path"]))

        if not cpath.is_file():
            print("skip missing", cpath)
            continue

        if not meta.get("review_col") or not meta.get("frame_col"):
            print("skip bad_columns", cpath)
            continue

        print("audit candidate file:", cpath)

        try:
            counts = load_candidate_counts_for_gaps(
                candidate_path=cpath,
                meta=meta,
                gaps=gaps,
                chunk_size=args.chunk_size,
            )
            counts["candidate_filename"] = cpath.name
            per_file_rows.append(counts)
        except Exception as e:
            print("candidate audit failed", cpath, repr(e))

    if per_file_rows:
        per_file = pd.concat(per_file_rows, ignore_index=True)
    else:
        per_file = pd.DataFrame(columns=[
            "_gap_uid",
            "candidate_file",
            "candidate_filename",
            "candidate_count",
            "candidate_frame_count",
            "candidate_best_score",
        ])

    out_per_file = out_dir / "candidate_gap_counts_per_file_005D6B.csv"
    per_file.to_csv(out_per_file, index=False, encoding="utf-8")

    agg_rows = []

    if not per_file.empty:
        for uid, g in per_file.groupby("_gap_uid", dropna=False):
            best_scores = to_num(g["candidate_best_score"]).dropna()

            files_with = g[to_num(g["candidate_count"]).fillna(0).astype(int).gt(0)]["candidate_filename"].astype(str).tolist()

            agg_rows.append({
                "_gap_uid": int(uid),
                "candidate_count_total_005D6B": int(to_num(g["candidate_count"]).fillna(0).sum()),
                "candidate_frame_count_total_005D6B": int(to_num(g["candidate_frame_count"]).fillna(0).sum()),
                "candidate_best_score_005D6B": "" if best_scores.empty else round(float(best_scores.max()), 6),
                "candidate_files_with_hits_005D6B": "|".join(files_with[:12]),
            })

    agg = pd.DataFrame(agg_rows)

    merged = triage.merge(agg, on="_gap_uid", how="left")

    for c in [
        "candidate_count_total_005D6B",
        "candidate_frame_count_total_005D6B",
    ]:
        if c not in merged.columns:
            merged[c] = 0
        merged[c] = to_num(merged[c]).fillna(0).astype(int)

    for c in [
        "candidate_best_score_005D6B",
        "candidate_files_with_hits_005D6B",
    ]:
        if c not in merged.columns:
            merged[c] = ""
        merged[c] = merged[c].fillna("")

    merged["candidate_gap_diagnosis_005D6B"] = merged.apply(classify_candidate_audit, axis=1)

    out_gap_audit = out_dir / "ball_tracking_gap_candidate_audit_005D6B.csv"
    out_top = out_dir / "ball_tracking_gap_candidate_audit_top_005D6B.csv"
    out_json = out_dir / "candidate_gap_audit_summary_005D6B.json"

    merged.to_csv(out_gap_audit, index=False, encoding="utf-8")

    top = merged.sort_values(["gap_len", "candidate_count_total_005D6B"], ascending=[False, False]).head(120).copy()
    top.to_csv(out_top, index=False, encoding="utf-8")

    diagnosis_counts = merged["candidate_gap_diagnosis_005D6B"].astype(str).value_counts().to_dict()
    selected_files = [str(x.get("path", "")) for x in selected]

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "audit_candidate_reservoir_availability_inside_tracking_gaps",
        "triage": str(triage_path),
        "search_root": str(search_root),
        "gaps_total": int(len(merged)),
        "target_gaps_audited": int(len(gaps)),
        "inventory_files": int(len(inventory)),
        "selected_candidate_files": selected_files,
        "diagnosis_counts": {str(k): int(v) for k, v in diagnosis_counts.items()},
        "target_candidate_hits": int(
            (
                merged["gap_triage_kind_005D6A"].astype(str).isin(target_kinds)
                & merged["candidate_count_total_005D6B"].astype(int).gt(0)
            ).sum()
        ),
        "target_candidate_empty": int(
            (
                merged["gap_triage_kind_005D6A"].astype(str).isin(target_kinds)
                & merged["candidate_count_total_005D6B"].astype(int).eq(0)
            ).sum()
        ),
        "outputs": {
            "inventory": str(out_inventory),
            "per_file": str(out_per_file),
            "gap_audit": str(out_gap_audit),
            "top": str(out_top),
        },
        "next": "If candidates exist in gaps, integrate candidate-level reranking/Viterbi. If no candidates exist, improve detector/reservoir with motion/upscale/multiscale."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005D6B status=OK")
    print("gaps_total=", summary["gaps_total"])
    print("target_gaps_audited=", summary["target_gaps_audited"])
    print("inventory_files=", summary["inventory_files"])
    print("selected_candidate_files=", json.dumps(summary["selected_candidate_files"], ensure_ascii=False))
    print("diagnosis_counts=", json.dumps(summary["diagnosis_counts"], ensure_ascii=False))
    print("target_candidate_hits=", summary["target_candidate_hits"])
    print("target_candidate_empty=", summary["target_candidate_empty"])
    print("wrote", out_inventory)
    print("wrote", out_per_file)
    print("wrote", out_gap_audit)
    print("wrote", out_top)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
