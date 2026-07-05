from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005F2_resolve_clean_raw_video"


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def safe_video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    info = {
        "path": str(path),
        "exists": path.exists(),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else None,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def read_csv_rows(path: Path, limit=20000):
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, r in enumerate(reader):
            if i >= limit:
                break
            rows.append(r)
        return rows, list(reader.fieldnames or [])


def collect_paths_from_csv(track_csv: Path):
    rows, fields = read_csv_rows(track_csv)
    candidates = set()

    interesting = [
        c for c in fields
        if any(k in c.lower() for k in ["video", "path", "file", "source", "clip", "mp4"])
    ]

    for r in rows:
        for c in interesting:
            v = str(r.get(c, "")).strip().strip('"')
            if ".mp4" in v.lower():
                # Cas simple : valeur entière = chemin.
                if v.lower().endswith(".mp4"):
                    candidates.add(v)
                else:
                    # Cas texte avec chemin dedans.
                    parts = v.replace("\\", "/").split()
                    for p in parts:
                        if ".mp4" in p.lower():
                            candidates.add(p.strip(",;"))

    return candidates


def collect_paths_from_json(root: Path, sequence_key: str, max_files=5000):
    candidates = set()
    scanned = 0

    for p in root.rglob("*.json"):
        scanned += 1
        if scanned > max_files:
            break

        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if sequence_key and sequence_key not in txt:
            continue

        try:
            data = json.loads(txt)
        except Exception:
            continue

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and ".mp4" in v.lower():
                        candidates.add(v)
                    else:
                        walk(v)
            elif isinstance(obj, list):
                for x in obj:
                    walk(x)

        walk(data)

    return candidates


def collect_mp4_by_name(root: Path, sequence_key: str, max_files=20000):
    candidates = set()
    scanned = 0

    for p in root.rglob("*.mp4"):
        scanned += 1
        if scanned > max_files:
            break

        s = str(p)
        name = p.name.lower()

        if sequence_key.lower() in s.lower():
            candidates.add(str(p))
            continue

        # On évite les vidéos overlay/runs sauf si on n'a rien d'autre ensuite.
        if any(k in name for k in ["rly0074", "gap_focus", "s01"]):
            candidates.add(str(p))

    return candidates


def resolve_path(raw: str, root: Path):
    p = Path(raw)

    if p.exists():
        return p

    p2 = root / raw
    if p2.exists():
        return p2

    # Normalisation slash Windows.
    raw2 = raw.replace("\\", "/")
    p3 = root / raw2
    if p3.exists():
        return p3

    return None


def classify_candidate(path: Path):
    s = str(path).lower()
    score = 0
    tags = []

    bad = ["overlay", "audit", "gap", "review", "trajectory", "selected", "contact", "005d", "005e", "005f", "runs"]
    good = ["clips", "raw", "dataset", "videos", "source"]

    for k in good:
        if k in s:
            score += 3
            tags.append(f"good:{k}")

    for k in bad:
        if k in s:
            score -= 2
            tags.append(f"bad:{k}")

    if path.name.lower().endswith(".mp4"):
        score += 1

    return score, "|".join(tags)


def frame_sample(cap, frame_no, width=320):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, img = cap.read()
    if not ok or img is None:
        return None
    h, w = img.shape[:2]
    scale = width / max(1, w)
    out = cv2.resize(img, (width, int(h * scale)))
    return out


def make_contact_sheet(candidates, out_path: Path):
    thumbs = []
    labels = []

    for idx, row in enumerate(candidates[:24]):
        path = Path(row["path"])
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            continue

        frames = row["frames"]
        sample_frames = [
            max(0, min(frames - 1, int(frames * 0.18))),
            max(0, min(frames - 1, int(frames * 0.50))),
            max(0, min(frames - 1, int(frames * 0.82))),
        ]

        local = []
        for fr in sample_frames:
            img = frame_sample(cap, fr, width=300)
            if img is not None:
                local.append(img)

        cap.release()

        if not local:
            continue

        h = max(x.shape[0] for x in local)
        panel = np.zeros((h + 42, 300 * len(local), 3), dtype=np.uint8)

        for i, img in enumerate(local):
            y = 0
            x = i * 300
            panel[y:y+img.shape[0], x:x+img.shape[1]] = img

        label = f"#{idx} score={row['clean_score']} {path.name[:58]}"
        cv2.putText(panel, label, (5, h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)

        thumbs.append(panel)
        labels.append(label)

    if not thumbs:
        return False

    width = max(t.shape[1] for t in thumbs)
    height = sum(t.shape[0] + 12 for t in thumbs)
    sheet = np.zeros((height, width, 3), dtype=np.uint8)

    y = 0
    for t in thumbs:
        sheet[y:y+t.shape[0], 0:t.shape[1]] = t
        y += t.shape[0] + 12

    return imwrite_unicode(out_path, sheet)


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--track-csv", default="runs/rally_ball_table_relink_005D4/ball_points_table_relinked_005D4.csv")
    ap.add_argument("--sequence-key", default="-0bM0t0qS8Q")
    ap.add_argument("--out-dir", default="runs/005F2_resolve_clean_raw_video")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    track_csv = Path(args.track_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_candidates = set()
    raw_candidates |= collect_paths_from_csv(track_csv)
    raw_candidates |= collect_paths_from_json(root / "runs", args.sequence_key)
    raw_candidates |= collect_mp4_by_name(root, args.sequence_key)

    resolved = []
    seen = set()

    for raw in sorted(raw_candidates):
        p = resolve_path(raw, root)
        if p is None:
            continue

        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)

        info = safe_video_info(p)
        if info is None:
            continue

        clean_score, tags = classify_candidate(p)
        info["clean_score"] = clean_score
        info["tags"] = tags
        info["is_probably_overlay"] = int(clean_score < 0)
        resolved.append(info)

    resolved.sort(key=lambda r: (r["clean_score"], r["size_mb"] or 0), reverse=True)

    csv_path = out_dir / "005F2_video_candidates.csv"
    contact_path = out_dir / "005F2_video_candidates_contact_sheet.jpg"
    summary_path = out_dir / "005F2_resolve_clean_raw_video_summary.json"

    fields = [
        "clean_score", "is_probably_overlay", "tags",
        "path", "size_mb", "frames", "fps", "width", "height",
    ]
    write_csv(csv_path, resolved, fields)

    make_contact_sheet(resolved, contact_path)

    summary = {
        "patch": PATCH_ID,
        "root": str(root),
        "track_csv": str(track_csv),
        "sequence_key": args.sequence_key,
        "candidate_count": len(resolved),
        "csv": str(csv_path),
        "contact_sheet": str(contact_path),
        "top_candidates": resolved[:10],
        "note": "Choisir une vidéo sans overlay incrusté. Les vidéos gap_focus/review/runs sont suspectes.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"root      = {root}")
    print(f"track_csv = {track_csv}")
    print(f"sequence  = {args.sequence_key}")
    print("=" * 72)
    print("")
    print("OK 005F2")
    print(f"summary       = {summary_path}")
    print(f"csv           = {csv_path}")
    print(f"contact_sheet = {contact_path}")
    print("")
    print(f"candidate_count={len(resolved)}")
    print("")
    print("TOP candidates:")
    for i, r in enumerate(resolved[:10]):
        print(f"#{i} score={r['clean_score']} frames={r['frames']} {r['width']}x{r['height']} size={r['size_mb']}MB")
        print(f"   {r['path']}")
        print(f"   tags={r['tags']}")
    print("")
    if resolved:
        print("Commande de rendu ensuite, à adapter avec le bon --video :")
        print("python scripts\\005F1_tracklet_only_ball_trajectory_view.py --video \"CHEMIN_VIDEO_CLEAN.mp4\"")
    else:
        print("Aucun candidat trouvé. Il faudra fournir manuellement le chemin du MP4 brut.")


if __name__ == "__main__":
    main()
