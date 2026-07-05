from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


PRIORITY_IDS = [
    "R0004",  # faux propre, piège principal
    "R0018",  # vraie balle rejetée par géométrie
    "R0008",
    "R0019",
    "R0020",
    "R0014",
    "R0009",
    "R0022",
    "R0015",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in reader]


def find_png_for_review_id(contact_dir: Path, review_id: str) -> Path | None:
    matches = sorted(contact_dir.glob(f"{review_id}_*.png"))
    if matches:
        return matches[0]
    return None


def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img

    scale = width / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def draw_header(width: int, text1: str, text2: str) -> np.ndarray:
    header = np.zeros((74, width, 3), dtype=np.uint8)
    header[:, :] = (18, 21, 29)

    cv2.putText(
        header,
        text1[:140],
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 248, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        header,
        text2[:170],
        (14, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 180, 200),
        1,
        cv2.LINE_AA,
    )

    return header


def pad_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img

    out = np.zeros((h, width, 3), dtype=np.uint8)
    out[:, :] = (10, 12, 18)
    out[:, :w] = img
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="runs/batch_001E/appearance_features_001L.csv")
    parser.add_argument("--contact-dir", default="runs/batch_001E/contact_sheets_001M")
    parser.add_argument("--out-jpg", default="runs/batch_001E/priority_contact_sheets_001M3.jpg")
    parser.add_argument("--max-width", type=int, default=1500)
    parser.add_argument("--quality", type=int, default=88)
    args = parser.parse_args()

    features_path = Path(args.features)
    contact_dir = Path(args.contact_dir)
    out_jpg = Path(args.out_jpg)

    rows = read_csv(features_path)
    by_id = {row.get("review_id", ""): row for row in rows}

    blocks = []
    used = []

    for review_id in PRIORITY_IDS:
        row = by_id.get(review_id)
        if not row:
            print(f"[001M3] missing row: {review_id}")
            continue

        png = find_png_for_review_id(contact_dir, review_id)
        if png is None:
            print(f"[001M3] missing png: {review_id}")
            continue

        img = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[001M3] unreadable png: {png}")
            continue

        img = resize_to_width(img, args.max_width)

        text1 = (
            f"{review_id} | {row.get('human_label_fr', '')} | "
            f"target={row.get('target_class', '')}"
        )
        text2 = (
            f"falseJ={row.get('false_track_score_001J', '')} "
            f"keepJ={row.get('keep_score_001J', '')} "
            f"app={row.get('appearance_guess_001L', '')} | "
            f"{row.get('clip_id', '')} / {row.get('segment_name', '')}"
        )

        header = draw_header(img.shape[1], text1, text2)
        block = np.vstack([header, img])
        blocks.append(block)
        used.append(review_id)

    if not blocks:
        raise SystemExit("[001M3] no blocks generated")

    width = max(b.shape[1] for b in blocks)
    sep = np.zeros((18, width, 3), dtype=np.uint8)
    sep[:, :] = (5, 7, 11)

    padded = []
    for block in blocks:
        padded.append(pad_width(block, width))
        padded.append(sep)

    collage = np.vstack(padded[:-1])

    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(out_jpg),
        collage,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)],
    )

    if not ok:
        raise SystemExit(f"[001M3] failed to write {out_jpg}")

    print(f"[001M3] used ids : {used}")
    print(f"[001M3] wrote    : {out_jpg}")
    print(f"[001M3] shape    : {collage.shape[1]}x{collage.shape[0]}")
    print(f"[001M3] size MB  : {out_jpg.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
    