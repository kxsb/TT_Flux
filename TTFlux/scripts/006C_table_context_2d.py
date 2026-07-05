from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "006C_table_context_2d_from_clean_segment"


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def draw_text_bg(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1):
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 5, y - th - 8), (x + tw + 5, y + base + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 50.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def read_frame(cap, frame_no):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ok, img = cap.read()
    if not ok or img is None:
        return None
    return img


def blue_table_mask(img):
    """
    Heuristique volontairement simple :
    la table est souvent la plus grande surface bleu/cyan saturée dans la zone centrale.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Bleu/cyan table. Plage large pour compenser compression.
    lower = np.array([85, 35, 35], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # On limite à la scène de jeu, pas score/bords.
    roi = np.zeros_like(mask)
    roi[int(h * 0.18):int(h * 0.82), int(w * 0.08):int(w * 0.92)] = 255
    mask = cv2.bitwise_and(mask, roi)

    # Nettoyage.
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)

    return mask


def largest_component(mask):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None

    best = None
    best_area = 0

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area > best_area:
            best_area = area
            best = i

    if best is None:
        return None

    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def contour_polygon(component_mask):
    cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))

    if area <= 1000:
        return None

    hull = cv2.convexHull(cnt)
    eps = 0.025 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, eps, True)

    pts = approx.reshape(-1, 2).astype(np.float32)

    if len(pts) < 4:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect).astype(np.float32)

    # Si plus de 4 points, on prend le rectangle min area comme fallback stable.
    if len(pts) != 4:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect).astype(np.float32)

    return pts, area


def order_quad_points(pts):
    pts = np.asarray(pts, dtype=np.float32)

    # ordre : top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    return ordered


def quad_quality(quad, area, img_w, img_h):
    xs = quad[:, 0]
    ys = quad[:, 1]
    w = float(xs.max() - xs.min())
    h = float(ys.max() - ys.min())

    if w <= 1 or h <= 1:
        return -999.0

    center_x = float(xs.mean())
    center_y = float(ys.mean())

    # La table devrait être large, pas trop haute, et située vers centre/bas.
    score = 0.0
    score += min(50.0, w / img_w * 80.0)
    score += min(35.0, area / (img_w * img_h) * 600.0)
    score -= abs(center_x - img_w / 2) / img_w * 30.0
    score -= max(0.0, 0.28 * img_h - center_y) / img_h * 40.0
    score -= max(0.0, h - 0.45 * img_h) / img_h * 50.0

    return score


def detect_table_on_frame(img):
    h, w = img.shape[:2]
    mask = blue_table_mask(img)
    comp = largest_component(mask)

    if comp is None:
        return None

    poly = contour_polygon(comp)
    if poly is None:
        return None

    pts, area = poly
    quad = order_quad_points(pts)
    score = quad_quality(quad, area, w, h)

    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))

    return {
        "quad": quad,
        "area": area,
        "score": score,
        "bbox": [int(x), int(y), int(bw), int(bh)],
        "mask_pixels": int(np.count_nonzero(comp)),
    }


def median_quad(detections):
    if not detections:
        return None

    quads = np.stack([d["quad"] for d in detections], axis=0)
    med = np.median(quads, axis=0).astype(np.float32)
    return med


def compute_net_line(quad):
    """
    Approx simple : filet = segment reliant les milieux des deux côtés gauche/droit
    sur la table, entre bord haut et bord bas.
    """
    tl, tr, br, bl = quad

    left_mid = (tl + bl) * 0.5
    right_mid = (tr + br) * 0.5

    # Alternative visuelle : ligne centrale horizontale de perspective.
    return np.array([left_mid, right_mid], dtype=np.float32)


def point_to_poly_distance(px, py, poly):
    p = np.array([px, py], dtype=np.float32)
    poly = np.asarray(poly, dtype=np.float32)

    inside = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(px), float(py)), False)

    min_dist = 1e9
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        denom = float(np.dot(ab, ab)) + 1e-9
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        dist = float(np.linalg.norm(p - proj))
        min_dist = min(min_dist, dist)

    return -min_dist if inside >= 0 else min_dist


def draw_context(img, quad, net_line, label=""):
    out = img.copy()

    q = quad.astype(np.int32)
    cv2.polylines(out, [q.reshape(-1, 1, 2)], True, (0, 255, 255), 3)

    nl = net_line.astype(np.int32)
    cv2.line(out, tuple(nl[0]), tuple(nl[1]), (255, 255, 255), 2)

    for i, p in enumerate(q):
        cv2.circle(out, tuple(p), 7, (0, 255, 255), -1)
        cv2.putText(out, str(i), tuple(p + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    draw_text_bg(out, f"{PATCH_ID} {label}", (12, 28), scale=0.58)

    return out


def make_review_video(video_path: Path, out_video: Path, quad, net_line, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if max_frames is None or max_frames <= 0:
        max_frames = total
    else:
        max_frames = min(max_frames, total)

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire {out_video}")

    written = 0

    while written < max_frames:
        ok, img = cap.read()
        if not ok or img is None:
            break

        out = draw_context(img, quad, net_line, label=f"f={written}")
        writer.write(out)
        written += 1

    cap.release()
    writer.release()

    return {
        "fps": fps,
        "width": w,
        "height": h,
        "total_frames": total,
        "written_frames": written,
        "duration_sec": written / fps if fps else None,
    }


def make_contact_sheet(video_path: Path, frame_nos, quad, net_line, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    thumbs = []

    for fr in frame_nos:
        img = read_frame(cap, fr)
        if img is None:
            continue

        out = draw_context(img, quad, net_line, label=f"f={fr}")
        h, w = out.shape[:2]
        thumb_w = 420
        scale = thumb_w / w
        thumb = cv2.resize(out, (thumb_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)

    cap.release()

    if not thumbs:
        return

    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    th = max(t.shape[0] for t in thumbs)
    tw = max(t.shape[1] for t in thumbs)

    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

    for i, t in enumerate(thumbs):
        r = i // cols
        c = i % cols
        y = r * th
        x = c * tw
        sheet[y:y+t.shape[0], x:x+t.shape[1]] = t

    imwrite_unicode(out_path, sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="runs/005F3_align_clean_video/005F3_clean_aligned_segment.mp4")
    ap.add_argument("--out-dir", default="runs/006C_table_context_2d")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--max-review-frames", type=int, default=500)
    args = ap.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    info = video_info(video_path)
    total = info["frames"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    # Échantillons répartis. On évite les toutes premières/dernières frames.
    frame_nos = []
    for i in range(args.samples):
        t = (i + 0.5) / args.samples
        fr = int(round(t * (total - 1)))
        frame_nos.append(fr)

    detections = []
    failed = []

    for fr in frame_nos:
        img = read_frame(cap, fr)
        if img is None:
            failed.append(fr)
            continue

        det = detect_table_on_frame(img)
        if det is None:
            failed.append(fr)
            continue

        det["frame"] = fr
        detections.append(det)

    cap.release()

    if not detections:
        raise RuntimeError("Aucune table détectée. Il faudra passer en mode manuel.")

    # Garder les détections les mieux scorées avant médiane.
    detections_sorted = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep_n = max(3, min(len(detections_sorted), int(round(len(detections_sorted) * 0.65))))
    kept = detections_sorted[:keep_n]

    quad = median_quad(kept)
    net_line = compute_net_line(quad)

    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    bbox = [int(x), int(y), int(bw), int(bh)]

    context = {
        "patch": PATCH_ID,
        "video": str(video_path),
        "video_info": info,
        "samples_requested": args.samples,
        "sample_frames": frame_nos,
        "detection_count": len(detections),
        "failed_frames": failed,
        "kept_detection_count": len(kept),
        "table_quad_order": "top_left, top_right, bottom_right, bottom_left",
        "table_quad": [[float(x), float(y)] for x, y in quad],
        "table_bbox": bbox,
        "net_line": [[float(x), float(y)] for x, y in net_line],
        "detection_scores": [
            {
                "frame": int(d["frame"]),
                "score": float(d["score"]),
                "area": float(d["area"]),
                "bbox": d["bbox"],
                "mask_pixels": int(d["mask_pixels"]),
            }
            for d in detections_sorted
        ],
        "note": "Contexte table 2D heuristique. À valider visuellement avant features candidates.",
    }

    json_path = out_dir / "006C_table_context_2d.json"
    json_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV corners.
    corners_csv = out_dir / "006C_table_corners.csv"
    with corners_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["corner", "x", "y"])
        wr.writeheader()
        for name, p in zip(["top_left", "top_right", "bottom_right", "bottom_left"], quad):
            wr.writerow({"corner": name, "x": float(p[0]), "y": float(p[1])})

    contact_path = out_dir / "006C_table_context_contact_sheet.jpg"
    review_video = out_dir / "006C_table_context_review.mp4"

    make_contact_sheet(video_path, frame_nos[:12], quad, net_line, contact_path)
    video_meta = make_review_video(video_path, review_video, quad, net_line, max_frames=args.max_review_frames)

    # Ajout méta review.
    context["contact_sheet"] = str(contact_path)
    context["review_video"] = str(review_video)
    context["review_video_meta"] = video_meta
    json_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"video = {video_path}")
    print("=" * 72)
    print("")
    print("OK 006C")
    print(f"context_json = {json_path}")
    print(f"corners_csv  = {corners_csv}")
    print(f"contact      = {contact_path}")
    print(f"review_video = {review_video}")
    print("")
    print(f"detection_count={len(detections)} / samples={args.samples}")
    print(f"kept_detection_count={len(kept)}")
    print(f"table_quad={context['table_quad']}")
    print(f"net_line={context['net_line']}")


if __name__ == "__main__":
    main()
