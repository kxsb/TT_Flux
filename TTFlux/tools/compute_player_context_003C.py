from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PLAYER_FEATURES = [
    "has_player_context_003C",
    "player_motion_area_ratio_003C",
    "player_motion_component_count_003C",
    "player_motion_bbox_area_ratio_003C",
    "inside_player_motion_mask_ratio_003C",
    "near_player_motion_mask_ratio_003C",
    "inside_hand_or_racket_proxy_ratio_003C",
    "near_foot_or_floor_proxy_ratio_003C",
    "occlusion_entry_exit_count_003C",
    "candidate_on_body_risk_003C",
    "player_context_confidence_003C",
]


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value or "").strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in csv.DictReader(f, dialect=dialect)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "review_id",
        "target_class",
        "final_decision_002B",
        "proposed_decision_003B4",
        "contextual_reject_003B4",
        "clip_id_003B2",
        "segment_name",
        "first_frame_003C",
        "last_frame_003C",
        "track_csv_resolved_003C",
        *PLAYER_FEATURES,
    ]

    cols = []
    seen = set()

    for c in preferred:
        cols.append(c)
        seen.add(c)

    for row in rows:
        for k in row:
            if k not in seen:
                cols.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 92) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def resolve_existing_path(value: str, roots: list[Path]) -> Path | None:
    value = str(value or "").strip().strip('"')

    if not value:
        return None

    p = Path(value)

    if p.is_absolute() and p.exists():
        return p

    for root in roots:
        q = root / p
        if q.exists():
            return q

    return None


def infer_frame_range(row: dict[str, Any]) -> tuple[int | None, int | None]:
    for a, b in [
        ("first_frame", "last_frame"),
        ("frame_first", "frame_last"),
        ("start_frame", "end_frame"),
        ("first_frame_003C", "last_frame_003C"),
    ]:
        if row.get(a) and row.get(b):
            try:
                return int(float(row[a])), int(float(row[b]))
            except Exception:
                pass

    blob = " ".join(str(v) for v in row.values())
    m = re.search(r"f(\d+)_to_f(\d+)", blob)

    if m:
        return int(m.group(1)), int(m.group(2))

    return None, None


def find_track_csv(row: dict[str, Any], roots: list[Path], run_dir: Path) -> Path | None:
    for col in [
        "track_csv_resolved_003B2",
        "track_csv_resolved_002G2",
        "track_csv_resolved",
        "validation_csv",
        "segment_csv",
        "csv_path",
        "csv",
    ]:
        p = resolve_existing_path(str(row.get(col, "")), roots)
        if p is not None:
            return p

    segment_name = str(row.get("segment_name", "")).strip()

    if not segment_name:
        return None

    hits = [p for p in run_dir.rglob(f"{segment_name}.csv") if p.is_file()]

    if hits:
        return sorted(hits, key=lambda p: len(str(p)))[0]

    hits = [p for p in run_dir.rglob("*.csv") if segment_name in p.name]

    if hits:
        return sorted(hits, key=lambda p: len(str(p)))[0]

    return None


def load_track_points(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.exists():
        return []

    rows = read_csv(path)
    points = []

    for r in rows:
        frame = fnum(r.get("frame") or r.get("frame_idx") or r.get("f"), None)
        x = fnum(r.get("x") or r.get("cx") or r.get("ball_x"), None)
        y = fnum(r.get("y") or r.get("cy") or r.get("ball_y"), None)

        if frame is None or x is None or y is None:
            continue

        points.append({
            "frame": float(frame),
            "x": float(x),
            "y": float(y),
        })

    dedup: dict[int, dict[str, float]] = {}

    for p in points:
        dedup[int(round(p["frame"]))] = p

    return [dedup[k] for k in sorted(dedup)]


def load_payload(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    clips = data.get("clips", [])

    out = {}

    for clip in clips:
        if not isinstance(clip, dict):
            continue

        cid = str(clip.get("clip_id", ""))
        if cid:
            out[cid] = clip

    return out


def load_table_models(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    data = read_json(path)
    models = data.get("table_models", {})

    return models if isinstance(models, dict) else {}


def read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if frame_count > 0:
        frame_idx = max(0, min(int(frame_idx), frame_count - 1))
    else:
        frame_idx = max(0, int(frame_idx))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()

    if not ok or img is None:
        return None

    return img


def sample_frames(video_path: Path, first_frame: int, last_frame: int, sample_count: int = 11) -> tuple[list[int], list[np.ndarray]]:
    if last_frame < first_frame:
        first_frame, last_frame = last_frame, first_frame

    if last_frame == first_frame:
        indices = [first_frame]
    else:
        indices = np.linspace(first_frame, last_frame, sample_count)
        indices = [int(round(x)) for x in indices]

    indices = sorted(set(indices))
    frames = []

    for idx in indices:
        img = read_video_frame(video_path, idx)
        if img is not None:
            frames.append(img)

    return indices[:len(frames)], frames


def valid_point(p: Any) -> bool:
    return isinstance(p, dict) and p.get("x") is not None and p.get("y") is not None


def table_mask_from_model(model: dict[str, Any] | None, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if not model or not model.get("has_table_model"):
        return mask

    quad = model.get("table_quad")

    if not isinstance(quad, list) or len(quad) < 4 or not all(valid_point(p) for p in quad[:4]):
        return mask

    pts = np.array([[float(p["x"]), float(p["y"])] for p in quad[:4]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    return mask


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask, k, iterations=1)


def clean_motion_mask(mask: np.ndarray, min_area: int, max_area_ratio: float = 0.45) -> tuple[np.ndarray, int, float]:
    h, w = mask.shape[:2]
    total = max(1, h * w)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = np.zeros_like(mask)
    comp_count = 0
    bbox_area = 0

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        if area < min_area:
            continue

        if area / total > max_area_ratio:
            continue

        comp = (labels == i)
        out[comp] = 255
        comp_count += 1
        bbox_area += bw * bh

    bbox_area_ratio = bbox_area / total

    return out, comp_count, bbox_area_ratio


def build_player_motion_masks(
    frames: list[np.ndarray],
    table_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not frames:
        return (
            np.zeros((1, 1), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
            {
                "motion_area_ratio": 0.0,
                "component_count": 0,
                "bbox_area_ratio": 0.0,
                "confidence": 0.0,
            },
        )

    h, w = frames[0].shape[:2]

    if len(frames) < 2:
        base = np.zeros((h, w), dtype=np.uint8)
    else:
        masks = []

        grays = [
            cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            for f in frames
        ]

        for a, b in zip(grays[:-1], grays[1:]):
            diff = cv2.absdiff(a, b)
            diff = cv2.GaussianBlur(diff, (7, 7), 0)

            otsu_thresh, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold = max(14.0, min(42.0, float(otsu_thresh)))
            m = (diff >= threshold).astype(np.uint8) * 255

            masks.append(m)

        base = np.maximum.reduce(masks) if masks else np.zeros((h, w), dtype=np.uint8)

    # Nettoyage morphologique.
    k_open = np.ones((3, 3), np.uint8)
    k_close = np.ones((9, 9), np.uint8)

    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, k_open, iterations=1)
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, k_close, iterations=2)

    # On retire la surface de table pour éviter de confondre table/logos/reflets avec joueur.
    if table_mask is not None and table_mask.shape == base.shape and table_mask.max() > 0:
        table_core = cv2.erode(table_mask, np.ones((7, 7), np.uint8), iterations=1)
        base[table_core > 0] = 0

    min_area = max(350, int(0.0008 * h * w))
    player_mask, comp_count, bbox_area_ratio = clean_motion_mask(base, min_area=min_area)

    player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)

    near_player_mask = dilate(player_mask, radius=35)

    motion_area_ratio = float((player_mask > 0).mean())

    # Confiance : trop peu ou trop de mouvement = moins fiable.
    if motion_area_ratio <= 0.002:
        confidence = 0.25
    elif motion_area_ratio >= 0.30:
        confidence = 0.35
    else:
        confidence = 0.75

    meta = {
        "motion_area_ratio": round(motion_area_ratio, 6),
        "component_count": int(comp_count),
        "bbox_area_ratio": round(float(bbox_area_ratio), 6),
        "confidence": round(float(confidence), 6),
    }

    return player_mask, near_player_mask, base, meta


def make_proxy_masks(
    player_mask: np.ndarray,
    near_player_mask: np.ndarray,
    table_mask: np.ndarray,
    table_model: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = player_mask.shape[:2]

    if table_mask is not None and table_mask.shape == player_mask.shape and table_mask.max() > 0:
        table_near = dilate(table_mask, radius=130)
        hand_proxy = cv2.bitwise_and(near_player_mask, table_near)

        # Zone pieds/sol : en-dessous du bord avant de table.
        quad = table_model.get("table_quad") if isinstance(table_model, dict) else None
        if isinstance(quad, list) and len(quad) >= 2 and all(valid_point(p) for p in quad[:2]):
            front_y = max(float(quad[0]["y"]), float(quad[1]["y"]))
            floor_y = int(max(0, min(h - 1, front_y + 25)))
        else:
            floor_y = int(h * 0.58)

        floor_zone = np.zeros((h, w), dtype=np.uint8)
        floor_zone[floor_y:, :] = 255
        foot_proxy = cv2.bitwise_and(near_player_mask, floor_zone)
    else:
        # Fallback sans table : bras/mains dans zone centrale/haute, pieds dans zone basse.
        upper = np.zeros((h, w), dtype=np.uint8)
        upper[int(h * 0.15):int(h * 0.72), :] = 255
        hand_proxy = cv2.bitwise_and(near_player_mask, upper)

        lower = np.zeros((h, w), dtype=np.uint8)
        lower[int(h * 0.58):, :] = 255
        foot_proxy = cv2.bitwise_and(near_player_mask, lower)

    return hand_proxy, foot_proxy


def point_in_mask(mask: np.ndarray, x: float, y: float) -> int:
    h, w = mask.shape[:2]
    xi = int(round(x))
    yi = int(round(y))

    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return 0

    return 1 if mask[yi, xi] > 0 else 0


def transitions(values: list[int]) -> int:
    if not values:
        return 0

    n = 0
    prev = values[0]

    for v in values[1:]:
        if v != prev:
            n += 1
            prev = v

    return n


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_features(
    points: list[dict[str, float]],
    player_mask: np.ndarray,
    near_player_mask: np.ndarray,
    hand_proxy: np.ndarray,
    foot_proxy: np.ndarray,
    meta: dict[str, Any],
) -> dict[str, Any]:
    if not points:
        return {
            "has_player_context_003C": 0,
            "player_motion_area_ratio_003C": meta.get("motion_area_ratio", 0.0),
            "player_motion_component_count_003C": meta.get("component_count", 0),
            "player_motion_bbox_area_ratio_003C": meta.get("bbox_area_ratio", 0.0),
            "inside_player_motion_mask_ratio_003C": "",
            "near_player_motion_mask_ratio_003C": "",
            "inside_hand_or_racket_proxy_ratio_003C": "",
            "near_foot_or_floor_proxy_ratio_003C": "",
            "occlusion_entry_exit_count_003C": "",
            "candidate_on_body_risk_003C": "",
            "player_context_confidence_003C": meta.get("confidence", 0.0),
        }

    inside = []
    near = []
    hand = []
    foot = []

    for p in points:
        x = float(p["x"])
        y = float(p["y"])

        inside.append(point_in_mask(player_mask, x, y))
        near.append(point_in_mask(near_player_mask, x, y))
        hand.append(point_in_mask(hand_proxy, x, y))
        foot.append(point_in_mask(foot_proxy, x, y))

    inside_ratio = mean([float(x) for x in inside])
    near_ratio = mean([float(x) for x in near])
    hand_ratio = mean([float(x) for x in hand])
    foot_ratio = mean([float(x) for x in foot])
    trans = transitions(near)

    risk = (
        0.38 * inside_ratio
        + 0.27 * near_ratio
        + 0.25 * hand_ratio
        + 0.10 * foot_ratio
    )

    return {
        "has_player_context_003C": 1,
        "player_motion_area_ratio_003C": meta.get("motion_area_ratio", 0.0),
        "player_motion_component_count_003C": meta.get("component_count", 0),
        "player_motion_bbox_area_ratio_003C": meta.get("bbox_area_ratio", 0.0),
        "inside_player_motion_mask_ratio_003C": round(inside_ratio, 6),
        "near_player_motion_mask_ratio_003C": round(near_ratio, 6),
        "inside_hand_or_racket_proxy_ratio_003C": round(hand_ratio, 6),
        "near_foot_or_floor_proxy_ratio_003C": round(foot_ratio, 6),
        "occlusion_entry_exit_count_003C": int(trans),
        "candidate_on_body_risk_003C": round(float(risk), 6),
        "player_context_confidence_003C": meta.get("confidence", 0.0),
    }


def draw_table(out: np.ndarray, model: dict[str, Any] | None) -> None:
    if not model or not model.get("has_table_model"):
        return

    quad = model.get("table_quad")

    if not isinstance(quad, list) or len(quad) < 4 or not all(valid_point(p) for p in quad[:4]):
        return

    pts = np.array([[float(p["x"]), float(p["y"])] for p in quad[:4]], dtype=np.int32)
    cv2.polylines(out, [pts], True, (80, 230, 120), 2, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    player_mask: np.ndarray,
    near_player_mask: np.ndarray,
    hand_proxy: np.ndarray,
    foot_proxy: np.ndarray,
    points: list[dict[str, float]],
    table_model: dict[str, Any] | None,
    title: str,
) -> np.ndarray:
    out = frame.copy()

    # Masque joueur principal.
    green = np.zeros_like(out)
    green[:, :, 1] = 255
    out = np.where(player_mask[:, :, None] > 0, cv2.addWeighted(out, 0.55, green, 0.45, 0), out)

    # Proxy main/raquette.
    blue = np.zeros_like(out)
    blue[:, :, 0] = 255
    out = np.where(hand_proxy[:, :, None] > 0, cv2.addWeighted(out, 0.70, blue, 0.30, 0), out)

    # Proxy pieds/sol.
    red = np.zeros_like(out)
    red[:, :, 2] = 255
    out = np.where(foot_proxy[:, :, None] > 0, cv2.addWeighted(out, 0.78, red, 0.22, 0), out)

    draw_table(out, table_model)

    if len(points) >= 2:
        pts = np.array([[int(round(p["x"])), int(round(p["y"]))] for p in points], dtype=np.int32)
        cv2.polylines(out, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)

    for p in points:
        x = int(round(p["x"]))
        y = int(round(p["y"]))
        cv2.circle(out, (x, y), 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, title[:120], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def target_class(row: dict[str, Any]) -> str:
    v = str(row.get("target_class", "")).strip()
    if v:
        return v

    human = str(row.get("human_decision_002A", ""))
    final = str(row.get("final_decision_002B", ""))
    source = str(row.get("final_source_002B", ""))

    if human == "reject_human_verified":
        return "human_reject"
    if human == "review_partial_human_verified":
        return "human_partial"
    if final == "keep":
        return "keep"
    if final == "reject" and source != "human_002A":
        return "auto_reject"
    if final == "review":
        return "auto_review"
    return "other"


def process_rows(
    rows: list[dict[str, str]],
    payload_by_clip: dict[str, dict[str, Any]],
    table_models: dict[str, Any],
    run_dir: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    roots = [Path.cwd(), run_dir, run_dir.resolve()]
    out_rows = []
    overlay_dir = out_dir / "overlays"

    for row in rows:
        item: dict[str, Any] = dict(row)

        rid = str(item.get("review_id", "")).strip()
        clip_id = str(item.get("clip_id_003B2", "")).strip()
        segment_name = str(item.get("segment_name", "")).strip()

        item["target_class"] = target_class(item)

        track_csv = find_track_csv(item, roots, run_dir)
        points = load_track_points(track_csv)

        first_frame, last_frame = infer_frame_range(item)

        if first_frame is None or last_frame is None:
            if points:
                first_frame = int(min(p["frame"] for p in points))
                last_frame = int(max(p["frame"] for p in points))
            else:
                first_frame = 0
                last_frame = 0

        item["first_frame_003C"] = first_frame
        item["last_frame_003C"] = last_frame
        item["track_csv_resolved_003C"] = str(track_csv) if track_csv else ""

        clip = payload_by_clip.get(clip_id)
        table_model = table_models.get(clip_id)

        if not clip:
            for f in PLAYER_FEATURES:
                item[f] = "" if f != "has_player_context_003C" else 0
            item["player_context_error_003C"] = "missing_clip_payload"
            out_rows.append(item)
            continue

        video_path = Path(str(clip.get("video_path", "")))

        if not video_path.exists():
            for f in PLAYER_FEATURES:
                item[f] = "" if f != "has_player_context_003C" else 0
            item["player_context_error_003C"] = f"missing_video:{video_path}"
            out_rows.append(item)
            continue

        sample_indices, frames = sample_frames(video_path, first_frame, last_frame, sample_count=11)

        if not frames:
            for f in PLAYER_FEATURES:
                item[f] = "" if f != "has_player_context_003C" else 0
            item["player_context_error_003C"] = "no_frames"
            out_rows.append(item)
            continue

        h, w = frames[0].shape[:2]
        table_mask = table_mask_from_model(table_model, (h, w))

        player_mask, near_player_mask, raw_motion, meta = build_player_motion_masks(frames, table_mask)
        hand_proxy, foot_proxy = make_proxy_masks(player_mask, near_player_mask, table_mask, table_model)

        feats = compute_features(points, player_mask, near_player_mask, hand_proxy, foot_proxy, meta)
        item.update(feats)

        mid_frame = int(round((first_frame + last_frame) / 2))
        mid_img = read_video_frame(video_path, mid_frame)

        if mid_img is not None:
            title = f"{rid} {item['target_class']} {segment_name}"
            overlay = draw_overlay(
                frame=mid_img,
                player_mask=player_mask,
                near_player_mask=near_player_mask,
                hand_proxy=hand_proxy,
                foot_proxy=foot_proxy,
                points=points,
                table_model=table_model,
                title=title,
            )

            safe_id = rid or segment_name or f"row_{len(out_rows) + 1:03d}"
            overlay_path = overlay_dir / f"player_context_{safe_id}.jpg"

            if imwrite_unicode(overlay_path, overlay):
                item["player_context_overlay_003C"] = str(overlay_path.relative_to(out_dir))
            else:
                item["player_context_overlay_003C"] = ""
        else:
            item["player_context_overlay_003C"] = ""

        item["player_context_sample_frames_003C"] = ",".join(str(x) for x in sample_indices)
        item["player_context_error_003C"] = ""

        out_rows.append(item)

    return out_rows


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}

    for r in rows:
        v = str(r.get(key, "")).strip()
        out[v] = out.get(v, 0) + 1

    return dict(sorted(out.items()))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_target = count_by(rows, "target_class")
    means_by_target: dict[str, dict[str, float]] = {}

    for target in sorted(by_target):
        subset = [r for r in rows if str(r.get("target_class", "")) == target]
        means_by_target[target] = {}

        for f in PLAYER_FEATURES:
            vals = []

            for r in subset:
                x = fnum(r.get(f), None)
                if x is not None:
                    vals.append(float(x))

            means_by_target[target][f] = round(mean(vals), 6) if vals else 0.0

    high_risk = [
        {
            "review_id": r.get("review_id", ""),
            "target_class": r.get("target_class", ""),
            "segment_name": r.get("segment_name", ""),
            "candidate_on_body_risk_003C": r.get("candidate_on_body_risk_003C", ""),
            "near_player_motion_mask_ratio_003C": r.get("near_player_motion_mask_ratio_003C", ""),
            "inside_hand_or_racket_proxy_ratio_003C": r.get("inside_hand_or_racket_proxy_ratio_003C", ""),
        }
        for r in rows
        if (fnum(r.get("candidate_on_body_risk_003C"), 0) or 0) >= 0.35
    ]

    return {
        "version": "003C",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(rows),
        "by_target_class": by_target,
        "means_by_target": means_by_target,
        "high_body_risk_count": len(high_risk),
        "high_body_risk_rows": high_risk,
        "warning": [
            "003C is a coarse motion-mask player context, not a full human pose model.",
            "It detects moving player-like regions and proxy zones near table/feet.",
            "Static players may be under-detected.",
            "Camera motion may create false player masks."
        ],
    }


def write_html(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    target_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["by_target_class"].items()
    )

    segment_rows = []

    for r in rows:
        cls = str(r.get("target_class", "")).replace("_", "-")
        risk = fnum(r.get("candidate_on_body_risk_003C"), 0) or 0

        if risk >= 0.35:
            cls += " high-body-risk"

        overlay = str(r.get("player_context_overlay_003C", ""))
        overlay_html = f'<a href="{html.escape(overlay)}">overlay</a>' if overlay else ""

        segment_rows.append(f"""
<tr class="{html.escape(cls)}">
<td>{html.escape(str(r.get("review_id", "")))}</td>
<td>{html.escape(str(r.get("target_class", "")))}</td>
<td>{html.escape(str(r.get("contextual_reject_003B4", "")))}</td>
<td>{html.escape(str(r.get("clip_id_003B2", "")))}</td>
<td><code>{html.escape(str(r.get("segment_name", "")))}</code></td>
<td>{html.escape(str(r.get("has_player_context_003C", "")))}</td>
<td>{html.escape(str(r.get("player_motion_area_ratio_003C", "")))}</td>
<td>{html.escape(str(r.get("player_motion_component_count_003C", "")))}</td>
<td>{html.escape(str(r.get("inside_player_motion_mask_ratio_003C", "")))}</td>
<td>{html.escape(str(r.get("near_player_motion_mask_ratio_003C", "")))}</td>
<td>{html.escape(str(r.get("inside_hand_or_racket_proxy_ratio_003C", "")))}</td>
<td>{html.escape(str(r.get("near_foot_or_floor_proxy_ratio_003C", "")))}</td>
<td>{html.escape(str(r.get("occlusion_entry_exit_count_003C", "")))}</td>
<td>{html.escape(str(r.get("candidate_on_body_risk_003C", "")))}</td>
<td>{html.escape(str(r.get("player_context_confidence_003C", "")))}</td>
<td>{overlay_html}</td>
</tr>
""")

    mean_sections = []

    for target, feats in summary["means_by_target"].items():
        body = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in feats.items()
        )

        mean_sections.append(f"""
<section>
<h2>Moyennes joueur · {html.escape(target)}</h2>
<table>
<thead><tr><th>feature</th><th>mean</th></tr></thead>
<tbody>{body}</tbody>
</table>
</section>
""")

    high_rows = "".join(
        f"<tr><td>{html.escape(str(r['review_id']))}</td><td>{html.escape(str(r['target_class']))}</td><td><code>{html.escape(str(r['segment_name']))}</code></td><td>{html.escape(str(r['candidate_on_body_risk_003C']))}</td><td>{html.escape(str(r['near_player_motion_mask_ratio_003C']))}</td><td>{html.escape(str(r['inside_hand_or_racket_proxy_ratio_003C']))}</td></tr>"
        for r in summary["high_body_risk_rows"]
    )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux player context 003C</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}}
th{{background:#20242e;position:sticky;top:0}}
code,pre,a{{color:#dce4ff}}
pre{{white-space:pre-wrap;word-break:break-word;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.human-reject td{{background:rgba(255,80,80,.09)}}
.human-partial td{{background:rgba(255,200,80,.08)}}
.keep td{{background:rgba(100,255,150,.055)}}
.auto-reject td{{background:rgba(255,80,80,.045)}}
.auto-review td{{background:rgba(150,170,255,.045)}}
.high-body-risk td{{outline:1px solid rgba(255,211,122,.45)}}
.warn{{color:#ffd37a}}
</style>
</head>
<body>
<h1>TTFlux · player context 003C</h1>

<section>
<h2>Résumé</h2>
<table>
<tr><th>rows</th><td>{summary["rows"]}</td></tr>
<tr><th>high_body_risk_count</th><td>{summary["high_body_risk_count"]}</td></tr>
<tr><th>warning</th><td class="warn"><pre>{html.escape(json.dumps(summary["warning"], indent=2, ensure_ascii=False))}</pre></td></tr>
</table>

<table>
<thead><tr><th>target</th><th>count</th></tr></thead>
<tbody>{target_rows}</tbody>
</table>
</section>

<section>
<h2>Segments à risque corps/joueur</h2>
<table>
<thead><tr><th>ID</th><th>target</th><th>segment</th><th>body risk</th><th>near player</th><th>hand/racket proxy</th></tr></thead>
<tbody>{high_rows}</tbody>
</table>
</section>

<section>
<h2>Segments · player_context</h2>
<table>
<thead>
<tr>
<th>ID</th><th>target</th><th>ctx reject</th><th>clip</th><th>segment</th>
<th>ctx</th><th>motion area</th><th>components</th><th>inside</th><th>near</th>
<th>hand/racket</th><th>foot/floor</th><th>entry/exit</th><th>body risk</th><th>confidence</th><th>overlay</th>
</tr>
</thead>
<tbody>{''.join(segment_rows)}</tbody>
</table>
</section>

{''.join(mean_sections)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="runs/batch_001E/contextual_arbiter_003B4/contextual_arbiter_003B4_all.csv")
    parser.add_argument("--payload", default="runs/batch_001E/table_scene_003A/table_annotation_payload_003A.json")
    parser.add_argument("--table-models", default="runs/batch_001E/table_bootstrap_003B1B/table_models_curated_003B1B.json")
    parser.add_argument("--run-dir", default="runs/batch_001E")
    parser.add_argument("--out-dir", default="runs/batch_001E/player_context_003C")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    payload_path = Path(args.payload)
    table_models_path = Path(args.table_models)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(input_csv)
    payload_by_clip = load_payload(payload_path)
    table_models = load_table_models(table_models_path)

    out_rows = process_rows(
        rows=rows,
        payload_by_clip=payload_by_clip,
        table_models=table_models,
        run_dir=run_dir,
        out_dir=out_dir,
    )

    summary = summarize(out_rows)

    csv_path = out_dir / "player_context_features_003C.csv"
    json_path = out_dir / "player_context_summary_003C.json"
    html_path = out_dir / "player_context_003C.html"

    write_csv(csv_path, out_rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html(html_path, out_rows, summary)

    print(f"[003C] rows                 : {summary['rows']}")
    print(f"[003C] by target            : {summary['by_target_class']}")
    print(f"[003C] high body risk       : {summary['high_body_risk_count']}")
    print(f"[003C] out dir              : {out_dir}")
    print(f"[003C] html                 : {html_path}")


if __name__ == "__main__":
    main()
