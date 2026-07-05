from __future__ import annotations

import argparse
import json
import os
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "003J1B"

BAD_STATUSES = {"BAD_OR_UNRELIABLE", "WEAK", "MISSING_MODEL"}

VIDEO_COL_PRIORITY = [
    "mp4",
    "table_mp4",
    "video",
    "overlay_mp4",
    "segment_mp4",
]


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    buf.tofile(str(path))


def resolve_path(value, project_root: Path, run_dir: Path) -> Path | None:
    if not isinstance(value, str):
        return None

    raw = value.strip().strip('"').strip("'")
    if not raw:
        return None

    raw = raw.replace("\\", os.sep)
    p = Path(raw)

    candidates = [p] if p.is_absolute() else [project_root / p, run_dir / p]

    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            pass

    return None


def choose_video(row: pd.Series, project_root: Path, run_dir: Path) -> tuple[Path | None, str]:
    for col in VIDEO_COL_PRIORITY:
        if col not in row.index:
            continue

        p = resolve_path(row.get(col), project_root, run_dir)
        if p and p.suffix.lower() in {".mp4", ".webm", ".mov", ".avi"}:
            return p, col

    return None, ""


def sample_frames(video_path: Path, max_frames: int = 15) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total <= 0:
        indices = list(range(max_frames))
    else:
        indices = np.linspace(0, max(0, total - 1), num=min(max_frames, total), dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)

    cap.release()
    return frames


def median_frame(frames: list[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None

    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)

    resized = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
    stack = np.stack(resized, axis=0).astype(np.float32)

    return np.median(stack, axis=0).astype(np.uint8)


def order_points_around_centroid(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Ordre géométrique stable, sans labels sémantiques.
    Retourne les points triés autour du centroïde.
    Ce n'est PAS une vérité métier ; juste un polygone temporaire.
    """
    if len(points) != 4:
        return points

    pts = np.array(points, dtype=np.float32)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    def angle(p):
        return math.atan2(float(p[1]) - cy, float(p[0]) - cx)

    ordered = sorted(points, key=angle)

    # On force un sens visuel plus stable : départ = point le plus haut puis gauche.
    start_i = min(range(4), key=lambda i: (ordered[i][1], ordered[i][0]))
    ordered = ordered[start_i:] + ordered[:start_i]

    return ordered


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) != 4:
        return 0.0
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(arr)))


def draw_overlay(
    frame: np.ndarray,
    raw_points: list[tuple[float, float]],
    ordered_points: list[tuple[float, float]] | None,
    title: str,
) -> np.ndarray:
    vis = frame.copy()

    # Polygone auto-ordonné en jaune.
    if ordered_points and len(ordered_points) == 4:
        pts = np.array(ordered_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=3)

    # Points bruts en cercles, sans gauche/droite/haut/bas.
    for i, (x, y) in enumerate(raw_points):
        cv2.circle(vis, (int(x), int(y)), 9, (0, 255, 255), -1)
        cv2.circle(vis, (int(x), int(y)), 11, (0, 0, 0), 2)

        label = f"P{i + 1}"
        cv2.putText(vis, label, (int(x) + 12, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, label, (int(x) + 12, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    lines = [
        title,
        "Clique 4 points/corners visibles de la table, ordre libre : P1 P2 P3 P4",
        "s=save | r=reset | u=undo | n=skip | q=quit",
        "Jaune = polygone auto-ordonne temporaire",
    ]

    y = 28
    for line in lines:
        cv2.putText(vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24

    return vis


def fit_to_screen(img: np.ndarray, max_w: int = 1400, max_h: int = 900) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)

    if scale >= 0.999:
        return img.copy(), 1.0

    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


class PointAnnotator:
    def __init__(self, frame: np.ndarray, review_id: str, target: str, old_status: str):
        self.frame = frame
        self.review_id = review_id
        self.target = target
        self.old_status = old_status
        self.points: list[tuple[float, float]] = []
        self.scale = 1.0
        self.window = f"TTFlux 003J1B unordered table points - {review_id}"

    def refresh(self):
        ordered = order_points_around_centroid(self.points) if len(self.points) == 4 else None
        title = f"{self.review_id} | {self.target} | old_table={self.old_status}"
        overlay = draw_overlay(self.frame, self.points, ordered, title)
        display, self.scale = fit_to_screen(overlay)
        cv2.imshow(self.window, display)

    def mouse_cb(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                ox = float(x) / max(1e-9, self.scale)
                oy = float(y) / max(1e-9, self.scale)
                self.points.append((ox, oy))
                self.refresh()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self.refresh()

    def run(self) -> tuple[str, list[tuple[float, float]], list[tuple[float, float]]]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse_cb)
        self.refresh()

        action = "skipped"

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord("s"):
                if len(self.points) == 4:
                    action = "saved"
                    break
                print(f"{self.review_id}: besoin de 4 points, actuel={len(self.points)}")

            elif key == ord("r"):
                self.points = []
                self.refresh()

            elif key == ord("u"):
                if self.points:
                    self.points.pop()
                    self.refresh()

            elif key == ord("n"):
                action = "skipped"
                break

            elif key == ord("q") or key == 27:
                action = "quit"
                break

        cv2.destroyWindow(self.window)

        ordered = order_points_around_centroid(self.points) if len(self.points) == 4 else []
        return action, self.points, ordered


def write_html(path: Path, rows: list[dict], summary: dict) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section,.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card.saved{border-color:rgba(116,217,159,.75);box-shadow:inset 4px 0 0 rgba(116,217,159,.85)}
.card.skipped{border-color:rgba(255,200,80,.65)}
.card.error{border-color:rgba(255,80,80,.75)}
h1,h2,h3{margin-top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{width:250px;background:#20242e}
.grid{display:grid;grid-template-columns:430px 1fr;gap:16px}
@media(max-width:950px){.grid{grid-template-columns:1fr}}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.saved{border-color:rgba(116,217,159,.8);background:rgba(116,217,159,.08)}
.badge.skipped{border-color:rgba(255,200,80,.8);background:rgba(255,200,80,.08)}
img{display:block;width:900px;max-width:100%;border:1px solid #2b303b;border-radius:10px;background:#05060a}
.muted{color:#aab2c5}
"""

    cards = []

    for r in rows:
        action = r.get("annotation_action", "")
        cls = "saved" if action == "saved" else "skipped"

        overlay = r.get("overlay_png_rel", "")

        cols = [
            "review_id",
            "target_class",
            "table_reality_status_003I",
            "would_reject_shadow_003G",
            "annotation_action",
            "manual_points_mode_003J1B",
            "manual_table_area_ratio_ordered_003J1B",
            "manual_table_valid_003J1B",
            "video_path",
        ]

        trs = []
        for c in cols:
            trs.append(f"<tr><th>{c}</th><td>{r.get(c, '')}</td></tr>")

        if overlay:
            img = f"<a href='{overlay}'><img src='{overlay}'></a>"
        else:
            img = "<p class='muted'>Pas d'overlay.</p>"

        cards.append(f"""
<div class="card {cls}">
<h2>{r.get('review_id', '')}</h2>
<div><span class="badge {cls}">{action}</span></div>
<div class="grid">
  <div>
    <h3>Données</h3>
    <table><tbody>{''.join(trs)}</tbody></table>
  </div>
  <div>
    <h3>Overlay</h3>
    {img}
  </div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003J1B unordered table points</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003J1B unordered table points</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

{''.join(cards)}

</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def existing_annotations(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    if "review_id" not in df.columns:
        return {}

    return {str(r["review_id"]): r.to_dict() for _, r in df.iterrows() if str(r.get("review_id", ""))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_001E")
    ap.add_argument("--only-rejects", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--ids", default="", help="liste de review ids separee par virgules, ex: R0022,R0019")
    args = ap.parse_args()

    project_root = Path.cwd()
    run_dir = Path(args.run)

    input_csv = run_dir / "table_reality_audit_003I.csv"
    if not input_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {input_csv}")

    df = pd.read_csv(input_csv)

    required = ["review_id_003G", "target_class_003G", "table_reality_status_003I"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Colonnes manquantes: " + ", ".join(missing))

    df["review_id_003G"] = df["review_id_003G"].astype(str)

    ids_filter_003J1B = [x.strip() for x in str(args.ids or "").split(",") if x.strip()]
    if ids_filter_003J1B:
        df = df[df["review_id_003G"].isin(ids_filter_003J1B)].copy()

    if not args.all:
        df = df[df["table_reality_status_003I"].isin(BAD_STATUSES)].copy()

    if args.only_rejects:
        if "would_reject_shadow_003G" not in df.columns:
            raise SystemExit("Colonne would_reject_shadow_003G manquante.")
        df = df[df["would_reject_shadow_003G"].astype(str).str.lower().eq("true")].copy()

    status_prio = {
        "BAD_OR_UNRELIABLE": 0,
        "WEAK": 1,
        "MISSING_MODEL": 2,
        "OK_OR_NOT_ENOUGH_EVIDENCE": 3,
    }

    if "would_reject_shadow_003G" in df.columns:
        df["_reject_prio"] = df["would_reject_shadow_003G"].astype(str).str.lower().map(lambda x: 0 if x == "true" else 1)
    else:
        df["_reject_prio"] = 1

    df["_status_prio"] = df["table_reality_status_003I"].map(lambda x: status_prio.get(str(x), 9))
    df = df.sort_values(["_reject_prio", "_status_prio", "review_id_003G"]).drop(columns=["_reject_prio", "_status_prio"])

    if args.max > 0:
        df = df.head(args.max).copy()

    out_dir = run_dir / "manual_table_points_003J1B_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = run_dir / "manual_table_points_003J1B.csv"
    previous = existing_annotations(out_csv) if args.resume else {}

    rows: list[dict] = list(previous.values())
    seen = set(previous.keys())

    print("")
    print("003J1B unordered table points")
    print("Clique 4 points visibles de la table, dans n'importe quel ordre.")
    print("Touches : s=save | r=reset | u=undo | n=skip | q=quit")
    print("")

    quit_requested = False

    for _, row in df.iterrows():
        rid = str(row["review_id_003G"])

        if args.resume and rid in seen:
            print(f"{rid}: deja annote, skip")
            continue

        target = str(row.get("target_class_003G", ""))
        old_status = str(row.get("table_reality_status_003I", ""))

        video_path, video_col = choose_video(row, project_root, run_dir)

        record = {
            "version": VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "review_id": rid,
            "target_class": target,
            "table_reality_status_003I": old_status,
            "would_reject_shadow_003G": row.get("would_reject_shadow_003G", ""),
            "video_source_col": video_col,
            "video_path": str(video_path) if video_path else "",
            "annotation_action": "",
            "manual_points_mode_003J1B": "unordered_4_points_plus_auto_ordered_polygon",
            "manual_table_valid_003J1B": False,
            "manual_table_area_px_ordered_003J1B": "",
            "manual_table_area_ratio_ordered_003J1B": "",
            "image_w": "",
            "image_h": "",
            "raw_p1_x": "", "raw_p1_y": "",
            "raw_p2_x": "", "raw_p2_y": "",
            "raw_p3_x": "", "raw_p3_y": "",
            "raw_p4_x": "", "raw_p4_y": "",
            "ordered_q1_x": "", "ordered_q1_y": "",
            "ordered_q2_x": "", "ordered_q2_y": "",
            "ordered_q3_x": "", "ordered_q3_y": "",
            "ordered_q4_x": "", "ordered_q4_y": "",
            "overlay_png": "",
            "overlay_png_rel": "",
        }

        if not video_path:
            print(f"{rid}: no video")
            record["annotation_action"] = "no_video"
            rows.append(record)
            continue

        frames = sample_frames(video_path)
        med = median_frame(frames)

        if med is None:
            print(f"{rid}: no frame")
            record["annotation_action"] = "no_frame"
            rows.append(record)
            continue

        print(f"{rid}: {target} | old_table={old_status} | {video_path.name}")

        annot = PointAnnotator(med, rid, target, old_status)
        action, raw_points, ordered_points = annot.run()

        record["annotation_action"] = action
        record["image_h"] = int(med.shape[0])
        record["image_w"] = int(med.shape[1])

        if action == "quit":
            quit_requested = True
            print("quit requested")
            break

        if action == "saved" and len(raw_points) == 4 and len(ordered_points) == 4:
            area = polygon_area(ordered_points)
            area_ratio = area / max(1.0, float(med.shape[0] * med.shape[1]))

            record["manual_table_valid_003J1B"] = True
            record["manual_table_area_px_ordered_003J1B"] = round(area, 3)
            record["manual_table_area_ratio_ordered_003J1B"] = round(area_ratio, 6)

            for i, (x, y) in enumerate(raw_points, start=1):
                record[f"raw_p{i}_x"] = round(float(x), 3)
                record[f"raw_p{i}_y"] = round(float(y), 3)

            for i, (x, y) in enumerate(ordered_points, start=1):
                record[f"ordered_q{i}_x"] = round(float(x), 3)
                record[f"ordered_q{i}_y"] = round(float(y), 3)

            overlay = draw_overlay(med, raw_points, ordered_points, f"{rid} | MANUAL_TABLE_POINTS_003J1B")
            overlay_path = out_dir / f"{rid}_manual_table_points_003J1B.png"
            imwrite_unicode(overlay_path, overlay)

            record["overlay_png"] = str(overlay_path)
            record["overlay_png_rel"] = os.path.relpath(overlay_path, run_dir).replace("\\", "/")

            print(f"{rid}: saved area_ratio={record['manual_table_area_ratio_ordered_003J1B']}")

        else:
            print(f"{rid}: {action}")

        rows.append(record)

        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    saved_df = out_df[out_df["annotation_action"].eq("saved")] if len(out_df) else pd.DataFrame()

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "unordered_manual_table_points_only_no_arbiter_change",
        "source_csv": str(input_csv),
        "rows_requested": int(len(df)),
        "rows_written": int(len(out_df)),
        "saved_count": int(len(saved_df)),
        "saved_ids": saved_df["review_id"].astype(str).tolist() if len(saved_df) else [],
        "quit_requested": bool(quit_requested),
        "next_step": (
            "003J2B relira manual_table_points_003J1B.csv, utilisera les ordered_q* "
            "pour recalculer temporairement les distances/inside, et gardera les raw_p* "
            "comme labels non sémantiques pour futur apprentissage."
        ),
    }

    out_json = run_dir / "manual_table_points_summary_003J1B.json"
    out_html = run_dir / "manual_table_points_003J1B.html"

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, rows, summary)

    print("")
    print("003J1B done")
    print("saved_count=", summary["saved_count"])
    print("saved_ids=", ",".join(summary["saved_ids"]) or "-")
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
