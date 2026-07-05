from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "003J1"

BAD_STATUSES = {"BAD_OR_UNRELIABLE", "WEAK", "MISSING_MODEL"}

VIDEO_COL_PRIORITY = [
    "mp4",
    "table_mp4",
    "video",
    "overlay_mp4",
    "segment_mp4",
]

CLICK_LABELS = [
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
]


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
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

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(project_root / p)
        candidates.append(run_dir / p)

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
    

def sample_frames(video_path: Path, max_frames: int = 15) -> tuple[list[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"error": "VideoCapture failed"}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

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

    return frames, {
        "frame_count": total,
        "fps": fps,
        "width": width,
        "height": height,
        "sampled": len(frames),
    }


def median_frame(frames: list[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None

    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    resized = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
    stack = np.stack(resized, axis=0).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) != 4:
        return 0.0

    arr = np.array(points, dtype=np.float32)
    area = cv2.contourArea(arr.reshape(-1, 1, 2))
    return float(abs(area))


def draw_overlay(frame: np.ndarray, points: list[tuple[float, float]], title: str) -> np.ndarray:
    vis = frame.copy()

    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        closed = len(points) == 4
        cv2.polylines(vis, [pts], isClosed=closed, color=(0, 255, 255), thickness=3)

    for i, (x, y) in enumerate(points):
        label = CLICK_LABELS[i] if i < len(CLICK_LABELS) else str(i + 1)
        cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 255), -1)
        cv2.circle(vis, (int(x), int(y)), 10, (0, 0, 0), 2)
        cv2.putText(
            vis,
            f"{i+1}:{label}",
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"{i+1}:{label}",
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    help_lines = [
        title,
        "Click order: 1 top_left, 2 top_right, 3 bottom_right, 4 bottom_left",
        "s=save | r=reset | u=undo | n=skip | q=quit",
    ]

    y = 28
    for line in help_lines:
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


class Annotator:
    def __init__(self, frame: np.ndarray, review_id: str, target: str, status: str):
        self.frame = frame
        self.review_id = review_id
        self.target = target
        self.status = status
        self.points: list[tuple[float, float]] = []
        self.action = None

        self.base_display, self.scale = fit_to_screen(frame)

        self.window = f"TTFlux 003J1 table corners - {review_id}"

    def refresh(self):
        title = f"{self.review_id} | {self.target} | old_table={self.status}"
        overlay = draw_overlay(self.frame, self.points, title)
        display, _ = fit_to_screen(overlay)
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

    def run(self) -> tuple[str, list[tuple[float, float]]]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse_cb)

        self.refresh()

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord("s"):
                if len(self.points) == 4:
                    self.action = "saved"
                    break
                else:
                    print(f"{self.review_id}: besoin de 4 points avant save, actuel={len(self.points)}")

            elif key == ord("r"):
                self.points = []
                self.refresh()

            elif key == ord("u"):
                if self.points:
                    self.points.pop()
                    self.refresh()

            elif key == ord("n"):
                self.action = "skipped"
                break

            elif key == ord("q"):
                self.action = "quit"
                break

            # Escape
            elif key == 27:
                self.action = "quit"
                break

        cv2.destroyWindow(self.window)
        return self.action or "skipped", self.points


def write_html(path: Path, rows: list[dict], summary: dict, run_dir: Path) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section,.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card.saved{border-color:rgba(116,217,159,.75);box-shadow:inset 4px 0 0 rgba(116,217,159,.85)}
.card.skipped{border-color:rgba(255,200,80,.65)}
.card.error{border-color:rgba(255,80,80,.75);box-shadow:inset 4px 0 0 rgba(255,80,80,.85)}
h1,h2,h3{margin-top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{width:240px;background:#20242e}
.grid{display:grid;grid-template-columns:430px 1fr;gap:16px}
@media(max-width:950px){.grid{grid-template-columns:1fr}}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.saved{border-color:rgba(116,217,159,.8);background:rgba(116,217,159,.08)}
.badge.skipped{border-color:rgba(255,200,80,.8);background:rgba(255,200,80,.08)}
.badge.error{border-color:rgba(255,80,80,.8);background:rgba(255,80,80,.08)}
img{display:block;width:900px;max-width:100%;border:1px solid #2b303b;border-radius:10px;background:#05060a}
.muted{color:#aab2c5}
"""

    cards = []

    for r in rows:
        action = r.get("annotation_action", "")
        cls = "saved" if action == "saved" else "skipped" if action == "skipped" else "error"
        overlay = r.get("overlay_png_rel", "")

        metric_cols = [
            "review_id",
            "target_class",
            "table_reality_status_003I",
            "would_reject_shadow_003G",
            "annotation_action",
            "table_area_ratio_manual_003J1",
            "manual_table_valid_003J1",
            "video_path",
        ]

        trs = []
        for c in metric_cols:
            trs.append(f"<tr><th>{c}</th><td>{r.get(c, '')}</td></tr>")

        img = ""
        if overlay:
            img = f"<a href='{overlay}'><img src='{overlay}'></a>"
        else:
            img = "<p class='muted'>Pas d'overlay sauvegardé.</p>"

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
    <h3>Overlay manuel</h3>
    {img}
  </div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003J1 manual table corner bootstrap</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003J1 manual table corner bootstrap</h1>

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

    out = {}
    for _, r in df.iterrows():
        rid = str(r.get("review_id", ""))
        if rid:
            out[rid] = r.to_dict()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_001E")
    ap.add_argument("--rejects-first", action="store_true", default=True)
    ap.add_argument("--all", action="store_true", help="annoter toutes les lignes, pas seulement BAD/WEAK/MISSING")
    ap.add_argument("--only-rejects", action="store_true", help="annoter uniquement les would_reject")
    ap.add_argument("--resume", action="store_true", help="ne pas redemander les lignes deja sauvegardees")
    ap.add_argument("--max", type=int, default=0, help="limite optionnelle de lignes a annoter")
    args = ap.parse_args()

    project_root = Path.cwd()
    run_dir = Path(args.run)

    input_csv = run_dir / "table_reality_audit_003I.csv"
    input_json = run_dir / "table_reality_audit_summary_003I.json"

    if not input_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {input_csv}")
    if not input_json.is_file():
        raise SystemExit(f"Fichier introuvable: {input_json}")

    df = pd.read_csv(input_csv)

    required = ["review_id_003G", "target_class_003G", "table_reality_status_003I"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Colonnes manquantes: " + ", ".join(missing))

    df["review_id_003G"] = df["review_id_003G"].astype(str)

    if not args.all:
        df = df[df["table_reality_status_003I"].isin(BAD_STATUSES)].copy()

    if args.only_rejects:
        if "would_reject_shadow_003G" not in df.columns:
            raise SystemExit("Colonne would_reject_shadow_003G manquante.")
        df = df[df["would_reject_shadow_003G"].map(lambda x: str(x).lower() == "true")].copy()

    # Tri : les rejets simulés d'abord, puis table bad/weak/missing.
    def reject_prio(x):
        return 0 if str(x).lower() == "true" else 1

    status_prio = {
        "BAD_OR_UNRELIABLE": 0,
        "WEAK": 1,
        "MISSING_MODEL": 2,
        "OK_OR_NOT_ENOUGH_EVIDENCE": 3,
    }

    if "would_reject_shadow_003G" in df.columns:
        df["_reject_prio"] = df["would_reject_shadow_003G"].map(reject_prio)
    else:
        df["_reject_prio"] = 1

    df["_status_prio"] = df["table_reality_status_003I"].map(lambda x: status_prio.get(str(x), 9))
    df = df.sort_values(["_reject_prio", "_status_prio", "review_id_003G"]).drop(columns=["_reject_prio", "_status_prio"])

    if args.max and args.max > 0:
        df = df.head(args.max).copy()

    out_dir = run_dir / "manual_table_corner_bootstrap_003J1_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = run_dir / "manual_table_corners_003J1.csv"
    previous = existing_annotations(out_csv) if args.resume else {}

    rows: list[dict] = []
    rows.extend(previous.values())

    seen = set(previous.keys())

    print("")
    print("003J1 manual table corner bootstrap")
    print("Ordre clics : 1 top_left, 2 top_right, 3 bottom_right, 4 bottom_left")
    print("Touches : s=save | r=reset | u=undo | n=skip | q=quit")
    print("")

    quit_requested = False

    for _, row in df.iterrows():
        rid = str(row["review_id_003G"])

        if args.resume and rid in seen:
            print(f"{rid}: deja annote, skip (--resume)")
            continue

        video_path, video_col = choose_video(row, project_root, run_dir)
        target = str(row.get("target_class_003G", ""))
        old_status = str(row.get("table_reality_status_003I", ""))

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
            "manual_table_valid_003J1": False,
            "table_area_px_manual_003J1": "",
            "table_area_ratio_manual_003J1": "",
            "image_w": "",
            "image_h": "",
            "top_left_x": "",
            "top_left_y": "",
            "top_right_x": "",
            "top_right_y": "",
            "bottom_right_x": "",
            "bottom_right_y": "",
            "bottom_left_x": "",
            "bottom_left_y": "",
            "overlay_png": "",
            "overlay_png_rel": "",
            "note": "",
        }

        if not video_path:
            print(f"{rid}: no video")
            record["annotation_action"] = "no_video"
            rows.append(record)
            continue

        frames, meta = sample_frames(video_path)
        med = median_frame(frames)

        if med is None:
            print(f"{rid}: no frame")
            record["annotation_action"] = "no_frame"
            record["note"] = json.dumps(meta, ensure_ascii=False)
            rows.append(record)
            continue

        print(f"{rid}: {target} | old_table={old_status} | {video_path.name}")

        annot = Annotator(med, rid, target, old_status)
        action, points = annot.run()

        record["annotation_action"] = action
        record["image_h"] = int(med.shape[0])
        record["image_w"] = int(med.shape[1])

        if action == "quit":
            print("quit requested")
            quit_requested = True
            break

        if action == "saved" and len(points) == 4:
            area = polygon_area(points)
            area_ratio = area / max(1.0, float(med.shape[0] * med.shape[1]))

            record["manual_table_valid_003J1"] = True
            record["table_area_px_manual_003J1"] = round(area, 3)
            record["table_area_ratio_manual_003J1"] = round(area_ratio, 6)

            for label, (x, y) in zip(CLICK_LABELS, points):
                record[f"{label}_x"] = round(float(x), 3)
                record[f"{label}_y"] = round(float(y), 3)

            overlay = draw_overlay(med, points, f"{rid} | MANUAL_TABLE_003J1")
            overlay_path = out_dir / f"{rid}_manual_table_overlay_003J1.png"
            imwrite_unicode(overlay_path, overlay)

            record["overlay_png"] = str(overlay_path)
            record["overlay_png_rel"] = os.path.relpath(overlay_path, run_dir).replace("\\", "/")

            print(f"{rid}: saved area_ratio={record['table_area_ratio_manual_003J1']}")

        else:
            print(f"{rid}: {action}")

        rows.append(record)

        # Sauvegarde progressive à chaque ligne.
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

        summary = {
            "version": VERSION,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "policy": "manual_table_corner_bootstrap_only_no_arbiter_change",
            "source_csv": str(input_csv),
            "output_csv": str(out_csv),
            "rows_total_current": len(rows),
            "saved_count_current": sum(1 for r in rows if r.get("annotation_action") == "saved"),
            "skipped_count_current": sum(1 for r in rows if r.get("annotation_action") == "skipped"),
            "quit_requested": quit_requested,
        }
        (run_dir / "manual_table_corners_summary_003J1.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    saved = out_df[out_df.get("annotation_action", "").eq("saved")] if len(out_df) else pd.DataFrame()
    skipped = out_df[out_df.get("annotation_action", "").eq("skipped")] if len(out_df) else pd.DataFrame()

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "manual_table_corner_bootstrap_only_no_arbiter_change",
        "source_csv": str(input_csv),
        "rows_requested": int(len(df)),
        "rows_written": int(len(out_df)),
        "saved_count": int(len(saved)),
        "saved_ids": saved["review_id"].astype(str).tolist() if len(saved) else [],
        "skipped_count": int(len(skipped)),
        "skipped_ids": skipped["review_id"].astype(str).tolist() if len(skipped) else [],
        "quit_requested": bool(quit_requested),
        "next_step": (
            "003J2 doit relire manual_table_corners_003J1.csv et recalculer "
            "inside/distance/table_context avec les coins manuels."
        ),
    }

    out_json = run_dir / "manual_table_corners_summary_003J1.json"
    out_html = run_dir / "manual_table_corners_003J1.html"

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, rows, summary, run_dir)

    print("")
    print("003J1 done")
    print("saved_count=", summary["saved_count"])
    print("saved_ids=", ",".join(summary["saved_ids"]) or "-")
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
