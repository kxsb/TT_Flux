from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "006A_human_frame_label_tool"


FIELDS = [
    "session_id",
    "mode",
    "video_path",
    "video_frame",
    "timestamp_sec",
    "sample_index",
    "visible",
    "x",
    "y",
    "label",
    "action",
    "saved_at",
    "patch",
]


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        wr.writeheader()
        for r in sorted(rows, key=lambda x: int(x.get("sample_index", 0))):
            wr.writerow({k: r.get(k, "") for k in FIELDS})


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


def sample_frames(total_frames: int, n: int, seed: int, start: int, end: int):
    start = max(0, start)
    end = min(total_frames - 1, end if end >= 0 else total_frames - 1)
    pool = list(range(start, end + 1))
    if not pool:
        raise RuntimeError("Aucune frame disponible pour sampling.")
    rnd = random.Random(seed)
    if n >= len(pool):
        return pool
    return sorted(rnd.sample(pool, n))


def draw_text_bg(img, text, org, scale=0.58, color=(255, 255, 255), thickness=1):
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 5, y - th - 8), (x + tw + 5, y + base + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class LabelTool:
    def __init__(self, args):
        self.args = args
        self.video_path = Path(args.video)
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.info = video_info(self.video_path)
        self.fps = self.info["fps"]
        self.mode = args.mode

        self.session_path = self.out_dir / f"006A_session_{self.mode}.json"
        self.csv_path = self.out_dir / f"006A_{self.mode}_labels.csv"

        self.session_id = f"{PATCH_ID}_{self.mode}_{int(time.time())}"

        if self.session_path.exists() and not args.new:
            self.session = json.loads(self.session_path.read_text(encoding="utf-8"))
            self.frames = [int(x) for x in self.session["frames"]]
            self.session_id = self.session.get("session_id", self.session_id)
        else:
            self.frames = sample_frames(
                total_frames=self.info["frames"],
                n=args.frames,
                seed=args.seed,
                start=args.start,
                end=args.end,
            )
            self.session = {
                "patch": PATCH_ID,
                "session_id": self.session_id,
                "mode": self.mode,
                "video_path": str(self.video_path),
                "video_info": self.info,
                "frames": self.frames,
                "seed": args.seed,
                "created_at": now_str(),
            }
            self.session_path.write_text(json.dumps(self.session, ensure_ascii=False, indent=2), encoding="utf-8")

        self.rows = read_csv(self.csv_path)
        self.labels = {}
        for r in self.rows:
            try:
                fr = int(r["video_frame"])
                self.labels[fr] = r
            except Exception:
                pass

        self.idx = 0
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir vidéo: {self.video_path}")

        self.window = f"006A {self.mode}"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        self.scale = 1.0
        self.disp_w = self.info["width"]
        self.disp_h = self.info["height"]

        if args.display_width > 0 and self.info["width"] > args.display_width:
            self.scale = args.display_width / float(self.info["width"])
            self.disp_w = int(round(self.info["width"] * self.scale))
            self.disp_h = int(round(self.info["height"] * self.scale))

        cv2.resizeWindow(self.window, self.disp_w, self.disp_h)
        cv2.setMouseCallback(self.window, self.on_mouse)

    def current_frame_no(self):
        return self.frames[self.idx]

    def current_sample_index(self):
        return self.idx + 1

    def get_or_create_label(self, fr: int):
        if fr not in self.labels:
            self.labels[fr] = {
                "session_id": self.session_id,
                "mode": self.mode,
                "video_path": str(self.video_path),
                "video_frame": str(fr),
                "timestamp_sec": f"{fr / self.fps:.6f}",
                "sample_index": str(self.current_sample_index()),
                "visible": "",
                "x": "",
                "y": "",
                "label": "",
                "action": "",
                "saved_at": "",
                "patch": PATCH_ID,
            }
        else:
            self.labels[fr]["sample_index"] = str(self.current_sample_index())
        return self.labels[fr]

    def persist(self):
        rows = list(self.labels.values())
        write_csv(self.csv_path, rows)

    def save_current(self, action="save"):
        fr = self.current_frame_no()
        r = self.get_or_create_label(fr)

        if not r.get("label"):
            r["label"] = "unsure"
            r["visible"] = ""
            r["x"] = ""
            r["y"] = ""

        r["action"] = action
        r["saved_at"] = now_str()
        self.persist()

    def delete_current(self):
        fr = self.current_frame_no()
        if fr in self.labels:
            del self.labels[fr]
        self.persist()

    def mark_visible(self):
        fr = self.current_frame_no()
        r = self.get_or_create_label(fr)
        r["visible"] = "1"
        r["label"] = "visible"
        if self.mode == "visibility":
            r["x"] = ""
            r["y"] = ""
        r["action"] = "visible"
        r["saved_at"] = now_str()
        self.persist()

    def mark_invisible(self):
        fr = self.current_frame_no()
        r = self.get_or_create_label(fr)
        r["visible"] = "0"
        r["label"] = "invisible"
        r["x"] = ""
        r["y"] = ""
        r["action"] = "invisible"
        r["saved_at"] = now_str()
        self.persist()

    def mark_unsure(self):
        fr = self.current_frame_no()
        r = self.get_or_create_label(fr)
        r["visible"] = ""
        r["label"] = "unsure"
        r["x"] = ""
        r["y"] = ""
        r["action"] = "unsure"
        r["saved_at"] = now_str()
        self.persist()

    def next(self):
        self.idx = min(len(self.frames) - 1, self.idx + 1)

    def prev(self):
        self.idx = max(0, self.idx - 1)

    def on_mouse(self, event, x, y, flags, param):
        if self.mode != "click":
            return

        fr = self.current_frame_no()

        if event == cv2.EVENT_LBUTTONDOWN:
            ox = x / self.scale
            oy = y / self.scale
            r = self.get_or_create_label(fr)
            r["visible"] = "1"
            r["label"] = "clicked_ball"
            r["x"] = f"{ox:.3f}"
            r["y"] = f"{oy:.3f}"
            r["action"] = "click_pending"
            r["saved_at"] = ""

        elif event == cv2.EVENT_RBUTTONDOWN:
            r = self.get_or_create_label(fr)
            r["visible"] = ""
            r["label"] = ""
            r["x"] = ""
            r["y"] = ""
            r["action"] = "reset_pending"
            r["saved_at"] = ""

    def read_current_image(self):
        fr = self.current_frame_no()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = self.cap.read()
        if not ok or img is None:
            img = np.zeros((self.info["height"], self.info["width"], 3), dtype=np.uint8)
            draw_text_bg(img, f"NO FRAME {fr}", (30, 60), scale=1.0, color=(0, 0, 255), thickness=2)
        return img

    def draw(self):
        fr = self.current_frame_no()
        img = self.read_current_image()
        out = img.copy()

        r = self.labels.get(fr)

        # Point cliqué, si présent.
        if r and r.get("x") and r.get("y"):
            try:
                x = int(round(float(r["x"])))
                y = int(round(float(r["y"])))
                cv2.drawMarker(out, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.circle(out, (x, y), 15, (0, 255, 255), 2)
            except Exception:
                pass

        status = "unlabeled"
        if r and r.get("label"):
            status = r["label"]

        color = (230, 230, 230)
        if status in {"visible", "clicked_ball"}:
            color = (0, 255, 255)
        elif status == "invisible":
            color = (80, 80, 255)
        elif status == "unsure":
            color = (180, 180, 180)

        header = (
            f"{PATCH_ID} | mode={self.mode} | "
            f"{self.idx + 1}/{len(self.frames)} | frame={fr} | t={fr / self.fps:.2f}s | {status}"
        )
        controls = "A prev | E next | C visible | V invisible | U unsure | R reset | D save+next | Q save+quit"
        if self.mode == "click":
            controls += " | left click ball | right click reset"

        draw_text_bg(out, header, (12, 28), scale=0.58, color=color)
        draw_text_bg(out, controls, (12, self.info["height"] - 16), scale=0.50, color=(230, 230, 230))

        if self.scale != 1.0:
            out = cv2.resize(out, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)

        return out

    def run(self):
        print("=" * 72)
        print(f"PATCH {PATCH_ID}")
        print(f"mode  = {self.mode}")
        print(f"video = {self.video_path}")
        print(f"out   = {self.csv_path}")
        print("=" * 72)
        print("")
        print("Contrôles : A précédent | E suivant | C visible | V invisible | U unsure | R reset | D save+next | Q save+quit")
        if self.mode == "click":
            print("Mode click : clic gauche sur la balle, clic droit reset.")
        print("")

        running = True

        while running:
            cv2.imshow(self.window, self.draw())
            key = cv2.waitKeyEx(0)

            low = key & 0xFF
            ch = chr(low).lower() if 0 <= low < 128 else ""

            if ch == "a" or key in [2424832]:
                self.prev()

            elif ch == "e" or key in [2555904]:
                self.next()

            elif ch == "r":
                self.delete_current()

            elif ch == "c":
                self.mark_visible()
                if self.mode == "visibility":
                    self.next()

            elif ch == "v":
                self.mark_invisible()
                self.next()

            elif ch == "u":
                self.mark_unsure()
                self.next()

            elif ch == "d":
                self.save_current(action="save_next")
                self.next()

            elif ch == "q" or key == 27:
                self.save_current(action="save_quit")
                running = False

        self.cap.release()
        cv2.destroyAllWindows()

        labeled = len(self.labels)
        visible = sum(1 for r in self.labels.values() if r.get("visible") == "1")
        invisible = sum(1 for r in self.labels.values() if r.get("visible") == "0")
        unsure = sum(1 for r in self.labels.values() if r.get("label") == "unsure")

        summary = {
            "patch": PATCH_ID,
            "mode": self.mode,
            "video": str(self.video_path),
            "session": str(self.session_path),
            "labels_csv": str(self.csv_path),
            "sample_count": len(self.frames),
            "labeled_count": labeled,
            "visible_count": visible,
            "invisible_count": invisible,
            "unsure_count": unsure,
            "video_info": self.info,
        }

        summary_path = self.out_dir / f"006A_{self.mode}_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print("")
        print("OK 006A")
        print(f"labels_csv = {self.csv_path}")
        print(f"summary    = {summary_path}")
        print(f"labeled={labeled} visible={visible} invisible={invisible} unsure={unsure}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="runs/005F3_align_clean_video/005F3_clean_aligned_segment.mp4")
    ap.add_argument("--mode", choices=["visibility", "click"], default="visibility")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--seed", type=int, default=606)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--display-width", type=int, default=1100)
    ap.add_argument("--out-dir", default="runs/006A_human_frame_labels")
    ap.add_argument("--new", action="store_true", help="Créer une nouvelle session au lieu de reprendre.")
    args = ap.parse_args()

    tool = LabelTool(args)
    tool.run()


if __name__ == "__main__":
    main()
