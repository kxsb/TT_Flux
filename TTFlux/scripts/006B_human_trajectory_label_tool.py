from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "006B_human_trajectory_label_tool"

FIELDS = [
    "session_id",
    "video_path",
    "frame",
    "timestamp_sec",
    "visible",
    "x",
    "y",
    "label",
    "point_type",
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
        for r in sorted(rows, key=lambda x: int(float(x.get("frame", 0)))):
            wr.writerow({k: r.get(k, "") for k in FIELDS})


def draw_text_bg(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1):
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 5, y - th - 8), (x + tw + 5, y + base + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class TrajectoryTool:
    def __init__(self, args):
        self.args = args
        self.video_path = Path(args.video)
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir vidéo: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 50.0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        self.start_frame = max(0, int(args.start))
        if args.end >= 0:
            self.end_frame = min(self.total_frames - 1, int(args.end))
        else:
            self.end_frame = min(
                self.total_frames - 1,
                self.start_frame + int(round(args.seconds * self.fps)) - 1,
            )

        if self.end_frame <= self.start_frame:
            raise RuntimeError("Fenêtre vidéo invalide.")

        self.frame = self.start_frame
        self.session_id = f"{PATCH_ID}_{int(time.time())}"

        suffix = f"f{self.start_frame:04d}_{self.end_frame:04d}"
        self.csv_path = self.out_dir / f"006B_trajectory_labels_{suffix}.csv"
        self.summary_path = self.out_dir / f"006B_trajectory_summary_{suffix}.json"

        self.labels = {}

        if self.csv_path.exists() and not args.new:
            for r in read_csv(self.csv_path):
                try:
                    fr = int(float(r["frame"]))
                    self.labels[fr] = r
                except Exception:
                    pass

        self.scale = 1.0
        self.disp_w = self.w
        self.disp_h = self.h

        if args.display_width > 0 and self.w > args.display_width:
            self.scale = args.display_width / float(self.w)
            self.disp_w = int(round(self.w * self.scale))
            self.disp_h = int(round(self.h * self.scale))

        self.window = "006B trajectory labels"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.disp_w, self.disp_h)
        cv2.setMouseCallback(self.window, self.on_mouse)

    def get_label(self, fr):
        if fr not in self.labels:
            self.labels[fr] = {
                "session_id": self.session_id,
                "video_path": str(self.video_path),
                "frame": str(fr),
                "timestamp_sec": f"{fr / self.fps:.6f}",
                "visible": "",
                "x": "",
                "y": "",
                "label": "",
                "point_type": "",
                "saved_at": "",
                "patch": PATCH_ID,
            }
        return self.labels[fr]

    def save(self):
        rows = list(self.labels.values())
        write_csv(self.csv_path, rows)

        manual = [r for r in rows if r.get("point_type") == "manual"]
        interpolated = [r for r in rows if r.get("point_type") == "interpolated"]
        invisible = [r for r in rows if r.get("label") == "invisible"]
        unsure = [r for r in rows if r.get("label") == "unsure"]

        summary = {
            "patch": PATCH_ID,
            "video_path": str(self.video_path),
            "csv": str(self.csv_path),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "fps": self.fps,
            "duration_sec": (self.end_frame - self.start_frame + 1) / self.fps,
            "label_count": len(rows),
            "manual_count": len(manual),
            "interpolated_count": len(interpolated),
            "invisible_count": len(invisible),
            "unsure_count": len(unsure),
        }

        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ox = x / self.scale
            oy = y / self.scale

            r = self.get_label(self.frame)
            r["visible"] = "1"
            r["x"] = f"{ox:.3f}"
            r["y"] = f"{oy:.3f}"
            r["label"] = "ball"
            r["point_type"] = "manual"
            r["saved_at"] = now_str()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.reset_current()

    def reset_current(self):
        if self.frame in self.labels:
            del self.labels[self.frame]

    def mark_invisible(self):
        r = self.get_label(self.frame)
        r["visible"] = "0"
        r["x"] = ""
        r["y"] = ""
        r["label"] = "invisible"
        r["point_type"] = "manual"
        r["saved_at"] = now_str()

    def mark_unsure(self):
        r = self.get_label(self.frame)
        r["visible"] = ""
        r["x"] = ""
        r["y"] = ""
        r["label"] = "unsure"
        r["point_type"] = "manual"
        r["saved_at"] = now_str()

    def step(self, delta):
        self.frame = max(self.start_frame, min(self.end_frame, self.frame + delta))

    def interpolate(self):
        manual_points = []

        for fr, r in self.labels.items():
            if (
                r.get("point_type") == "manual"
                and r.get("visible") == "1"
                and r.get("x")
                and r.get("y")
            ):
                manual_points.append((int(fr), float(r["x"]), float(r["y"])))

        manual_points.sort()

        for (f0, x0, y0), (f1, x1, y1) in zip(manual_points[:-1], manual_points[1:]):
            if f1 <= f0 + 1:
                continue

            for fr in range(f0 + 1, f1):
                existing = self.labels.get(fr)
                if existing and existing.get("point_type") == "manual":
                    continue

                t = (fr - f0) / float(f1 - f0)
                x = x0 * (1 - t) + x1 * t
                y = y0 * (1 - t) + y1 * t

                r = self.get_label(fr)
                r["visible"] = "1"
                r["x"] = f"{x:.3f}"
                r["y"] = f"{y:.3f}"
                r["label"] = "ball"
                r["point_type"] = "interpolated"
                r["saved_at"] = now_str()

    def read_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame)
        ok, img = self.cap.read()

        if not ok or img is None:
            img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            draw_text_bg(img, f"NO FRAME {self.frame}", (30, 60), scale=1.0, color=(0, 0, 255), thickness=2)

        return img

    def draw_trajectory(self, img):
        out = img.copy()

        prev = None

        for fr in range(self.frame - 45, self.frame + 46):
            r = self.labels.get(fr)

            if not r or r.get("visible") != "1" or not r.get("x") or not r.get("y"):
                prev = None
                continue

            x = int(round(float(r["x"])))
            y = int(round(float(r["y"])))
            ptype = r.get("point_type", "")

            if ptype == "manual":
                color = (0, 255, 255)
                radius = 6
            else:
                color = (0, 150, 255)
                radius = 3

            if prev is not None:
                cv2.line(out, prev, (x, y), (0, 180, 255), 2)

            cv2.circle(out, (x, y), radius, color, -1)
            prev = (x, y)

        r = self.labels.get(self.frame)
        if r and r.get("visible") == "1" and r.get("x") and r.get("y"):
            x = int(round(float(r["x"])))
            y = int(round(float(r["y"])))
            cv2.circle(out, (x, y), 16, (255, 255, 255), 2)
            cv2.drawMarker(out, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

        return out

    def draw_timeline(self, img):
        out = img.copy()

        y = self.h - 34
        x1 = 120
        x2 = self.w - 120
        cv2.rectangle(out, (x1, y), (x2, y + 8), (60, 60, 60), -1)

        span = max(1, self.end_frame - self.start_frame)

        for fr, r in self.labels.items():
            if fr < self.start_frame or fr > self.end_frame:
                continue

            t = (fr - self.start_frame) / span
            x = int(round(x1 + t * (x2 - x1)))

            if r.get("visible") == "1":
                col = (0, 255, 255)
            elif r.get("visible") == "0":
                col = (80, 80, 255)
            else:
                col = (160, 160, 160)

            cv2.line(out, (x, y - 7), (x, y + 14), col, 1)

        tcur = (self.frame - self.start_frame) / span
        xcur = int(round(x1 + tcur * (x2 - x1)))
        cv2.line(out, (xcur, y - 14), (xcur, y + 22), (255, 255, 255), 2)

        return out

    def draw(self):
        img = self.read_frame()
        out = self.draw_trajectory(img)
        out = self.draw_timeline(out)

        r = self.labels.get(self.frame)
        status = "unlabeled"
        if r and r.get("label"):
            status = f"{r.get('label')} / {r.get('point_type')}"

        header = (
            f"{PATCH_ID} | frame={self.frame} | "
            f"{self.frame - self.start_frame + 1}/{self.end_frame - self.start_frame + 1} | "
            f"t={self.frame / self.fps:.2f}s | {status}"
        )
        controls = "A/Left prev | E/Right next | click ball | V invisible | U unsure | R reset | S interpolate | D save | Q save+quit"

        draw_text_bg(out, header, (12, 28), scale=0.55)
        draw_text_bg(out, controls, (12, self.h - 58), scale=0.48)

        if self.scale != 1.0:
            out = cv2.resize(out, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)

        return out

    def run(self):
        print("=" * 72)
        print(f"PATCH {PATCH_ID}")
        print(f"video = {self.video_path}")
        print(f"range = {self.start_frame}->{self.end_frame}")
        print(f"out   = {self.csv_path}")
        print("=" * 72)
        print("")
        print("Contrôles : A/← prev | E/→ next | clic gauche balle | clic droit reset | V invisible | U unsure | R reset | S interpolate | D save | Q save+quit")
        print("")

        running = True

        while running:
            cv2.imshow(self.window, self.draw())
            key = cv2.waitKeyEx(0)

            low = key & 0xFF
            ch = chr(low).lower() if 0 <= low < 128 else ""

            if ch == "a" or key == 2424832:
                self.step(-1)

            elif ch == "e" or key == 2555904:
                self.step(1)

            elif ch == "v":
                self.mark_invisible()
                self.step(1)

            elif ch == "u":
                self.mark_unsure()
                self.step(1)

            elif ch == "r":
                self.reset_current()

            elif ch == "s":
                self.interpolate()
                self.save()

            elif ch == "d":
                self.save()

            elif ch == "q" or key == 27:
                self.save()
                running = False

        self.cap.release()
        cv2.destroyAllWindows()

        rows = list(self.labels.values())
        manual = sum(1 for r in rows if r.get("point_type") == "manual")
        inter = sum(1 for r in rows if r.get("point_type") == "interpolated")
        invisible = sum(1 for r in rows if r.get("label") == "invisible")
        unsure = sum(1 for r in rows if r.get("label") == "unsure")

        print("")
        print("OK 006B")
        print(f"csv     = {self.csv_path}")
        print(f"summary = {self.summary_path}")
        print(f"labels={len(rows)} manual={manual} interpolated={inter} invisible={invisible} unsure={unsure}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="runs/005F3_align_clean_video/005F3_clean_aligned_segment.mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--display-width", type=int, default=1100)
    ap.add_argument("--out-dir", default="runs/006B_human_trajectory_labels")
    ap.add_argument("--new", action="store_true")
    args = ap.parse_args()

    tool = TrajectoryTool(args)
    tool.run()


if __name__ == "__main__":
    main()
