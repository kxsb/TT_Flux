import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import shutil
import threading
import time
import cv2
from PIL import Image, ImageTk


APP_ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = APP_ROOT / "videos"
RUNS_DIR = APP_ROOT / "runs"


class TTFluxV3App:
    def __init__(self, root):
        self.root = root
        self.root.title("TTFlux V3")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self.video_path = None
        self.analysis_video_path = None
        self.cap = None
        self.playing = False
        self.frame_delay_ms = 30

        self.build_ui()

    def build_ui(self):
        self.root.configure(bg="#15171c")

        top = tk.Frame(self.root, bg="#15171c")
        top.pack(fill="x", padx=16, pady=12)

        title = tk.Label(
            top,
            text="TTFlux V3",
            fg="#f2f2f2",
            bg="#15171c",
            font=("Segoe UI", 20, "bold"),
        )
        title.pack(side="left")

        self.import_btn = tk.Button(
            top,
            text="Importer",
            command=self.import_video,
            width=14,
            height=2,
        )
        self.import_btn.pack(side="right", padx=6)

        self.analyze_btn = tk.Button(
            top,
            text="Analyser",
            command=self.analyze_video,
            width=14,
            height=2,
            state="disabled",
        )
        self.analyze_btn.pack(side="right", padx=6)

        self.status = tk.Label(
            self.root,
            text="Aucune vidéo importée.",
            fg="#cfcfcf",
            bg="#15171c",
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.status.pack(fill="x", padx=16)

        body = tk.Frame(self.root, bg="#15171c")
        body.pack(fill="both", expand=True, padx=16, pady=12)

        self.video_label = tk.Label(
            body,
            bg="#050608",
            fg="#999",
            text="Retour vidéo analyse",
            font=("Segoe UI", 16),
        )
        self.video_label.pack(fill="both", expand=True)

        bottom = tk.Frame(self.root, bg="#15171c")
        bottom.pack(fill="x", padx=16, pady=(0, 12))

        self.path_label = tk.Label(
            bottom,
            text="",
            fg="#8f98a8",
            bg="#15171c",
            anchor="w",
            font=("Segoe UI", 9),
        )
        self.path_label.pack(fill="x")

    def import_video(self):
        src = filedialog.askopenfilename(
            title="Importer une vidéo",
            filetypes=[
                ("Vidéos", "*.mp4 *.avi *.mov *.mkv"),
                ("Tous les fichiers", "*.*"),
            ],
        )

        if not src:
            return

        src = Path(src)
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

        dst = VIDEOS_DIR / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

        self.video_path = dst
        self.analysis_video_path = None

        self.status.config(text=f"Vidéo importée : {dst.name}")
        self.path_label.config(text=str(dst))
        self.analyze_btn.config(state="normal")

        self.load_video(dst)

    def analyze_video(self):
        if not self.video_path:
            messagebox.showwarning("TTFlux V3", "Importe d'abord une vidéo.")
            return

        self.analyze_btn.config(state="disabled")
        self.import_btn.config(state="disabled")
        self.status.config(text="Analyse en cours...")

        thread = threading.Thread(target=self.fake_analysis_worker, daemon=True)
        thread.start()

    def fake_analysis_worker(self):
        try:
            run_id = time.strftime("run_%Y%m%d_%H%M%S")
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            output_path = run_dir / "analysis_preview.mp4"
            self.make_mock_analysis_video(self.video_path, output_path)

            self.analysis_video_path = output_path

            self.root.after(0, lambda: self.status.config(text=f"Analyse terminée : {output_path.name}"))
            self.root.after(0, lambda: self.path_label.config(text=str(output_path)))
            self.root.after(0, lambda: self.load_video(output_path))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erreur analyse", str(e)))
            self.root.after(0, lambda: self.status.config(text="Erreur pendant l'analyse."))

        finally:
            self.root.after(0, lambda: self.analyze_btn.config(state="normal"))
            self.root.after(0, lambda: self.import_btn.config(state="normal"))

    def make_mock_analysis_video(self, input_path, output_path):
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la vidéo importée.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # MOCK analyse : overlay simple. Le vrai tracking viendra ici.
            cv2.rectangle(frame, (20, 20), (430, 82), (0, 0, 0), -1)
            cv2.putText(
                frame,
                "TTFlux V3 - analyse preview",
                (36, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Faux point mobile pour simuler un retour de tracking.
            x = int((frame_idx * 7) % max(w, 1))
            y = int(h * 0.45 + 80 * __import__("math").sin(frame_idx / 12))
            cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

    def load_video(self, path):
        self.stop_video()

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            messagebox.showerror("TTFlux V3", "Impossible de lire la vidéo.")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_delay_ms = max(10, int(1000 / fps))

        self.playing = True
        self.play_loop()

    def stop_video(self):
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def play_loop(self):
        if not self.playing or self.cap is None:
            return

        ok, frame = self.cap.read()

        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()

        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            label_w = max(self.video_label.winfo_width(), 320)
            label_h = max(self.video_label.winfo_height(), 240)

            img_h, img_w = frame.shape[:2]
            scale = min(label_w / img_w, label_h / img_h)
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))

            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)

            self.video_label.config(image=photo, text="")
            self.video_label.image = photo

        self.root.after(self.frame_delay_ms, self.play_loop)

    def on_close(self):
        self.stop_video()
        self.root.destroy()


def main():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    app = TTFluxV3App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
