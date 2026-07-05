import json
import shutil
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import cv2
    from PIL import Image, ImageTk
except Exception as exc:
    cv2 = None
    Image = None
    ImageTk = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


APP_ROOT = Path(__file__).resolve().parent
VIDEOS_RAW = APP_ROOT / "videos_raw"
RUNS_DIR = APP_ROOT / "runs"

VIDEOS_RAW.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)


class VideoPlayer:
    def __init__(self, parent):
        self.parent = parent
        self.label = ttk.Label(parent, text="Aucune vidéo", anchor="center")
        self.label.pack(fill="both", expand=True)

        self.cap = None
        self.path = None
        self.after_id = None
        self.playing = False
        self.fps = 25
        self.frame_delay_ms = 40
        self.last_frame_time = 0

    def stop(self):
        self.playing = False
        if self.after_id is not None:
            try:
                self.parent.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def open(self, path):
        self.stop()

        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))

        if not self.cap.isOpened():
            self.label.configure(text=f"Impossible d'ouvrir la vidéo:\n{self.path}")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 1:
            self.fps = fps
        else:
            self.fps = 25

        self.frame_delay_ms = max(15, int(1000 / self.fps))
        self.playing = True
        self.last_frame_time = time.time()
        self._next_frame()

    def _next_frame(self):
        if not self.playing or self.cap is None:
            return

        ok, frame = self.cap.read()

        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                self.label.configure(text="Fin / vidéo illisible")
                self.stop()
                return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        widget_w = max(320, self.label.winfo_width())
        widget_h = max(240, self.label.winfo_height())

        h, w = frame.shape[:2]
        scale = min(widget_w / w, widget_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(img)

        self.label.configure(image=photo, text="")
        self.label.image = photo

        self.after_id = self.parent.after(self.frame_delay_ms, self._next_frame)


class TTFluxV3App:
    def __init__(self, root):
        self.root = root
        self.root.title("TTFlux V3")
        self.root.geometry("1100x700")
        self.root.minsize(900, 560)

        self.selected_video = None
        self.analysis_video = None
        self.current_run_dir = None

        self._build_ui()

        if IMPORT_ERROR is not None:
            messagebox.showerror(
                "Dépendance manquante",
                "Impossible de charger OpenCV/Pillow.\n\n"
                "Lance :\n"
                "python -m pip install -r requirements.txt\n\n"
                f"Détail : {IMPORT_ERROR}"
            )

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, width=260)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        title = ttk.Label(left, text="TTFlux V3", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w", pady=(0, 16))

        self.btn_import = ttk.Button(left, text="Importer", command=self.import_video)
        self.btn_import.pack(fill="x", pady=4)

        self.btn_analyze = ttk.Button(left, text="Analyser", command=self.analyze_video, state="disabled")
        self.btn_analyze.pack(fill="x", pady=4)

        ttk.Separator(left).pack(fill="x", pady=16)

        self.status = tk.StringVar(value="Statut : aucune vidéo importée")
        status_label = ttk.Label(left, textvariable=self.status, wraplength=235, justify="left")
        status_label.pack(anchor="w", fill="x")

        ttk.Separator(left).pack(fill="x", pady=16)

        self.path_label = ttk.Label(left, text="Vidéo : -", wraplength=235, justify="left")
        self.path_label.pack(anchor="w", fill="x")

        self.run_label = ttk.Label(left, text="Run : -", wraplength=235, justify="left")
        self.run_label.pack(anchor="w", fill="x", pady=(10, 0))

        video_box = ttk.LabelFrame(right, text="Retour vidéo")
        video_box.pack(fill="both", expand=True)

        self.player = VideoPlayer(video_box)

    def import_video(self):
        path = filedialog.askopenfilename(
            title="Importer une vidéo",
            filetypes=[
                ("Vidéos", "*.mp4 *.mov *.avi *.mkv *.webm"),
                ("Tous les fichiers", "*.*"),
            ],
        )

        if not path:
            return

        src = Path(path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = VIDEOS_RAW / f"{timestamp}_{src.name}"

        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            messagebox.showerror("Erreur import", str(exc))
            return

        self.selected_video = dst
        self.analysis_video = None
        self.current_run_dir = None

        self.status.set("Statut : vidéo importée")
        self.path_label.configure(text=f"Vidéo : {dst.name}")
        self.run_label.configure(text="Run : -")
        self.btn_analyze.configure(state="normal")

        self.player.open(dst)

    def analyze_video(self):
        if self.selected_video is None:
            messagebox.showwarning("Analyse impossible", "Importe d'abord une vidéo.")
            return

        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        output_video = run_dir / "analysis_return.mp4"
        summary_json = run_dir / "summary.json"

        self.status.set("Statut : analyse en cours...")
        self.root.update_idletasks()

        try:
            # Placeholder V3.
            # Ici on branchera ensuite le vrai tracking balle.
            shutil.copy2(self.selected_video, output_video)

            summary = {
                "ttflux_version": "V3",
                "analysis_status": "placeholder",
                "message": "Analyse non branchée : retour vidéo = copie de la vidéo importée.",
                "input_video": str(self.selected_video),
                "output_video": str(output_video),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }

            summary_json.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        except Exception as exc:
            self.status.set("Statut : erreur analyse")
            messagebox.showerror("Erreur analyse", str(exc))
            return

        self.analysis_video = output_video
        self.current_run_dir = run_dir

        self.status.set("Statut : analyse terminée")
        self.run_label.configure(text=f"Run : {run_dir.name}")
        self.player.open(output_video)


def main():
    root = tk.Tk()
    app = TTFluxV3App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
