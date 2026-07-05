from pathlib import Path
import json
import cv2

roots = [
    Path(r"C:\Users\micka\pingcoach_probe"),
    Path(r"C:\Users\micka\Desktop\développement\ping"),
    Path.cwd(),
]

patterns = [
    "dataset2_manifest*.json",
    "normalized_720p50.mp4",
    "source.mp4",
    "*.mp4",
]

print("004A2_SOURCE_LOCATOR")
print("cwd =", Path.cwd())

for root in roots:
    print("")
    print("ROOT", root, "exists=", root.exists())

    if not root.exists():
        continue

    for pat in patterns:
        found = list(root.rglob(pat))
        print(" pattern", pat, "count=", len(found))

        for p in found[:40]:
            kind = "mp4" if p.suffix.lower() == ".mp4" else "json"
            extra = ""

            if kind == "mp4":
                cap = cv2.VideoCapture(str(p))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    sec = frames / fps if fps else 0
                    extra = f" | {w}x{h} fps={fps:.2f} min={sec/60:.2f}"
                else:
                    extra = " | opencv_open_failed"
                cap.release()

            print("  ", p, extra)
