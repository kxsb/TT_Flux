from pathlib import Path

p = Path("tools/evaluate_crop_separability_002E.py")
text = p.read_text(encoding="utf-8")

helper = r'''
def imread_unicode(path: Path) -> np.ndarray | None:
    """Lecture image compatible chemins Windows avec accents."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 92) -> bool:
    """Écriture image compatible chemins Windows avec accents."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False
'''

if "def imread_unicode(" not in text:
    marker = "def extract_features(img: np.ndarray) -> np.ndarray:"
    if marker not in text:
        raise SystemExit("marker not found: extract_features")
    text = text.replace(marker, helper + "\n\n" + marker, 1)

text = text.replace(
    "img = cv2.imread(str(crop_path))",
    "img = imread_unicode(crop_path)",
)

text = text.replace(
    "img = cv2.imread(str(crop_path))",
    "img = imread_unicode(crop_path)",
)

text = text.replace(
    "cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])",
    "imwrite_unicode(path, sheet, quality=92)",
)

# Sécurité : message clair si aucun crop n'est lisible.
old = "return usable, np.vstack(feats), np.array(labels, dtype=np.int32)"
new = """if not feats:
        raise SystemExit(
            "[002E] Aucun crop lisible. Vérifie le manifest 002D et les chemins. "
            "Si le chemin contient un accent, ce patch 002E2 doit être appliqué."
        )

    return usable, np.vstack(feats), np.array(labels, dtype=np.int32)"""

if old in text:
    text = text.replace(old, new, 1)

p.write_text(text, encoding="utf-8")
print("[002E2] patched:", p)
