# PATCH 001M2 - Embed contact sheet PNGs into a portable HTML file
#
# Reads patch_review_001M.html and inlines every local PNG as base64.
# Output is a single self-contained HTML file that can be uploaded/shared.
#
# Usage:
#   python tools\embed_contact_sheets_001M2.py ^
#     --in-html runs\batch_001E\patch_review_001M.html ^
#     --out-html runs\batch_001E\patch_review_001M2_embedded.html

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
from pathlib import Path


IMG_RE = re.compile(r'(<img\b[^>]*\bsrc=")([^"]+)(")', re.IGNORECASE)


def path_to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def embed_images(in_html: Path, out_html: Path) -> tuple[int, list[str]]:
    text = in_html.read_text(encoding="utf-8")
    base_dir = in_html.parent
    missing: list[str] = []
    embedded_count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal embedded_count

        prefix = match.group(1)
        src = match.group(2)
        suffix = match.group(3)

        if src.startswith("data:") or src.startswith("http://") or src.startswith("https://"):
            return match.group(0)

        img_path = (base_dir / src.replace("/", "\\")).resolve()

        if not img_path.exists():
            missing.append(src)
            return match.group(0)

        data_uri = path_to_data_uri(img_path)
        embedded_count += 1

        return f"{prefix}{data_uri}{suffix}"

    out = IMG_RE.sub(repl, text)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(out, encoding="utf-8")

    return embedded_count, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-html", default="runs/batch_001E/patch_review_001M.html")
    parser.add_argument("--out-html", default="runs/batch_001E/patch_review_001M2_embedded.html")
    args = parser.parse_args()

    in_html = Path(args.in_html)
    out_html = Path(args.out_html)

    count, missing = embed_images(in_html, out_html)

    print(f"[001M2] input HTML  : {in_html}")
    print(f"[001M2] output HTML : {out_html}")
    print(f"[001M2] embedded    : {count}")
    print(f"[001M2] missing     : {len(missing)}")

    if missing:
        print("[001M2] missing files:")
        for item in missing[:20]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()