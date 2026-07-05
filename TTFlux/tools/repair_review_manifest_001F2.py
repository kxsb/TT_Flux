# PATCH 001F2 — Repair TTFlux review manifest identity
#
# Fixes review_manifest_001F.csv when:
# - clip_id was inferred as validation_XX_f... instead of the batch clip folder
# - segment_idx stayed at 0
#
# Adds light review assistance columns:
# - segment_name
# - review_guess
# - risk_flags
#
# Usage:
#   python tools\repair_review_manifest_001F2.py ^
#     --in-csv runs\batch_001E\review_manifest_001F.csv ^
#     --out-csv runs\batch_001E\review_manifest_001F2_repaired.csv ^
#     --out-html runs\batch_001E\review_manifest_001F2_repaired.html

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from pathlib import Path
from typing import Any


VALIDATION_RE = re.compile(
    r"(validation)_(\d+)_f(\d+)_to_f(\d+)_n(\d+)",
    re.IGNORECASE,
)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text or text.lower() == "nan":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def clean_path(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    # In case previous manifest contains multiple files joined by " | ".
    return text.split("|")[0].strip()


def split_rel_path(path_text: str) -> list[str]:
    path_text = path_text.replace("\\", "/").strip("/")
    return [p for p in path_text.split("/") if p]


def infer_clip_id(row: dict[str, str]) -> str:
    for key in ["mp4", "csv", "overlay_png"]:
        path_text = clean_path(row.get(key, ""))
        parts = split_rel_path(path_text)
        if len(parts) >= 2:
            return parts[0]

    # Fallback: keep previous value if we cannot recover.
    return str(row.get("clip_id", "")).strip()


def infer_segment_name(row: dict[str, str]) -> str:
    for key in ["mp4", "csv", "overlay_png"]:
        path_text = clean_path(row.get(key, ""))
        if not path_text:
            continue
        stem = Path(path_text.replace("\\", "/")).stem
        if stem:
            return stem

    return str(row.get("clip_id", "")).strip()


def infer_segment_idx(segment_name: str, fallback: Any = "") -> str:
    match = VALIDATION_RE.search(segment_name)
    if match:
        return str(int(match.group(2)))

    text = str(fallback).strip()
    if text and text.lower() != "nan":
        return text

    return "0"


def parse_validation_frames(segment_name: str) -> tuple[str, str, str]:
    match = VALIDATION_RE.search(segment_name)
    if not match:
        return "", "", ""
    first_frame = str(int(match.group(3)))
    last_frame = str(int(match.group(4)))
    n_points = str(int(match.group(5)))
    return first_frame, last_frame, n_points


def review_flags(row: dict[str, str]) -> tuple[str, str]:
    density = safe_float(row.get("density"))
    travel = safe_float(row.get("travel"))
    x_range = safe_float(row.get("x_range"))
    y_range = safe_float(row.get("y_range"))
    n_points = safe_float(row.get("n_points"))

    flags: list[str] = []
    risk = 0

    if density is not None:
        if density < 0.25:
            flags.append("very_low_density")
            risk += 3
        elif density < 0.33:
            flags.append("low_density")
            risk += 2

    if n_points is not None:
        if n_points < 16:
            flags.append("few_points")
            risk += 1
        elif n_points >= 26:
            flags.append("many_points")
            risk -= 1

    if travel is not None:
        if travel > 900:
            flags.append("huge_travel")
            risk += 3
        elif travel > 700:
            flags.append("high_travel")
            risk += 2

    if x_range is not None:
        if x_range > 650:
            flags.append("huge_x_range")
            risk += 3
        elif x_range > 420:
            flags.append("wide_x_range")
            risk += 2

    if y_range is not None:
        if y_range > 380:
            flags.append("huge_y_range")
            risk += 3
        elif y_range > 220:
            flags.append("wide_y_range")
            risk += 2

    if x_range is not None and y_range is not None:
        if x_range < 90 and y_range < 90:
            flags.append("compact_track")
            risk -= 1

    if risk >= 6:
        guess = "high_risk"
    elif risk >= 3:
        guess = "medium_risk"
    elif risk <= 0:
        guess = "plausible"
    else:
        guess = "check"

    if not flags:
        flags.append("none")

    return guess, ",".join(flags)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    base_columns = [
        "review_id",
        "clip_id",
        "segment_idx",
        "segment_name",
        "first_frame",
        "last_frame",
        "n_points",
        "score",
        "density",
        "travel",
        "x_range",
        "y_range",
        "mp4",
        "overlay_png",
        "csv",
        "review_guess",
        "risk_flags",
        "review_status",
        "notes",
    ]

    extra = []
    seen = set(base_columns)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                extra.append(key)

    columns = base_columns + extra

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def html_link(path_text: str, label: str) -> str:
    path_text = clean_path(path_text)
    if not path_text:
        return ""
    href = html.escape(path_text.replace("\\", "/"))
    return f'<a href="{href}">{html.escape(label)}</a>'


def write_html(path: Path, rows: list[dict[str, str]], csv_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    table_rows = []

    for row in rows:
        guess = row.get("review_guess", "")
        tr_class = guess.replace("_", "-")

        table_rows.append(
            f"""
            <tr class="{html.escape(tr_class)}">
              <td>{html.escape(row.get("review_id", ""))}</td>
              <td><code>{html.escape(row.get("clip_id", ""))}</code></td>
              <td>{html.escape(row.get("segment_idx", ""))}</td>
              <td><code>{html.escape(row.get("segment_name", ""))}</code></td>
              <td>{html.escape(row.get("first_frame", ""))} → {html.escape(row.get("last_frame", ""))}</td>
              <td>{html.escape(row.get("n_points", ""))}</td>
              <td>{html.escape(row.get("density", ""))}</td>
              <td>{html.escape(row.get("travel", ""))}</td>
              <td>{html.escape(row.get("x_range", ""))} / {html.escape(row.get("y_range", ""))}</td>
              <td class="guess">{html.escape(row.get("review_guess", ""))}</td>
              <td class="flags">{html.escape(row.get("risk_flags", ""))}</td>
              <td>{html_link(row.get("mp4", ""), "mp4")}</td>
              <td>{html_link(row.get("csv", ""), "csv")}</td>
              <td class="review"></td>
              <td></td>
            </tr>
            """
        )

    counts = {}
    for row in rows:
        key = row.get("review_guess", "unknown")
        counts[key] = counts.get(key, 0) + 1

    counts_text = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items()))

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux review manifest 001F2 repaired</title>
<style>
  :root {{
    --bg: #111318;
    --panel: #181b22;
    --line: #2b303b;
    --text: #edf0f7;
    --muted: #9ea7b8;
    --accent: #83d4ff;
    --risk: #ff8a8a;
    --medium: #ffd37a;
    --ok: #9effb3;
  }}
  body {{
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  h1 {{
    margin: 0 0 8px;
    font-size: 24px;
  }}
  .meta {{
    color: var(--muted);
    margin-bottom: 20px;
  }}
  a {{
    color: var(--accent);
    text-decoration: none;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    border: 1px solid var(--line);
  }}
  th, td {{
    border-bottom: 1px solid var(--line);
    padding: 8px 10px;
    vertical-align: top;
    font-size: 13px;
  }}
  th {{
    position: sticky;
    top: 0;
    background: #20242e;
    z-index: 2;
    text-align: left;
  }}
  code {{
    color: #d7e4ff;
    font-size: 12px;
  }}
  tr.high-risk td {{
    background: rgba(255, 80, 80, 0.08);
  }}
  tr.medium-risk td {{
    background: rgba(255, 200, 80, 0.07);
  }}
  tr.plausible td {{
    background: rgba(100, 255, 150, 0.035);
  }}
  .guess {{
    font-weight: 700;
    white-space: nowrap;
  }}
  .flags {{
    color: var(--muted);
    max-width: 260px;
  }}
  .review {{
    min-width: 90px;
  }}
</style>
</head>
<body>
  <h1>TTFlux review manifest 001F2 repaired</h1>
  <div class="meta">
    Segments : {len(rows)} · {html.escape(counts_text)} · CSV :
    <a href="{html.escape(csv_name)}">{html.escape(csv_name)}</a>
  </div>

  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>clip_id</th>
        <th>segment</th>
        <th>segment_name</th>
        <th>frames</th>
        <th>points</th>
        <th>density</th>
        <th>travel</th>
        <th>x/y range</th>
        <th>guess</th>
        <th>flags</th>
        <th>mp4</th>
        <th>csv</th>
        <th>review</th>
        <th>notes</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def repair_manifest(in_csv: Path, out_csv: Path, out_html: Path | None) -> None:
    rows = read_manifest(in_csv)

    repaired = []
    for i, row in enumerate(rows, start=1):
        row = {k: "" if v is None else str(v) for k, v in row.items()}

        clip_id = infer_clip_id(row)
        segment_name = infer_segment_name(row)
        segment_idx = infer_segment_idx(segment_name, row.get("segment_idx", ""))

        first_from_name, last_from_name, n_from_name = parse_validation_frames(segment_name)

        row["review_id"] = row.get("review_id", "") or f"R{i:04d}"
        row["clip_id"] = clip_id
        row["segment_idx"] = segment_idx
        row["segment_name"] = segment_name

        # Keep computed values if present, but fill from filename if missing.
        row["first_frame"] = row.get("first_frame", "").strip() or first_from_name
        row["last_frame"] = row.get("last_frame", "").strip() or last_from_name
        row["n_points"] = row.get("n_points", "").strip() or n_from_name

        guess, flags = review_flags(row)
        row["review_guess"] = guess
        row["risk_flags"] = flags

        repaired.append(row)

    def sort_key(row: dict[str, str]) -> tuple[str, int, int]:
        try:
            seg = int(row.get("segment_idx", "0"))
        except ValueError:
            seg = 0
        try:
            first = int(float(row.get("first_frame", "0")))
        except ValueError:
            first = 0
        return row.get("clip_id", ""), seg, first

    repaired.sort(key=sort_key)

    # Rewrite review_id after sorting.
    for i, row in enumerate(repaired, start=1):
        row["review_id"] = f"R{i:04d}"

    write_csv(out_csv, repaired)

    if out_html:
        write_html(out_html, repaired, out_csv.name)

    counts = {}
    for row in repaired:
        key = row["review_guess"]
        counts[key] = counts.get(key, 0) + 1

    print(f"[001F2] input     : {in_csv}")
    print(f"[001F2] segments  : {len(repaired)}")
    print(f"[001F2] guesses   : {counts}")
    print(f"[001F2] wrote CSV : {out_csv}")
    if out_html:
        print(f"[001F2] wrote HTML: {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", default="runs/batch_001E/review_manifest_001F.csv")
    parser.add_argument("--out-csv", default="runs/batch_001E/review_manifest_001F2_repaired.csv")
    parser.add_argument("--out-html", default="runs/batch_001E/review_manifest_001F2_repaired.html")
    args = parser.parse_args()

    repair_manifest(
        in_csv=Path(args.in_csv),
        out_csv=Path(args.out_csv),
        out_html=Path(args.out_html) if args.out_html else None,
    )


if __name__ == "__main__":
    main()
    