from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


def load_tracks(path: Path) -> dict[str, list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "track_id", "track_rank", "frame", "x", "y",
            "prediction_error_px",
        }
        missing = required.difference(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "Colonnes absentes : " + ", ".join(sorted(missing))
            )

        for row in reader:
            track_id = row["track_id"]
            tracks[track_id].append(
                {
                    "track_id": track_id,
                    "rank": int(row["track_rank"]),
                    "frame": int(row["frame"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "prediction_error_px": float(
                        row["prediction_error_px"]
                    ),
                }
            )

    for points in tracks.values():
        points.sort(key=lambda point: point["frame"])

    return dict(tracks)


def load_metrics(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["track_id"]: item
        for item in payload.get("tracks", [])
        if isinstance(item, dict) and item.get("track_id")
    }


def build_rows(
    tracks: dict[str, list[dict[str, Any]]],
    metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for track_id, points in tracks.items():
        metric = metrics.get(track_id, {})
        first_frame = int(metric.get("first_frame", points[0]["frame"]))
        last_frame = int(metric.get("last_frame", points[-1]["frame"]))

        rows.append(
            {
                "track_id": track_id,
                "rank": int(metric.get("rank", points[0]["rank"])),
                "point_count": int(metric.get("point_count", len(points))),
                "first_frame": first_frame,
                "last_frame": last_frame,
                "span_frames": int(
                    metric.get(
                        "span_frames",
                        last_frame - first_frame + 1,
                    )
                ),
                "score": float(metric.get("score", 0.0)),
                "mean_prediction_error_px": float(
                    metric.get(
                        "mean_prediction_error_px",
                        sum(
                            point["prediction_error_px"]
                            for point in points
                        )
                        / len(points),
                    )
                ),
                "clip_path": f"clips/{track_id}.mp4",
                "label": "",
                "notes": "",
            }
        )

    rows.sort(key=lambda row: (row["rank"], row["track_id"]))
    return rows


def find_source_video(run_dir: Path) -> Path:
    for name in ("source_clip.mp4", "segment.mp4", "clip.mp4"):
        path = run_dir / name
        if path.is_file():
            return path

    candidates = [
        path for path in run_dir.glob("*.mp4")
        if not path.name.startswith("overlay_")
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        "Segment source introuvable ou ambigu dans le run."
    )


def draw_overlay(
    frame: Any,
    track_id: str,
    points: list[dict[str, Any]],
    frame_index: int,
) -> Any:
    output = frame.copy()
    visible = [
        point for point in points
        if point["frame"] <= frame_index
    ]

    for previous, current in zip(visible, visible[1:]):
        cv2.line(
            output,
            (round(previous["x"]), round(previous["y"])),
            (round(current["x"]), round(current["y"])),
            (255, 220, 0),
            3,
            cv2.LINE_AA,
        )

    current = next(
        (
            point for point in points
            if point["frame"] == frame_index
        ),
        None,
    )

    if current:
        center = (round(current["x"]), round(current["y"]))
        cv2.circle(output, center, 13, (0, 255, 255), 3)
        cv2.circle(output, center, 3, (0, 0, 255), -1)

    cv2.rectangle(output, (12, 12), (330, 62), (8, 12, 18), -1)
    cv2.putText(
        output,
        f"{track_id} | frame {frame_index}",
        (24, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (235, 240, 247),
        2,
        cv2.LINE_AA,
    )
    return output


def export_clips(
    source_video: Path,
    output_dir: Path,
    tracks: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    padding_frames: int,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source_video))

    if not capture.isOpened():
        raise RuntimeError(f"Vidéo illisible : {source_video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    review_fps = min(fps, 25.0)

    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    try:
        for row in rows:
            track_id = row["track_id"]
            points = tracks[track_id]
            first = max(0, row["first_frame"] - padding_frames)
            last = min(
                frame_count - 1,
                row["last_frame"] + padding_frames,
            )

            capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            writer = cv2.VideoWriter(
                str(output_dir / row["clip_path"]),
                cv2.VideoWriter_fourcc(*"mp4v"),
                review_fps,
                (width, height),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Écriture impossible : {row['clip_path']}"
                )

            try:
                frame_index = first

                while frame_index <= last:
                    ok, frame = capture.read()

                    if not ok:
                        break

                    writer.write(
                        draw_overlay(
                            frame,
                            track_id,
                            points,
                            frame_index,
                        )
                    )
                    frame_index += 1
            finally:
                writer.release()
    finally:
        capture.release()

    return {
        "source_video": source_video.name,
        "source_fps": fps,
        "review_fps": review_fps,
        "exported_tracks": len(rows),
        "padding_frames": padding_frames,
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else [
        "track_id", "rank", "point_count", "first_frame",
        "last_frame", "span_frames", "score",
        "mean_prediction_error_px", "clip_path", "label", "notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_html(rows: list[dict[str, Any]], run_id: str) -> str:
    cards = []

    for row in rows:
        track_id = html.escape(row["track_id"])
        cards.append(
            f"""
<article data-id="{track_id}">
<header><b>{track_id}</b><span>rang {row['rank']} ·
{row['point_count']} points · {row['span_frames']} frames ·
erreur {row['mean_prediction_error_px']:.2f}px</span></header>
<video controls preload="metadata" src="{html.escape(row['clip_path'])}"></video>
<div class="labels">
<button data-label="ball">Balle</button>
<button data-label="not_ball">Pas balle</button>
<button data-label="uncertain">Incertain</button>
</div>
<textarea placeholder="Note facultative"></textarea>
</article>"""
        )

    cards_html = "\n".join(cards)
    safe_run_id = html.escape(run_id)

    return f"""<!doctype html>
<html lang="fr">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TTFlux — revue tracklets</title>
<style>
:root{{color-scheme:dark;font-family:system-ui;background:#0b0e13;color:#e7edf5}}
body{{margin:0}} .top{{position:sticky;top:0;z-index:2;display:flex;
justify-content:space-between;align-items:center;padding:14px 18px;
background:#0b0e13ee;border-bottom:1px solid #29313b}}
h1{{margin:0;font-size:18px}} .top span,header span{{color:#8d99a8;font-size:11px}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));
gap:12px;padding:14px}} article{{padding:11px;border:1px solid #27303b;
border-radius:10px;background:#11161d}} article header{{display:flex;
justify-content:space-between;gap:10px;margin-bottom:8px}} video{{width:100%;
max-height:340px;background:#000}} .labels{{display:grid;
grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}}
button,textarea{{color:#dfe7f0;border:1px solid #354150;border-radius:7px;
background:#171d25}} button{{padding:8px;cursor:pointer}}
button.active{{border-color:#77a7d9;background:#294766}} textarea{{width:100%;
margin-top:7px;padding:7px;box-sizing:border-box}} #export{{padding:9px 12px}}
</style>
<header class="top"><div><h1>Revue des tracklets</h1>
<span>{safe_run_id} — la forme n'est pas utilisée comme vérité</span></div>
<div><span id="progress"></span> <button id="export">Exporter CSV</button></div>
</header>
<main>{cards_html}</main>
<script>
const key="ttflux-review:{safe_run_id}";
const state=JSON.parse(localStorage.getItem(key)||"{{}}");
const save=()=>{{localStorage.setItem(key,JSON.stringify(state));progress()}};
const progress=()=>{{const n=document.querySelectorAll("article").length;
const d=Object.values(state).filter(x=>x.label).length;
document.querySelector("#progress").textContent=`${{d}} / ${{n}}`;}}
document.querySelectorAll("article").forEach(card=>{{
 const id=card.dataset.id, text=card.querySelector("textarea");
 const item=state[id]||{{label:"",notes:""}}; text.value=item.notes||"";
 card.querySelectorAll("[data-label]").forEach(b=>{{
  if(b.dataset.label===item.label)b.classList.add("active");
  b.onclick=()=>{{card.querySelectorAll("[data-label]").forEach(x=>x.classList.remove("active"));
   b.classList.add("active");state[id]={{label:b.dataset.label,notes:text.value}};save();}};
 }});
 text.oninput=()=>{{state[id]={{label:(state[id]||{{}}).label||"",notes:text.value}};save();}};
}});
document.querySelector("#export").onclick=()=>{{
 const lines=[["track_id","label","notes"]];
 document.querySelectorAll("article").forEach(card=>{{const x=state[card.dataset.id]||{{label:"",notes:""}};
 lines.push([card.dataset.id,x.label,x.notes]);}});
 const q=x=>`"${{String(x??"").replaceAll('"','""')}}"`;
 const blob=new Blob([lines.map(r=>r.map(q).join(",")).join("\\r\\n")],
 {{type:"text/csv;charset=utf-8"}});const a=document.createElement("a");
 a.href=URL.createObjectURL(blob);a.download="tracklet_labels.csv";a.click();
 URL.revokeObjectURL(a.href);
}};
progress();
</script>
</html>"""


def generate_review(
    run_dir: Path,
    padding_frames: int = 10,
) -> Path:
    tracks = load_tracks(run_dir / "tracks_probe.csv")
    metrics = load_metrics(run_dir / "tracks_metrics.json")
    rows = build_rows(tracks, metrics)
    output_dir = run_dir / "tracklet_review"
    output_dir.mkdir(exist_ok=True)

    summary = export_clips(
        find_source_video(run_dir),
        output_dir,
        tracks,
        rows,
        padding_frames,
    )

    write_manifest(
        output_dir / "tracklet_review_manifest.csv",
        rows,
    )
    (output_dir / "index.html").write_text(
        render_html(rows, run_dir.name),
        encoding="utf-8",
    )
    (output_dir / "review_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "track_count": len(rows),
                "labels": ["ball", "not_ball", "uncertain"],
                "video": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--padding-frames", type=int, default=10)
    args = parser.parse_args()

    output = generate_review(
        args.run_dir.resolve(),
        args.padding_frames,
    )

    print(f"Revue créée : {output}")
    print(f"Page : {output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
