from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttflux.config import TTFluxConfig
from ttflux.overlay import (
    make_segment_png,
    make_segment_video,
)
from ttflux.segments import (
    find_candidate_segments,
    select_non_overlapping,
)
from ttflux.track_io import (
    read_track_points,
    write_segment_csv,
)


def _segment_dict(
    idx: int,
    segment,
    png_path: Path,
    mp4_path: Path,
    csv_path: Path,
) -> dict:
    return {
        "idx": idx,
        "first_frame": segment.first_frame,
        "last_frame": segment.last_frame,
        "n_points": len(segment.points),
        "span": segment.span,
        "density": segment.density,
        "travel": segment.travel,
        "x_range": segment.x_range,
        "y_range": segment.y_range,
        "score": segment.score,
        "png_path": str(png_path),
        "mp4_path": str(mp4_path),
        "csv_path": str(csv_path),
    }


def run_validation(
    config_path: str | Path,
    out_dir: str | Path,
) -> dict:
    config = TTFluxConfig.from_json(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config.video_path.exists():
        raise FileNotFoundError(
            f"Vidéo introuvable: {config.video_path}"
        )

    if not config.track_csv.exists():
        raise FileNotFoundError(
            f"CSV introuvable: {config.track_csv}"
        )

    points = read_track_points(
        config.track_csv,
        config.clip_id,
    )

    if not points:
        raise RuntimeError(
            f"Aucun point pour clip_id={config.clip_id}"
        )

    candidates = find_candidate_segments(
        points,
        config.window_span,
        config.min_points,
    )

    selected = select_non_overlapping(
        candidates,
        config.top_k,
        config.non_overlap_margin,
    )

    if not selected:
        raise RuntimeError("Aucun segment sélectionné")

    summary = {
        "project": "TTFlux",
        "mode": "validation",
        "config_path": str(config_path),
        "clip_id": config.clip_id,
        "total_points_for_clip": len(points),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected": [],
    }

    for idx, segment in enumerate(selected, start=1):
        base = (
            f"validation_{idx:02d}"
            f"_f{segment.first_frame}"
            f"_to_f{segment.last_frame}"
            f"_n{len(segment.points)}"
        )

        png_path = out_dir / f"{base}.png"
        mp4_path = out_dir / f"{base}.mp4"
        csv_path = out_dir / f"{base}.csv"

        title = (
            f"TTFlux validation {idx:02d} | "
            f"frames {segment.first_frame}->{segment.last_frame} | "
            f"points={len(segment.points)}"
        )

        subtitle = (
            f"density={segment.density:.3f} | "
            f"travel={segment.travel:.1f} | "
            f"score={segment.score:.2f}"
        )

        make_segment_png(
            config.video_path,
            segment,
            png_path,
            title,
            subtitle,
        )

        make_segment_video(
            video_path=config.video_path,
            segment=segment,
            path=mp4_path,
            fps=config.fps,
            size=(config.frame_width, config.frame_height),
            trail_keep_last=config.trail_keep_last,
            title_prefix=f"TTFlux validation {idx:02d}",
        )

        write_segment_csv(csv_path, segment.points)

        summary["selected"].append(
            _segment_dict(
                idx,
                segment,
                png_path,
                mp4_path,
                csv_path,
            )
        )

    summary_path = out_dir / "validation_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ttflux",
    )

    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
    )

    p_val = sub.add_parser(
        "validate",
        help="Generate validation overlays",
    )

    p_val.add_argument(
        "--config",
        required=True,
    )

    p_val.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    if args.cmd == "validate":
        summary = run_validation(
            args.config,
            args.out,
        )

        print("TTFLUX_VALIDATE_OK")
        print(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
