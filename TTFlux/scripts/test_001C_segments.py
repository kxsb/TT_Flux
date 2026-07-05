from ttflux.config import TTFluxConfig
from ttflux.track_io import read_track_points
from ttflux.segments import (
    find_candidate_segments,
    select_non_overlapping,
)


def main():
    cfg = TTFluxConfig.from_json(
        "configs/pingcoach_v62_m2forqbqazc_s01.json"
    )

    points = read_track_points(
        cfg.track_csv,
        cfg.clip_id,
    )

    candidates = find_candidate_segments(
        points,
        cfg.window_span,
        cfg.min_points,
    )

    selected = select_non_overlapping(
        candidates,
        cfg.top_k,
        cfg.non_overlap_margin,
    )

    print("TTFLUX_001C_OK")
    print("points =", len(points))
    print("candidates =", len(candidates))
    print("selected =", len(selected))

    for i, seg in enumerate(selected, start=1):
        print(
            i,
            "frames",
            seg.first_frame,
            "to",
            seg.last_frame,
            "points",
            len(seg.points),
            "score",
            round(seg.score, 3),
        )


if __name__ == "__main__":
    main()
