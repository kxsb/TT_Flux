from pathlib import Path

import pytest

from ttflux.analysis.tracks import (
    CandidatePoint,
    Hypothesis,
    TrackConfig,
    build_tracklets,
    evaluate_link,
    read_candidates,
)


def point(
    frame: int,
    x: float,
    y: float,
    candidate_id: str | None = None,
) -> CandidatePoint:
    return CandidatePoint(
        candidate_id=candidate_id or f"c{frame}",
        frame=frame,
        time_s=frame / 50.0,
        x=x,
        y=y,
        score=0.8,
        area=20.0,
        mean_brightness=220.0,
        bbox_w=5.0,
        bbox_h=5.0,
        rank=1,
    )


def test_read_candidates_requires_core_columns(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text("candidate_id,frame\na,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Colonnes"):
        read_candidates(path)


def test_first_link_can_establish_velocity() -> None:
    config = TrackConfig(max_jump_per_frame=60.0)
    hypothesis = Hypothesis(points=[point(0, 10.0, 10.0)])

    result = evaluate_link(
        hypothesis,
        point(1, 55.0, 10.0),
        config,
    )

    assert result is not None
    assert result[1] == 0.0


def test_abrupt_acceleration_breaks_tracklet() -> None:
    config = TrackConfig(
        prediction_error_limit_px=100.0,
        max_acceleration_px_per_frame2=12.0,
    )
    hypothesis = Hypothesis(
        points=[
            point(0, 0.0, 0.0, "a"),
            point(1, 10.0, 0.0, "b"),
        ]
    )

    assert evaluate_link(
        hypothesis,
        point(2, 45.0, 0.0, "c"),
        config,
    ) is None


def test_tracklets_are_bounded_by_max_span() -> None:
    candidates = [
        point(frame, 100.0 + 3.0 * frame, 200.0 + frame)
        for frame in range(60)
    ]
    config = TrackConfig(
        max_track_span_frames=18,
        max_output_tracks=8,
        min_points=5,
        seed_stride_frames=3,
    )

    tracks, counters = build_tracklets(candidates, config)

    assert tracks
    assert counters["eligible_hypotheses"] > 0
    assert max(track.span_frames for track in tracks) <= 18


def test_tracklets_do_not_cross_scene_cut() -> None:
    candidates = [
        point(frame, 50.0 + 2.0 * frame, 100.0)
        for frame in range(30)
    ]
    config = TrackConfig(
        max_track_span_frames=30,
        max_output_tracks=10,
        min_points=4,
        seed_stride_frames=3,
    )

    tracks, _ = build_tracklets(
        candidates,
        config,
        scene_cuts={15},
    )

    assert tracks
    assert all(
        not (track.first_frame < 15 <= track.last_frame)
        for track in tracks
    )
