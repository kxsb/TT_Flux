from pathlib import Path

from ttflux.video.catalog import (
    parse_frame_rate,
    stable_video_id,
)


def test_stable_video_id_is_reproducible() -> None:
    path = Path("matchs") / "video_01.mp4"

    assert stable_video_id(path) == stable_video_id(path)


def test_stable_video_id_changes_with_path() -> None:
    first = stable_video_id(Path("a.mp4"))
    second = stable_video_id(Path("b.mp4"))

    assert first != second


def test_parse_frame_rate() -> None:
    assert parse_frame_rate("120/1") == 120.0
    assert parse_frame_rate("30000/1001") == 29.97
    assert parse_frame_rate("0/0") is None