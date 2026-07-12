"""
Compatibility layer for the historical analysis namespace.

The canonical temporal tracking implementation now lives under
ttflux.tracking.temporal.tracklets.
"""

from importlib import import_module
from typing import Any

from ttflux.tracking.temporal.tracklets import (
    TRACK_CSV_FIELDS,
    CandidatePoint,
    Hypothesis,
    TrackConfig,
    analyze_tracks,
    build_tracklets,
    detect_scene_cuts,
    evaluate_link,
    group_candidates,
    read_candidates,
    summarize_track,
)


_CANONICAL_MODULE = import_module(
    "ttflux.tracking.temporal.tracklets"
)


__all__ = [
    "TRACK_CSV_FIELDS",
    "CandidatePoint",
    "Hypothesis",
    "TrackConfig",
    "analyze_tracks",
    "build_tracklets",
    "detect_scene_cuts",
    "evaluate_link",
    "group_candidates",
    "read_candidates",
    "summarize_track",
]


def __getattr__(name: str) -> Any:
    """
    Preserve access to historical internal helpers during migration.
    """

    try:
        return getattr(
            _CANONICAL_MODULE,
            name,
        )
    except AttributeError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | set(dir(_CANONICAL_MODULE))
    )
