from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "BallCandidateScorer",
    "BallTrackingArtifacts",
    "BallTrackingConfig",
    "BallTrackingEngine",
    "BallTrackingResult",
    "CandidateConfig",
    "HeuristicV1BallCandidateScorer",
    "TrackConfig",
]


_PUBLIC_IMPORTS = {
    name: (
        "ttflux.tracking.api",
        name,
    )
    for name in __all__
}


def __getattr__(name: str) -> Any:
    """
    Resolve public tracking objects only when they are requested.

    This keeps compatibility imports lightweight and prevents
    circular imports while internal modules are being reorganized.
    """

    target = _PUBLIC_IMPORTS.get(name)

    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, attribute_name = target
    module = import_module(module_name)
    value = getattr(module, attribute_name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | set(__all__)
    )
