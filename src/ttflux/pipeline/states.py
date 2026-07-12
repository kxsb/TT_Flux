from __future__ import annotations

from typing import Final, Literal, TypeAlias, TypeGuard


RunStatus: TypeAlias = Literal[
    "created",
    "running",
    "completed",
    "failed",
]


RUN_CREATED: Final[RunStatus] = "created"
RUN_RUNNING: Final[RunStatus] = "running"
RUN_COMPLETED: Final[RunStatus] = "completed"
RUN_FAILED: Final[RunStatus] = "failed"


RUN_STATUSES: Final[frozenset[RunStatus]] = frozenset(
    {
        RUN_CREATED,
        RUN_RUNNING,
        RUN_COMPLETED,
        RUN_FAILED,
    }
)

TERMINAL_RUN_STATUSES: Final[frozenset[RunStatus]] = (
    frozenset(
        {
            RUN_COMPLETED,
            RUN_FAILED,
        }
    )
)


def is_run_status(value: object) -> TypeGuard[RunStatus]:
    return (
        isinstance(value, str)
        and value in RUN_STATUSES
    )


__all__ = [
    "RUN_COMPLETED",
    "RUN_CREATED",
    "RUN_FAILED",
    "RUN_RUNNING",
    "RUN_STATUSES",
    "TERMINAL_RUN_STATUSES",
    "RunStatus",
    "is_run_status",
]
