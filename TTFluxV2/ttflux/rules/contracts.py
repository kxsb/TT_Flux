from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RallyEvent:
    frame: int
    event_type: str
    player: str | None = None
    confidence: float = 0.0
    notes: str = ""
