from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SceneObjectKind = Literal["table", "net", "scoreboard", "player", "camera", "unknown"]


@dataclass
class SceneObject:
    frame: int
    kind: SceneObjectKind
    confidence: float
    payload: dict
    source: str
