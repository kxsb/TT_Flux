from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TTFluxConfig:
    name: str
    video_path: Path
    track_csv: Path
    clip_id: str
    fps: float
    frame_width: int
    frame_height: int
    window_span: int
    min_points: int
    top_k: int
    non_overlap_margin: int
    trail_keep_last: int

    @classmethod
    def from_json(cls, path: str | Path) -> "TTFluxConfig":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8-sig"))

        return cls(
            name=data["name"],
            video_path=Path(data["video_path"]),
            track_csv=Path(data["track_csv"]),
            clip_id=data["clip_id"],
            fps=float(data.get("fps", 50.0)),
            frame_width=int(data.get("frame_width", 1280)),
            frame_height=int(data.get("frame_height", 720)),
            window_span=int(data.get("window_span", 80)),
            min_points=int(data.get("min_points", 14)),
            top_k=int(data.get("top_k", 3)),
            non_overlap_margin=int(
                data.get("non_overlap_margin", 25)
            ),
            trail_keep_last=int(data.get("trail_keep_last", 18)),
        )
