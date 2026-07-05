from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

BallStatus = Literal["detected", "linked", "interpolated", "missing", "rejected"]


@dataclass(slots=True)
class VideoAsset:
    path: str
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    duration_sec: float | None = None


@dataclass(slots=True)
class BallObservation:
    frame: int
    timestamp_ms: float
    x: float | None
    y: float | None
    confidence: float
    source: str
    status: BallStatus
    gap_id: str | None = None
    track_id: str = "ball_0"


@dataclass(slots=True)
class TableContext:
    status: str = "NO_TABLE_CONTEXT"
    source: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisRun:
    run_id: str
    root: str
    status: str
    created_at: str
    input_video: str | None
    outputs: dict[str, str] = field(default_factory=dict)

    @classmethod
    def new(cls, root: Path, input_video: Path | None, status: str = "created") -> "AnalysisRun":
        now = datetime.now().isoformat(timespec="seconds")
        run_id = datetime.now().strftime("analysis_%Y%m%d_%H%M%S")
        run_dir = root / "runs" / run_id
        return cls(
            run_id=run_id,
            root=str(run_dir),
            status=status,
            created_at=now,
            input_video=str(input_video) if input_video else None,
            outputs={
                "input_manifest": str(run_dir / "input_manifest.json"),
                "video_qc": str(run_dir / "video_qc.json"),
                "raw_candidates": str(run_dir / "raw_candidates.csv"),
                "ball_track": str(run_dir / "ball_track.csv"),
                "overlay": str(run_dir / "overlay.mp4"),
                "metrics": str(run_dir / "metrics.json"),
                "report": str(run_dir / "report.html"),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
