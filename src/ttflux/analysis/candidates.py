"""
Compatibility imports for the historical analysis namespace.

The canonical candidate generator now lives under
ttflux.tracking.candidates.generator.
"""

from ttflux.tracking.candidates.generator import (
    CSV_FIELDS,
    CandidateConfig,
    analyze_candidates,
    detect_frame_candidates,
    summarize_candidate_counts,
)

__all__ = [
    "CSV_FIELDS",
    "CandidateConfig",
    "analyze_candidates",
    "detect_frame_candidates",
    "summarize_candidate_counts",
]
