"""
Compatibility alias for the historical analysis namespace.

Run lifecycle orchestration now lives under
ttflux.pipeline.runs.

This module name resolves to the canonical module object so that
historical imports and monkeypatch-based tests retain the same
global state.
"""

from importlib import import_module
import sys


_CANONICAL_MODULE = import_module(
    "ttflux.pipeline.runs"
)

sys.modules[__name__] = _CANONICAL_MODULE
