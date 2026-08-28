"""Needle Time Machine tracing primitives."""

from .events import TraceEvent, TensorInfo
from .trace import Tracer, trace, get_tracer
from .snapshot import SnapshotStore
from .replay import ReplayCursor
from .needle_adapter import NeedleAdapter, instrument_block_stack
from .ui import serve

__all__ = [
    "NeedleAdapter",
    "NeedleRuntime",
    "ReplayCursor",
    "SnapshotStore",
    "TensorInfo",
    "TraceEvent",
    "Tracer",
    "get_tracer",
    "instrument_block_stack",
    "serve",
    "trace",
]


def __getattr__(name):
    if name == "NeedleRuntime":
        from .needle_runtime import NeedleRuntime
        return NeedleRuntime
    raise AttributeError(name)
