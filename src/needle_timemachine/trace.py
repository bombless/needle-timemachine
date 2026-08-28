"""Low-level instrumentation API.

The tracer is intentionally framework-agnostic. It can be used around normal
Python functions today and later wired into Needle's Flax modules. It never
changes the returned value, which is important for numerical equivalence tests.
"""

from __future__ import annotations

import contextvars
import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from .events import TensorInfo, TraceEvent
from .snapshot import SnapshotStore

_CURRENT: contextvars.ContextVar[Optional["Tracer"]] = contextvars.ContextVar(
    "needle_timemachine_tracer", default=None
)


def _tensor_info(value: Any, summarize: bool = True) -> Optional[TensorInfo]:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    shape = tuple(int(x) for x in value.shape)
    dtype = str(value.dtype)
    nbytes = None
    try:
        nbytes = math.prod(shape) * int(value.dtype.itemsize)
    except (AttributeError, TypeError, ValueError):
        pass
    if not summarize:
        return TensorInfo(shape, dtype, nbytes=nbytes)
    try:
        import numpy as np
        arr = np.asarray(value)
        if arr.size == 0:
            return TensorInfo(shape, dtype, nbytes=nbytes)
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else arr
        if finite.size:
            return TensorInfo(
                shape, dtype, nbytes=nbytes,
                min=float(np.min(finite)), max=float(np.max(finite)),
                mean=float(np.mean(finite)),
            )
    except (ImportError, TypeError, ValueError):
        pass
    return TensorInfo(shape, dtype, nbytes=nbytes)


class Tracer:
    """Collect execution events and optional snapshots."""

    def __init__(self, *, snapshot_store: Optional[SnapshotStore] = None,
                 snapshot_every: int = 0, summarize_tensors: bool = True):
        self.events = []
        self.snapshots = snapshot_store or SnapshotStore()
        self.snapshot_every = snapshot_every
        self.summarize_tensors = summarize_tensors
        self._step = 0
        self._token = None

    @property
    def step(self) -> int:
        return self._step

    def reset(self) -> None:
        self.events.clear()
        self._step = 0

    def emit(self, op: str, *, layer: Optional[int] = None,
             name: Optional[str] = None, phase: str = "forward",
             tensors: Optional[Dict[str, Any]] = None,
             state: Optional[Dict[str, Any]] = None,
             metadata: Optional[Dict[str, Any]] = None) -> TraceEvent:
        self._step += 1
        infos = {}
        for key, value in (tensors or {}).items():
            info = _tensor_info(value, self.summarize_tensors)
            if info is not None:
                infos[key] = info

        snapshot_id = None
        if state is not None and self.snapshot_every and self._step % self.snapshot_every == 0:
            snapshot_id = self.snapshots.put(self._step, state)

        event = TraceEvent(
            step=self._step, op=op, layer=layer, name=name, phase=phase,
            tensors=infos, snapshot_id=snapshot_id, metadata=metadata or {},
        )
        self.events.append(event)
        return event

    @contextmanager
    def session(self) -> Iterator["Tracer"]:
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def trace_value(self, op: str, value: Any, **kwargs: Any) -> Any:
        tensors = kwargs.pop("tensors", None) or {"output": value}
        self.emit(op, tensors=tensors, **kwargs)
        return value


def get_tracer() -> Optional[Tracer]:
    return _CURRENT.get()


@contextmanager
def trace(*, snapshot_store: Optional[SnapshotStore] = None,
          snapshot_every: int = 0, summarize_tensors: bool = True) -> Iterator[Tracer]:
    tracer = Tracer(snapshot_store=snapshot_store,
                    snapshot_every=snapshot_every,
                    summarize_tensors=summarize_tensors)
    with tracer.session():
        yield tracer
