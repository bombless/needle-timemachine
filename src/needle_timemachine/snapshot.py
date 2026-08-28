"""Snapshot storage for replay without reverse execution."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass
class Snapshot:
    id: str
    step: int
    state: Dict[str, Any]


class SnapshotStore:
    """A deliberately simple in-memory snapshot store for the first prototype.

    Values are copied at capture time. NumPy/JAX arrays are converted to NumPy
    when possible, while ordinary Python values are deep-copied. This keeps the
    replay layer independent from JAX and makes later disk persistence easy.
    """

    def __init__(self, max_snapshots: Optional[int] = None):
        self.max_snapshots = max_snapshots
        self._items: Dict[str, Snapshot] = {}
        self._order = []

    @staticmethod
    def _copy_value(value: Any) -> Any:
        try:
            import numpy as np
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                return np.array(value, copy=True)
        except ImportError:
            pass
        return copy.deepcopy(value)

    def put(self, step: int, state: Mapping[str, Any], snapshot_id: Optional[str] = None) -> str:
        sid = snapshot_id or uuid.uuid4().hex
        snapshot = Snapshot(sid, step, {k: self._copy_value(v) for k, v in state.items()})
        self._items[sid] = snapshot
        self._order.append(sid)
        if self.max_snapshots is not None:
            while len(self._order) > self.max_snapshots:
                old = self._order.pop(0)
                self._items.pop(old, None)
        return sid

    def get(self, snapshot_id: str) -> Snapshot:
        return self._items[snapshot_id]

    def latest_at_or_before(self, step: int) -> Optional[Snapshot]:
        for sid in reversed(self._order):
            item = self._items[sid]
            if item.step <= step:
                return item
        return None

    def __len__(self) -> int:
        return len(self._items)

    def ids(self) -> Iterable[str]:
        return tuple(self._order)
