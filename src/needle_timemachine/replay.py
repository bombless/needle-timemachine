"""Timeline navigation for the Time Machine."""

from __future__ import annotations

from typing import Optional

from .events import TraceEvent
from .snapshot import Snapshot, SnapshotStore


class ReplayCursor:
    """Navigate a completed trace without re-running the model."""

    def __init__(self, events, snapshots: Optional[SnapshotStore] = None):
        self.events = list(events)
        self.snapshots = snapshots or SnapshotStore()
        self.position = -1

    @property
    def current(self) -> Optional[TraceEvent]:
        if self.position < 0 or self.position >= len(self.events):
            return None
        return self.events[self.position]

    def seek_step(self, step: int) -> Optional[TraceEvent]:
        self.position = -1
        for i, event in enumerate(self.events):
            if event.step >= step:
                self.position = i
                break
        if self.position == -1 and self.events:
            self.position = len(self.events) - 1
        return self.current

    def forward(self, count: int = 1) -> Optional[TraceEvent]:
        if not self.events:
            return None
        self.position = min(len(self.events) - 1, self.position + max(1, count))
        return self.current

    def backward(self, count: int = 1) -> Optional[TraceEvent]:
        if not self.events:
            return None
        self.position = max(0, self.position - max(1, count))
        return self.current

    def snapshot(self) -> Optional[Snapshot]:
        event = self.current
        if event is not None and event.snapshot_id:
            return self.snapshots.get(event.snapshot_id)
        return self.snapshots.latest_at_or_before(event.step) if event else None
