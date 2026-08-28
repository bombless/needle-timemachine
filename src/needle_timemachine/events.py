"""Serializable metadata for one observable model execution step."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class TensorInfo:
    """Small, JSON-friendly description of a tensor/array."""

    shape: Tuple[int, ...]
    dtype: str
    nbytes: Optional[int] = None
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEvent:
    """A point on the Time Machine timeline.

    `step` is a monotonically increasing execution index. `layer` is optional
    because some events (embedding, logits, decoding) are outside a block.
    `snapshot_id` points into SnapshotStore when a state was retained.
    """

    step: int
    op: str
    layer: Optional[int] = None
    name: Optional[str] = None
    phase: str = "forward"
    tensors: Dict[str, TensorInfo] = field(default_factory=dict)
    snapshot_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tensors"] = {k: v.to_dict() for k, v in self.tensors.items()}
        return data
