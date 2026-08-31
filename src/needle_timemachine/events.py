"""Serializable metadata for one observable model execution step."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class TensorInfo:
    shape: Tuple[int, ...]
    dtype: str
    nbytes: Optional[int] = None
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TensorPayload:
    """Browser-replayable tensor, encoded as little-endian float32."""

    shape: Tuple[int, ...]
    dtype: str
    encoding: str
    data: str
    sum: float
    sum_squares: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEvent:
    step: int
    op: str
    layer: Optional[int] = None
    name: Optional[str] = None
    phase: str = "forward"
    tensors: Dict[str, TensorInfo] = field(default_factory=dict)
    values: Dict[str, TensorPayload] = field(default_factory=dict)
    snapshot_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tensors"] = {k: v.to_dict() for k, v in self.tensors.items()}
        data["values"] = {k: v.to_dict() for k, v in self.values.items()}
        return data
