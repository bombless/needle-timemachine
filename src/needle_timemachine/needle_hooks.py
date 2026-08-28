"""Runtime bridge for optional Needle JAX execution tracing.

The upstream Needle checkout remains untouched by the package itself. A small
source patch installed by ``tools/patch_needle_trace.py`` adds a hook setter
and calls this bridge through ``jax.debug.callback``. That means callbacks see
runtime values even when Needle is inside ``jit``/``scan``/``remat``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .trace import Tracer


OP_NAMES = {
    1: "layer.input",
    2: "norm.output",
    3: "attention.q",
    4: "attention.k",
    5: "attention.v",
    6: "attention.output",
    7: "attention.projected",
    8: "residual.attention",
    9: "mlp.input",
    10: "mlp.activation",
    11: "layer.output",
    12: "attention.scores",
    13: "attention.softmax",
    14: "mlp.hadamard1",
    15: "mlp.silu",
    16: "mlp.hadamard2",
}


class NeedleRuntimeHook:
    """Convert Needle runtime callbacks into Time Machine events."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def __call__(self, op_id: Any, layer: Any, value: Any) -> None:
        try:
            op_id = int(op_id)
        except (TypeError, ValueError):
            op_id = -1
        try:
            layer = int(layer)
        except (TypeError, ValueError):
            layer = None
        op = OP_NAMES.get(op_id, f"needle.op.{op_id}")
        self.tracer.emit(
            op,
            layer=layer,
            name=f"layer.{layer}.{op}" if layer is not None else op,
            tensors={"value": value},
            metadata={"runtime": True, "needle_op_id": op_id},
        )


@contextmanager
def installed(architecture: Any, tracer: Tracer) -> Iterator[NeedleRuntimeHook]:
    """Install a runtime hook on the patched upstream architecture module."""
    hook = NeedleRuntimeHook(tracer)
    setter = getattr(architecture, "set_timemachine_hook", None)
    if setter is None:
        raise RuntimeError(
            "Needle source is not patched for Time Machine hooks. "
            "Run tools/patch_needle_trace.py against the local Needle checkout."
        )
    setter(hook)
    try:
        yield hook
    finally:
        setter(None)
