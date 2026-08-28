"""Adapter for tracing the upstream Needle Flax model.

The layer-level integration uses Needle's public ``hidden_states`` API. The
upstream implementation already returns the residual stream after every
Transformer block, so this gives us a faithful layer timeline without
rewriting or monkey-patching Flax's ``nn.scan``.
"""

from __future__ import annotations

from typing import Any, Literal

from .trace import Tracer

TraceLevel = Literal["none", "layer"]


class NeedleAdapter:
    """Trace high-level and layer-level Needle execution.

    ``trace_level="layer"`` is the first real integration mode. It records the
    embedding output and every Transformer block output. ``trace_level="none"``
    only records the model boundary events and is useful for cheap baselines.
    """

    def __init__(self, model: Any, tracer: Tracer, *, trace_level: TraceLevel = "layer"):
        if trace_level not in ("none", "layer"):
            raise ValueError("trace_level must be 'none' or 'layer'")
        self.model = model
        self.tracer = tracer
        self.trace_level = trace_level

    def __call__(self, tokens: Any, **kwargs: Any) -> Any:
        self.tracer.emit(
            "model_input", name="model.input", tensors={"tokens": tokens},
            metadata={"shape_only": True},
        )
        output = self.model(tokens, **kwargs)
        if isinstance(output, tuple):
            tensors = {f"output_{i}": value for i, value in enumerate(output)}
        else:
            tensors = {"logits": output}
        self.tracer.emit("model_output", name="model.output", tensors=tensors)
        return output

    def hidden_states(self, tokens: Any, **kwargs: Any) -> Any:
        """Run Needle's real hidden-state path and optionally emit its timeline."""
        self.tracer.emit(
            "model_input", name="model.hidden_states.input",
            tensors={"tokens": tokens}, metadata={"shape_only": True},
        )

        cells = self.model.hidden_states(tokens, **kwargs)
        if self.trace_level == "none":
            self.tracer.emit(
                "hidden_states_output", name="model.hidden_states.output",
                tensors={"hidden": cells}, metadata={"layer_trace": False},
            )
            return cells

        if not hasattr(cells, "shape") or len(cells.shape) != 4:
            raise ValueError(
                "Needle hidden_states must return [batch, sequence, states, hidden]; "
                f"got shape={getattr(cells, 'shape', None)!r}"
            )

        num_states = int(cells.shape[2])
        self.tracer.emit(
            "embedding_output", layer=None, name="embedding.output",
            tensors={"hidden": cells[:, :, 0, :]},
            metadata={"state_index": 0, "layer_type": "embedding"},
        )
        for state_index in range(1, num_states):
            layer = state_index - 1
            self.tracer.emit(
                "layer_output", layer=layer, name=f"layer.{layer}.output",
                tensors={"hidden": cells[:, :, state_index, :]},
                metadata={"state_index": state_index, "layer_type": "transformer"},
            )

        self.tracer.emit(
            "hidden_states_output", name="model.hidden_states.output",
            tensors={"hidden": cells},
            metadata={"layer_count": max(0, num_states - 1)},
        )
        return cells


def instrument_block_stack(stack: Any, tracer: Tracer) -> Any:
    """Return a callable wrapper around a Stack boundary.

    This remains a non-invasive escape hatch for experiments. The preferred
    layer-level path is :meth:`NeedleAdapter.hidden_states`, which preserves the
    upstream ``nn.scan`` execution exactly.
    """
    original = stack

    def call(x: Any, *args: Any, **kwargs: Any) -> Any:
        tracer.emit("stack_input", name="stack", tensors={"input": x})
        out = original(x, *args, **kwargs)
        if isinstance(out, tuple):
            tensors = {f"output_{i}": value for i, value in enumerate(out)}
        else:
            tensors = {"output": out}
        tracer.emit("stack_output", name="stack", tensors=tensors)
        return out

    return call
