"""Adapter for tracing the upstream Needle Flax model.

The first useful integration point is Needle's public ``hidden_states`` API.
Needle already exposes the residual stream after every scanned Transformer
block, so we can obtain a real per-layer timeline without rewriting Flax's
``nn.scan`` implementation.
"""

from __future__ import annotations

from typing import Any

from .trace import Tracer


class NeedleAdapter:
    """Trace high-level and layer-level Needle execution."""

    def __init__(self, model: Any, tracer: Tracer):
        self.model = model
        self.tracer = tracer

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
        """Run Needle's real hidden-state path and emit one event per layer.

        The upstream implementation returns ``[B, T, L, C]`` where state 0 is
        the embedding output and each following state is the output of one
        Transformer block. We therefore get a real per-layer timeline without
        replacing or monkey-patching Flax's ``nn.scan`` module.
        """
        self.tracer.emit(
            "model_input", name="model.hidden_states.input",
            tensors={"tokens": tokens}, metadata={"shape_only": True},
        )

        cells = self.model.hidden_states(tokens, **kwargs)
        if not hasattr(cells, "shape") or len(cells.shape) != 4:
            self.tracer.emit(
                "hidden_states_output", name="model.hidden_states.output",
                tensors={"hidden": cells},
                metadata={"layer_trace": False},
            )
            return cells

        num_states = int(cells.shape[2])
        self.tracer.emit(
            "embedding_output", layer=0, name="embedding.output",
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

    The preferred layer-level path is :meth:`NeedleAdapter.hidden_states`,
    because it preserves the upstream ``nn.scan`` execution exactly.
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
