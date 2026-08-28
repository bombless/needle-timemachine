"""Adapter for tracing the upstream Needle Flax model."""

from __future__ import annotations

from typing import Any, Literal

from .trace import Tracer

TraceLevel = Literal["none", "layer", "op"]


class NeedleAdapter:
    """Trace Needle's model boundary, layers, or runtime operations."""

    def __init__(self, model: Any, tracer: Tracer, *, trace_level: TraceLevel = "layer"):
        if trace_level not in ("none", "layer", "op"):
            raise ValueError("trace_level must be 'none', 'layer', or 'op'")
        self.model = model
        self.tracer = tracer
        self.trace_level = trace_level

    def __call__(self, tokens: Any, **kwargs: Any) -> Any:
        self.tracer.emit("model_input", name="model.input", tensors={"tokens": tokens}, metadata={"shape_only": True})
        output = self.model(tokens, **kwargs)
        tensors = {f"output_{i}": value for i, value in enumerate(output)} if isinstance(output, tuple) else {"logits": output}
        self.tracer.emit("model_output", name="model.output", tensors=tensors)
        return output

    def hidden_states(self, tokens: Any, **kwargs: Any) -> Any:
        self.tracer.emit("model_input", name="model.hidden_states.input", tensors={"tokens": tokens}, metadata={"shape_only": True})
        cells = self.model.hidden_states(tokens, **kwargs)
        return self.record_hidden_states(tokens, cells, emit_input=False)

    def record_hidden_states(self, tokens: Any, cells: Any, *, emit_input: bool = True) -> Any:
        """Record a precomputed Needle hidden-state tensor.

        Needle returns [batch, sequence, states, hidden], where state 0 is the
        embedding output and state i>0 is the output of Transformer layer i-1.
        """
        if emit_input:
            self.tracer.emit("model_input", name="model.hidden_states.input", tensors={"tokens": tokens}, metadata={"shape_only": True})
        if self.trace_level == "none":
            self.tracer.emit("hidden_states_output", name="model.hidden_states.output", tensors={"hidden": cells}, metadata={"layer_trace": False})
            return cells
        if not hasattr(cells, "shape") or len(cells.shape) != 4:
            raise ValueError("Needle hidden_states must return [batch, sequence, states, hidden]; got shape=%r" % (getattr(cells, "shape", None),))
        num_states = int(cells.shape[2])
        self.tracer.emit("embedding_output", layer=None, name="embedding.output", tensors={"hidden": cells[:, :, 0, :]}, metadata={"state_index": 0, "layer_type": "embedding"})
        if self.trace_level == "layer":
            for state_index in range(1, num_states):
                layer = state_index - 1
                self.tracer.emit("layer_output", layer=layer, name=f"layer.{layer}.output", tensors={"hidden": cells[:, :, state_index, :]}, metadata={"state_index": state_index, "layer_type": "transformer"})
        self.tracer.emit("hidden_states_output", name="model.hidden_states.output", tensors={"hidden": cells}, metadata={"layer_count": max(0, num_states - 1)})
        return cells


def instrument_block_stack(stack: Any, tracer: Tracer) -> Any:
    """Return a non-invasive callable wrapper around a Stack boundary."""
    original = stack

    def call(x: Any, *args: Any, **kwargs: Any) -> Any:
        tracer.emit("stack_input", name="stack", tensors={"input": x})
        out = original(x, *args, **kwargs)
        tensors = {f"output_{i}": value for i, value in enumerate(out)} if isinstance(out, tuple) else {"output": out}
        tracer.emit("stack_output", name="stack", tensors=tensors)
        return out

    return call
