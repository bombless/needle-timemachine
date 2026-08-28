"""Small adapter for the upstream Needle Flax model.

This module deliberately does not fork Needle. It provides stable trace points
around the public model call and hidden-state API while we work toward finer
operation-level instrumentation inside architecture.py.
"""

from __future__ import annotations

from typing import Any, Optional

from .trace import Tracer


class NeedleAdapter:
    """Trace high-level Needle execution without modifying model semantics."""

    def __init__(self, model: Any, tracer: Tracer):
        self.model = model
        self.tracer = tracer

    def __call__(self, tokens: Any, **kwargs: Any) -> Any:
        self.tracer.emit(
            "embedding_input", name="model.input", tensors={"tokens": tokens},
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
        self.tracer.emit("hidden_states_input", name="model.hidden_states", tensors={"tokens": tokens})
        output = self.model.hidden_states(tokens, **kwargs)
        self.tracer.emit("hidden_states_output", name="model.hidden_states.output", tensors={"hidden": output})
        return output


def instrument_block_stack(stack: Any, tracer: Tracer) -> Any:
    """Return a callable wrapper for Stack.

    Fine-grained layer instrumentation is intentionally deferred until we can
    safely account for Flax `nn.scan` and JIT semantics. This boundary gives us
    a working adapter now and a single integration point for the next step.
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
