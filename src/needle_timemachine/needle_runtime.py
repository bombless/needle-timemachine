"""Optional integration with the upstream Needle Python source tree."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .needle_adapter import NeedleAdapter
from .trace import Tracer


@dataclass
class NeedleRuntime:
    model: Any
    params: Any
    tokenizer: Any
    config: Any
    adapter: NeedleAdapter

    def hidden_states(self, tokens: Any, **kwargs: Any) -> Any:
        """Run the real upstream model's hidden-state path through the tracer."""
        variables = {"params": self.params}
        hidden_states = self.model.apply(
            variables,
            tokens,
            method=self.model.hidden_states,
            **kwargs,
        )
        return self.adapter.record_hidden_states(tokens, hidden_states, emit_input=False)

    def logits(self, tokens: Any, **kwargs: Any) -> Any:
        """Run the ordinary Needle forward path, recording model boundaries."""
        return self.adapter(tokens, **kwargs)


def load_needle_checkpoint(
    checkpoint: str | Path,
    *,
    needle_source: str | Path,
    tracer: Tracer,
    trace_level: str = "layer",
) -> NeedleRuntime:
    """Load a real upstream Needle checkpoint without copying Needle sources."""
    import sys

    source = str(Path(needle_source).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)

    run = import_module("needle.model.run")
    architecture = import_module("needle.model.architecture")
    tokenizer_mod = import_module("needle.model.tokenizer")

    params, config = run.load_checkpoint(str(checkpoint))
    model = architecture.SimpleAttentionNetwork(config)
    tokenizer = tokenizer_mod.get_tokenizer(config.vocab_size)
    adapter = NeedleAdapter(model, tracer, trace_level=trace_level)
    return NeedleRuntime(model, params, tokenizer, config, adapter)
