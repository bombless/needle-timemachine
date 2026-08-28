"""Optional integration with the upstream Needle Python source tree.

This module deliberately keeps Needle an external source dependency. It loads a
Needle checkpoint with the upstream loader, constructs the upstream model, and
runs the real ``hidden_states`` path through :class:`NeedleAdapter`.
"""

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
        # hidden_states is an un-jitted model method; this is intentional for
        # the first Time Machine prototype because it preserves observability.
        variables = {"params": self.params}
        hidden_states = self.model.apply(
            variables,
            tokens,
            method=self.model.hidden_states,
            **kwargs,
        )
        # Feed the already computed states through the adapter's event encoder.
        return self.adapter.record_hidden_states(tokens, hidden_states)


def load_needle_checkpoint(
    checkpoint: str | Path,
    *,
    needle_source: str | Path,
    tracer: Tracer,
    trace_level: str = "layer",
) -> NeedleRuntime:
    """Load a real upstream Needle checkpoint without copying Needle sources.

    ``needle_source`` is the local checkout of ``cactus-compute/needle``. It is
    added to ``sys.path`` only for this call, allowing the adapter repository to
    remain independent from the upstream project.
    """
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
