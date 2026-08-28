"""Optional integration with the upstream Needle Python source tree."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .needle_adapter import NeedleAdapter
from .needle_hooks import installed as install_runtime_hook
from .trace import Tracer


@dataclass
class NeedleRuntime:
    model: Any
    params: Any
    tokenizer: Any
    config: Any
    adapter: NeedleAdapter
    architecture: Any
    trace_level: str = "layer"

    def hidden_states(self, tokens: Any, **kwargs: Any) -> Any:
        """Run the real upstream model's hidden-state path through the tracer."""
        variables = {"params": self.params}
        hook_context = install_runtime_hook(self.architecture, self.adapter.tracer) if self.trace_level == "op" else None
        if hook_context is None:
            hidden_states = self.model.apply(
                variables,
                tokens,
                method=self.model.hidden_states,
                **kwargs,
            )
        else:
            with hook_context:
                hidden_states = self.model.apply(
                    variables,
                    tokens,
                    method=self.model.hidden_states,
                    **kwargs,
                )
        return self.adapter.record_hidden_states(tokens, hidden_states, emit_input=False)

    def logits(self, tokens: Any, **kwargs: Any) -> Any:
        """Run the ordinary Needle forward path through Flax's apply API."""
        variables = {"params": self.params}
        self.adapter.tracer.emit(
            "model_input",
            name="model.input",
            tensors={"tokens": tokens},
            metadata={"shape_only": True},
        )
        output = self.model.apply(variables, tokens, **kwargs)
        tensors = (
            {f"output_{i}": value for i, value in enumerate(output)}
            if isinstance(output, tuple)
            else {"logits": output}
        )
        self.adapter.tracer.emit("model_output", name="model.output", tensors=tensors)
        return output


def _checkpoint_error(path: str, exc: ValueError) -> ValueError:
    return ValueError(
        f"{path} cannot be loaded by the current Needle 2 runtime.\n\n"
        f"The file appears to be the legacy Cactus-Compute/needle checkpoint, "
        f"not the Needle 2 format-v2 checkpoint required by the current source.\n\n"
        f"Use the Needle 2 weights from Cactus-Compute/needle2 instead. The current "
        f"Hugging Face layout is checkpoints/needle2.pkl. You can simply pass "
        f"--checkpoint needle2.pkl and Needle's loader will download it automatically.\n\n"
        f"Original error: {exc}"
    )


def load_needle_checkpoint(
    checkpoint: str | Path,
    *,
    needle_source: str | Path,
    tracer: Tracer,
    trace_level: str = "layer",
) -> NeedleRuntime:
    """Load a real upstream Needle 2 checkpoint without copying Needle sources.

    A missing checkpoint name is resolved by the upstream loader from its
    Cactus-Compute/needle2 Hugging Face repository. In particular,
    ``--checkpoint needle2.pkl`` resolves to ``checkpoints/needle2.pkl``.
    """
    import sys

    source = str(Path(needle_source).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)

    run = import_module("needle.model.run")
    architecture = import_module("needle.model.architecture")
    tokenizer_mod = import_module("needle.model.tokenizer")

    try:
        params, config = run.load_checkpoint(str(checkpoint))
    except ValueError as exc:
        if "not a format-v2 checkpoint" in str(exc):
            raise _checkpoint_error(str(checkpoint), exc) from exc
        raise

    model = architecture.SimpleAttentionNetwork(config)
    tokenizer = tokenizer_mod.get_tokenizer(config.vocab_size)
    adapter = NeedleAdapter(model, tracer, trace_level=trace_level)
    return NeedleRuntime(model, params, tokenizer, config, adapter, architecture, trace_level)
