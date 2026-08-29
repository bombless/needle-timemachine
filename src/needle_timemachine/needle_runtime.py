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
    kv_patch: Any = None
    kv_cache: Any = None

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

    def kv_prefill(self, tokens: Any, *, quant: bool = False) -> Any:
        if self.kv_patch is None:
            raise RuntimeError("KV cache patch is not installed")
        self.kv_patch.set_active(True)
        self.kv_patch.set_position(0)
        output, updates = self.model.apply(
            {"params": self.params, "cache": self.kv_cache}, tokens,
            quant=quant, use_kv_cache=True, mutable=["cache"],
        )
        self.kv_cache = updates["cache"]
        return output

    def kv_decode(self, token: Any, history: Any, position: int, *, quant: bool = False) -> Any:
        if self.kv_patch is None:
            raise RuntimeError("KV cache patch is not installed")
        self.kv_patch.set_active(True)
        self.kv_patch.set_position(position)
        output, updates = self.model.apply(
            {"params": self.params, "cache": self.kv_cache}, token,
            method=self.model.cached_logits, engram_tokens=history,
            quant=quant, mutable=["cache"],
        )
        self.kv_cache = updates["cache"]
        return output


def _load_kv_patch(architecture: Any, max_seq_len: int):
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools import patch_kv_cache
    patch_kv_cache.install(architecture, max_seq_len)
    patch_kv_cache.configure(max_seq_len, 0)
    patch_kv_cache.patch_runtime(NeedleRuntime)
    return patch_kv_cache


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
    """Load Needle 2 and install the repo-local KV-cache patch.

    A missing checkpoint name is resolved by the upstream loader from its
    Cactus-Compute/needle2 Hugging Face repository. In particular,
    ``--checkpoint needle2.pkl`` resolves to ``checkpoints/needle2.pkl``.
    """
    import sys
    import jax
    import jax.numpy as jnp

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

    patch = _load_kv_patch(architecture, config.max_seq_len)
    patch.set_active(True)
    cache_variables = model.init(
        jax.random.key(0), jnp.zeros((1, 1), dtype=jnp.int32), use_kv_cache=True
    )
    kv_cache = cache_variables["cache"]
    patch.set_active(False)
    return NeedleRuntime(
        model, params, tokenizer, config, adapter, architecture, trace_level,
        kv_patch=patch, kv_cache=kv_cache,
    )
