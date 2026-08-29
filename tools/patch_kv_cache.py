"""Runtime KV-cache patch for the upstream Needle submodule."""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

_ENABLED = False
_ACTIVE = False
_MAX_SEQ_LEN = 0
_CACHE_POS = 0
_ORIGINAL_SCAN = None
_ORIGINAL_MHA = None
_ORIGINAL_MODEL_CALL = None
_INSTALLED = False


def configure(max_seq_len: int, cache_pos: int = 0) -> None:
    global _MAX_SEQ_LEN, _CACHE_POS
    _MAX_SEQ_LEN, _CACHE_POS = int(max_seq_len), int(cache_pos)


def set_position(cache_pos: int) -> None:
    global _CACHE_POS
    _CACHE_POS = int(cache_pos)


def set_active(active: bool) -> None:
    global _ACTIVE
    _ACTIVE = bool(active)


def _cache_init(shape, dtype=jnp.bfloat16):
    return jnp.zeros(shape, dtype=dtype)


def install(architecture: Any, max_seq_len: int) -> None:
    """Install the patch without modifying the Needle checkout."""
    global _INSTALLED, _ENABLED, _MAX_SEQ_LEN
    global _ORIGINAL_SCAN, _ORIGINAL_MHA, _ORIGINAL_MODEL_CALL
    if _INSTALLED:
        _MAX_SEQ_LEN = int(max_seq_len)
        return

    _MAX_SEQ_LEN = int(max_seq_len)
    from flax.linen.module import wrap_method_once

    nn = architecture.nn

    def compact_method(function):
        # Runtime replacement must retain both compact semantics and Flax's
        # Module state wrapper. The latter is required inside nn.scan.
        return wrap_method_once(nn.compact(function))
    _ORIGINAL_SCAN = nn.scan
    _ORIGINAL_MHA = architecture.MultiHeadAttention.__call__
    _ORIGINAL_MODEL_CALL = architecture.SimpleAttentionNetwork.__call__

    def scan_with_cache(*args, **kwargs):
        if not _ACTIVE:
            return _ORIGINAL_SCAN(*args, **kwargs)
        axes = dict(kwargs.get("variable_axes", {}))
        axes["cache"] = 0
        kwargs["variable_axes"] = axes
        return _ORIGINAL_SCAN(*args, **kwargs)

    nn.scan = scan_with_cache

    def mha(self, x, mask=None, rope=None, quant=False, use_kv_cache=False, **kwargs):
        """MHA wrapper accepting Needle tracing kwargs such as trace_layer."""
        if not _ACTIVE:
            return _ORIGINAL_MHA(self, x, mask=mask, rope=rope, quant=quant, **kwargs)

        attn_dim = self.attn_dim or self.d_model
        head_dim = attn_dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim
        batch, tokens = x.shape[0], x.shape[1]
        capacity, position = _MAX_SEQ_LEN, _CACHE_POS
        if position + tokens > capacity:
            raise ValueError(
                f"KV cache overflow: position={position}, tokens={tokens}, capacity={capacity}"
            )

        x = architecture._aq(x, quant)
        q = architecture.nn.Dense(attn_dim, dtype=self.dtype, use_bias=False,
                                  kernel_init=architecture.default_init(), name="q_proj")(x)
        k = architecture.nn.Dense(kv_dim, dtype=self.dtype, use_bias=False,
                                  kernel_init=architecture.default_init(), name="k_proj")(x)
        v = architecture.nn.Dense(kv_dim, dtype=self.dtype, use_bias=False,
                                  kernel_init=architecture.default_init(), name="v_proj")(x)
        q = q.reshape(batch, tokens, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, tokens, self.num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, tokens, self.num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        q = architecture.ZCRMSNorm(dtype=self.dtype, name="q_norm")(q)
        k = architecture.ZCRMSNorm(dtype=self.dtype, name="k_norm")(k)

        if rope is not None:
            cos, sin = rope
            half = q.shape[-1] // 2
            c = cos[position:position + tokens][None, None]
            s = sin[position:position + tokens][None, None]
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k[..., :half], k[..., half:]
            q = jnp.concatenate([q1 * c - q2 * s, q2 * c + q1 * s], axis=-1).astype(q.dtype)
            k = jnp.concatenate([k1 * c - k2 * s, k2 * c + k1 * s], axis=-1).astype(k.dtype)

        k = architecture._quantize.maybe_quant_kv(k, quant)
        v = architecture._quantize.maybe_quant_kv(v, quant)
        cache_k = self.variable("cache", "k", _cache_init,
                                (batch, self.num_kv_heads, capacity, head_dim), self.dtype)
        cache_v = self.variable("cache", "v", _cache_init,
                                (batch, self.num_kv_heads, capacity, head_dim), self.dtype)
        cache_k.value = cache_k.value.at[:, :, position:position + tokens, :].set(k)
        cache_v.value = cache_v.value.at[:, :, position:position + tokens, :].set(v)
        keys, values = cache_k.value, cache_v.value
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            keys = jnp.repeat(keys, repeats, axis=1)
            values = jnp.repeat(values, repeats, axis=1)

        key_positions = jnp.arange(capacity)[None, None, None, :]
        query_positions = (position + jnp.arange(tokens))[None, None, :, None]
        cache_mask = jnp.broadcast_to(key_positions <= query_positions,
                                      (batch, 1, tokens, capacity))
        if self.flash:
            implementation = (
                "cudnn" if architecture.jax.default_backend() == "gpu"
                and q.dtype in (jnp.bfloat16, jnp.float16) else None
            )
            output = architecture.jax.nn.dot_product_attention(
                q.transpose(0, 2, 1, 3), keys.transpose(0, 2, 1, 3),
                values.transpose(0, 2, 1, 3), mask=cache_mask,
                implementation=implementation,
            ).reshape(batch, tokens, attn_dim)
        else:
            scale = jnp.sqrt(jnp.float32(head_dim))
            weights = jnp.matmul(q, keys.transpose(0, 1, 3, 2)) / scale
            weights = jnp.where(cache_mask, weights, jnp.finfo(weights.dtype).min)
            weights = architecture.nn.softmax(weights, axis=-1)
            output = jnp.matmul(weights, values).transpose(0, 2, 1, 3).reshape(
                batch, tokens, attn_dim
            )
        output = output * architecture.nn.sigmoid(
            architecture.nn.Dense(attn_dim, dtype=self.dtype, use_bias=False,
                                  kernel_init=architecture.default_init(), name="gate_proj")(x)
        )
        output = architecture._aq(output, quant)
        return architecture.nn.Dense(
            self.d_model, dtype=self.dtype, use_bias=False,
            kernel_init=architecture.residual_init(self.num_layers), name="out_proj"
        )(output)

    # The upstream methods are @nn.compact. Preserve that context after
    # replacing them, otherwise Dense/Norm variables cannot be initialized.
    architecture.MultiHeadAttention.__call__ = compact_method(mha)

    def model_call(self, tokens, mask=None, quant=False, return_mtp=False,
                   use_kv_cache=False, **kwargs):
        if not use_kv_cache:
            return _ORIGINAL_MODEL_CALL(self, tokens, mask=mask, quant=quant,
                                        return_mtp=return_mtp, **kwargs)
        if mask is None:
            mask = architecture.make_causal_mask(tokens.shape[1])
        x = self.embedding(tokens) * self.embed_scale
        rope = self._rope(_MAX_SEQ_LEN)
        engram_kv = self._engram_kv(tokens, mask, quant)
        x, _ = self.stack(x, mask=mask, rope=rope, engram_kv=engram_kv, quant=quant)
        return architecture._aq(x, quant).astype(jnp.float32) @ self.embedding.embedding.T

    architecture.SimpleAttentionNetwork.__call__ = compact_method(model_call)

    def cached_logits(self, tokens, engram_tokens=None, quant=False, **kwargs):
        if tokens.shape[1] != 1:
            raise ValueError("cached_logits expects exactly one token")
        history = tokens if engram_tokens is None else engram_tokens
        full_mask = architecture.make_causal_mask(history.shape[1])
        engram_kv = self._engram_kv(history, full_mask, quant)
        engram_kv = (engram_kv[0][:, :, -1:], engram_kv[1][:, :, -1:])
        x = self.embedding(tokens) * self.embed_scale
        rope = self._rope(_MAX_SEQ_LEN)
        x, _ = self.stack(x, mask=full_mask, rope=rope, engram_kv=engram_kv, quant=quant)
        return architecture._aq(x, quant).astype(jnp.float32) @ self.embedding.embedding.T

    architecture.SimpleAttentionNetwork.cached_logits = compact_method(cached_logits)
    _ENABLED = _INSTALLED = True


def patch_runtime(runtime_cls: Any) -> None:
    """Make ``NeedleRuntime.logits`` transparently use prefill and decode."""
    if getattr(runtime_cls, "_kv_cache_patched", False):
        return
    original = runtime_cls.logits

    def logits(self, tokens, **kwargs):
        array = np.asarray(tokens)
        if (getattr(self, "kv_patch", None) is None or array.ndim != 2
                or array.shape[0] != 1):
            return original(self, tokens, **kwargs)
        ids = array[0].astype(np.int32).tolist()
        previous = getattr(self, "_kv_tokens", None)
        if previous == ids:
            return self._kv_last_logits
        if previous is not None and len(ids) == len(previous) + 1 and ids[:-1] == previous:
            output = self.kv_decode(array[:, -1:], array, len(previous),
                                    quant=bool(kwargs.get("quant", False)))
        else:
            output = self.kv_prefill(array, quant=bool(kwargs.get("quant", False)))
        self._kv_tokens = ids
        self._kv_last_logits = output
        return output

    runtime_cls.logits = logits
    runtime_cls._kv_cache_patched = True


def enabled() -> bool:
    return _ENABLED
