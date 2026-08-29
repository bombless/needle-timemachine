"""Runtime KV-cache patch for the upstream Needle submodule."""
from __future__ import annotations
from typing import Any
import jax.numpy as jnp

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


def _cache_init(_, shape, dtype=jnp.bfloat16):
    return jnp.zeros(shape, dtype=dtype)


def install(architecture: Any, max_seq_len: int) -> None:
    """Install the patch without modifying the Needle git submodule."""
    global _INSTALLED, _ENABLED, _MAX_SEQ_LEN, _ORIGINAL_SCAN, _ORIGINAL_MHA, _ORIGINAL_MODEL_CALL
    if _INSTALLED:
        _MAX_SEQ_LEN = int(max_seq_len)
        return
    _MAX_SEQ_LEN = int(max_seq_len)
    nn = architecture.nn
    _ORIGINAL_SCAN = nn.scan
    _ORIGINAL_MHA = architecture.MultiHeadAttention.__call__
    _ORIGINAL_MODEL_CALL = architecture.SimpleAttentionNetwork.__call__

    def scan_with_cache(*args, **kwargs):
        if not _ACTIVE:
            return _ORIGINAL_SCAN(*args, **kwargs)
        carry = kwargs.get("variable_carry", False)
        if carry is False:
            kwargs["variable_carry"] = "cache"
        elif isinstance(carry, str):
            kwargs["variable_carry"] = (carry, "cache") if carry != "cache" else carry
        elif "cache" not in carry:
            kwargs["variable_carry"] = tuple(carry) + ("cache",)
        return _ORIGINAL_SCAN(*args, **kwargs)
    nn.scan = scan_with_cache

    def mha(self, x, mask=None, rope=None, quant=False, use_kv_cache=False):
        if not _ACTIVE:
            return _ORIGINAL_MHA(self, x, mask=mask, rope=rope, quant=quant)
        attn_dim = self.attn_dim or self.d_model
        head_dim = attn_dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim
        B, T = x.shape[0], x.shape[1]
        capacity, pos = _MAX_SEQ_LEN, _CACHE_POS
        if pos + T > capacity:
            raise ValueError(f"KV cache overflow: position={pos}, tokens={T}, capacity={capacity}")

        x = architecture._aq(x, quant)
        q = architecture.nn.Dense(attn_dim, dtype=self.dtype, use_bias=False, kernel_init=architecture.default_init(), name="q_proj")(x)
        k = architecture.nn.Dense(kv_dim, dtype=self.dtype, use_bias=False, kernel_init=architecture.default_init(), name="k_proj")(x)
        v = architecture.nn.Dense(kv_dim, dtype=self.dtype, use_bias=False, kernel_init=architecture.default_init(), name="v_proj")(x)
        q = q.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        q = architecture.ZCRMSNorm(dtype=self.dtype, name="q_norm")(q)
        k = architecture.ZCRMSNorm(dtype=self.dtype, name="k_norm")(k)
        if rope is not None:
            cos, sin = rope
            half = q.shape[-1] // 2
            c, s = cos[pos:pos + T][None, None], sin[pos:pos + T][None, None]
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k[..., :half], k[..., half:]
            q = jnp.concatenate([q1 * c - q2 * s, q2 * c + q1 * s], axis=-1).astype(q.dtype)
            k = jnp.concatenate([k1 * c - k2 * s, k2 * c + k1 * s], axis=-1).astype(k.dtype)
        k = architecture._quantize.maybe_quant_kv(k, quant)
        v = architecture._quantize.maybe_quant_kv(v, quant)
        ck = self.variable("cache", "k", _cache_init, (B, self.num_kv_heads, capacity, head_dim), self.dtype)
        cv = self.variable("cache", "v", _cache_init, (B, self.num_kv_heads, capacity, head_dim), self.dtype)
        ck.value = ck.value.at[:, :, pos:pos + T, :].set(k)
        cv.value = cv.value.at[:, :, pos:pos + T, :].set(v)
        k_all, v_all = ck.value, cv.value
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            k_all = jnp.repeat(k_all, repeats, axis=1)
            v_all = jnp.repeat(v_all, repeats, axis=1)
        key_pos = jnp.arange(capacity)[None, None, None, :]
        query_pos = (pos + jnp.arange(T))[None, None, :, None]
        cache_mask = jnp.broadcast_to(key_pos <= query_pos, (B, 1, T, capacity))
        if self.flash:
            impl = "cudnn" if architecture.jax.default_backend() == "gpu" and q.dtype in (jnp.bfloat16, jnp.float16) else None
            out = architecture.jax.nn.dot_product_attention(q.transpose(0, 2, 1, 3), k_all.transpose(0, 2, 1, 3), v_all.transpose(0, 2, 1, 3), mask=cache_mask, implementation=impl).reshape(B, T, attn_dim)
        else:
            scale = jnp.sqrt(jnp.float32(head_dim))
            weights = jnp.matmul(q, k_all.transpose(0, 1, 3, 2)) / scale
            weights = jnp.where(cache_mask, weights, jnp.finfo(weights.dtype).min)
            weights = architecture.nn.softmax(weights, axis=-1)
            out = jnp.matmul(weights, v_all).transpose(0, 2, 1, 3).reshape(B, T, attn_dim)
        out = out * architecture.nn.sigmoid(architecture.nn.Dense(attn_dim, dtype=self.dtype, use_bias=False, kernel_init=architecture.default_init(), name="gate_proj")(x))
        out = architecture._aq(out, quant)
        return architecture.nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=architecture.residual_init(self.num_layers), name="out_proj")(out)
    architecture.MultiHeadAttention.__call__ = mha

    def model_call(self, tokens, mask=None, quant=False, return_mtp=False, use_kv_cache=False):
        if not use_kv_cache:
            return _ORIGINAL_MODEL_CALL(self, tokens, mask=mask, quant=quant, return_mtp=return_mtp)
        if mask is None:
            mask = architecture.make_causal_mask(tokens.shape[1])
        x = self.embedding(tokens) * self.embed_scale
        rope = self._rope(_MAX_SEQ_LEN)
        engram_kv = self._engram_kv(tokens, mask, quant)
        x, _ = self.stack(x, mask=mask, rope=rope, engram_kv=engram_kv, quant=quant)
        return architecture._aq(x, quant).astype(jnp.float32) @ self.embedding.embedding.T
    architecture.SimpleAttentionNetwork.__call__ = model_call

    def cached_logits(self, tokens, quant=False):
        if tokens.shape[1] != 1:
            raise ValueError("cached_logits expects exactly one token")
        return self.__call__(tokens, quant=quant, use_kv_cache=True)
    architecture.SimpleAttentionNetwork.cached_logits = cached_logits
    _ENABLED = _INSTALLED = True


def enabled() -> bool:
    return _ENABLED
