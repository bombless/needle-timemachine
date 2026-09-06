"""Explicit NumPy/JAX forward used to verify the traced Needle execution.

This intentionally does not call ``model.apply``.  It mirrors the arithmetic in
``forward.js`` so the trace can be checked against an independent forward path.
"""
from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


def _flat(tree: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(tree, dict) or hasattr(tree, "items"):
        for key, value in tree.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            out.update(_flat(value, path))
    else:
        out[prefix] = tree
    return out


def flatten_params(params: Any) -> dict[str, Any]:
    """Flatten a Flax parameter tree using the slash paths used by forward.js."""
    root = params["params"] if isinstance(params, dict) and "params" in params else params
    return _flat(root)


def _w(w: dict[str, Any], name: str) -> jnp.ndarray:
    if name in w:
        return jnp.asarray(w[name], dtype=jnp.float64)
    # Flax scans can retain an extra ``layers``/``block`` container.  The
    # flattened checkpoint normally already matches this path; this fallback
    # makes the verifier tolerant of an additional leading stack scope.
    suffix = "/" + name
    for key, value in w.items():
        if key.endswith(suffix):
            return jnp.asarray(value, dtype=jnp.float64)
    raise KeyError(f"reference forward parameter not found: {name}")


def _cfg(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default)


def rms_unit(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x = x.astype(jnp.float64)
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


def zcrms(x: jnp.ndarray, scale: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    return (1.0 + scale) * rms_unit(x, eps)


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(x)


def rope_freqs(head_dim: int, seq_len: int, theta: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    half = head_dim // 2
    j = jnp.arange(half, dtype=jnp.float64)
    freq = 1.0 / theta ** ((2.0 * j) / head_dim)
    a = jnp.arange(seq_len, dtype=jnp.float64)[:, None] * freq[None, :]
    return jnp.cos(a), jnp.sin(a)


def rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    h = x.shape[-1] // 2
    c = cos[: x.shape[2]][None, None, :, :]
    s = sin[: x.shape[2]][None, None, :, :]
    x1, x2 = x[..., :h], x[..., h:]
    return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def shift_right(x: jnp.ndarray, offset: int) -> jnp.ndarray:
    if offset == 0:
        return x
    return jnp.pad(x, ((0, 0), (offset, 0), (0, 0)))[:, : x.shape[1]]


def sinkhorn(logits: jnp.ndarray, iters: int = 20) -> jnp.ndarray:
    x = logits
    for _ in range(iters):
        x = x - jax.nn.logsumexp(x, axis=-1, keepdims=True)
        x = x - jax.nn.logsumexp(x, axis=-2, keepdims=True)
    return jnp.exp(x)


def walsh(n: int) -> jnp.ndarray:
    h = jnp.ones((1, 1), dtype=jnp.float64)
    while h.shape[0] < n:
        h = jnp.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def engram_indices(tokens: jnp.ndarray, orders: tuple[int, ...], heads: int, slots: int) -> jnp.ndarray:
    u = tokens.astype(jnp.uint32)
    all_idx = []
    seed0, prime = 0x9E3779B9, 0x01000193
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (seed0 * (oi * heads + h + 1)) & 0xFFFFFFFF
            acc = jnp.full_like(u, jnp.uint32(seed))
            for j in range(order):
                shifted = jnp.pad(u, ((0, 0), (j, 0)))[:, : u.shape[1]] if j else u
                acc = (acc ^ shifted) * jnp.uint32(prime)
            acc = acc ^ (acc >> jnp.uint32(15))
            all_idx.append((acc % jnp.uint32(slots)).astype(jnp.int32))
    return jnp.stack(all_idx, axis=-1)


def make_engram_kv(tokens: jnp.ndarray, cfg: Any, w: dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    layers = tuple(_cfg(cfg, "engram_layers", ()))
    if not layers:
        return None
    orders = tuple(_cfg(cfg, "engram_orders", (2, 3)))
    heads = int(_cfg(cfg, "engram_heads", 0) or max(1, int(cfg.d_model) // (len(orders) * 128)))
    sub_dim = int(cfg.d_model) // (len(orders) * heads)
    slots = int(cfg.engram_slots)
    idx = engram_indices(tokens, orders, heads, slots)
    num_tables = len(orders) * heads
    ks, vs = [], []
    max_order = max(orders)
    t = tokens.shape[1]
    for site in range(len(layers)):
        table = _w(w, f"engrams_{site}/embedding")
        fetched = []
        for j in range(num_tables):
            one = table[j]
            gathered = one[idx[..., j]]
            order = orders[j // heads]
            ok = (jnp.arange(t)[None, :] >= order - 1)[..., None]
            fetched.append(gathered * ok)
        e = jnp.stack(fetched, axis=2).reshape(tokens.shape[0], t, num_tables * sub_dim)
        k = e @ _w(w, f"engrams_{site}/key_proj/kernel")
        v = e @ _w(w, f"engrams_{site}/value_proj/kernel")
        taps = _w(w, f"engrams_{site}/taps")
        vv = jnp.zeros_like(v)
        for j in range(4):
            shifted = shift_right(v, j * max_order)
            ok = (jnp.arange(t)[None, :] >= j * max_order)[..., None]
            vv = vv + shifted * taps[j][None, None, :] * ok
        ks.append(k)
        vs.append(vv)
    return jnp.stack(ks, axis=0), jnp.stack(vs, axis=0)


def attention(x: jnp.ndarray, layer: int, cfg: Any, w: dict[str, Any], cos: jnp.ndarray, sin: jnp.ndarray, causal: jnp.ndarray) -> jnp.ndarray:
    p = "stack/layers/block/self_attn"
    c, heads, kv_heads = int(cfg.d_model), int(cfg.num_heads), int(cfg.num_kv_heads)
    attn_dim = int(_cfg(cfg, "attn_dim", 0) or c)
    hd = attn_dim // heads
    qproj = _w(w, f"{p}/q_proj/kernel")[layer]
    kproj = _w(w, f"{p}/k_proj/kernel")[layer]
    vproj = _w(w, f"{p}/v_proj/kernel")[layer]
    qnorm = _w(w, f"{p}/q_norm/scale")[layer]
    knorm = _w(w, f"{p}/k_norm/scale")[layer]
    q0 = (x @ qproj).reshape(x.shape[0], x.shape[1], heads, hd).transpose(0, 2, 1, 3)
    k0 = (x @ kproj).reshape(x.shape[0], x.shape[1], kv_heads, hd).transpose(0, 2, 1, 3)
    v0 = (x @ vproj).reshape(x.shape[0], x.shape[1], kv_heads, hd).transpose(0, 2, 1, 3)
    q = rope(zcrms(q0, qnorm), cos, sin)
    k = rope(zcrms(k0, knorm), cos, sin)
    repeat = heads // kv_heads
    if repeat > 1:
        k = jnp.repeat(k, repeat, axis=1)
        v0 = jnp.repeat(v0, repeat, axis=1)
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / math.sqrt(hd)
    scores = jnp.where(causal, scores, -1e30)
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(probs, v0).transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], attn_dim)
    gate = _w(w, "stack/layers/block/self_attn/gate_proj/kernel")[layer]
    out = out * sigmoid(x @ gate)
    outproj = _w(w, "stack/layers/block/self_attn/out_proj/kernel")[layer]
    return out @ outproj


def hadamard_mlp(x: jnp.ndarray, layer: int, cfg: Any, w: dict[str, Any], H: jnp.ndarray) -> jnp.ndarray:
    p = "stack/layers/block/hadamard_mlp"
    d1 = _w(w, f"{p}/d1")[layer]
    d2 = _w(w, f"{p}/d2")[layer]
    d3 = _w(w, f"{p}/d3")[layer]
    z = (x * d1) @ H
    z = jax.nn.silu(z * d2) @ H
    return z * d3


def forward(tokens: Any, cfg: Any, params: Any) -> jnp.ndarray:
    """Run the explicit forward.js-equivalent computation and return logits."""
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    w = flatten_params(params)
    b, t = tokens.shape
    c, n = int(cfg.d_model), int(cfg.mhc_lanes)
    embed = _w(w, "embedding/embedding")[tokens] * math.sqrt(c)
    hd = int(_cfg(cfg, "attn_dim", 0) or c) // int(cfg.num_heads)
    cos, sin = rope_freqs(hd, t, float(_cfg(cfg, "rope_theta", 100000.0)))
    causal = jnp.tril(jnp.ones((t, t), dtype=bool))[None, None, :, :]
    engram = make_engram_kv(tokens, cfg, w)
    x = jnp.broadcast_to(embed[:, :, None, :], (b, t, n, c)).astype(jnp.float64)
    H = walsh(1 << (c - 1).bit_length())
    for layer in range(int(cfg.num_layers)):
        nx = rms_unit(x.reshape(b, t, n * c))
        phi_pre = _w(w, "stack/mhc_phi_pre")[layer]
        phi_post = _w(w, "stack/mhc_phi_post")[layer]
        phi_res = _w(w, "stack/mhc_phi_res")[layer]
        a_pre = _w(w, "stack/mhc_a_pre")[layer]
        a_post = _w(w, "stack/mhc_a_post")[layer]
        a_res = _w(w, "stack/mhc_a_res")[layer]
        b_pre = _w(w, "stack/mhc_b_pre")[layer]
        b_post = _w(w, "stack/mhc_b_post")[layer]
        b_res = _w(w, "stack/mhc_b_res")[layer]
        active = layer % n
        pre_off = jnp.where(jnp.arange(n) == active, 4.0, -4.0)
        post_off = jnp.where(jnp.arange(n) == active, 0.0, -4.0)
        hpre = sigmoid(a_pre * jnp.einsum("btc,cn->btn", nx, phi_pre) + b_pre + pre_off)
        u = jnp.einsum("btn,btnc->btc", hpre, x)
        block_input = u
        if engram is not None:
            ek, ev = engram
            alpha = sigmoid(jnp.einsum("btd,sbtd->sbt", rms_unit(u), rms_unit(ek)) / math.sqrt(c))
            flags = jnp.asarray([float(site_layer == layer) for site_layer in cfg.engram_layers])
            block_input = u + jnp.einsum("s,sbt,sbtd->btd", flags, alpha, ev)
        pre = zcrms(block_input, _w(w, "stack/layers/block/ZCRMSNorm_0/scale")[layer])
        attn = attention(pre, layer, cfg, w, cos, sin, causal)
        post = zcrms(attn, _w(w, "stack/layers/block/post_attn_norm/scale")[layer])
        gate = sigmoid(_w(w, "stack/layers/block/attn_gate")[layer])
        after_attn = block_input + post * gate
        pre_h = zcrms(after_attn, _w(w, "stack/layers/block/pre_hada_norm/scale")[layer])
        block_output = hadamard_mlp(pre_h, layer, cfg, w, H) + after_attn
        y = block_output - u
        hpost = 2.0 * sigmoid(a_post * jnp.einsum("btc,cn->btn", nx, phi_post) + b_post + post_off)
        res = jnp.einsum("btc,cn->btn", nx, phi_res)
        hres = sinkhorn((a_res * res).reshape(b, t, n, n) + b_res)
        mixed = jnp.einsum("btij,btjc->btic", hres, x)
        x = mixed + jnp.einsum("btn,btc->btnc", hpost, y)
    x = jnp.mean(x, axis=2)
    x = zcrms(x, _w(w, "stack/final_norm/scale"))
    return x @ _w(w, "embedding/embedding").T


def verify(tokens: Any, cfg: Any, params: Any, actual: Any) -> dict[str, float]:
    expected = jnp.asarray(forward(tokens, cfg, params), dtype=jnp.float64)
    got = jnp.asarray(actual, dtype=jnp.float64)
    if expected.shape != got.shape:
        raise ValueError(f"reference logits shape {expected.shape} != Needle logits shape {got.shape}")
    diff = expected - got
    max_abs = float(jnp.max(jnp.abs(diff)))
    max_rel = float(jnp.max(jnp.abs(diff) / jnp.maximum(1e-6, jnp.abs(expected))))
    rmse = float(jnp.sqrt(jnp.mean(diff * diff)))
    cosine = float(jnp.sum(expected * got) / jnp.sqrt(jnp.sum(expected * expected) * jnp.sum(got * got)))
    return {"max_abs": max_abs, "max_rel": max_rel, "rmse": rmse, "cosine": cosine}
