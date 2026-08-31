"""Build a WebGPU-arithmetic-compatible model from an unquantized Needle 2 checkpoint."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np


def _flatten(tree: Any, prefix: tuple[str, ...] = ()):
    if isinstance(tree, dict):
        out = []
        for key, value in tree.items():
            out.extend(_flatten(value, prefix + (str(key),)))
        return out
    try:
        return [(prefix, np.asarray(tree))]
    except Exception:
        return []


def _find(leaves, suffix, contains=()):
    matches = [(p, v) for p, v in leaves
               if len(p) >= len(suffix) and p[-len(suffix):] == suffix
               and all(x in p for x in contains)]
    if len(matches) != 1:
        raise KeyError(f"expected one {'/'.join(suffix)}, got {[('/'.join(p), v.shape) for p,v in matches]}")
    return np.asarray(matches[0][1], dtype=np.float32)


def _scanned(leaves, module, name, layers):
    matches = [(p, v) for p, v in leaves
               if p[-1] == name and module in p and v.ndim >= 1 and v.shape[0] == layers]
    if len(matches) != 1:
        raise KeyError(f"expected one scanned {module}/{name}, got {[('/'.join(p), v.shape) for p,v in matches]}")
    return np.asarray(matches[0][1], dtype=np.float32)


def _dense(x):
    x = np.asarray(x, dtype=np.float32)
    return x.T


def _engram(leaves, site, name):
    matches = [(p, v) for p, v in leaves if p[-1] == name and any(
        tag in "/".join(p) for tag in (f"engrams_{site}", f"engrams/{site}", f"engrams[{site}]"))]
    if len(matches) == 1:
        return np.asarray(matches[0][1], dtype=np.float32)
    all_matches = [v for p, v in leaves if p[-1] == name]
    if len(all_matches) > site:
        return np.asarray(all_matches[site], dtype=np.float32)
    raise KeyError(f"missing engram {site}/{name}")


class UnquantizedWebGPUModel:
    """CactModel-compatible weights, populated directly from Needle 2 .pkl."""
    def __init__(self, checkpoint: str | Path, needle_source: str | Path):
        import sys
        source = str(Path(needle_source).resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        from needle.model import run, tokenizer as tokenizer_mod
        params, config = run.load_checkpoint(str(checkpoint))
        leaves = _flatten(params)
        self.config = config
        self.tokenizer = tokenizer_mod.get_tokenizer(config.vocab_size)

        D = int(config.d_model); L = int(config.num_layers)
        H = int(config.num_heads); KVH = int(config.num_kv_heads)
        Hd = int((config.attn_dim or D) // H); lanes = int(config.mhc_lanes)
        orders = tuple(config.engram_orders)
        heads = int(config.engram_heads or max(1, D // (len(orders) * 128)))
        sub = int(D // (len(orders) * heads))
        self.g = {
            "vocab_size": int(config.vocab_size), "d_model": D, "num_heads": H,
            "num_kv_heads": KVH, "num_layers": L, "head_dim": Hd,
            "max_seq_len": int(config.max_seq_len), "hada_n": 1 << (D - 1).bit_length(),
            "mhc_lanes": lanes, "engram_slots": int(config.engram_slots),
            "engram_sub_dim": sub, "num_engram_tables": len(orders) * heads,
            "engram_conv_taps": 4, "engram_conv_dilation": max(orders),
            "engram_orders": list(orders), "engram_layers": list(config.engram_layers),
            "rope_theta": float(config.rope_theta), "kv_window": int(getattr(config, "kv_window", 0)),
            "kv_bits": int(getattr(config, "kv_bits", 8)),
        }

        w = [_find(leaves, ("embedding", "embedding"))]
        norm = _scanned(leaves, "norm", "scale", L)
        q = _scanned(leaves, "q_proj", "kernel", L)
        k = _scanned(leaves, "k_proj", "kernel", L)
        v = _scanned(leaves, "v_proj", "kernel", L)
        qn = _scanned(leaves, "q_norm", "scale", L)
        kn = _scanned(leaves, "k_norm", "scale", L)
        gate = _scanned(leaves, "gate_proj", "kernel", L)
        out = _scanned(leaves, "out_proj", "kernel", L)
        post = _scanned(leaves, "post_attn_norm", "scale", L)
        ag = _scanned(leaves, "attn_gate", "attn_gate", L) if False else _find(leaves, ("stack", "layers", "block", "attn_gate"))
        pre = _scanned(leaves, "pre_hada_norm", "scale", L)
        d1 = _scanned(leaves, "hadamard_mlp", "d1", L)
        d2 = _scanned(leaves, "hadamard_mlp", "d2", L)
        d3 = _scanned(leaves, "hadamard_mlp", "d3", L)
        for i in range(L):
            w += [norm[i], _dense(q[i]), _dense(k[i]), _dense(v[i]), qn[i], kn[i], _dense(gate[i]),
                  _dense(out[i]), post[i], np.asarray(ag[i]).reshape(-1), pre[i], d1[i], d2[i], d3[i]]

        for name in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res"):
            w.append(_find(leaves, ("stack", name)))
        w += [_dense(_find(leaves, ("stack", "mhc_phi_pre"))),
              _dense(_find(leaves, ("stack", "mhc_phi_post"))),
              _dense(_find(leaves, ("stack", "mhc_phi_res")))]
        for site in range(len(config.engram_layers)):
            w += [_engram(leaves, site, "embedding"), _dense(_engram(leaves, site, "key_proj")),
                  _dense(_engram(leaves, site, "value_proj")), _engram(leaves, site, "taps")]
        w.append(_find(leaves, ("stack", "final_norm", "scale")))
        self.weights = [np.asarray(x, dtype=np.float32).reshape(-1) for x in w]
        expected = 1 + 14 * L + 9 + 4 * len(config.engram_layers) + 1
        if len(self.weights) != expected:
            raise RuntimeError(f"mapped {len(self.weights)} tensors, expected {expected}")
