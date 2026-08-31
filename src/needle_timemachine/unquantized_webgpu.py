"""Adapt an unquantized Needle 2 checkpoint to the WebGPU arithmetic debugger.

The WebGPU debugger consumes the same flattened, output-major weight layout as
``webgpu_runtime.run_webgpu``.  Needle's Flax Dense kernels are input-major,
so this adapter transposes those kernels while preserving embedding tables and
vector parameters.  No quantization/dequantization is performed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _flatten(tree: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], np.ndarray]]:
    if isinstance(tree, dict):
        out: list[tuple[tuple[str, ...], np.ndarray]] = []
        for key, value in tree.items():
            out.extend(_flatten(value, prefix + (str(key),)))
        return out
    try:
        return [(prefix, np.asarray(tree))]
    except Exception:
        return []


def _leaf(leaves: list[tuple[tuple[str, ...], np.ndarray]], suffix: tuple[str, ...], *, contains: tuple[str, ...] = ()) -> np.ndarray:
    matches = [v for p, v in leaves if len(p) >= len(suffix) and p[-len(suffix):] == suffix
               and all(part in p for part in contains)]
    if not matches:
        paths = ["/".join(p) for p, _ in leaves]
        raise KeyError(f"missing parameter {'/'.join(suffix)}; available paths: {paths[:20]}")
    if len(matches) != 1:
        raise KeyError(f"ambiguous parameter {'/'.join(suffix)} ({len(matches)} matches)")
    return np.asarray(matches[0], dtype=np.float32)


def _layer_leaf(leaves: list[tuple[tuple[str, ...], np.ndarray]], name: str, layers: int) -> np.ndarray:
    # Scan parameters are stored with a leading layer axis by Flax.
    candidates = [(p, v) for p, v in leaves if p[-1] == name and v.ndim >= 1 and v.shape[0] == layers]
    if len(candidates) != 1:
        paths = ["/".join(p) for p, _ in candidates]
        raise KeyError(f"missing/ambiguous scanned parameter {name}: {paths}")
    return candidates[0][1]


def _dense_layer_weight(value: np.ndarray) -> np.ndarray:
    # Flax nn.Dense stores [in, out]; the WebGPU debugger uses [out, in].
    return np.asarray(value, dtype=np.float32).transpose(tuple(range(value.ndim - 2)) + (value.ndim - 1, value.ndim - 2))


def _engram_leaf(leaves: list[tuple[tuple[str, ...], np.ndarray]], site: int, name: str) -> np.ndarray:
    tags = (f"engrams_{site}", f"engrams/{site}", f"engrams[{site}]")
    for p, v in leaves:
        if p[-1] != name:
            continue
        if any(tag in "/".join(p) for tag in tags):
            return np.asarray(v, dtype=np.float32)
    # Flax may use a numeric list component in the path; fall back to the
    # occurrence order of this leaf name, which is stable for this module list.
    matches = [v for p, v in leaves if p[-1] == name]
    if len(matches) > site:
        return np.asarray(matches[site], dtype=np.float32)
    raise KeyError(f"missing engram parameter {site}/{name}")


class UnquantizedWebGPUModel:
    """CactModel-compatible view backed directly by Needle 2 ``.pkl`` params."""

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

        D = int(config.d_model)
        L = int(config.num_layers)
        H = int(config.num_heads)
        KVH = int(config.num_kv_heads)
        Hd = int((config.attn_dim or D) // H)
        lanes = int(config.mhc_lanes)
        orders = tuple(config.engram_orders)
        heads = int(config.engram_heads or max(1, D // (len(orders) * 128)))
        sub_dim = int(D // (len(orders) * heads))
        hada_n = 1 << (D - 1).bit_length()

        self.g = {
            "vocab_size": int(config.vocab_size), "d_model": D, "num_heads": H,
            "num_kv_heads": KVH, "num_layers": L, "head_dim": Hd,
            "max_seq_len": int(config.max_seq_len), "hada_n": hada_n,
            "mhc_lanes": lanes, "engram_slots": int(config.engram_slots),
            "engram_sub_dim": sub_dim, "num_engram_tables": len(orders) * heads,
            "engram_conv_taps": 4, "engram_conv_dilation": max(orders),
            "engram_orders": list(orders), "engram_layers": list(config.engram_layers),
            "rope_theta": float(config.rope_theta),
            "kv_window": int(getattr(config, "kv_window", 0)),
            "kv_bits": int(getattr(config, "kv_bits", 8)),
        }

        weights: list[np.ndarray] = []
        weights.append(_leaf(leaves, ("embedding", "embedding")))

        # Match the exact order expected by run_webgpu().
        layer_params = {
            "norm": _layer_leaf(leaves, "scale", L),
        }
        # There are several scanned scale leaves; resolve each by its parent
        # module name rather than terminal name alone.
        def scanned(module: str, name: str) -> np.ndarray:
            matches = [(p, v) for p, v in leaves if p[-1] == name and module in p and v.ndim >= 1 and v.shape[0] == L]
            if len(matches) != 1:
                raise KeyError(f"missing/ambiguous scanned {module}/{name}: {[p for p, _ in matches]}")
            return matches[0][1]

        for l in range(L):
            weights.extend([
                scanned("layers", "scale")[:, :][l],
                _dense_layer_weight(scanned("q_proj", "kernel")[l]),
                _dense_layer_weight(scanned("k_proj", "kernel")[l]),
                _dense_layer_weight(scanned("v_proj", "kernel")[l]),
                scanned("q_norm", "scale")[l],
                scanned("k_norm", "scale")[l],
                _dense_layer_weight(scanned("gate_proj", "kernel")[l]),
                _dense_layer_weight(scanned("out_proj", "kernel")[l]),
                scanned("post_attn_norm", "scale")[l],
                np.asarray(_leaf(leaves, ("stack", "layers", "block", "attn_gate")))[l:l+1] if False else np.asarray(_layer_leaf(leaves, "attn_gate", L)[l]).reshape(-1),
                scanned("pre_hada_norm", "scale")[l],
                scanned("d1", "d1")[l],
                scanned("d2", "d2")[l],
                scanned("d3", "d3")[l],
            ])

        for name in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res"):
            weights.append(_leaf(leaves, ("stack", name)))
        weights.append(_dense_layer_weight(_leaf(leaves, ("stack", "mhc_phi_pre"))))
        weights.append(_dense_layer_weight(_leaf(leaves, ("stack", "mhc_phi_post"))))
        weights.append(_dense_layer_weight(_leaf(leaves, ("stack", "mhc_phi_res"))))

        for site in range(len(config.engram_layers)):
            weights.append(_engram_leaf(leaves, site, "embedding"))
            weights.append(_dense_layer_weight(_engram_leaf(leaves, site, "key_proj")))
            weights.append(_dense_layer_weight(_engram_leaf(leaves, site, "value_proj")))
            weights.append(_engram_leaf(leaves, site, "taps"))

        weights.append(_leaf(leaves, ("stack", "final_norm", "scale")))
        self.weights = [np.asarray(v, dtype=np.float32).reshape(-1) for v in weights]

        expected = 1 + 14 * L + 9 + 4 * len(config.engram_layers) + 1
        if len(self.weights) != expected:
            raise RuntimeError(f"internal weight mapping produced {len(self.weights)} tensors; expected {expected}")

    def tokenizer_bytes(self) -> bytes:
        # Reuse the normal Needle tokenizer in trace_needle; this method exists
        # only so callers can treat this model like CactModel.
        raise TypeError("unquantized Needle tokenizer is not stored as .cact RAW bytes")
