"""Patch a local cactus-compute/needle checkout for Time Machine tracing.

Usage on Windows::

    py tools/patch_needle_trace.py d:\\needle2\\needle

The patch is deliberately kept outside the Needle submodule. It adds a tiny
hook API and runtime callbacks to the local checkout. The patch is idempotent
and creates ``architecture.py.timemachine.bak`` before the first change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# --- needle-timemachine runtime hooks ---"

HEADER = '''\n\n# --- needle-timemachine runtime hooks ---\n_TIMEMACHINE_HOOK = None\n\n\ndef set_timemachine_hook(hook):\n    """Install a host callback used only by the optional debugger."""\n    global _TIMEMACHINE_HOOK\n    _TIMEMACHINE_HOOK = hook\n\n\ndef _tm_emit(op_id, layer, value):\n    hook = _TIMEMACHINE_HOOK\n    if hook is not None:\n        jax.debug.callback(hook, jnp.asarray(op_id, dtype=jnp.int32),\n                           jnp.asarray(layer, dtype=jnp.int32), value,\n                           ordered=True)\n\n'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"anchor not found: {label}")
    return text.replace(old, new, 1)


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("already patched:", path)
        return

    backup = path.with_suffix(path.suffix + ".timemachine.bak")
    backup.write_text(text, encoding="utf-8")

    text = replace_once(
        text,
        "from .quantize import fake_quant_act\n",
        "from .quantize import fake_quant_act\n" + HEADER,
        "hook header",
    )

    text = replace_once(
        text,
        "def __call__(self, x, mask=None, rope=None, quant=False):\n        attn_dim = self.attn_dim or self.d_model\n",
        "def __call__(self, x, mask=None, rope=None, quant=False, trace_layer=None):\n        attn_dim = self.attn_dim or self.d_model\n",
        "attention signature",
    )
    text = replace_once(
        text,
        "        v = nn.Dense(kv_dim, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name=\"v_proj\")(x)\n",
        "        v = nn.Dense(kv_dim, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name=\"v_proj\")(x)\n        if trace_layer is not None:\n            _tm_emit(3, trace_layer, q)\n            _tm_emit(4, trace_layer, k)\n            _tm_emit(5, trace_layer, v)\n",
        "qkv hook",
    )
    text = replace_once(
        text,
        "            out = out.reshape(B, -1, attn_dim)\n        else:\n",
        "            out = out.reshape(B, -1, attn_dim)\n            if trace_layer is not None:\n                _tm_emit(6, trace_layer, out)\n        else:\n",
        "flash attention hook",
    )
    text = replace_once(
        text,
        "            out = out.transpose(0, 2, 1, 3).reshape(B, -1, attn_dim)\n\n        out = out * nn.sigmoid(\n",
        "            out = out.transpose(0, 2, 1, 3).reshape(B, -1, attn_dim)\n            if trace_layer is not None:\n                _tm_emit(6, trace_layer, out)\n\n        out = out * nn.sigmoid(\n",
        "non-flash attention hook",
    )
    text = replace_once(
        text,
        "        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name=\"out_proj\")(out)\n",
        "        projected = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name=\"out_proj\")(out)\n        if trace_layer is not None:\n            _tm_emit(7, trace_layer, projected)\n        return projected\n",
        "attention projection hook",
    )

    text = replace_once(
        text,
        "    def __call__(self, x, mask=None, rope=None, quant=False, engram_kv=None, site_flags=None):\n",
        "    def __call__(self, x, mask=None, rope=None, quant=False, engram_kv=None, site_flags=None, trace_layer=None):\n",
        "block signature",
    )
    text = replace_once(
        text,
        "        skip = x\n        x = ZCRMSNorm(dtype=self.dtype)(x)\n",
        "        skip = x\n        if trace_layer is not None:\n            _tm_emit(1, trace_layer, x)\n        x = ZCRMSNorm(dtype=self.dtype)(x)\n        if trace_layer is not None:\n            _tm_emit(2, trace_layer, x)\n",
        "block input norm hook",
    )
    text = replace_once(
        text,
        "                               name=\"self_attn\")(x, mask=mask, rope=rope, quant=quant)\n",
        "                               name=\"self_attn\")(x, mask=mask, rope=rope, quant=quant, trace_layer=trace_layer)\n",
        "block attention call",
    )
    text = replace_once(
        text,
        "        x = skip + self._gate(\"attn_gate\") * x\n\n        skip = x\n",
        "        x = skip + self._gate(\"attn_gate\") * x\n        if trace_layer is not None:\n            _tm_emit(8, trace_layer, x)\n\n        skip = x\n",
        "attention residual hook",
    )
    text = replace_once(
        text,
        "        x = ZCRMSNorm(dtype=self.dtype, name=\"pre_hada_norm\")(x)\n        x = HadamardMLP(self.d_model, self.dtype, name=\"hadamard_mlp\")(x)\n        return skip + x\n",
        "        x = ZCRMSNorm(dtype=self.dtype, name=\"pre_hada_norm\")(x)\n        if trace_layer is not None:\n            _tm_emit(9, trace_layer, x)\n        x = HadamardMLP(self.d_model, self.dtype, name=\"hadamard_mlp\")(x)\n        if trace_layer is not None:\n            _tm_emit(10, trace_layer, x)\n        result = skip + x\n        if trace_layer is not None:\n            _tm_emit(11, trace_layer, result)\n        return result\n",
        "mlp hooks",
    )

    text = replace_once(
        text,
        "        site_flags, hc = xs\n",
        "        site_flags, layer_id, hc = xs\n",
        "scan xs",
    )
    text = replace_once(
        text,
        "                  engram_kv=engram_kv, site_flags=site_flags) - u\n",
        "                  engram_kv=engram_kv, site_flags=site_flags, trace_layer=layer_id) - u\n",
        "scan block call",
    )
    text = replace_once(
        text,
        "        ScanBlock = nn.scan(\n",
        "        layer_ids = jnp.arange(L, dtype=jnp.int32)\n\n        ScanBlock = nn.scan(\n",
        "layer ids",
    )
    # The first in_axes entry already scans the whole ``xs`` pytree. Adding
    # layer_ids inside that pytree therefore preserves the original scan ABI.
    text = replace_once(
        text,
        "        )(x, (site_flags, hc), mask, rope, quant, engram_kv)\n",
        "        )(x, (site_flags, layer_ids, hc), mask, rope, quant, engram_kv)\n",
        "scan call",
    )

    path.write_text(text, encoding="utf-8")
    print("patched:", path)
    print("backup:", backup)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("needle_root", type=Path)
    args = ap.parse_args()
    path = args.needle_root / "needle" / "model" / "architecture.py"
    if not path.exists():
        raise SystemExit(f"Needle architecture.py not found: {path}")
    patch(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
