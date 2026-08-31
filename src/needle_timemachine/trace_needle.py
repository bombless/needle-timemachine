"""CLI for producing and serving a real Needle Time Machine trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cases import BASE_CASE
from .needle_runtime import load_needle_checkpoint
from .trace import Tracer
from .webgpu_runtime import CactModel, run_webgpu


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "needle2.pkl"
DEFAULT_NEEDLE_SOURCE = PROJECT_ROOT / "needle"


class _CactTokenizer:
    """Tokenizer matching needle-webgpu's RAW SentencePiece dump."""
    def __init__(self, data: bytes):
        import struct
        self.parts = []
        self.scores = []
        self.types = []
        self.first = {}
        count = struct.unpack_from('<I', data, 0)[0]
        offset = 24
        for index in range(count):
            score = struct.unpack_from('<f', data, offset)[0]
            kind = data[offset + 4]
            size = struct.unpack_from('<H', data, offset + 5)[0]
            offset += 7
            part = data[offset:offset + size].decode('utf-8')
            offset += size
            self.parts.append(part)
            self.scores.append(score)
            self.types.append(kind)
            self.first.setdefault(part[0], []).append(index)

    def encode(self, text: str) -> list[int]:
        chars = list('▁' + text.replace(' ', '▁'))
        inf = 1e30
        dp = [inf] * (len(chars) + 1); prev = [-1] * (len(chars) + 1); piece = [-1] * (len(chars) + 1)
        dp[0] = 0
        for i, char in enumerate(chars):
            for index in self.first.get(char, []):
                value = list(self.parts[index])
                if chars[i:i + len(value)] == value and dp[i] - self.scores[index] < dp[i + len(value)]:
                    dp[i + len(value)] = dp[i] - self.scores[index]; prev[i + len(value)] = i; piece[i + len(value)] = index
            if dp[i + 1] >= inf:
                for byte in char.encode('utf-8'):
                    token = f'<0x{byte:02X}>'
                    if token in self.parts:
                        dp[i + 1] = dp[i] + 20; prev[i + 1] = i; piece[i + 1] = self.parts.index(token); break
        result = []
        i = len(chars)
        while i > 0 and prev[i] < i:
            result.append(piece[i]); i = prev[i]
        return result[::-1]

    def decode(self, ids: list[int]) -> str:
        out = ''
        for index in ids:
            part = self.parts[index] if 0 <= index < len(self.parts) else ''
            if self.types[index] == 4 and part.startswith('<0x'):
                out += chr(int(part[3:5], 16))
            else:
                out += part
        return out.replace('▁', ' ').strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _find_logits(value: Any) -> np.ndarray:
    """Find the logits array in Needle's forward return value."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(value)
    if isinstance(value, (list, tuple)):
        arrays = []
        for item in value:
            try:
                arrays.append(_find_logits(item))
            except TypeError:
                continue
        if arrays:
            for array in arrays:
                if array.ndim >= 2:
                    return array
            return arrays[0]
    raise TypeError(f"Could not find logits array in model output of type {type(value)!r}")


def _top_k_probabilities(logits: Any, top_k: int, tokenizer: Any) -> list[dict[str, Any]]:
    array = _find_logits(logits)
    if array.ndim == 1:
        final_logits = np.asarray(array, dtype=np.float64)
    elif array.ndim >= 2:
        final_logits = np.asarray(array[0, -1], dtype=np.float64)
    else:
        raise ValueError(f"Expected logits with at least 1 dimension, got {array.shape}")
    shifted = final_logits - np.max(final_logits)
    exp = np.exp(shifted)
    probabilities = exp / np.sum(exp)
    k = min(max(1, int(top_k)), probabilities.size)
    indices = np.argpartition(probabilities, -k)[-k:]
    indices = indices[np.argsort(probabilities[indices])[::-1]]

    results = []
    for index in indices:
        token_id = int(index)
        try:
            token_text = tokenizer.decode([token_id])
        except Exception:
            token_text = ""
        token_bytes_hex = token_text.encode("utf-8").hex(" ")
        results.append({
            "token_id": token_id,
            "token_text": token_text,
            "token_bytes_hex": token_bytes_hex,
            "probability": float(probabilities[index]),
        })
    return results


def _prompt_token_list(token_ids: list[int], tokenizer: Any) -> list[dict[str, Any]]:
    """Build a list of token dicts for the prompt (including BOS)."""
    results = []
    for token_id in token_ids:
        try:
            token_text = tokenizer.decode([token_id])
        except Exception:
            token_text = ""
        token_bytes_hex = token_text.encode("utf-8").hex(" ")
        results.append({
            "token_id": token_id,
            "token_text": token_text,
            "token_bytes_hex": token_bytes_hex,
        })
    return results


def write_trace(
    path: Path,
    tracer: Tracer,
    *,
    checkpoint: str,
    prompt: str,
    config: Any,
    token_ids: list[int] | None = None,
    tokenizer: Any = None,
) -> None:
    prompt_tokens = _prompt_token_list(token_ids, tokenizer) if token_ids and tokenizer else []
    payload = {
        "format": "needle-timemachine.trace/v1",
        "checkpoint": checkpoint,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "config": _jsonable(vars(config)) if hasattr(config, "__dict__") else _jsonable(config),
        "events": [event.to_dict() for event in tracer.events],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace Needle JAX or needle-webgpu arithmetic.")
    parser.add_argument("--backend", choices=("jax", "webgpu"), default="jax", help="inference path to reproduce")
    parser.add_argument("--cact", help=".cact file required by --backend webgpu")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help=f"Needle checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--needle-source",
        default=str(DEFAULT_NEEDLE_SOURCE),
        help=f"Local cactus-compute/needle checkout (default: {DEFAULT_NEEDLE_SOURCE})",
    )
    parser.add_argument("--prompt", help="Prompt to tokenize (defaults to cases.py BASE_CASE prompt and tools")
    parser.add_argument("--output", default="traces/run.json", help="Output trace JSON path")
    parser.add_argument(
        "--trace-level",
        choices=("none", "layer", "op"),
        default="layer",
        help="layer records hidden states; op records runtime attention/MLP values",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of final-token probabilities to show in the UI")
    parser.add_argument("--serve", action="store_true", help="Serve the trace in the local timeline UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def _needle_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style function tools to Needle's compact tool schema."""
    normalized = []
    for tool in tools:
        function = tool.get("function") if tool.get("type") == "function" else None
        normalized.append(function if isinstance(function, dict) else tool)
    return normalized


def _default_prompt(*, browser: bool = False) -> str:
    """Build the default prompt; browser mode matches engine-finetune.ts exactly."""
    tools_json = json.dumps(_needle_tools(BASE_CASE.tools), separators=(",", ":"), ensure_ascii=False)
    if browser:
        return f"<|im_start|>user<tools>{tools_json}</tools>{BASE_CASE.prompt}<|im_end|><|im_start|>assistant"
    return (
        "<|im_start|>user\n"
        f"<tools>{tools_json}</tools>\n"
        f"{BASE_CASE.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    prompt = args.prompt if args.prompt is not None else _default_prompt(browser=args.backend == "webgpu")
    tracer = Tracer()
    runtime = None
    cact = None
    if args.backend == "webgpu":
        if not args.cact:
            raise ValueError("--cact is required with --backend webgpu")
        cact = CactModel(args.cact)
        tokenizer = _CactTokenizer(cact.tokenizer_bytes())
        max_seq_len = cact.g["max_seq_len"]
        token_ids = [2] + tokenizer.encode(prompt)
    else:
        runtime = load_needle_checkpoint(args.checkpoint, needle_source=args.needle_source, tracer=tracer, trace_level=args.trace_level)
        tokenizer = runtime.tokenizer
        max_seq_len = runtime.config.max_seq_len
        token_ids = [runtime.tokenizer.bos_token_id] + runtime.tokenizer.encode(prompt)
    if len(token_ids) > max_seq_len:
        raise ValueError(f"Prompt token count {len(token_ids)} exceeds max_seq_len={max_seq_len}")

    tokens = np.asarray([token_ids], dtype=np.int32)
    print("Needle Time Machine")
    print("-------------------")
    print(f"tokens:     {len(token_ids)}")
    print(f"trace:      {args.trace_level}")
    print(f"backend:    {args.backend}")
    print(f"top-k:      {args.top_k}")
    print(f"model:      {args.cact if args.backend == 'webgpu' else args.checkpoint}")

    if args.backend == "webgpu":
        logits = run_webgpu(cact, token_ids, tracer, trace_level=args.trace_level)
    else:
        runtime.hidden_states(tokens)
        logits = runtime.logits(tokens)
    top_k = _top_k_probabilities(logits, args.top_k, tokenizer)
    tracer.emit(
        "probability_output",
        name="model.output.probabilities",
        metadata={"top_k": top_k, "position": "final", "source": "logits"},
    )
    write_trace(
        Path(args.output),
        tracer,
        checkpoint=str(args.cact if args.backend == "webgpu" else args.checkpoint),
        prompt=prompt,
        config=(cact.g if args.backend == "webgpu" else runtime.config),
        token_ids=token_ids,
        tokenizer=tokenizer,
    )

    layer_events = [e for e in tracer.events if e.op == "layer_output"]
    runtime_events = [e for e in tracer.events if e.metadata.get("runtime")]
    print(f"layers:     {len(layer_events)}")
    print(f"runtime:    {len(runtime_events)}")
    print(f"events:     {len(tracer.events)}")
    print(f"saved:      {args.output}")

    if args.serve:
        from .ui import serve
        serve(Path(args.output), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
