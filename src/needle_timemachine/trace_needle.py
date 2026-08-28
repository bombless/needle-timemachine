"""CLI for producing and serving a real Needle Time Machine trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .needle_runtime import load_needle_checkpoint
from .trace import Tracer


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


def write_trace(path: Path, tracer: Tracer, *, checkpoint: str, prompt: str, config: Any) -> None:
    payload = {
        "format": "needle-timemachine.trace/v1",
        "checkpoint": checkpoint,
        "prompt": prompt,
        "config": _jsonable(vars(config)) if hasattr(config, "__dict__") else str(config),
        "events": [event.to_dict() for event in tracer.events],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace a real Needle checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Needle checkpoint path")
    parser.add_argument("--needle-source", required=True, help="Local cactus-compute/needle checkout")
    parser.add_argument("--prompt", required=True, help="Prompt to tokenize")
    parser.add_argument("--output", default="traces/run.json", help="Output trace JSON path")
    parser.add_argument("--trace-level", choices=("none", "layer"), default="layer")
    parser.add_argument("--serve", action="store_true", help="Serve the trace in the local timeline UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tracer = Tracer()
    runtime = load_needle_checkpoint(
        args.checkpoint,
        needle_source=args.needle_source,
        tracer=tracer,
        trace_level=args.trace_level,
    )
    token_ids = [runtime.tokenizer.bos_token_id] + runtime.tokenizer.encode(args.prompt)
    if len(token_ids) > runtime.config.max_seq_len:
        raise ValueError(
            f"Prompt token count {len(token_ids)} exceeds max_seq_len={runtime.config.max_seq_len}"
        )

    tokens = np.asarray([token_ids], dtype=np.int32)
    print("Needle Time Machine")
    print("-------------------")
    print(f"tokens:     {len(token_ids)}")
    print(f"trace:      {args.trace_level}")
    print(f"checkpoint: {args.checkpoint}")

    runtime.hidden_states(tokens)
    write_trace(
        Path(args.output),
        tracer,
        checkpoint=str(args.checkpoint),
        prompt=args.prompt,
        config=runtime.config,
    )

    layer_events = [e for e in tracer.events if e.op == "layer_output"]
    print(f"layers:     {len(layer_events)}")
    print(f"events:     {len(tracer.events)}")
    print(f"saved:      {args.output}")

    if args.serve:
        from .ui import serve
        serve(Path(args.output), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
