"""CLI and browser-facing APIs for tracing and verifying a real Needle checkpoint."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cases import BASE_CASE
from .needle_runtime import load_needle_checkpoint
from .trace import Tracer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "needle2.pkl"
DEFAULT_NEEDLE_SOURCE = PROJECT_ROOT / "needle"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    if hasattr(value, "item"):
        try: return value.item()
        except (TypeError, ValueError): pass
    return str(value)


def _tensor_for_browser(value: Any) -> dict[str, Any]:
    arr = np.asarray(value, dtype=np.float32)
    arr = np.asarray(arr, dtype="<f4", order="C")
    return {"shape": [int(x) for x in arr.shape], "dtype": "float32", "encoding": "base64-f32-le", "data": base64.b64encode(arr.tobytes(order="C")).decode("ascii")}


def _serialize_params(tree: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(tree, dict):
        for key, value in tree.items(): out.update(_serialize_params(value, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree): out.update(_serialize_params(value, f"{prefix}/{i}" if prefix else str(i)))
    elif hasattr(tree, "shape") and hasattr(tree, "dtype"):
        out[prefix] = _tensor_for_browser(tree)
    else: raise TypeError(f"Unsupported checkpoint parameter leaf at {prefix!r}: {type(tree)!r}")
    return out


def _find_logits(value: Any) -> np.ndarray:
    if hasattr(value, "shape") and hasattr(value, "dtype"): return np.asarray(value)
    if isinstance(value, (list, tuple)):
        arrays=[]
        for item in value:
            try: arrays.append(_find_logits(item))
            except TypeError: pass
        if arrays:
            return next((a for a in arrays if a.ndim>=2), arrays[0])
    raise TypeError(f"Could not find logits array in model output of type {type(value)!r}")


def _top_k_probabilities(logits: Any, top_k: int, tokenizer: Any) -> list[dict[str, Any]]:
    array=_find_logits(logits)
    if array.ndim<2: raise ValueError(f"Expected logits with at least 2 dimensions, got {array.shape}")
    final_logits=np.asarray(array[0,-1],dtype=np.float64); shifted=final_logits-np.max(final_logits); exp=np.exp(shifted); probabilities=exp/np.sum(exp)
    k=min(max(1,int(top_k)),probabilities.size); indices=np.argpartition(probabilities,-k)[-k:]; indices=indices[np.argsort(probabilities[indices])[::-1]]
    rows=[]
    for index in indices:
        text=tokenizer.decode([int(index)])
        rows.append({"token_id":int(index),"token_text":text,"token_bytes_hex":text.encode("utf-8").hex(" "),"probability":float(probabilities[index])})
    return rows


def _prompt_token_list(token_ids: list[int], tokenizer: Any) -> list[dict[str, Any]]:
    rows=[]
    for token_id in token_ids:
        text=tokenizer.decode([token_id]); rows.append({"token_id":int(token_id),"token_text":text,"token_bytes_hex":text.encode("utf-8").hex(" ")})
    return rows


def write_trace(path: Path, tracer: Tracer, *, checkpoint: str, prompt: str, config: Any, token_ids: list[int] | None = None, tokenizer: Any = None) -> None:
    payload={"format":"needle-timemachine.trace/v1","checkpoint":checkpoint,"prompt":prompt,"prompt_tokens":_prompt_token_list(token_ids,tokenizer) if token_ids and tokenizer else [],"config":_jsonable(vars(config)) if hasattr(config,"__dict__") else str(config),"events":[event.to_dict() for event in tracer.events]}
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")


def export_checkpoint_for_browser(checkpoint_bytes: bytes, *, filename: str, prompt: str, needle_source: str | Path, trace_level: str = "layer", top_k: int = 5) -> dict[str, Any]:
    """Load weights, tokenize the prompt, and expose the complete parameter tree to jax-js."""
    import tempfile
    if not checkpoint_bytes: raise ValueError("The uploaded checkpoint is empty")
    if not filename.lower().endswith(".pkl"): raise ValueError("Please upload a .pkl Needle checkpoint")
    if top_k<1: raise ValueError("top_k must be >= 1")
    with tempfile.TemporaryDirectory(prefix="needle-browser-") as tmp:
        path=Path(tmp)/"checkpoint.pkl"; path.write_bytes(checkpoint_bytes); tracer=Tracer()
        runtime=load_needle_checkpoint(path,needle_source=needle_source,tracer=tracer,trace_level=trace_level)
        token_ids=[runtime.tokenizer.bos_token_id]+runtime.tokenizer.encode(prompt)
        if len(token_ids)>runtime.config.max_seq_len: raise ValueError(f"Prompt token count {len(token_ids)} exceeds max_seq_len={runtime.config.max_seq_len}")
        tokens=np.asarray([token_ids],dtype=np.int32); runtime.hidden_states(tokens); logits=runtime.logits(tokens)
        top=_top_k_probabilities(logits,top_k,runtime.tokenizer); tracer.emit("probability_output",name="model.output.probabilities",metadata={"top_k":top,"position":"final","source":"logits"})
        logits_array=_find_logits(logits)
        return {"format":"needle-timemachine.browser-verification/v1","filename":filename,"checkpoint_bytes":len(checkpoint_bytes),"prompt":prompt,"token_ids":token_ids,"prompt_tokens":_prompt_token_list(token_ids,runtime.tokenizer),"config":_jsonable(vars(runtime.config)),"weights":_serialize_params(runtime.params),"reference":{"events":[event.to_dict() for event in tracer.events],"top_k":top,"logits":_tensor_for_browser(logits_array)}}


def build_arg_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Trace a real Needle checkpoint."); p.add_argument("--checkpoint",default=str(DEFAULT_CHECKPOINT)); p.add_argument("--needle-source",default=str(DEFAULT_NEEDLE_SOURCE)); p.add_argument("--prompt"); p.add_argument("--output",default="traces/run.json"); p.add_argument("--trace-level",choices=("none","layer","op"),default="layer"); p.add_argument("--top-k",type=int,default=5); p.add_argument("--serve",action="store_true"); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8765); return p


def _needle_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]: return [tool.get("function") if tool.get("type")=="function" and isinstance(tool.get("function"),dict) else tool for tool in tools]

def _default_prompt() -> str:
    tools_json=json.dumps(_needle_tools(BASE_CASE.tools),separators=(",",":"),ensure_ascii=False); return "<|im_start|>user\n"+f"<tools>{tools_json}</tools>\n"+f"{BASE_CASE.prompt}<|im_end|>\n<|im_start|>assistant\n"


def main(argv:list[str]|None=None)->int:
    args=build_arg_parser().parse_args(argv); prompt=args.prompt if args.prompt is not None else _default_prompt(); tracer=Tracer(); runtime=load_needle_checkpoint(args.checkpoint,needle_source=args.needle_source,tracer=tracer,trace_level=args.trace_level); token_ids=[runtime.tokenizer.bos_token_id]+runtime.tokenizer.encode(prompt)
    if len(token_ids)>runtime.config.max_seq_len: raise ValueError(f"Prompt token count {len(token_ids)} exceeds max_seq_len={runtime.config.max_seq_len}")
    tokens=np.asarray([token_ids],dtype=np.int32); runtime.hidden_states(tokens); logits=runtime.logits(tokens); tracer.emit("probability_output",name="model.output.probabilities",metadata={"top_k":_top_k_probabilities(logits,args.top_k,runtime.tokenizer),"position":"final","source":"logits"}); write_trace(Path(args.output),tracer,checkpoint=str(args.checkpoint),prompt=prompt,config=runtime.config,token_ids=token_ids,tokenizer=runtime.tokenizer)
    if args.serve:
        from .ui import serve
        serve(Path(args.output),args.host,args.port,verification_runner=export_checkpoint_for_browser,needle_source=args.needle_source)
    return 0

if __name__=="__main__": raise SystemExit(main())
