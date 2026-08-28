"""CLI for producing and serving a real Needle Time Machine trace."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
import numpy as np
from .needle_runtime import load_needle_checkpoint
from .trace import Tracer

def _jsonable(value:Any)->Any:
    if isinstance(value,dict): return {str(k):_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_jsonable(v) for v in value]
    if isinstance(value,np.generic): return value.item()
    if hasattr(value,"item"):
        try:return value.item()
        except (TypeError,ValueError):pass
    return str(value)

def _find_logits(value:Any)->np.ndarray:
    if hasattr(value,"shape") and hasattr(value,"dtype"):return np.asarray(value)
    if isinstance(value,(list,tuple)):
        arrays=[]
        for item in value:
            try:arrays.append(_find_logits(item))
            except TypeError:continue
        if arrays:
            return next((a for a in arrays if a.ndim>=2),arrays[0])
    raise TypeError(f"Could not find logits array in model output of type {type(value)!r}")

def _top_k_probabilities(logits:Any,top_k:int,tokenizer:Any)->list[dict[str,Any]]:
    array=_find_logits(logits)
    if array.ndim<2:raise ValueError(f"Expected logits with at least 2 dimensions, got {array.shape}")
    final=np.asarray(array[0,-1],dtype=np.float64);final-=np.max(final);p=np.exp(final);p/=np.sum(p);k=min(max(1,int(top_k)),p.size)
    idx=np.argpartition(p,-k)[-k:];idx=idx[np.argsort(p[idx])[::-1]]
    out=[]
    for i in idx:
        tid=int(i)
        try:text=tokenizer.decode([tid])
        except Exception:text=""
        out.append({"token_id":tid,"token_text":text,"token_bytes_hex":text.encode("utf-8",errors="replace").hex(" ").upper(),"probability":float(p[i])})
    return out

def write_trace(path:Path,tracer:Tracer,*,checkpoint:str,prompt:str,config:Any)->None:
    payload={"format":"needle-timemachine.trace/v1","checkpoint":checkpoint,"prompt":prompt,"config":_jsonable(vars(config)) if hasattr(config,"__dict__") else str(config),"events":[e.to_dict() for e in tracer.events]}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")

def build_arg_parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Trace Needle and serve the interactive playground.")
    p.add_argument("--checkpoint",required=True);p.add_argument("--needle-source",required=True);p.add_argument("--prompt",required=True);p.add_argument("--output",default="traces/run.json");p.add_argument("--trace-level",choices=("none","layer","op"),default="layer");p.add_argument("--top-k",type=int,default=5);p.add_argument("--serve",action="store_true");p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8765);return p

def main(argv:list[str]|None=None)->int:
    args=build_arg_parser().parse_args(argv)
    if args.top_k<1:raise ValueError("--top-k must be >= 1")
    tracer=Tracer();runtime=load_needle_checkpoint(args.checkpoint,needle_source=args.needle_source,tracer=tracer,trace_level=args.trace_level)
    ids=[runtime.tokenizer.bos_token_id]+runtime.tokenizer.encode(args.prompt)
    if len(ids)>runtime.config.max_seq_len:raise ValueError(f"Prompt token count {len(ids)} exceeds max_seq_len={runtime.config.max_seq_len}")
    tokens=np.asarray([ids],dtype=np.int32);print("Needle Time Machine\n-------------------");print(f"tokens:     {len(ids)}\ntrace:      {args.trace_level}\ntop-k:      {args.top_k}\ncheckpoint: {args.checkpoint}")
    runtime.hidden_states(tokens);logits=runtime.logits(tokens);top=_top_k_probabilities(logits,args.top_k,runtime.tokenizer)
    tracer.emit("probability_output",name="model.output.probabilities",metadata={"top_k":top,"position":"final","source":"logits"})
    write_trace(Path(args.output),tracer,checkpoint=str(args.checkpoint),prompt=args.prompt,config=runtime.config)
    print(f"layers:     {len([e for e in tracer.events if e.op=='layer_output'])}\nruntime:    {len([e for e in tracer.events if e.metadata.get('runtime')])}\nevents:     {len(tracer.events)}\nsaved:      {args.output}")
    if args.serve:
        from .ui import serve
        serve(Path(args.output),args.host,args.port,runtime=runtime)
    return 0
if __name__=="__main__":raise SystemExit(main())
