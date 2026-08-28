"""Browser UI for Needle Time Machine traces and interactive experiments."""
from __future__ import annotations
import argparse, json, logging, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from .experiments import TEMPLATES, run_experiment

log = logging.getLogger("needle_timemachine.ui")

_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Needle Time Machine</title><style>
:root{font-family:ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f5f7fb}body{margin:0}.app{max-width:1280px;margin:auto;padding:24px}.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,input,textarea{font:inherit}button{padding:8px 12px;border:1px solid #cbd2df;border-radius:8px;background:white;cursor:pointer}button:hover{background:#eef2f7}.primary{background:#172033;color:white;border-color:#172033}.card{background:white;border:1px solid #dde2eb;border-radius:12px;padding:16px;margin-top:16px;box-shadow:0 1px 2px #0000000b}.editor,.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.field{display:flex;flex-direction:column;gap:6px}.full{grid-column:1/-1}label{font-weight:600;font-size:13px}textarea{width:100%;box-sizing:border-box;min-height:120px;border:1px solid #cbd2df;border-radius:8px;padding:10px;font-family:ui-monospace,monospace;font-size:13px;resize:vertical}#tools{min-height:250px}select,input{border:1px solid #cbd2df;border-radius:8px;padding:8px}.muted{color:#697586}.error{color:#a22;font-weight:600;white-space:pre-wrap}.timeline{position:relative;height:72px;overflow-x:auto;padding:18px 8px 0}.track{position:relative;height:3px;background:#ccd3df;top:25px;min-width:100%}.dot{position:absolute;top:18px;width:14px;height:14px;border-radius:50%;border:2px solid white;background:#697586;box-shadow:0 0 0 1px #aeb7c5;transform:translateX(-50%);cursor:pointer}.dot.current{background:#e05a33;transform:translateX(-50%) scale(1.25)}#slider{width:100%}.kv{display:grid;grid-template-columns:130px 1fr;gap:7px;font-family:ui-monospace,monospace;font-size:13px}.kv div{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow:auto;max-height:300px;background:#f7f8fa;padding:12px;border-radius:8px}.probability-table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}.probability-table th,.probability-table td{text-align:left;padding:8px;border-bottom:1px solid #e5e9f0}.probability-table th:nth-child(5),.probability-table td:nth-child(5){text-align:right}.probability-bar{height:8px;border-radius:4px;background:#e5e9f0;overflow:hidden}.probability-fill{height:100%;background:#3d63dd}.probability-value{font-family:ui-monospace,monospace;min-width:80px;text-align:right}.token-text,.token-hex{font-family:ui-monospace,monospace;white-space:pre-wrap}.template-row{display:flex;gap:8px;flex-wrap:wrap}.template{display:flex;flex-direction:column;align-items:flex-start;min-width:190px}.template small{font-weight:normal;color:#697586;text-align:left;margin-top:3px}.badge{font-size:12px;padding:3px 7px;border-radius:999px;background:#eef2f7}.chosen{font-family:ui-monospace,monospace;padding:10px;background:#f7f8fa;border-radius:8px;overflow-wrap:anywhere}.hidden{display:none}@media(max-width:800px){.editor,.grid{grid-template-columns:1fr}.full{grid-column:auto}.probability-table{font-size:12px}.probability-table th,.probability-table td{padding:5px}}
</style></head><body><main class="app"><div class="bar"><h1 style="margin:0">Needle Time Machine</h1><span id="summary" class="muted"></span></div><section class="card"><div class="bar"><h2 style="margin:0">Experiment Playground</h2><span class="badge">Prompt + Tools + System facts</span></div><p class="muted">Choose a starter experiment, edit any input, then run Needle and inspect every generated token.</p><div id="templates" class="template-row"></div><div class="editor" style="margin-top:14px"><div class="field"><label for="prompt">Prompt</label><textarea id="prompt"></textarea></div><div class="field"><label for="system">System facts / instructions</label><textarea id="system"></textarea></div><div class="field full"><label for="tools">Tools / JSON Schema</label><textarea id="tools" spellcheck="false"></textarea></div></div><div class="bar" style="margin-top:12px"><button class="primary" id="run">▶ Run & Trace</button><label>Top-K <input id="topk" type="number" min="1" max="50" value="5" style="width:65px"></label><label>Max new tokens <input id="maxnew" type="number" min="1" max="256" value="48" style="width:75px"></label><span id="status" class="muted"></span></div><div id="error" class="error"></div></section><section id="result" class="hidden"><div class="card"><div class="bar"><h2 style="margin:0">Generated output</h2><span id="resultSummary" class="muted"></span></div><pre id="output"></pre></div><div class="card"><div class="bar"><button id="first">|&lt;</button><button id="prev">&lt;</button><button id="next">&gt;</button><button id="last">&gt;|</button><span id="step" class="badge"></span></div><input id="slider" type="range" min="0" max="0" value="0"><div class="timeline"><div class="track" id="track"></div></div></div><div class="grid"><section class="card"><h2>Chosen token</h2><div id="chosen" class="chosen"></div><h3>Top-K probabilities</h3><div id="probabilities"></div></section><section class="card"><h2>Experiment input</h2><div id="input" class="kv"></div><h3>Rendered Needle prompt</h3><pre id="rendered"></pre></section></div></section><section class="card"><h2>Existing trace</h2><p class="muted">The original trace timeline remains available when you load a trace produced by the CLI.</p></section></main><script>
let data=null,pos=0;const $=id=>document.getElementById(id);const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function templates(){let ts=await(await fetch('/api/templates')).json();$('templates').innerHTML=ts.map(t=>`<button class="template" data-id="${esc(t.id)}"><b>${esc(t.name)}</b><small>${esc(t.description)}</small></button>`).join('');ts.forEach(t=>document.querySelector(`[data-id="${CSS.escape(t.id)}"]`).onclick=()=>apply(t));if(ts[0])apply(ts[0])}function apply(t){$('prompt').value=t.prompt;$('system').value=t.system||'';$('tools').value=JSON.stringify(t.tools||[],null,2);$('error').textContent=''}
async function run(){ $('error').textContent='';$('status').textContent='Running…';$('run').disabled=true;try{let tools=JSON.parse($('tools').value||'[]');if(!Array.isArray(tools))throw Error('Tools must be a JSON array.');let r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:$('prompt').value,system:$('system').value,tools,top_k:+$('topk').value,max_new_tokens:+$('maxnew').value})});let j=await r.json();if(!r.ok)throw Error(j.error||'Run failed');data=j;pos=0;$('result').classList.remove('hidden');$('output').textContent=j.generated_text||'(empty)';$('resultSummary').textContent=`${j.events.length} generated steps`;$('rendered').textContent=j.rendered_prompt;$('input').innerHTML=[['prompt',j.query],['system',j.system||'—'],['tools',JSON.stringify(j.tools)],['steps',j.events.length]].map(([k,v])=>`<div class="muted">${esc(k)}</div><div>${esc(v)}</div>`).join('');$('slider').max=Math.max(0,j.events.length-1);draw();render(0);$('status').textContent='Done';$('result').scrollIntoView({behavior:'smooth',block:'start'})}catch(e){$('error').textContent=e.message;$('status').textContent='Failed'}finally{$('run').disabled=false}}
function draw(){let t=$('track');t.innerHTML='';(data?.events||[]).forEach((e,i)=>{let d=document.createElement('button');d.className='dot';d.title=`Step ${i+1}: ${e.chosen.token_text}`;d.style.left=(data.events.length<2?50:i*100/(data.events.length-1))+'%';d.onclick=()=>render(i);t.appendChild(d)})}function render(i){if(!data?.events?.length)return;pos=Math.max(0,Math.min(i,data.events.length-1));let e=data.events[pos],c=e.chosen;$('slider').value=pos;$('step').textContent=`Step ${pos+1} / ${data.events.length}`;$('chosen').innerHTML=`<b>${esc(c.token_text||'∅')}</b> · ID ${c.token_id} · <code>0x${c.token_id.toString(16).toUpperCase()}</code> · bytes <code>${esc(c.token_bytes_hex||'')}</code> · <b>${(c.probability*100).toFixed(4)}%</b>`;let top=e.top_k||[];$('probabilities').innerHTML='<table class="probability-table"><thead><tr><th>Rank</th><th>Token ID</th><th>Token text</th><th>Token bytes hex</th><th>Probability</th><th>Distribution</th></tr></thead><tbody>'+top.map((x,n)=>`<tr><td>${n+1}</td><td><code>${x.token_id}</code> / <code>0x${x.token_id.toString(16).toUpperCase()}</code></td><td class="token-text">${esc(x.token_text||'')}</td><td class="token-hex"><code>${esc(x.token_bytes_hex||'')}</code></td><td class="probability-value">${(x.probability*100).toFixed(4)}%</td><td><div class="probability-bar"><div class="probability-fill" style="width:${Math.min(100,x.probability*100)}%"></div></div></td></tr>`).join('')+'</tbody></table>';document.querySelectorAll('.dot').forEach((d,n)=>d.classList.toggle('current',n===pos))}
$('run').onclick=run;$('first').onclick=()=>render(0);$('prev').onclick=()=>render(pos-1);$('next').onclick=()=>render(pos+1);$('last').onclick=()=>render(data?.events?.length-1);$('slider').oninput=e=>render(+e.target.value);templates();</script></body></html>'''

def build_arg_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Serve the Needle Time Machine browser UI.")
    p.add_argument("trace",type=Path,nargs="?",help="Optional existing trace JSON")
    p.add_argument("--checkpoint",help="Needle checkpoint for interactive Run")
    p.add_argument("--needle-source",help="Local cactus-compute/needle checkout")
    p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8765)
    return p

def serve(trace_path: Path|None=None,host:str="127.0.0.1",port:int=8765,*,runtime:Any=None)->None:
    payload=trace_path.read_text(encoding="utf-8") if trace_path and trace_path.exists() else json.dumps({"events":[]})
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self)->None:
            if self.path in ("/","/index.html"): body=_HTML.encode();ct="text/html; charset=utf-8"
            elif self.path=="/trace.json": body=payload.encode();ct="application/json; charset=utf-8"
            elif self.path=="/api/templates": body=json.dumps(TEMPLATES,ensure_ascii=False).encode();ct="application/json; charset=utf-8"
            else:self.send_error(404);return
            self.send_response(200);self.send_header("Content-Type",ct);self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self)->None:
            if self.path!="/api/run":self.send_error(404);return
            started=time.perf_counter();client=self.client_address[0] if self.client_address else "?"
            log.info("[request] POST /api/run client=%s",client)
            if runtime is None:
                log.error("[request] runtime is None")
                self._json(503,{"error":"Interactive Run unavailable: start through trace_needle with --checkpoint and --needle-source."});return
            try:
                n=int(self.headers.get("Content-Length","0"));log.info("[request] reading body bytes=%d",n)
                req=json.loads(self.rfile.read(n).decode());query=req.get("prompt","");tools=req.get("tools") or [];system=req.get("system","") or "";max_new=int(req.get("max_new_tokens",48));top_k=int(req.get("top_k",5))
                log.info("[request] parsed prompt_chars=%d system_chars=%d tools=%d top_k=%d max_new_tokens=%d",len(query),len(system),len(tools),top_k,max_new)
                log.info("[run] entering run_experiment")
                result=run_experiment(runtime,query,tools,system,max_new_tokens=max_new,top_k=top_k)
                log.info("[run] completed events=%d generated_chars=%d elapsed=%.2fs",len(result.get("events",[])),len(result.get("generated_text", "")),time.perf_counter()-started)
                self._json(200,result)
            except Exception as exc:
                log.exception("[run] FAILED after %.2fs",time.perf_counter()-started)
                self._json(500,{"error":f"{type(exc).__name__}: {exc}"})
        def _json(self,status:int,obj:Any)->None:
            body=json.dumps(obj,ensure_ascii=False).encode();self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8");self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,fmt:str,*args:Any)->None:return
    server=ThreadingHTTPServer((host,port),Handler);log.info("[server] UI listening on http://%s:%d",host,port)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

def main(argv:list[str]|None=None)->int:
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s",force=True)
    args=build_arg_parser().parse_args(argv);log.info("[startup] checkpoint=%s needle_source=%s",args.checkpoint,args.needle_source);runtime=None
    if args.checkpoint and args.needle_source:
        from .needle_runtime import load_needle_checkpoint
        from .trace import Tracer
        log.info("[startup] loading Needle checkpoint")
        runtime=load_needle_checkpoint(args.checkpoint,needle_source=args.needle_source,tracer=Tracer(),trace_level="none")
        log.info("[startup] checkpoint loaded")
    if args.trace and not args.trace.exists():raise SystemExit(f"trace not found: {args.trace}")
    serve(args.trace,args.host,args.port,runtime=runtime);return 0

if __name__=="__main__":raise SystemExit(main())
