"""Browser-side WebGPU comparator for the Needle Time Machine trace.

This deliberately imports the *exact* TypeScript engine from needle-webui through
Vite instead of reimplementing its math in Python.  The page wraps Runtime.mm so
we can compare the WebGPU GEMM boundaries with the JAX trace produced by
trace_needle --trace-level op.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Needle WebGPU Compare</title>
<style>
body{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#f5f7fb;color:#172033}.app{max-width:1280px;margin:auto;padding:20px}.card{background:#fff;border:1px solid #dde2eb;border-radius:12px;padding:16px;margin:14px 0}button,input{font:inherit}button{padding:8px 12px;border:1px solid #cbd2df;border-radius:8px;background:#fff;cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.muted{color:#687385}pre{white-space:pre-wrap;overflow:auto;max-height:360px;background:#f7f8fa;padding:10px;border-radius:8px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:7px;border-bottom:1px solid #e5e9f0}td.num,th.num{text-align:right;font-family:ui-monospace,monospace}.ok{font-weight:700}.bad{font-weight:700}.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.small{font-size:12px}
@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style></head><body><main class="app">
<div class="bar"><h1 style="margin:0">Needle WebGPU Compare</h1><span id="status" class="muted">等待模型…</span></div>
<div class="card"><div class="bar"><label>.cact 模型 <input id="model" type="file" accept=".cact,.bin"></label><button id="run">运行 WebGPU</button><span id="gpu" class="muted"></span></div><p class="small muted">输入 token IDs 来自参考 trace.json，因此不会重新 tokenization；WebGPU 使用 needle-webui/src/engine.ts 的原始实现。</p></div>
<div class="grid"><section class="card"><h2>Reference</h2><div id="refInfo" class="muted"></div><pre id="refTop">{}</pre></section><section class="card"><h2>WebGPU</h2><div id="gpuInfo" class="muted"></div><pre id="gpuTop">{}</pre></section></div>
<section class="card"><h2>逐边界比较</h2><p class="muted small">重点看第一处明显变大的差异：q/k/v 与 projected 是最有价值的边界；若它们都接近而最终 logits 偏离，则问题通常在 attention / mHC / Hadamard / dtype。</p><div id="table">尚未运行。</div></section>
<section class="card"><h2>WebGPU GEMM 调用</h2><pre id="calls">[]</pre></section>
<script type="module">
const engine = await import('/@fs/WEBUI_ENGINE@')
const $=id=>document.getElementById(id)
let reference=null, wrapped=false, calls=[]
function stats(a){let min=Infinity,max=-Infinity,sum=0,n=0;for(const x of a){if(!Number.isFinite(x))continue;min=Math.min(min,x);max=Math.max(max,x);sum+=x;n++}return {n,min,max,mean:n?sum/n:NaN}}
function refEvent(layer,name){const es=(reference?.events||[]);return es.find(e=>e.layer===layer&&e.op===name&&e.metadata?.runtime)
}
function delta(a,b){if(!a||!b)return null;return {mean:Math.abs(a.mean-b.mean),min:Math.abs(a.min-b.min),max:Math.abs(a.max-b.max)}}
function fmt(x){return Number.isFinite(x)?x.toExponential(4):String(x)}
function renderRows(){
 const rows=[]
 for(const c of calls){if(!c.layer||!c.name)continue;const r=refEvent(c.layer,c.name);const d=delta(c.stats,r?.stats);rows.push({layer:c.layer,boundary:c.name,weight:c.weightIndex,webgpu:c.stats,reference:r?.stats||null,delta:d})}
 if(!rows.length){$('table').textContent='没有可映射的 q/k/v/projected GEMM。';return}
 $('table').innerHTML='<table><thead><tr><th>Layer</th><th>Boundary</th><th>Weight</th><th class="num">GPU mean</th><th class="num">REF mean</th><th class="num">Δmean</th><th class="num">Δmin</th><th class="num">Δmax</th></tr></thead><tbody>'+rows.map(r=>{const d=r.delta;const bad=!d||d.mean>1e-2;return `<tr><td>${r.layer}</td><td>${r.boundary}</td><td>${r.weight}</td><td class="num">${fmt(r.webgpu.mean)}</td><td class="num">${fmt(r.reference?.mean)}</td><td class="num ${bad?'bad':'ok'}">${fmt(d?.mean)}</td><td class="num">${fmt(d?.min)}</td><td class="num">${fmt(d?.max)}</td></tr>`}).join('')+'</tbody></table>'
}
function installWrapper(){if(wrapped)return;wrapped=true;const proto=engine.Runtime.prototype;const orig=proto.mm;proto.mm=async function(a,m,k,wi,n){const out=await orig.call(this,a,m,k,wi,n);const s=stats(out);let layer=null,name=null;for(const l of [0]){};const L=this.m.g.num_layers;for(let l=0;l<L;l++){const base=1+14*l;if(wi===base+1){layer=l;name='attention.q'}else if(wi===base+2){layer=l;name='attention.k'}else if(wi===base+3){layer=l;name='attention.v'}else if(wi===base+7){layer=l;name='attention.projected'}};calls.push({weightIndex:wi,m,k,n,stats:s,layer,name});$('calls').textContent=JSON.stringify(calls.slice(-80),null,2);return out}}
async function run(){
 const f=$('model').files?.[0];if(!f){$('status').textContent='请先选择 .cact 模型';return}
 if(!navigator.gpu){$('status').textContent='当前浏览器没有 WebGPU';return}
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('无法获取 WebGPU adapter');const dev=await adapter.requestDevice();
 installWrapper();calls=[];$('status').textContent='运行中…';
 const buf=await f.arrayBuffer();const m=new engine.Model(buf);const r=new engine.Runtime(dev,m);const counter={dispatches:0,flops:0,forwardMs:0};
 const ids=(reference?.prompt_tokens||[]).map(x=>Number(x.token_id));if(!ids.length)throw Error('reference trace 没有 prompt_tokens');
 const logits=await engine.generate(ids,m,r,layer=>{$('status').textContent=`WebGPU layer ${layer}/${m.g.num_layers}`},counter);
 const top=engine.topCandidates(logits,m.tok,10);$('gpuInfo').textContent=`${ids.length} tokens · ${counter.dispatches} dispatch · ${counter.flops} FLOPs`;$('gpuTop').textContent=JSON.stringify(top,null,2);$('status').textContent='完成';renderRows()
}
$('run').onclick=()=>run().catch(e=>{$('status').textContent='失败：'+(e?.message||e);console.error(e)})
reference=await (await fetch('/reference.json',{cache:'no-store'})).json();const refTop=(reference.events||[]).find(e=>e.op==='probability_output');$('refInfo').textContent=`${reference.prompt_tokens?.length||0} prompt tokens · ${reference.events?.length||0} events · ${reference.checkpoint||''}`;$('refTop').textContent=JSON.stringify(refTop?.metadata?.top_k||[],null,2);$('gpu').textContent='WebGPU comparator ready'
</script></main></body></html>'''


def _write_vite_project(root: Path, engine_path: Path) -> None:
    (root / "index.html").write_text(_HTML.replace("WEBUI_ENGINE@", engine_path.resolve().as_posix()), encoding="utf-8")
    (root / "package.json").write_text('{"private":true,"type":"module"}\n', encoding="utf-8")


def serve(reference_trace: Path, webui_source: Path, host: str, port: int) -> None:
    """Start a tiny HTTP endpoint and a Vite dev server for the TS engine."""
    engine_path = webui_source / "src" / "engine.ts"
    if not engine_path.exists():
        raise FileNotFoundError(f"needle-webui engine.ts not found: {engine_path}")
    if not (webui_source / "node_modules").exists():
        raise RuntimeError(f"{webui_source} has no node_modules; run npm install in needle-webui first")

    temp = Path(tempfile.mkdtemp(prefix="needle-webgpu-compare-"))
    _write_vite_project(temp, engine_path)
    ref_payload = reference_trace.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/reference.json":
                body, ctype = ref_payload, "application/json; charset=utf-8"
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, fmt: str, *args: Any) -> None:
            return

    api = ThreadingHTTPServer((host, 0), Handler)
    threading.Thread(target=api.serve_forever, daemon=True).start()
    api_port = api.server_address[1]
    # The Vite page is served on the public port. /reference.json is proxied by a
    # tiny Vite plugin below; this avoids cross-origin requests from the page.
    plugin = temp / "vite.config.js"
    plugin.write_text(f'''import {{ defineConfig }} from "vite"\nimport fs from "node:fs"\nexport default defineConfig({{plugins:[{{name:"tm-reference",configureServer(s){{s.middlewares.use("/reference.json",(_,res)=>{{const b=fs.readFileSync({json.dumps(str(reference_trace.resolve()))});res.setHeader("Content-Type","application/json");res.end(b)}})}}}}]}})\n''', encoding="utf-8")
    cmd = ["npx", "vite", "--host", host, "--port", str(port), "--strictPort"]
    print(f"Needle WebGPU comparator: http://{host}:{port}/")
    print(f"reference: {reference_trace}")
    print(f"webui engine: {engine_path}")
    proc = subprocess.Popen(cmd, cwd=temp, env=os.environ.copy())
    try:
        proc.wait()
    finally:
        api.shutdown(); api.server_close(); proc.terminate()


def main(argv: list[str] | None = None) -> int:
    raise SystemExit("Use trace_needle.py --webgpu-compare")
