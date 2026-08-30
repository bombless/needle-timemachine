"""Browser-side WebGPU comparator for the Needle Time Machine trace.

The page imports the exact TypeScript engine from needle-webui through Vite
instead of reimplementing its math in Python. Runtime.mm is wrapped so the
browser execution can be compared with the JAX op trace at GEMM boundaries.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Needle WebGPU Compare</title>
<style>
body{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#f5f7fb;color:#172033}.app{max-width:1280px;margin:auto;padding:20px}.card{background:#fff;border:1px solid #dde2eb;border-radius:12px;padding:16px;margin:14px 0}button,input{font:inherit}button{padding:8px 12px;border:1px solid #cbd2df;border-radius:8px;background:#fff;cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.muted{color:#687385}pre{white-space:pre-wrap;overflow:auto;max-height:360px;background:#f7f8fa;padding:10px;border-radius:8px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:7px;border-bottom:1px solid #e5e9f0}td.num,th.num{text-align:right;font-family:ui-monospace,monospace}.ok{font-weight:700}.bad{font-weight:700}.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.small{font-size:12px}@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style></head><body><main class="app">
<div class="bar"><h1 style="margin:0">Needle WebGPU Compare</h1><span id="status" class="muted">等待模型…</span></div>
<div class="card"><div class="bar"><label>.cact 模型 <input id="model" type="file" accept=".cact,.bin"></label><button id="run">运行 WebGPU</button><span id="gpu" class="muted"></span></div><p class="small muted">token IDs 直接来自 reference trace，因此输入完全固定；计算使用 needle-webui/src/engine.ts 的原始实现。</p></div>
<div class="grid"><section class="card"><h2>Reference / JAX</h2><div id="refInfo" class="muted"></div><pre id="refTop">{}</pre></section><section class="card"><h2>WebGPU</h2><div id="gpuInfo" class="muted"></div><pre id="gpuTop">{}</pre></section></div>
<section class="card"><h2>逐边界比较</h2><p class="muted small">看第一处明显增大的 Δ：q/k/v 和 projected 最有价值；若这些仍接近而 logits 偏离，优先检查 attention、mHC、Hadamard 或 dtype。</p><div id="table">尚未运行。</div></section>
<section class="card"><h2>WebGPU GEMM 调用</h2><pre id="calls">[]</pre></section>
<script type="module">
const engine = await import('/@fs/WEBUI_ENGINE@')
const $=id=>document.getElementById(id)
let reference=null,wrapped=false,calls=[]
function stats(a){let min=Infinity,max=-Infinity,sum=0,n=0;for(const x of a){if(!Number.isFinite(x))continue;min=Math.min(min,x);max=Math.max(max,x);sum+=x;n++}return {n,min,max,mean:n?sum/n:NaN}}
function refEvent(layer,name){return (reference?.events||[]).find(e=>e.layer===layer&&e.op===name&&e.metadata?.runtime)}
function refStats(e){const t=e?.tensors?.value;return t&&Number.isFinite(t.mean)?{n:t.n??0,min:t.min,max:t.max,mean:t.mean}:null}
function delta(a,b){if(!a||!b)return null;return {mean:Math.abs(a.mean-b.mean),min:Math.abs(a.min-b.min),max:Math.abs(a.max-b.max)}}
function fmt(x){return Number.isFinite(x)?x.toExponential(4):String(x)}
function renderRows(){const rows=[];for(const c of calls){if(c.layer===null||!c.name)continue;const r=refEvent(c.layer,c.name),rs=refStats(r),d=delta(c.stats,rs);rows.push({layer:c.layer,boundary:c.name,weight:c.weightIndex,webgpu:c.stats,reference:rs,delta:d})}if(!rows.length){$('table').textContent='没有可映射的 q/k/v/projected GEMM。';return}$('table').innerHTML='<table><thead><tr><th>Layer</th><th>Boundary</th><th>Weight</th><th class="num">GPU mean</th><th class="num">REF mean</th><th class="num">Δmean</th><th class="num">Δmin</th><th class="num">Δmax</th></tr></thead><tbody>'+rows.map(r=>{const d=r.delta,bad=!d||d.mean>1e-2;return `<tr><td>${r.layer}</td><td>${r.boundary}</td><td>${r.weight}</td><td class="num">${fmt(r.webgpu.mean)}</td><td class="num">${fmt(r.reference?.mean)}</td><td class="num ${bad?'bad':'ok'}">${fmt(d?.mean)}</td><td class="num">${fmt(d?.min)}</td><td class="num">${fmt(d?.max)}</td></tr>`}).join('')+'</tbody></table>'}
function installWrapper(){if(wrapped)return;wrapped=true;const proto=engine.Runtime.prototype,orig=proto.mm;proto.mm=async function(a,m,k,wi,n){const out=await orig.call(this,a,m,k,wi,n),s=stats(out);let layer=null,name=null;for(let l=0;l<this.m.g.num_layers;l++){const base=1+14*l;if(wi===base+1){layer=l;name='attention.q'}else if(wi===base+2){layer=l;name='attention.k'}else if(wi===base+3){layer=l;name='attention.v'}else if(wi===base+7){layer=l;name='attention.projected'}}calls.push({weightIndex:wi,m,k,n,stats:s,layer,name});$('calls').textContent=JSON.stringify(calls.slice(-80),null,2);return out}}
async function run(){const f=$('model').files?.[0];if(!f){$('status').textContent='请先选择 .cact 模型';return}if(!navigator.gpu){$('status').textContent='当前浏览器没有 WebGPU';return}const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('无法获取 WebGPU adapter');const dev=await adapter.requestDevice();installWrapper();calls=[];$('status').textContent='运行中…';const m=new engine.Model(await f.arrayBuffer()),r=new engine.Runtime(dev,m),counter={dispatches:0,flops:0,forwardMs:0},ids=(reference?.prompt_tokens||[]).map(x=>Number(x.token_id));if(!ids.length)throw Error('reference trace 没有 prompt_tokens');const logits=await engine.generate(ids,m,r,layer=>{$('status').textContent=`WebGPU layer ${layer}/${m.g.num_layers}`},counter);const top=engine.topCandidates(logits,m.tok,10);$('gpuInfo').textContent=`${ids.length} tokens · ${counter.dispatches} dispatch · ${counter.flops} FLOPs`;$('gpuTop').textContent=JSON.stringify(top,null,2);$('status').textContent='完成';renderRows()}
$('run').onclick=()=>run().catch(e=>{$('status').textContent='失败：'+(e?.message||e);console.error(e)})
reference=await (await fetch('/reference.json',{cache:'no-store'})).json();const refTop=(reference.events||[]).find(e=>e.op==='probability_output');$('refInfo').textContent=`${reference.prompt_tokens?.length||0} prompt tokens · ${reference.events?.length||0} events · ${reference.checkpoint||''}`;$('refTop').textContent=JSON.stringify(refTop?.metadata?.top_k||[],null,2);$('gpu').textContent='WebGPU comparator ready'
</script></main></body></html>'''


def _write_vite_project(root: Path, engine_path: Path, reference_trace: Path) -> None:
    (root / "index.html").write_text(_HTML.replace("WEBUI_ENGINE@", engine_path.resolve().as_posix()), encoding="utf-8")
    (root / "package.json").write_text('{"private":true,"type":"module"}\n', encoding="utf-8")
    ref = str(reference_trace.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    (root / "vite.config.js").write_text(
        'import { defineConfig } from "vite"\n'
        'import fs from "node:fs"\n'
        f'export default defineConfig({{plugins:[{{name:"tm-reference",configureServer(s){{s.middlewares.use("/reference.json",(_,res)=>{{const b=fs.readFileSync("{ref}");res.setHeader("Content-Type","application/json");res.end(b)}})}}}}]}})\n',
        encoding="utf-8",
    )


def serve(reference_trace: Path, webui_source: Path, host: str, port: int) -> None:
    """Launch Vite and serve a page that imports the exact webui engine.ts."""
    engine_path = webui_source / "src" / "engine.ts"
    if not engine_path.exists():
        raise FileNotFoundError(f"needle-webui engine.ts not found: {engine_path}")
    if not (webui_source / "node_modules").exists():
        raise RuntimeError(f"{webui_source} has no node_modules; run npm install in needle-webui first")
    if not reference_trace.exists():
        raise FileNotFoundError(f"reference trace not found: {reference_trace}")

    temp = Path(tempfile.mkdtemp(prefix="needle-webgpu-compare-"))
    _write_vite_project(temp, engine_path, reference_trace)
    vite = webui_source / "node_modules" / ".bin" / ("vite.cmd" if os.name == "nt" else "vite")
    if not vite.exists():
        raise FileNotFoundError(f"Vite executable not found: {vite}; run npm install in needle-webui")
    print(f"Needle WebGPU comparator: http://{host}:{port}/")
    print(f"reference: {reference_trace}")
    print(f"webui engine: {engine_path}")
    proc = subprocess.Popen(
        [str(vite), "--host", host, "--port", str(port), "--strictPort"],
        cwd=temp,
        env=os.environ.copy(),
    )
    try:
        proc.wait()
    finally:
        proc.terminate()
