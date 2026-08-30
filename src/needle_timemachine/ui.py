"""Tiny zero-dependency browser UI for Needle Time Machine traces."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Needle Time Machine</title>
<style>
:root{font-family:ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f5f7fb}
body{margin:0}.app{max-width:1200px;margin:auto;padding:24px}.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button,input{font:inherit}button{padding:8px 12px;border:1px solid #cbd2df;border-radius:8px;background:white;cursor:pointer}button:hover{background:#eef2f7}
#play{min-width:90px}.speed{display:flex;align-items:center;gap:8px}.card{background:white;border:1px solid #dde2eb;border-radius:12px;padding:16px;margin-top:16px;box-shadow:0 1px 2px #0000000b}
.timeline{position:relative;height:92px;overflow-x:auto;overflow-y:hidden;padding:18px 8px 0}.track{position:relative;height:3px;background:#ccd3df;top:30px;min-width:100%}
.dot{position:absolute;top:23px;width:14px;height:14px;border-radius:50%;border:2px solid white;background:#697586;box-shadow:0 0 0 1px #aeb7c5;transform:translateX(-50%);cursor:pointer}.dot.layer{background:#3d63dd}.dot.current{background:#e05a33;transform:translate(-50%,0) scale(1.25)}
#slider{width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.kv{display:grid;grid-template-columns:130px 1fr;gap:7px;font-family:ui-monospace,monospace;font-size:13px}.muted{color:#697586}pre{white-space:pre-wrap;overflow:auto;max-height:280px;background:#f7f8fa;padding:12px;border-radius:8px}
.probability-table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}.probability-table th,.probability-table td{text-align:left;padding:9px 10px;border-bottom:1px solid #e5e9f0}.probability-table th:nth-child(5),.probability-table td:nth-child(5){text-align:right}.probability-bar{height:8px;border-radius:4px;background:#e5e9f0;overflow:hidden}.probability-fill{height:100%;background:#3d63dd}.probability-value{font-family:ui-monospace,monospace;min-width:90px;text-align:right}.token-text{font-family:ui-monospace,monospace;white-space:pre-wrap}.token-hex{font-family:ui-monospace,monospace;word-break:break-word}.probability-empty{color:#697586}
@media(max-width:760px){.grid{grid-template-columns:1fr}.probability-table{font-size:12px}.probability-table th,.probability-table td{padding:7px 5px}}
</style></head>
<body><main class="app">
<div class="bar"><h1 style="margin:0">Needle Time Machine</h1><span id="summary" class="muted"></span></div>
<div class="card"><div class="bar"><button id="first">|&lt;</button><button id="prev">&lt;</button><button id="play">▶ Play</button><button id="next">&gt;</button><button id="last">&gt;|</button><label class="speed">speed <input id="speed" type="range" min="0.25" max="4" step="0.25" value="1"><span id="speedText">1×</span></label></div>
<input id="slider" type="range" min="0" max="0" value="0"><div class="timeline"><div class="track" id="track"></div></div></div>
<div class="grid"><section class="card"><h2 id="title">No event</h2><div id="details" class="kv"></div></section><section class="card"><h2>Tensor metadata</h2><pre id="tensors">{}</pre><h2>Event metadata</h2><pre id="metadata">{}</pre></section></div>
<section class="card"><div class="bar"><h2 style="margin:0">Prompt tokens</h2><span id="promptTokenSummary" class="muted"></span></div><div id="promptTokens"><p class="probability-empty">No prompt tokens recorded.</p></div></section>
<section class="card"><div class="bar"><h2 style="margin:0">Final token probabilities</h2><span id="probabilitySummary" class="muted"></span></div><div id="probabilities"><p class="probability-empty">No probability output recorded.</p></div></section>
</main><script>
let trace={events:[]}, pos=0, timer=null;
const $=id=>document.getElementById(id);
async function load(){trace=await (await fetch('/trace.json')).json();$('summary').textContent=`${trace.events.length} events · ${trace.checkpoint||'unknown checkpoint'}`;$('slider').max=Math.max(0,trace.events.length-1);drawDots();renderPromptTokens();render(0)}
function drawDots(){const t=$('track');trace.events.forEach((e,i)=>{const d=document.createElement('button');d.className='dot'+(e.op==='layer_output'?' layer':'');d.title=`${e.step}: ${e.op}`;d.style.left=(trace.events.length<2?50:i*100/(trace.events.length-1))+'%';d.onclick=()=>render(i);t.appendChild(d)})}
function escapeHtml(value){return String(value??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#039;')}
function renderProbabilities(event){
  const topK=event&&event.metadata&&Array.isArray(event.metadata.top_k)?event.metadata.top_k:null;
  if(!topK||!topK.length){$('probabilities').innerHTML='<p class="probability-empty">No probability output recorded for this event.</p>';$('probabilitySummary').textContent='';return}
  $('probabilitySummary').textContent=`top ${topK.length} · final sequence position`;
  $('probabilities').innerHTML='<table class="probability-table"><thead><tr><th>Rank</th><th>Token ID</th><th>Token 文本</th><th>Token bytes hex</th><th>概率</th><th style="width:28%">Distribution</th></tr></thead><tbody>'+topK.map((item,index)=>{
    const p=Number(item.probability)||0;
    const pct=(p*100).toFixed(4);
    const tokenId=Number(item.token_id);
    const tokenText=escapeHtml(item.token_text||'');
    const tokenHex=escapeHtml(item.token_bytes_hex||'');
    return `<tr><td>${index+1}</td><td><code>${escapeHtml(item.token_id)}</code></td><td class="token-text">${tokenText||'∅'}</td><td class="token-hex"><code>${tokenHex||'∅'}</code></td><td class="probability-value">${pct}%</td><td><div class="probability-bar"><div class="probability-fill" style="width:${Math.max(0,Math.min(100,p*100))}%"></div></div></td></tr>`;
  }).join('')+'</tbody></table>';
}
function renderPromptTokens(){
  const tokens=trace.prompt_tokens;
  if(!tokens||!tokens.length){$('promptTokens').innerHTML='<p class="probability-empty">No prompt tokens recorded.</p>';$('promptTokenSummary').textContent='';return}
  $('promptTokenSummary').textContent=`${tokens.length} tokens`;
  $('promptTokens').innerHTML='<table class="probability-table"><thead><tr><th>#</th><th>Token ID</th><th>Token 文本</th><th>Token bytes hex</th></tr></thead><tbody>'+tokens.map((item,index)=>{
    const tokenText=escapeHtml(item.token_text||'');
    const tokenHex=escapeHtml(item.token_bytes_hex||'');
    return `<tr><td>${index+1}</td><td><code>${escapeHtml(item.token_id)}</code></td><td class="token-text">${tokenText||'∅'}</td><td class="token-hex"><code>${tokenHex||'∅'}</code></td></tr>`;
  }).join('')+'</tbody></table>';
}
function render(i){if(!trace.events.length)return;pos=Math.max(0,Math.min(i,trace.events.length-1));const e=trace.events[pos];$('slider').value=pos;$('title').textContent=`Step ${e.step} · ${e.op}`;const rows=[['layer',e.layer??'—'],['name',e.name??'—'],['phase',e.phase],['snapshot',e.snapshot_id??'—']];$('details').innerHTML=rows.map(([k,v])=>`<div class="muted">${k}</div><div>${v}</div>`).join('');$('tensors').textContent=JSON.stringify(e.tensors||{},null,2);$('metadata').textContent=JSON.stringify(e.metadata||{},null,2);renderProbabilities(e);document.querySelectorAll('.dot').forEach((d,j)=>d.classList.toggle('current',j===pos))}
function step(n){render(pos+n)}
$('first').onclick=()=>render(0);$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('last').onclick=()=>render(trace.events.length-1);$('slider').oninput=e=>render(+e.target.value);
$('speed').oninput=e=>{$('speedText').textContent=e.target.value+'×';if(timer){stop();play()}};
function play(){if(timer)return;$('play').textContent='⏸ Pause';const tick=()=>{if(pos>=trace.events.length-1){stop();return}step(1);timer=setTimeout(tick,500/+$('speed').value)};tick()}
function stop(){clearTimeout(timer);timer=null;$('play').textContent='▶ Play'}$('play').onclick=()=>timer?stop():play();load();
</script></body></html>'''


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serve a Needle Time Machine trace in a browser.")
    p.add_argument("trace", type=Path, help="Trace JSON produced by trace_needle")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p


def serve(trace_path: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    payload = trace_path.read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path == "/index.html":
                body = _HTML.encode("utf-8")
                content_type = "text/html; charset=utf-8"
            elif self.path == "/trace.json":
                body = payload.encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Needle Time Machine UI: http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.trace.exists():
        raise SystemExit(f"trace not found: {args.trace}")
    json.loads(args.trace.read_text(encoding="utf-8"))
    serve(args.trace, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
