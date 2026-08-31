"""Tiny zero-dependency browser UI for Needle Time Machine traces and verification."""

from __future__ import annotations

import argparse
import cgi
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional


VerificationRunner = Callable[..., dict[str, Any]]

_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Needle Time Machine</title>
<style>
:root{font-family:ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f5f7fb}body{margin:0}.app{max-width:1200px;margin:auto;padding:24px}
.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}button,input,textarea,select{font:inherit}button{padding:8px 12px;border:1px solid #cbd2df;border-radius:8px;background:white;cursor:pointer}button:hover{background:#eef2f7}
.card{background:white;border:1px solid #dde2eb;border-radius:12px;padding:16px;margin-top:16px;box-shadow:0 1px 2px #0000000b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.field{display:flex;flex-direction:column;gap:6px;min-width:220px;flex:1}.field input,.field textarea,.field select{padding:8px;border:1px solid #cbd2df;border-radius:8px}.field textarea{min-height:90px;resize:vertical}
.timeline{position:relative;height:92px;overflow-x:auto;padding:18px 8px 0}.track{position:relative;height:3px;background:#ccd3df;top:30px;min-width:100%}.dot{position:absolute;top:23px;width:14px;height:14px;border-radius:50%;border:2px solid white;background:#697586;box-shadow:0 0 0 1px #aeb7c5;transform:translateX(-50%);cursor:pointer}.dot.layer{background:#3d63dd}.dot.current{background:#e05a33;transform:translate(-50%,0) scale(1.25)}
#slider{width:100%}.kv{display:grid;grid-template-columns:130px 1fr;gap:7px;font-family:ui-monospace,monospace;font-size:13px}.muted{color:#697586}pre{white-space:pre-wrap;overflow:auto;max-height:280px;background:#f7f8fa;padding:12px;border-radius:8px}
.token-table,.probability-table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}.token-table th,.token-table td,.probability-table th,.probability-table td{text-align:left;padding:8px 10px;border-bottom:1px solid #e5e9f0}.token-text,.token-hex{font-family:ui-monospace,monospace;white-space:pre-wrap}.token-hex{word-break:break-word}.probability-table th:nth-child(5),.probability-table td:nth-child(5){text-align:right}.probability-bar{height:8px;border-radius:4px;background:#e5e9f0;overflow:hidden}.probability-fill{height:100%;background:#3d63dd}.probability-value{font-family:ui-monospace,monospace;min-width:90px;text-align:right}.status{margin-top:10px;padding:10px;border-radius:8px;background:#f7f8fa;white-space:pre-wrap}.status.error{background:#fff0f0}.status.ok{background:#eef8f0}
@media(max-width:760px){.grid{grid-template-columns:1fr}.probability-table,.token-table{font-size:12px}.probability-table th,.probability-table td,.token-table th,.token-table td{padding:7px 5px}}
</style></head><body><main class="app">
<div class="bar"><h1 style="margin:0">Needle Time Machine</h1><span id="summary" class="muted"></span></div>
<section class="card"><h2>Verify a checkpoint</h2><p class="muted">Choose a <code>.pkl</code> Needle 2 checkpoint. The server loads the weights with the same JAX/Needle runtime as the CLI, tokenizes the prompt, then executes one complete hidden-state + logits forward pass.</p>
<form id="verifyForm"><div class="bar"><label class="field"><span>Checkpoint (.pkl)</span><input id="checkpoint" name="checkpoint" type="file" accept=".pkl,application/octet-stream" required></label><label class="field"><span>Trace level</span><select name="trace_level"><option value="layer" selected>layer</option><option value="op">op</option><option value="none">none</option></select></label><label class="field" style="max-width:120px"><span>Top-k</span><input name="top_k" type="number" min="1" value="5"></label></div>
<label class="field" style="margin-top:12px"><span>Prompt</span><textarea name="prompt" id="prompt">hello world</textarea></label><div class="bar" style="margin-top:12px"><button type="submit">Load weights &amp; run forward</button><span id="verifyStatus" class="status">No verification run yet.</span></div></form></section>
<section class="card"><div class="bar"><button id="first">|&lt;</button><button id="prev">&lt;</button><button id="play">▶ Play</button><button id="next">&gt;</button><button id="last">&gt;|</button><label>speed <input id="speed" type="range" min="0.25" max="4" step="0.25" value="1"><span id="speedText">1×</span></label></div><input id="slider" type="range" min="0" max="0" value="0"><div class="timeline"><div class="track" id="track"></div></div></section>
<div class="grid"><section class="card"><h2 id="title">No event</h2><div id="details" class="kv"></div></section><section class="card"><h2>Tensor metadata</h2><pre id="tensors">{}</pre><h2>Event metadata</h2><pre id="metadata">{}</pre></section></div>
<section class="card"><div class="bar"><h2 style="margin:0">Prompt tokens</h2><span id="promptSummary" class="muted"></span></div><div id="promptTokens"><p class="muted">No prompt tokenization recorded.</p></div></section>
<section class="card"><div class="bar"><h2 style="margin:0">Final token probabilities</h2><span id="probabilitySummary" class="muted"></span></div><div id="probabilities"><p class="muted">No probability output recorded.</p></div></section>
</main><script>
let trace={events:[]},pos=0,timer=null;const $=id=>document.getElementById(id);
async function load(){const r=await fetch('/trace.json',{cache:'no-store'});trace=await r.json();$('summary').textContent=`${trace.events.length} events · ${trace.checkpoint||trace.filename||'unknown checkpoint'}`;$('slider').max=Math.max(0,trace.events.length-1);renderPromptTokens();drawDots();render(0)}
function drawDots(){const t=$('track');t.innerHTML='';trace.events.forEach((e,i)=>{const d=document.createElement('button');d.className='dot'+(e.op==='layer_output'?' layer':'');d.title=`${e.step}: ${e.op}`;d.style.left=(trace.events.length<2?50:i*100/(trace.events.length-1))+'%';d.onclick=()=>render(i);t.appendChild(d)})}
function esc(v){return String(v??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#039;')}
function renderPromptTokens(){const tokens=Array.isArray(trace.prompt_tokens)?trace.prompt_tokens:[];if(!tokens.length){$('promptTokens').innerHTML='<p class="muted">No prompt tokenization recorded.</p>';return}$('promptSummary').textContent=`${tokens.length} tokens · includes BOS`;$('promptTokens').innerHTML='<table class="token-table"><thead><tr><th>Token ID</th><th>Token text</th><th>Token bytes hex</th></tr></thead><tbody>'+tokens.map(x=>`<tr><td><code>${esc(x.token_id)}</code></td><td class="token-text">${esc(x.token_text)||'∅'}</td><td class="token-hex"><code>${esc(x.token_bytes_hex)||'∅'}</code></td></tr>`).join('')+'</tbody></table>'}
function renderProbabilities(e){const topK=e?.metadata?.top_k;if(!Array.isArray(topK)||!topK.length){$('probabilities').innerHTML='<p class="muted">No probability output recorded for this event.</p>';return}$('probabilitySummary').textContent=`top ${topK.length} · final sequence position`;$('probabilities').innerHTML='<table class="probability-table"><thead><tr><th>Rank</th><th>Token ID</th><th>Token text</th><th>Token bytes hex</th><th>Probability</th><th style="width:28%">Distribution</th></tr></thead><tbody>'+topK.map((x,i)=>{const p=Number(x.probability)||0;return `<tr><td>${i+1}</td><td><code>${esc(x.token_id)}</code></td><td class="token-text">${esc(x.token_text)||'∅'}</td><td class="token-hex"><code>${esc(x.token_bytes_hex)||'∅'}</code></td><td class="probability-value">${(p*100).toFixed(4)}%</td><td><div class="probability-bar"><div class="probability-fill" style="width:${Math.max(0,Math.min(100,p*100))}%"></div></div></td></tr>`}).join('')+'</tbody></table>'}
function render(i){if(!trace.events.length)return;pos=Math.max(0,Math.min(i,trace.events.length-1));const e=trace.events[pos];$('slider').value=pos;$('title').textContent=`Step ${e.step} · ${e.op}`;$('details').innerHTML=[['layer',e.layer??'—'],['name',e.name??'—'],['phase',e.phase],['snapshot',e.snapshot_id??'—']].map(([k,v])=>`<div class="muted">${k}</div><div>${esc(v)}</div>`).join('');$('tensors').textContent=JSON.stringify(e.tensors||{},null,2);$('metadata').textContent=JSON.stringify(e.metadata||{},null,2);renderProbabilities(e);document.querySelectorAll('.dot').forEach((d,j)=>d.classList.toggle('current',j===pos))}
function step(n){render(pos+n)}$('first').onclick=()=>render(0);$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('last').onclick=()=>render(trace.events.length-1);$('slider').oninput=e=>render(+e.target.value);$('speed').oninput=e=>{$('speedText').textContent=e.target.value+'×';if(timer){stop();play()}};
function play(){if(timer)return;$('play').textContent='⏸ Pause';const tick=()=>{if(pos>=trace.events.length-1){stop();return}step(1);timer=setTimeout(tick,500/+$('speed').value)};tick()}function stop(){clearTimeout(timer);timer=null;$('play').textContent='▶ Play'}$('play').onclick=()=>timer?stop():play();
$('verifyForm').onsubmit=async ev=>{ev.preventDefault();const file=$('checkpoint').files[0];if(!file)return;const status=$('verifyStatus');status.className='status';status.textContent=`Uploading ${file.name} (${file.size.toLocaleString()} bytes) and running JAX forward…`;const form=new FormData(ev.target);try{const r=await fetch('/api/verify',{method:'POST',body:form});const data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);status.className='status ok';status.textContent=`✓ Forward pass complete · ${data.result.layer_count} layers · ${data.result.event_count} events · final top-${data.result.top_k.length} recorded`;await load()}catch(err){status.className='status error';status.textContent=`Verification failed: ${err.message}`}};load();
</script></body></html>'''


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serve a Needle Time Machine trace in a browser.")
    p.add_argument("trace", type=Path, help="Trace JSON produced by trace_needle")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p


def serve(
    trace_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    verification_runner: Optional[VerificationRunner] = None,
    needle_source: str | Path | None = None,
) -> None:
    payload = trace_path.read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, value: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            nonlocal payload
            if self.path == "/" or self.path == "/index.html":
                body = _HTML.encode("utf-8")
                content_type = "text/html; charset=utf-8"
            elif self.path == "/trace.json":
                body = payload.encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200);self.send_header("Content-Type",content_type);self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            nonlocal payload
            if self.path != "/api/verify" or verification_runner is None:
                self.send_error(404);return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    raise ValueError("empty request")
                # Pickle checkpoints are executable Python objects; this endpoint is
                # intended for the local UI and should not be exposed to untrusted users.
                if content_length > 2 * 1024 * 1024 * 1024:
                    raise ValueError("checkpoint upload exceeds the 2 GiB limit")
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={"REQUEST_METHOD":"POST", "CONTENT_TYPE":self.headers.get("Content-Type", "")},
                )
                if "checkpoint" not in form:
                    raise ValueError("checkpoint file is required")
                item = form["checkpoint"]
                checkpoint_bytes = item.file.read()
                prompt = form.getfirst("prompt", "hello world")
                trace_level = form.getfirst("trace_level", "layer")
                top_k = int(form.getfirst("top_k", "5"))
                result = verification_runner(
                    checkpoint_bytes,
                    filename=item.filename or "checkpoint.pkl",
                    prompt=prompt,
                    needle_source=needle_source,
                    trace_level=trace_level,
                    top_k=top_k,
                )
                payload = json.dumps(result, indent=2, ensure_ascii=False)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

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
