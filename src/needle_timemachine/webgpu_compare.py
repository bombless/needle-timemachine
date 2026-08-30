"""Browser-side WebGPU comparison harness for Needle WebUI.

The important property of this harness is that the actual inference is still
performed by bombless/needle-webui in the browser.  We only proxy its Vite
page, expose Runtime from webgpu-main.ts, and instrument Runtime.mm after the
module has loaded.  This makes the trace observe the same WebGPU path as the
real page instead of introducing a second Python implementation.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


_OVERLAY = r'''<script>
(() => {
  const state = { mm: [], started: 0, last: 0, saved: false };
  const $ = (id) => document.getElementById(id);
  function summary(a) {
    let min=Infinity,max=-Infinity,sum=0,sumsq=0,n=a?.length||0,bad=-1;
    for(let i=0;i<n;i++) { const v=Number(a[i]); if(!Number.isFinite(v)){bad=i;continue;} min=Math.min(min,v);max=Math.max(max,v);sum+=v;sumsq+=v*v; }
    return {length:n,min:n?min:null,max:n?max:null,mean:n?sum/n:null,rms:n?Math.sqrt(sumsq/n):null,first32:n?Array.from(a.slice(0,32)):[],firstNonFinite:bad};
  }
  function ui() {
    if ($('tmCompare')) return;
    const d=document.createElement('div'); d.id='tmCompare';
    d.style='position:fixed;right:14px;bottom:14px;z-index:99999;background:#111827;color:#fff;padding:12px;border-radius:10px;font:12px ui-monospace,monospace;box-shadow:0 4px 18px #0008;max-width:420px';
    d.innerHTML='<b>Needle Time Machine · WebGPU</b><div id="tmStatus" style="margin:6px 0">waiting for WebUI runtime…</div><button id="tmSave">Save WebGPU trace</button><button id="tmClear" style="margin-left:6px">Clear</button>';
    document.body.appendChild(d);
    $('tmSave').onclick=save; $('tmClear').onclick=()=>{state.mm=[];state.saved=false;status('cleared')};
  }
  function status(s){const e=$('tmStatus'); if(e)e.textContent=s}
  async function install(){
    ui();
    if(!globalThis.__needleWebUI){setTimeout(install,100);return}
    const R=globalThis.__needleWebUI.Runtime;
    if(!R || R.prototype.__tmWrapped){setTimeout(install,100);return}
    const original=R.prototype.mm;
    R.prototype.mm=async function(a,m,k,wi,n){
      const t0=performance.now();
      const out=await original.call(this,a,m,k,wi,n);
      state.mm.push({index:state.mm.length,wi,m,k,n,input:summary(a),output:summary(out),ms:performance.now()-t0});
      state.last=performance.now();
      status(`GEMM ${state.mm.length} · tensor ${wi} · ${m}x${k} · ${k}x${n}`);
      return out;
    };
    R.prototype.__tmWrapped=true;
    status('instrumented Runtime.mm · run the WebUI normally');
    observe();
  }
  function observe(){
    let quiet=null;
    const mo=new MutationObserver(()=>{
      clearTimeout(quiet);
      quiet=setTimeout(()=>{ if(state.mm.length && !state.saved) status(`captured ${state.mm.length} WebGPU GEMMs · click Save`) },700);
    });
    mo.observe(document.body,{subtree:true,childList:true,characterData:true});
  }
  async function save(){
    const promptIds=(()=>{try{return JSON.parse($('debugPrompt')?.textContent||'[]')}catch{return[]}})();
    const generated=(()=>{try{return JSON.parse($('debugGenerated')?.textContent||'[]')}catch{return[]}})();
    const steps=(()=>{try{return JSON.parse($('debugSteps')?.textContent||'[]')}catch{return[]}})();
    const latents=Array.isArray(globalThis.__needleLatents)?globalThis.__needleLatents.map(x=>({layer:x.layer,shape:x.shape,min:x.min,max:x.max,mean:x.mean,rms:x.rms,first32:Array.from(x.values?.slice(0,32)||[])})):[];
    const trace={format:'needle-timemachine.webgpu/v1',url:location.href,prompt_ids:promptIds,generated_ids:generated,steps,latents,mm:state.mm,created_at:new Date().toISOString()};
    const r=await fetch('/__timemachine__/webgpu-trace',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(trace)});
    const j=await r.json(); state.saved=true; status(j.message||'saved');
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',install); else install();
})();
</script>'''


def _proxy_request(base: str, path: str, method: str = "GET", body: bytes | None = None) -> tuple[int, dict[str, str], bytes]:
    url = base.rstrip("/") + "/" + path.lstrip("/")
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Accept", "*/*")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
            headers = {k: v for k, v in r.headers.items() if k.lower() in {"content-type", "cache-control", "etag"}}
            return r.status, headers, data
    except urllib.error.HTTPError as e:
        return e.code, {"Content-Type": "text/plain; charset=utf-8"}, e.read()


def serve_webui_compare(
    *,
    webui_url: str,
    host: str,
    port: int,
    output: Path,
    reference: Path | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    reference_payload = reference.read_bytes() if reference and reference.exists() else None

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path == "/__timemachine__/reference.json":
                if reference_payload is None:
                    self._send(404, "text/plain; charset=utf-8", b"no reference trace")
                else:
                    self._send(200, "application/json; charset=utf-8", reference_payload)
                return
            status, headers, body = _proxy_request(webui_url, parsed.path + ("?" + parsed.query if parsed.query else ""))
            if status >= 400:
                self._send(status, headers.get("Content-Type", "text/plain; charset=utf-8"), body)
                return
            ctype = headers.get("Content-Type", "application/octet-stream")
            if "text/html" in ctype:
                text = body.decode("utf-8", errors="replace")
                # DEBUG_LATENTS is evaluated while webgpu-main.ts is initialized,
                # so the flag must exist before the Vite module executes.
                marker = '<script>history.replaceState(null,"",location.pathname+"?DEBUG_LATENTS=1")</script>'
                text = text.replace("</head>", marker + "</head>")
                text = text.replace("</body>", _OVERLAY + "</body>")
                body = text.encode("utf-8")
            self._send(status, ctype, body)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path != "/__timemachine__/webgpu-trace":
                self._send(404, "text/plain; charset=utf-8", b"not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                self._send(200, "application/json; charset=utf-8", json.dumps({"ok": True, "message": f"saved {output}"}).encode())
            except Exception as exc:  # pragma: no cover - browser-only path
                self._send(400, "application/json; charset=utf-8", json.dumps({"ok": False, "error": str(exc)}).encode())

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Needle WebGPU compare page: http://{host}:{port}/")
    print(f"Proxy target:                 {webui_url}")
    print(f"WebGPU trace output:          {output}")
    if reference:
        print(f"Python reference trace:       {reference}")
    print("Open the compare URL, select the same .cact model in needle-webui, and run the same prompt.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
