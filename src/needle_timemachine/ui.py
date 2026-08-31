from __future__ import annotations
import argparse
import cgi
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

VerificationRunner = Callable[..., dict[str, Any]]
_HTML = open(Path(__file__).with_name("ui.html"), encoding="utf-8").read()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serve a Needle Time Machine trace in a browser.")
    p.add_argument("trace", type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p


def serve(trace_path: Path, host: str = "127.0.0.1", port: int = 8765, *, verification_runner: Optional[VerificationRunner] = None, needle_source: str | Path | None = None) -> None:
    payload = trace_path.read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _json(self, value: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body, content_type = _HTML.encode("utf-8"), "text/html; charset=utf-8"
            elif self.path == "/trace.json":
                body, content_type = payload.encode("utf-8"), "application/json; charset=utf-8"
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self) -> None:
            nonlocal payload
            if self.path != "/api/verify" or verification_runner is None:
                self.send_error(404); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0: raise ValueError("empty request")
                if length > 2 * 1024 * 1024 * 1024: raise ValueError("checkpoint upload exceeds the 2 GiB limit")
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD":"POST", "CONTENT_TYPE":self.headers.get("Content-Type", "")})
                if "checkpoint" not in form: raise ValueError("checkpoint file is required")
                item = form["checkpoint"]
                result = verification_runner(item.file.read(), filename=item.filename or "checkpoint.pkl", prompt=form.getfirst("prompt", "hello world"), needle_source=needle_source, trace_level=form.getfirst("trace_level", "layer"), top_k=int(form.getfirst("top_k", "5")))
                payload = json.dumps(result, ensure_ascii=False)
                self._json(result)
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

        def log_message(self, fmt: str, *args: Any) -> None: return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Needle Time Machine UI: http://{host}:{port}/")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.trace.exists(): raise SystemExit(f"trace not found: {args.trace}")
    json.loads(args.trace.read_text(encoding="utf-8")); serve(args.trace, args.host, args.port)
    return 0

if __name__ == "__main__": raise SystemExit(main())
