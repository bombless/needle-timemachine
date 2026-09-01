
from __future__ import annotations
import os
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


_HTML = open(os.path.join(os.path.dirname(__file__), "ui.html"), encoding="utf-8").read()

_JS = open(os.path.join(os.path.dirname(__file__), "ui.script.js"), encoding="utf-8").read()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serve a Needle Time Machine trace in a browser.")
    p.add_argument("trace", type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p


def make_server(trace_path: Path, host: str = "127.0.0.1", port: int = 8765, weights_provider=None) -> ThreadingHTTPServer:
    payload = trace_path.read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                body, content_type = _HTML.encode("utf-8"), "text/html; charset=utf-8"
            elif self.path == "/trace.json":
                body, content_type = payload.encode("utf-8"), "application/json; charset=utf-8"
            elif self.path == "/weights.json" and weights_provider is not None:
                body = json.dumps(weights_provider(), separators=(",", ":")).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif self.path == "/ui.script.js":
                body, content_type = _JS.encode("utf-8"), "application/javascript; charset=utf-8"
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve(trace_path: Path, host: str = "127.0.0.1", port: int = 8765, weights_provider=None) -> None:
    server = make_server(trace_path, host, port, weights_provider)
    print(f"Needle Time Machine UI: http://{host}:{port}/")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.trace.exists(): raise SystemExit(f"trace not found: {args.trace}")
    json.loads(args.trace.read_text(encoding="utf-8")); serve(args.trace, args.host, args.port); return 0


if __name__ == "__main__": raise SystemExit(main())
