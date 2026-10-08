"""Loopback-only HTTP demo. No payload logging, cookies, or persistent sessions."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

from .service import RequestError, handoff, match

HTML = Path(__file__).parent / "web" / "index.html"
MAX_BODY = 4096


def create_server(model, port=8765):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_args):
            pass

        def handle_one_request(self):
            try:
                super().handle_one_request()
            except ConnectionError:
                # Consent withdrawal can close the socket during a response.
                self.close_connection = True

        def reply(self, status, content, kind="application/json; charset=utf-8"):
            raw = content if isinstance(content, bytes) else json.dumps(content).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'unsafe-inline'; "
                             "style-src 'unsafe-inline'; connect-src 'self'; "
                             "base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(raw)

        def guard(self):
            port = self.server.server_port
            expected_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in expected_hosts:
                raise RequestError("Use the loopback URL shown in the quickstart.", 403)
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {f"http://{h}" for h in expected_hosts}:
                raise RequestError("Cross-origin requests are not allowed.", 403)
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise RequestError("Cross-site requests are not allowed.", 403)
            if urlsplit(self.path).query and self.path not in {
                "/?clawpilotTheme=light", "/?clawpilotTheme=dark"
            }:
                raise RequestError("Do not put input in URLs.")

        def do_GET(self):
            try:
                self.guard()
                if urlsplit(self.path).path == "/":
                    self.reply(200, HTML.read_bytes(), "text/html; charset=utf-8")
                elif self.path == "/health":
                    self.reply(200, {"status": "ok", "resources": len(model.catalog)})
                else:
                    self.reply(404, {"error": "Route not found."})
            except RequestError as exc:
                self.reply(exc.status, {"error": str(exc)})

        def do_POST(self):
            try:
                self.guard()
                if self.path not in {"/api/match", "/api/handoff"}:
                    raise RequestError("Route not found.", 404)
                if self.headers.get_content_type() != "application/json":
                    raise RequestError("Use Content-Type: application/json.", 415)
                if self.headers.get("Transfer-Encoding") is not None:
                    raise RequestError("Chunked requests are not supported.", 400)
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    raise RequestError("Invalid Content-Length.") from None
                if size <= 0 or size > MAX_BODY:
                    raise RequestError("Request body must be between 1 and 4096 bytes.", 413)
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise RequestError("Incomplete request body.")
                try:
                    payload = json.loads(raw)
                except (ValueError, UnicodeDecodeError, RecursionError):
                    raise RequestError("Send valid UTF-8 JSON.") from None
                result = match(model, payload) if self.path == "/api/match" else handoff(model, payload)
                self.reply(200, result)
            except RequestError as exc:
                self.reply(exc.status, {"error": str(exc)})

        def setup(self):
            super().setup()
            self.connection.settimeout(5)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
