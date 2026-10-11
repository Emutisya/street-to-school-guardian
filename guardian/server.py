"""Loopback-only HTTP demo. No payload logging, cookies, or persistent sessions."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

from .service import RequestError, browse, compare, handoff, match, plan, related

HTML = Path(__file__).parent / "web" / "index.html"
MAX_BODY = 4096


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON fields are not accepted.")
        result[key] = value
    return result


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
            target = self.requestline.split()[1]
            if not target.startswith("/") or target.startswith("//") or "#" in target:
                raise RequestError("Use an exact local route. Do not put input in URLs.")
            expected_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if len(self.headers.get_all("Host", [])) != 1:
                raise RequestError("Send exactly one loopback Host header.", 403)
            if self.headers.get("Host") not in expected_hosts:
                raise RequestError("Use the loopback URL shown in the quickstart.", 403)
            origin = self.headers.get("Origin")
            if len(self.headers.get_all("Origin", [])) > 1 or len(self.headers.get_all("Sec-Fetch-Site", [])) > 1:
                raise RequestError("Ambiguous browser origin headers are not allowed.", 403)
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
                elif self.path == "/ready":
                    try:
                        summary = model.summary()
                    except ValueError:
                        self.reply(503, {"status": "unavailable", "recovery": "Fit or reload the synthetic catalog model before using the finder."})
                    else:
                        self.reply(200, {"status": "ready", "catalog_sha256": summary["catalog_sha256"],
                                         "resources": summary["documents"], "vocabulary": summary["vocabulary_size"],
                                         "note": "Local retrieval readiness only; no real services or referrals."})
                else:
                    self.reply(404, {"error": "Route not found.", "code": "route_not_found"})
            except RequestError as exc:
                self.reply(exc.status, {"error": str(exc), "code": exc.code})

        def do_POST(self):
            try:
                self.guard()
                operations = {"/api/match": match, "/api/handoff": handoff,
                              "/api/browse": browse, "/api/compare": compare, "/api/plan": plan,
                              "/api/related": related}
                if self.path not in operations:
                    raise RequestError("Route not found.", 404, "route_not_found")
                if len(self.headers.get_all("Content-Type", [])) != 1 or self.headers.get_content_type() != "application/json":
                    raise RequestError("Use Content-Type: application/json.", 415)
                if self.headers.get("Transfer-Encoding") is not None:
                    raise RequestError("Chunked requests are not supported.", 400)
                try:
                    length = self.headers.get("Content-Length", "")
                    if (len(self.headers.get_all("Content-Length", [])) != 1
                            or not length.isascii() or not length.isdigit()):
                        raise ValueError("One decimal Content-Length is required.")
                    size = int(length)
                except ValueError:
                    raise RequestError("Invalid Content-Length.") from None
                if size <= 0 or size > MAX_BODY:
                    raise RequestError("Request body must be between 1 and 4096 bytes.", 413)
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise RequestError("Incomplete request body.")
                try:
                    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_fields)
                except (ValueError, UnicodeDecodeError, RecursionError):
                    raise RequestError("Send valid UTF-8 JSON.", code="invalid_json") from None
                result = operations[self.path](model, payload)
                self.reply(200, result)
            except TimeoutError:
                self.reply(408, {"error": "Request body timed out. Send a complete JSON request.", "code": "body_timeout"})
            except RequestError as exc:
                self.reply(exc.status, {"error": str(exc), "code": exc.code})

        def setup(self):
            super().setup()
            self.connection.settimeout(5)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
