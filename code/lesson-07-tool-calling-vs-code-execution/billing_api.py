"""A tiny local billing API over invoice_tools, so tools running INSIDE a Docker sandbox can reach
the data over HTTP (the container cannot import this repo's modules). Run: python billing_api.py
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from invoice_tools import list_invoices, send_email


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        q = parse_qs(urlparse(self.path).query)
        self._reply(list_invoices(q["month"][0]))

    def do_POST(self) -> None:
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._reply(send_email(data["to"], data["subject"], data["body"]))


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8765), Handler).serve_forever()
