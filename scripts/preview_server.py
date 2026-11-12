"""Serve the generated Hugo preview and an in-memory Waline API double.

Preview data is deliberately synthetic and is never sent to a production service.
"""
import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    counters = {}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *pos, **kwargs):
            super().__init__(*pos, directory=args.directory, **kwargs)

        def respond(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path != "/api/article":
                return super().do_GET()
            query = parse_qs(url.query)
            paths = query.get("path", [""])[0].split(",")
            rows = [dict(counters.setdefault(path, {"time": 1234, "reaction0": 56})) for path in paths]
            self.respond({"errno": 0, "errmsg": "", "data": rows})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/article":
                return self.respond({"errno": 404, "data": []}, 404)
            try:
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                field = data["type"]
                action = data.get("action", "inc")
                if field not in ("time", "reaction0") or action not in ("inc", "desc"):
                    raise ValueError("Invalid operation")
                row = counters.setdefault(data["path"], {"time": 1234, "reaction0": 56})
                row[field] = max(0, row[field] + (1 if action == "inc" else -1))
                self.respond({"errno": 0, "errmsg": "", "data": [{field: row[field]}]})
            except (KeyError, ValueError, TypeError):
                self.respond({"errno": 400, "errmsg": "Invalid request", "data": []}, 400)

    print(f"Preview with synthetic counts: http://127.0.0.1:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    run()
