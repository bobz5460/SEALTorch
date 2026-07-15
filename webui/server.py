#!/usr/bin/env python3
import json
import argparse
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "build" / "sealtorch_trace"
MODEL = ROOT / "src" / "mnist_mlp_gelu.json"
INDEX = Path(__file__).with_name("index.html")


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status, content_type, data):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_bytes(200, "text/html; charset=utf-8", INDEX.read_bytes())
        else:
            self.send_bytes(404, "text/plain; charset=utf-8", b"Not found\n")

    def do_POST(self):
        if self.path != "/trace":
            self.send_bytes(404, "text/plain; charset=utf-8", b"Not found\n")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            values = json.loads(self.rfile.read(length))
            if not isinstance(values, list) or len(values) != 784:
                raise ValueError("expected exactly 784 input values")
            if not TRACE.exists():
                raise RuntimeError("build/sealtorch_trace does not exist; run cmake --build build first")
            completed = subprocess.run(
                [str(TRACE), str(MODEL)],
                input=" ".join(str(float(value)) for value in values) + "\n",
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "trace executable failed")
            result = json.loads(completed.stdout)
            self.send_bytes(200, "application/json; charset=utf-8", json.dumps(result).encode())
        except Exception as error:
            self.send_bytes(400, "application/json; charset=utf-8", json.dumps({"error": str(error)}).encode())

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the SEALTorch trace UI")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (use 0.0.0.0 for a cloud/container port forward)",
    )
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"SEALTorch trace UI: http://{display_host}:{args.port}")
    server.serve_forever()
