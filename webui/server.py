#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build", "sealtorch_gui")
MODEL = os.path.join(ROOT, "src", "mnist_mlp.json")
THREADS = sys.argv[1] if len(sys.argv) > 1 else "4"
BACKEND = sys.argv[2] if len(sys.argv) > 2 else "packed"

worker = subprocess.Popen([BUILD, "--web-worker", MODEL, THREADS, BACKEND], cwd=ROOT,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, bufsize=1)
worker_lock = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def send_json(self, value, status=200):
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/health":
            self.send_json({"ok": worker.poll() is None})
            return
        if self.path != "/" and self.path != "/index.html":
            self.send_error(404)
            return
        with open(os.path.join(ROOT, "webui", "index.html"), "rb") as file:
            data = file.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            pixels = request["pixels"]
            if len(pixels) != 784:
                raise ValueError("pixels must contain 784 values")
            with worker_lock:
                backend = request.get("backend", "packed")
                if backend not in ("packed", "scalar"):
                    raise ValueError("backend must be packed or scalar")
                worker.stdin.write(json.dumps({"pixels": pixels, "backend": backend}) + "\n")
                worker.stdin.flush()
                result = json.loads(worker.stdout.readline())
            self.send_json(result)
        except Exception as error:
            self.send_json({"error": str(error)}, 400)

    def log_message(self, format, *args):
        return

if not os.path.exists(BUILD):
    print("Build SEALTorch first: cmake --build build -j2", file=sys.stderr)
    sys.exit(1)

print("SEALTorch WebUI: http://127.0.0.1:8080", flush=True)
ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
