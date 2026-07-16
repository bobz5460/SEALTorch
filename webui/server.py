#!/usr/bin/env python3
import json
import argparse
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "build" / "sealtorch_trace"
MODEL_DIR = ROOT / "src"
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
        elif self.path == "/models":
            models = sorted(path.name for path in MODEL_DIR.glob("*.json")
                            if path.name.startswith("mnist_mlp_"))
            self.send_bytes(200, "application/json; charset=utf-8", json.dumps({
                "models": models,
                "default": "mnist_mlp_gelu_improved.json" if "mnist_mlp_gelu_improved.json" in models else (models[0] if models else ""),
            }).encode())
        else:
            self.send_bytes(404, "text/plain; charset=utf-8", b"Not found\n")

    def do_POST(self):
        if self.path != "/trace":
            self.send_bytes(404, "text/plain; charset=utf-8", b"Not found\n")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if isinstance(request, list):
                values, model_name, activation = request, "mnist_mlp_gelu_improved.json", "gelu"
            else:
                values = request.get("pixels")
                model_name = request.get("model", "mnist_mlp_gelu_improved.json")
                activation = request.get("activation", "gelu")
            if not isinstance(values, list) or len(values) != 784:
                raise ValueError("expected exactly 784 input values")
            if activation not in {"relu", "gelu"}:
                raise ValueError("activation must be relu or gelu")
            model_path = MODEL_DIR / Path(str(model_name)).name
            if model_path.parent != MODEL_DIR or not model_path.name.startswith("mnist_mlp_") or model_path.suffix != ".json" or not model_path.exists():
                raise ValueError("unknown model")
            if not TRACE.exists():
                raise RuntimeError("build/sealtorch_trace does not exist; run cmake --build build first")
            completed = subprocess.run(
                [str(TRACE), str(model_path), "--activation", activation,
                 "--threads", str(self.server.max_concurrency)],
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
    parser.add_argument("--threads", type=int, default=4, help="maximum concurrent neurons per layer")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.max_concurrency = max(1, args.threads)
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"SEALTorch trace UI: http://{display_host}:{args.port}")
    server.serve_forever()
