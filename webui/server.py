#!/usr/bin/env python3
import json
import argparse
import os
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "build" / "sealtorch_trace"
VALIDATE = ROOT / "build" / "sealtorch_validate"
MODEL_DIR = ROOT / "src"
INDEX = Path(__file__).with_name("index.html")
JOBS = {}
JOBS_LOCK = threading.Lock()


def update_job(job_id, **values):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)


def run_validation(job_id, command, timeout):
    try:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, bufsize=1)
        for line in process.stderr:
            fields = line.split()
            if len(fields) == 3 and fields[0] == "progress":
                update_job(job_id, completed=int(fields[1]), total=int(fields[2]))
        stdout, _ = process.communicate(timeout=timeout)
        if process.returncode != 0:
            raise RuntimeError("validation executable failed")
        update_job(job_id, state="complete", result=json.loads(stdout), completed=None)
    except Exception as error:
        update_job(job_id, state="error", error=str(error), completed=None)


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
        elif self.path.startswith("/validate/status"):
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            job_id = query[3:] if query.startswith("id=") else ""
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {"state": "error", "error": "unknown validation job"}))
            self.send_bytes(200, "application/json; charset=utf-8", json.dumps(job).encode())
        else:
            self.send_bytes(404, "text/plain; charset=utf-8", b"Not found\n")

    def do_POST(self):
        if self.path not in {"/trace", "/validate"}:
            self.send_bytes(404, "text/plain; charset=utf-8", b"Not found\n")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if self.path == "/validate":
                count = int(request.get("count", 10))
                if count < 1 or count > 10000:
                    raise ValueError("count must be between 1 and 10000")
                if not VALIDATE.exists():
                    raise RuntimeError("build/sealtorch_validate does not exist; run cmake --build build first")
                models = request.get("models") or [p.name for p in MODEL_DIR.glob("mnist_mlp_*.json")]
                if not models or any(Path(str(name)).name != str(name) or not str(name).startswith("mnist_mlp_") or Path(str(name)).suffix != ".json" for name in models):
                    raise ValueError("invalid model selection")
                model_paths = []
                for name in models:
                    path = MODEL_DIR / str(name)
                    if not path.exists():
                        raise ValueError("unknown model: " + str(name))
                    model_paths += ["--model", str(path)]
                dataset = self.server.mnist_dir
                images = dataset / "t10k-images-idx3-ubyte"
                labels = dataset / "t10k-labels-idx1-ubyte"
                if not images.exists() or not labels.exists():
                    raise RuntimeError(f"MNIST raw test files not found in {dataset}; pass --mnist-dir")
                job_id = uuid.uuid4().hex
                with JOBS_LOCK:
                    JOBS[job_id] = {"state": "running", "completed": 0, "total": count * len(models),
                                    "started": time.time()}
                command = [str(VALIDATE), "--images", str(images), "--labels", str(labels),
                           "--count", str(count), "--threads", str(self.server.max_concurrency), *model_paths]
                threading.Thread(target=run_validation, args=(job_id, command, max(180, count * 30)), daemon=True).start()
                self.send_bytes(202, "application/json; charset=utf-8", json.dumps({"job_id": job_id}).encode())
                return
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
    parser.add_argument(
        "--mnist-dir",
        default=None,
        help="directory containing raw t10k MNIST IDX files (or set MNIST_DIR; default: ./data/MNIST/raw)",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.max_concurrency = max(1, args.threads)
    server.mnist_dir = Path(args.mnist_dir or os.environ.get("MNIST_DIR", ROOT / "data" / "MNIST" / "raw"))
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"SEALTorch trace UI: http://{display_host}:{args.port}")
    server.serve_forever()
