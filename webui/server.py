#!/usr/bin/env python3
"""Small local API for the SEALTorch research dashboard."""
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    import torch
except ModuleNotFoundError:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists() and pathlib.Path(sys.executable) != venv_python:
        os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])
    raise

BUILD = pathlib.Path(os.environ.get("SEALTORCH_BINARY", ROOT / "build" / "sealtorch_gui"))
RESULTS_DIR = ROOT / "results"
plaintext_models = {}
benchmark_jobs = {}


def plaintext_model(name):
    if name in plaintext_models:
        return plaintext_models[name]
    path = ROOT / "src" / ("mnist_mlp_gelu.json" if name == "gelu" else "mnist_mlp.json")
    artifact = json.loads(path.read_text())
    layers = artifact["model"]["layers"]
    tensors = artifact["tensors"]
    modules = []
    linear_index = 0
    for item in layers:
        if item["type"] == "Linear":
            key = ("network.1", "network.3", "network.5")[linear_index]
            module = torch.nn.Linear(item["in_features"], item["out_features"])
            with torch.no_grad():
                module.weight.copy_(torch.tensor(tensors[key + ".weight"]["data"], dtype=torch.float32))
                module.bias.copy_(torch.tensor(tensors[key + ".bias"]["data"], dtype=torch.float32))
            modules.append(module)
            linear_index += 1
        elif item["type"] == "ReLU":
            modules.append(torch.nn.ReLU())
        elif "GELU" in item["type"]:
            modules.append(torch.nn.GELU())
    model = torch.nn.Sequential(*modules).eval()
    plaintext_models[name] = model
    return model


def run_plaintext(pixels, name, requested_device):
    if requested_device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch has no available CUDA device")
    device = "cuda" if requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()) else "cpu"
    model = plaintext_model(name).to(device)
    value = torch.tensor(pixels, dtype=torch.float32, device=device)
    if device == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(value)
    if device == "cuda": torch.cuda.synchronize()
    return output.cpu().double().tolist(), (time.perf_counter() - started) * 1000, device


class HEWorker:
    """One warm CUDA context per HE configuration; replaces it on a change."""
    def __init__(self):
        self.process = None
        self.config_key = None
        self.lock = threading.Lock()

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try: self.process.wait(timeout=10)
            except subprocess.TimeoutExpired: self.process.kill()
        self.process = self.config_key = None

    def run(self, request):
        config_key = json.dumps({"model": request.get("model"), "config": request.get("config")}, sort_keys=True)
        with self.lock:
            if self.process is None or self.process.poll() is not None or self.config_key != config_key:
                self.close()
                self.process = subprocess.Popen([str(BUILD), "--web-worker"], cwd=ROOT,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                self.config_key = config_key
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if not line:
                    detail = self.process.stderr.read().strip() or "ciphertext worker exited before responding"
                    self.close()
                    raise RuntimeError(detail)
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)


he_worker = HEWorker()


def gpu_power_watts():
    """Best-effort telemetry; unavailable sensors intentionally return null."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True).stdout.splitlines()
        values = [float(value.strip()) for value in output if value.strip() not in ("N/A", "[Not Supported]")]
        return sum(values) if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def cpu_energy_uj():
    paths = list(pathlib.Path("/sys/class/powercap").glob("intel-rapl*/energy_uj"))
    try:
        return sum(int(path.read_text().strip()) for path in paths) or None
    except (OSError, ValueError):
        return None


def execute(request):
    config = request.get("config", {})
    engine = config.get("engine", "he")
    model = request.get("model", "relu")
    if engine not in ("he", "pytorch"):
        raise ValueError("engine must be he or pytorch")
    before_power, before_energy, started = gpu_power_watts(), cpu_energy_uj(), time.monotonic()
    if engine == "pytorch":
        output, runtime_ms, device = run_plaintext(request["pixels"], model, config.get("device", "auto"))
        result = {
            "engine": "pytorch", "device": device, "output": output,
            "execution_ms": runtime_ms, "plain_ms": runtime_ms,
            "output_ciphertext_bytes": 0, "setup_ms": 0,
        }
    else:
        if config.get("device", "auto") == "cpu":
            raise RuntimeError("HE CPU is unavailable: this FIDESlib build requires CUDA for ciphertext inference")
        result = he_worker.run(request)
        if "error" not in result:
            result["engine"] = "he"
            result["device"] = "cuda"
            result["output"] = result["encrypted"]
            result["execution_ms"] = result["encrypted_ms"]
    elapsed = time.monotonic() - started
    after_power, after_energy = gpu_power_watts(), cpu_energy_uj()
    if "error" not in result:
        result["telemetry"] = {
            "gpu_power_w": (before_power + after_power) / 2 if before_power is not None and after_power is not None else None,
            "cpu_power_w": ((after_energy - before_energy) / elapsed / 1_000_000
                            if before_energy is not None and after_energy is not None and elapsed else None),
            "wall_ms": elapsed * 1000,
        }
    return result


def read_idx(path, expected_magic):
    """Read standard MNIST IDX files without an extra dataset dependency."""
    import gzip
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as file:
        magic = int.from_bytes(file.read(4), "big")
        if magic != expected_magic:
            raise ValueError(f"unexpected MNIST IDX header in {path}")
        count = int.from_bytes(file.read(4), "big")
        if expected_magic == 2051:
            rows, columns = int.from_bytes(file.read(4), "big"), int.from_bytes(file.read(4), "big")
            if (rows, columns) != (28, 28): raise ValueError("MNIST images must be 28×28")
            return [list(file.read(784)) for _ in range(count)]
        return list(file.read(count))


def mnist_validation_set():
    roots = [ROOT / "data" / "MNIST" / "raw", ROOT / "data", ROOT / "mnist"]
    for root in roots:
        image = next((root / name for name in ("t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz") if (root / name).exists()), None)
        label = next((root / name for name in ("t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz") if (root / name).exists()), None)
        if image and label:
            return read_idx(image, 2051), read_idx(label, 2049)
    raise FileNotFoundError("MNIST test IDX files were not found. Put t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte under data/MNIST/raw/")


def benchmark_runner(job_id, config, limit, batch_size):
    job = benchmark_jobs[job_id]
    try:
        images, labels = mnist_validation_set()
        count = min(limit, len(labels)) if limit else len(labels)
        job.update({"status": "running", "total": count, "completed": 0})
        latencies, correct, predictions = [], 0, []
        for begin in range(0, count, batch_size):
            for index in range(begin, min(begin + batch_size, count)):
                pixels = [value / 255.0 for value in images[index]]
                result = execute({"pixels": pixels, "model": config.get("model", "relu"), "config": config})
                if "error" in result: raise RuntimeError(result["error"])
                prediction = max(range(10), key=lambda item: result["output"][item])
                correct += prediction == labels[index]
                latencies.append(result["execution_ms"])
                predictions.append({"index": index, "label": labels[index], "prediction": prediction, "latency_ms": result["execution_ms"]})
            job.update({"completed": min(begin + batch_size, count), "accuracy": correct / min(begin + batch_size, count)})
        summary = {"id": job_id, "created_at": job["created_at"], "finished_at": datetime.now(timezone.utc).isoformat(), "config": config, "samples": count, "batch_size": batch_size, "accuracy": correct / count if count else 0, "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0, "p50_latency_ms": sorted(latencies)[len(latencies)//2] if latencies else 0, "predictions": predictions}
        RESULTS_DIR.mkdir(exist_ok=True)
        destination = RESULTS_DIR / f"mnist-benchmark-{job_id}.json"
        destination.write_text(json.dumps(summary, indent=2))
        job.update({"status": "complete", "result_file": str(destination.relative_to(ROOT)), **summary})
    except Exception as error:
        job.update({"status": "failed", "error": str(error)})


def start_benchmark(config, limit, batch_size):
    if config.get("engine") not in ("he", "pytorch"):
        raise ValueError("choose an execution engine for the benchmark")
    job_id = uuid.uuid4().hex[:12]
    benchmark_jobs[job_id] = {"id": job_id, "status": "queued", "created_at": datetime.now(timezone.utc).isoformat(), "config": config}
    threading.Thread(target=benchmark_runner, args=(job_id, config, limit, batch_size), daemon=True).start()
    return benchmark_jobs[job_id]


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
            self.send_json({"ok": BUILD.exists()})
            return
        if self.path.startswith("/api/benchmark/"):
            job_id = self.path.rsplit("/", 1)[-1]
            if job_id not in benchmark_jobs:
                self.send_json({"error": "benchmark not found"}, 404)
            else:
                self.send_json(benchmark_jobs[job_id])
            return
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        data = (ROOT / "webui" / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if self.path == "/api/predict":
                if len(request.get("pixels", [])) != 784:
                    raise ValueError("pixels must contain 784 values")
                self.send_json(execute(request))
            elif self.path == "/api/compare":
                pixels = request.get("pixels", [])
                if len(pixels) != 784:
                    raise ValueError("pixels must contain 784 values")
                left = execute({"pixels": pixels, "model": request.get("left", {}).get("model", "relu"), "config": request.get("left", {})})
                right = execute({"pixels": pixels, "model": request.get("right", {}).get("model", "relu"), "config": request.get("right", {})})
                self.send_json({"left": left, "right": right})
            elif self.path == "/api/benchmark":
                config = request.get("config", {})
                limit = int(request.get("limit", 0))
                batch_size = max(1, int(request.get("batch_size", 32)))
                self.send_json(start_benchmark(config, limit, batch_size), 202)
            else:
                self.send_error(404)
        except Exception as error:
            self.send_json({"error": str(error)}, 400)

    def log_message(self, format, *args):
        return


if not BUILD.exists():
    print("Build SEALTorch first: cmake -S . -B build && cmake --build build -j2", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    print("SEALTorch WebUI: http://127.0.0.1:8080", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
