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
import urllib.request
from collections import OrderedDict
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

BUILD_DIR = pathlib.Path(os.environ.get("SEALTORCH_BUILD_DIR", ROOT / "build"))
HE_BINARY = pathlib.Path(os.environ.get("SEALTORCH_HE_BINARY", BUILD_DIR / "sealtorch_gui"))
RESULTS_DIR = ROOT / "results"
MNIST_DIR = ROOT / "data" / "MNIST" / "raw"
MAX_PLAINTEXT_MODELS = 2
plaintext_models = OrderedDict()
plaintext_models_lock = threading.Lock()
benchmark_jobs = {}


def plaintext_model(name, device):
    cache_key = (name, device)
    with plaintext_models_lock:
        if cache_key in plaintext_models:
            model = plaintext_models.pop(cache_key)
            plaintext_models[cache_key] = model
            return model, False

        paths = {
            "relu": "mnist_mlp.json",
            "gelu": "mnist_mlp_gelu.json",
            "lenet": "lenet.json",
        }
        if name not in paths:
            raise ValueError("model must be relu, gelu, or lenet")
        path = ROOT / "src" / paths[name]
        artifact = json.loads(path.read_text())
        layers = artifact["model"]["layers"]
        tensors = artifact["tensors"]
        modules = []
        linear_index = 0
        convolution_index = 0

        for item in layers:
            layer_type = item["type"]
            if layer_type == "Linear":
                prefixes = (
                    ("classifier.1", "classifier.3", "classifier.5")
                    if name == "lenet"
                    else ("network.1", "network.3", "network.5")
                )
                prefix = prefixes[linear_index]
                module = torch.nn.Linear(
                    item["in_features"], item["out_features"])
                copy_parameters(module, tensors, prefix)
                modules.append(module)
                linear_index += 1
            elif layer_type == "Conv2d":
                prefix = ("features.0", "features.3")[convolution_index]
                module = torch.nn.Conv2d(
                    item["in_channels"],
                    item["out_channels"],
                    item["kernel_size"],
                )
                copy_parameters(module, tensors, prefix)
                modules.append(module)
                convolution_index += 1
            elif layer_type == "AvgPool2d":
                modules.append(torch.nn.AvgPool2d(
                    item["kernel_size"],
                    item.get("stride", item["kernel_size"]),
                ))
            elif layer_type == "Flatten":
                modules.append(torch.nn.Flatten())
            elif layer_type == "ReLU":
                modules.append(torch.nn.ReLU())
            elif "GELU" in layer_type:
                modules.append(torch.nn.GELU())
            elif layer_type == "Tanh":
                modules.append(torch.nn.Tanh())

        model = torch.nn.Sequential(*modules).eval().to(device)
        plaintext_models[cache_key] = model
        while len(plaintext_models) > MAX_PLAINTEXT_MODELS:
            plaintext_models.popitem(last=False)
        if device == "cuda":
            torch.cuda.empty_cache()
        return model, True


def copy_parameters(module, tensors, prefix):
    with torch.no_grad():
        module.weight.copy_(torch.tensor(
            tensors[prefix + ".weight"]["data"], dtype=torch.float32))
        module.bias.copy_(torch.tensor(
            tensors[prefix + ".bias"]["data"], dtype=torch.float32))


def run_plaintext(pixels, name, requested_device):
    if requested_device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch has no available CUDA device")
    use_cuda = (
        requested_device == "cuda"
        or (requested_device == "auto" and torch.cuda.is_available())
    )
    device = "cuda" if use_cuda else "cpu"
    setup_started = time.perf_counter()
    model, initialized = plaintext_model(name, device)
    value = torch.tensor(pixels, dtype=torch.float32, device=device)
    if name == "lenet":
        value = value.reshape(1, 1, 28, 28)
    else:
        value = value.reshape(1, 784)

    # CUDA may defer allocation and kernel/module loading until the first
    # invocation.  Warm a newly cached model before starting the inference
    # timer so a cold load is reported as setup, not sample latency.
    if initialized:
        with torch.inference_mode():
            model(value)
        if device == "cuda":
            torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_started) * 1000 if initialized else 0
    if device == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(value)
    if device == "cuda": torch.cuda.synchronize()
    return output.reshape(-1).cpu().double().tolist(), (time.perf_counter() - started) * 1000, device, setup_ms


class HEWorker:
    """One warm HE context per configuration; replaces it on a change."""
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

    def run(self, request, binary):
        config_key = json.dumps({"model": request.get("model"), "config": request.get("config")}, sort_keys=True)
        with self.lock:
            worker_startup_ms = 0
            if self.process is None or self.process.poll() is not None or self.config_key != (binary, config_key):
                worker_startup_started = time.perf_counter()
                self.close()
                self.process = subprocess.Popen([str(binary), "--web-worker"], cwd=ROOT,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                self.config_key = (binary, config_key)
                worker_startup_ms = (time.perf_counter() - worker_startup_started) * 1000
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
                    response = json.loads(line)
                    # The worker measures context/model/key initialization.  Process
                    # startup is also setup work, so expose it separately from inference.
                    response["worker_startup_ms"] = worker_startup_ms
                    return response


he_worker = HEWorker()


def select_he_backend(requested_device):
    """Validate a provider choice handled by the single SEALTorch worker."""
    if requested_device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if not HE_BINARY.exists():
        raise RuntimeError(f"SEALTorch worker is not built ({HE_BINARY}). Run cmake --build build.")
    return HE_BINARY


def he_capabilities():
    if not HE_BINARY.exists():
        return {"cpu": False, "cuda": False}
    try:
        output = subprocess.run(
            [str(HE_BINARY), "--capabilities"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return json.loads(output.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {"cpu": True, "cuda": False}


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


def platform_power_watts():
    """Read the Intel Node Manager total-node meter when the host exposes it.

    This is deliberately not used as a CPU measurement: the kernel describes
    this domain as total node power, so it can include memory and other board
    components.  hwmon power values are expressed in microwatts.
    """
    try:
        for directory in pathlib.Path("/sys/class/hwmon").glob("hwmon*"):
            if directory.joinpath("name").read_text().strip() != "power_meter":
                continue
            value = int(directory.joinpath("power1_average").read_text().strip())
            return value / 1_000_000
    except (OSError, ValueError):
        pass
    return None


def cpu_energy_snapshot():
    """Return CPU *package* RAPL counters, without double-counting subdomains.

    A RAPL package directory may also contain DRAM/core subdomains.  Summing all
    ``energy_uj`` files counts the same work more than once, so select only
    domains named ``package-*``.  Some container hosts expose the files but do
    not grant read access; preserve that useful distinction for the UI.
    """
    roots = (pathlib.Path("/sys/class/powercap"),
             pathlib.Path("/sys/devices/virtual/powercap"))
    packages, seen, unreadable = [], set(), False
    for root in roots:
        if not root.exists():
            continue
        for energy_path in root.rglob("energy_uj"):
            directory = energy_path.parent
            try:
                resolved = str(directory.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                if not directory.joinpath("name").read_text().strip().lower().startswith("package-"):
                    continue
                packages.append({"path": energy_path,
                                 "max_path": directory / "max_energy_range_uj"})
            except OSError:
                unreadable = True
    if not packages:
        if unreadable:
            reason = "CPU package RAPL counters exist but are not readable"
        else:
            reason = "No CPU package RAPL counters are exposed by this host"
        return {"counters": None,
                "reason": reason}
    counters = []
    try:
        for package in packages:
            counters.append({"energy_uj": int(package["path"].read_text().strip()),
                             "max_energy_uj": int(package["max_path"].read_text().strip())})
    except (OSError, ValueError):
        return {"counters": None,
                "reason": "CPU package RAPL counters are present but not readable by this server"}
    return {"counters": counters, "reason": None}


def telemetry_snapshot():
    return {"gpu_power_w": gpu_power_watts(), "platform_power_w": platform_power_watts(),
            "cpu": cpu_energy_snapshot()}


def usage_telemetry(before, after, elapsed_seconds):
    """Energy is comparable across CPU/GPU runs; GPU energy is sampled estimate."""
    gpu_power = None
    if before["gpu_power_w"] is not None and after["gpu_power_w"] is not None:
        gpu_power = (before["gpu_power_w"] + after["gpu_power_w"]) / 2
    platform_power = None
    if before["platform_power_w"] is not None and after["platform_power_w"] is not None:
        platform_power = (before["platform_power_w"] + after["platform_power_w"]) / 2
    cpu_energy = None
    cpu_reason = after["cpu"]["reason"] or before["cpu"]["reason"]
    before_counters, after_counters = before["cpu"]["counters"], after["cpu"]["counters"]
    if before_counters is not None and after_counters is not None:
        if len(before_counters) != len(after_counters):
            cpu_reason = "CPU package counter set changed while measuring"
        else:
            # RAPL counters wrap at max_energy_range_uj. Correct wrapping is
            # essential for long benchmarks, and avoids dropping valid energy.
            deltas = []
            for start, end in zip(before_counters, after_counters):
                delta = end["energy_uj"] - start["energy_uj"]
                if delta < 0:
                    delta += end["max_energy_uj"]
                if delta < 0:
                    cpu_reason = "CPU package counter reset while measuring"
                    break
                deltas.append(delta)
            else:
                cpu_energy = sum(deltas) / 1_000_000
    gpu_energy = gpu_power * elapsed_seconds if gpu_power is not None else None
    cpu_power = None
    if cpu_energy is not None and elapsed_seconds:
        cpu_power = cpu_energy / elapsed_seconds
    known_energy = [
        value for value in (cpu_energy, gpu_energy)
        if value is not None
    ]
    known_power = [
        value for value in (cpu_power, gpu_power)
        if value is not None
    ]
    return {
        "gpu_power_w": gpu_power,
        "cpu_power_w": cpu_power,
        "gpu_energy_j": gpu_energy,
        "platform_power_w": platform_power,
        "platform_energy_j": (
            platform_power * elapsed_seconds
            if platform_power is not None
            else None
        ),
        "cpu_energy_j": cpu_energy,
        "total_energy_j": sum(known_energy) if known_energy else None,
        "total_power_w": sum(known_power) if known_power else None,
        "wall_ms": elapsed_seconds * 1000,
        "gpu_energy_method": (
            "two-point power estimate" if gpu_energy is not None else None
        ),
        "cpu_energy_method": (
            "RAPL CPU package counters" if cpu_energy is not None else None
        ),
        "cpu_power_reason": cpu_reason,
        "gpu_power_reason": (
            "NVIDIA power sensor is unavailable"
            if gpu_power is None
            else None
        ),
        "platform_power_method": (
            "Intel Node Manager total-node meter (1 s rolling average)"
            if platform_power is not None
            else None
        ),
        "platform_power_reason": (
            "No readable total-node power meter is exposed by this host"
            if platform_power is None
            else None
        ),
    }


def execute(request, collect_telemetry=True):
    config = request.get("config", {})
    engine = config.get("engine", "he")
    model = request.get("model", "relu")
    if engine not in ("he", "pytorch"):
        raise ValueError("engine must be he or pytorch")
    validate_pixels(request.get("pixels"))
    before = telemetry_snapshot() if collect_telemetry else None
    started = time.monotonic()
    if engine == "pytorch":
        output, runtime_ms, device, setup_ms = run_plaintext(
            request["pixels"], model, config.get("device", "auto"))
        result = {
            "engine": "pytorch", "device": device, "output": output,
            "execution_ms": runtime_ms, "plain_ms": runtime_ms,
            "output_ciphertext_bytes": 0, "setup_ms": setup_ms,
        }
    else:
        binary = select_he_backend(config.get("device", "auto"))
        result = he_worker.run(request, binary)
        if "error" not in result:
            result["engine"] = "he"
            # The native worker reports whether this request used SEAL CPU or
            # FIDESlib CUDA. Keep that observed value rather than echoing it.
            result["output"] = result["encrypted"]
            result["execution_ms"] = result["encrypted_ms"]
            result["setup_ms"] = (
                result.get("setup_ms", 0)
                + result.get("worker_startup_ms", 0)
            )
    elapsed = time.monotonic() - started
    if "error" not in result and collect_telemetry:
        result["telemetry"] = usage_telemetry(
            before, telemetry_snapshot(), elapsed)
    return result


def validate_pixels(pixels):
    if not isinstance(pixels, list) or len(pixels) != 784:
        raise ValueError("pixels must contain 784 values")
    for pixel in pixels:
        if not isinstance(pixel, (int, float)) or not 0 <= pixel <= 1:
            raise ValueError("pixels must be numbers between 0 and 1")


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
            rows = int.from_bytes(file.read(4), "big")
            columns = int.from_bytes(file.read(4), "big")
            if (rows, columns) != (28, 28):
                raise ValueError("MNIST images must be 28×28")
            data = file.read(count * 784)
            if len(data) != count * 784:
                raise ValueError(f"truncated MNIST image data in {path}")
            return [
                list(data[offset:offset + 784])
                for offset in range(0, len(data), 784)
            ]
        data = file.read(count)
        if len(data) != count:
            raise ValueError(f"truncated MNIST label data in {path}")
        return list(data)


def mnist_validation_set():
    roots = [MNIST_DIR, ROOT / "data", ROOT / "mnist"]
    image_names = (
        "t10k-images-idx3-ubyte",
        "t10k-images-idx3-ubyte.gz",
    )
    label_names = (
        "t10k-labels-idx1-ubyte",
        "t10k-labels-idx1-ubyte.gz",
    )
    for root in roots:
        image = next(
            (root / name for name in image_names if (root / name).exists()),
            None,
        )
        label = next(
            (root / name for name in label_names if (root / name).exists()),
            None,
        )
        if image and label:
            return read_idx(image, 2051), read_idx(label, 2049)
    raise FileNotFoundError(
        "MNIST test IDX files were not found. Put "
        "t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte "
        "under data/MNIST/raw/")


def mnist_data_status():
    try:
        images, labels = mnist_validation_set()
        counts = [0] * 10
        for label in labels:
            counts[label] += 1
        return {"loaded": True, "samples": len(labels), "label_counts": counts,
                "location": str(MNIST_DIR.relative_to(ROOT))}
    except (FileNotFoundError, OSError, ValueError) as error:
        return {
            "loaded": False,
            "error": str(error),
            "location": str(MNIST_DIR.relative_to(ROOT)),
        }


def download_mnist():
    """Download only the public MNIST test split used by this validation page."""
    MNIST_DIR.mkdir(parents=True, exist_ok=True)
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    names = ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    for name in names:
        destination = MNIST_DIR / name
        if not destination.exists():
            partial = destination.with_suffix(destination.suffix + ".part")
            urllib.request.urlretrieve(base + name, partial)
            partial.replace(destination)
    return mnist_data_status()


def percentile(values, fraction):
    """Linearly interpolated percentile for a pre-sorted, non-empty sequence."""
    if not values:
        return 0
    position = (len(values) - 1) * fraction
    lower, upper = int(position), min(int(position) + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def benchmark_runner(
        job_id,
        config,
        limit,
        batch_size,
        start_index=0,
        stride=1,
        shuffle_seed=None):
    job = benchmark_jobs[job_id]
    try:
        images, labels = mnist_validation_set()
        indices = list(range(max(0, start_index), len(labels), max(1, stride)))
        if shuffle_seed is not None:
            import random
            random.Random(shuffle_seed).shuffle(indices)
        count = min(limit, len(indices)) if limit else len(indices)
        indices = indices[:count]
        job.update({"status": "running", "total": count, "completed": 0})
        latencies, setup_times, correct, predictions = [], [], 0, []
        confusion = [[0] * 10 for _ in range(10)]
        power_before, benchmark_started = telemetry_snapshot(), time.monotonic()
        for begin in range(0, count, batch_size):
            for index in range(begin, min(begin + batch_size, count)):
                dataset_index = indices[index]
                pixels = [value / 255.0 for value in images[dataset_index]]
                result = execute(
                    {
                        "pixels": pixels,
                        "model": config.get("model", "relu"),
                        "config": config,
                    },
                    collect_telemetry=False,
                )
                if "error" in result:
                    raise RuntimeError(result["error"])
                prediction = max(
                    range(10),
                    key=lambda item: result["output"][item],
                )
                label = labels[dataset_index]
                if prediction == label:
                    correct += 1
                latencies.append(result["execution_ms"])
                setup_times.append(result.get("setup_ms", 0))
                confusion[label][prediction] += 1
                predictions.append({
                    "index": dataset_index,
                    "label": label,
                    "prediction": prediction,
                    "latency_ms": result["execution_ms"],
                    "setup_ms": result.get("setup_ms", 0),
                })
            done = min(begin + batch_size, count)
            mean_latency = sum(latencies) / len(latencies)
            initialization_events = sum(
                1 for setup_ms in setup_times if setup_ms > 0)
            progress = {
                "completed": done,
                "accuracy": correct / done,
                "mean_latency_ms": mean_latency,
            }
            job.update({
                **progress,
                "initialization_ms": sum(setup_times),
                "initialization_events": initialization_events,
                "progress": progress,
            })
        elapsed = time.monotonic() - benchmark_started
        latency_sorted = sorted(latencies)
        label_totals = [sum(row) for row in confusion]
        mean_latency = sum(latencies) / len(latencies) if latencies else 0
        if latencies:
            variance = sum(
                (latency - mean_latency) ** 2
                for latency in latencies
            ) / len(latencies)
            latency_stddev = variance ** 0.5
        else:
            latency_stddev = 0
        per_digit_accuracy = [
            confusion[digit][digit] / label_totals[digit]
            if label_totals[digit]
            else None
            for digit in range(10)
        ]
        initialization_events = sum(
            1 for setup_ms in setup_times if setup_ms > 0)
        summary = {
            "id": job_id,
            "created_at": job["created_at"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "samples": count,
            "batch_size": batch_size,
            "selection": {
                "start_index": start_index,
                "stride": stride,
                "shuffle_seed": shuffle_seed,
            },
            "accuracy": correct / count if count else 0,
            "correct": correct,
            "incorrect": count - correct,
            "mean_latency_ms": mean_latency,
            "min_latency_ms": latency_sorted[0] if latencies else 0,
            "max_latency_ms": latency_sorted[-1] if latencies else 0,
            "latency_stddev_ms": latency_stddev,
            "p50_latency_ms": percentile(latency_sorted, 0.50),
            "p95_latency_ms": percentile(latency_sorted, 0.95),
            "p99_latency_ms": percentile(latency_sorted, 0.99),
            "initialization_ms": sum(setup_times),
            "initialization_events": initialization_events,
            "throughput_per_s": count / elapsed if elapsed else 0,
            "confusion_matrix": confusion,
            "per_digit_accuracy": per_digit_accuracy,
            "latency_ms": latencies,
            "setup_ms": setup_times,
            "predictions": predictions,
            "telemetry": usage_telemetry(
                power_before, telemetry_snapshot(), elapsed),
        }
        RESULTS_DIR.mkdir(exist_ok=True)
        destination = RESULTS_DIR / f"mnist-benchmark-{job_id}.json"
        destination.write_text(json.dumps(summary, indent=2))
        job.update({
            "status": "complete",
            "result_file": str(destination.relative_to(ROOT)),
            **summary,
        })
    except Exception as error:
        job.update({"status": "failed", "error": str(error)})


def start_benchmark(
        config,
        limit,
        batch_size,
        start_index=0,
        stride=1,
        shuffle_seed=None):
    if config.get("engine") not in ("he", "pytorch"):
        raise ValueError("choose an execution engine for the benchmark")
    if limit < 0:
        raise ValueError("benchmark limit cannot be negative")
    job_id = uuid.uuid4().hex[:12]
    benchmark_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    arguments = (
        job_id,
        config,
        limit,
        batch_size,
        start_index,
        stride,
        shuffle_seed,
    )
    threading.Thread(
        target=benchmark_runner,
        args=arguments,
        daemon=True,
    ).start()
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
            capabilities = he_capabilities()
            self.send_json({
                "ok": capabilities["cpu"],
                "he_backends": capabilities,
            })
            return
        if self.path == "/api/mnist/data":
            self.send_json(mnist_data_status())
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
                self.send_json(execute(request))
            elif self.path == "/api/compare":
                pixels = request.get("pixels", [])
                validate_pixels(pixels)
                left_config = request.get("left", {})
                right_config = request.get("right", {})
                left = execute({
                    "pixels": pixels,
                    "model": left_config.get("model", "relu"),
                    "config": left_config,
                })
                right = execute({
                    "pixels": pixels,
                    "model": right_config.get("model", "relu"),
                    "config": right_config,
                })
                self.send_json({"left": left, "right": right})
            elif self.path == "/api/benchmark":
                config = request.get("config", {})
                limit = int(request.get("limit", 0))
                batch_size = max(1, int(request.get("batch_size", 32)))
                start_index = max(0, int(request.get("start_index", 0)))
                stride = max(1, int(request.get("stride", 1)))
                shuffle_seed = request.get("shuffle_seed")
                if shuffle_seed in (None, ""):
                    shuffle_seed = None
                else:
                    shuffle_seed = int(shuffle_seed)
                job = start_benchmark(
                    config,
                    limit,
                    batch_size,
                    start_index,
                    stride,
                    shuffle_seed,
                )
                self.send_json(job, 202)
            elif self.path == "/api/mnist/download":
                self.send_json(download_mnist())
            else:
                self.send_error(404)
        except Exception as error:
            self.send_json({"error": str(error)}, 400)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    if not HE_BINARY.exists():
        print(
            "Build SEALTorch first: "
            "cmake -S . -B build && cmake --build build -j2",
            file=sys.stderr,
        )
        sys.exit(1)
    print("SEALTorch WebUI: http://127.0.0.1:8080", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
