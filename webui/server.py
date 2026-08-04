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
import urllib.parse
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEBUI_DIR = pathlib.Path(__file__).resolve().parent
if str(WEBUI_DIR) not in sys.path:
    sys.path.insert(0, str(WEBUI_DIR))
from model_translator import load_export, native_artifact

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
LENET_ROOT = pathlib.Path(os.environ.get(
    "SEALTORCH_LENET_ROOT", ROOT.parent / "LeNet-5"))
NIST19_ROOT = pathlib.Path(os.environ.get(
    "SEALTORCH_NIST19_ROOT", LENET_ROOT / "data" / "nist19" / "by_class"))
NATIVE_MODELS = {
    "relu": ROOT / "src" / "mnist_mlp.json",
    "gelu": ROOT / "src" / "mnist_mlp_gelu.json",
    "lenet": ROOT / "src" / "lenet.json",
}


def available_models():
    """Expose one choice per trainer export, never separate .pt/.json choices."""
    models = {name: path for name, path in NATIVE_MODELS.items() if path.is_file()}
    if LENET_ROOT.is_dir():
        for path in sorted(LENET_ROOT.glob("exports/**/*.pt")):
            models[f"trainer:{path.stem}"] = path
    return models


def model_file(name):
    models = available_models()
    if name not in models:
        raise ValueError("unknown model; choose one advertised by /api/models")
    return models[name]


def is_trainer_model(name):
    return name.startswith("trainer:")


def is_native_model(name):
    return name in NATIVE_MODELS


def native_manifest(name):
    if not is_native_model(name):
        raise ValueError("unknown native model")
    return json.loads(model_file(name).read_text(encoding="utf-8"))


def model_classes(name):
    if is_trainer_model(name):
        return load_export(model_file(name)).manifest.get("classes", [])
    if is_native_model(name):
        return native_manifest(name)["model"]["outputs"].get("classes", [])
    raise ValueError("unknown model")


def model_preprocessing(name):
    if is_trainer_model(name):
        return load_export(model_file(name)).manifest.get("preprocessing", {})
    if is_native_model(name):
        preprocessing = native_manifest(name).get("preprocessing", {})
        return {"operations": [{
            "op": "normalize_foreground",
            "canvas_size": 28,
            "foreground_size": preprocessing.get("resize_ink_to_max", [20])[0],
            "threshold": preprocessing.get("crop_threshold", 20),
        }]}
    raise ValueError("unknown model")


def model_minimum_ring_dimension(name):
    """Return the smallest CKKS ring whose slot count fits every feature map."""
    if is_trainer_model(name):
        architecture = load_export(model_file(name)).manifest["architecture"]
        _, channels, height, width = architecture["input"]["shape"]
        layers = architecture["layers"]
    elif is_native_model(name):
        artifact = native_manifest(name)
        channels, height, width = artifact["model"]["input_shape"]
        layers = artifact["model"]["layers"]
    else:
        raise ValueError("unknown model")

    def pair(value, default=None):
        if value is None:
            value = default
        return (value, value) if isinstance(value, int) else tuple(value)

    widest = channels * height * width
    for layer in layers:
        op = layer.get("op", layer.get("type", "")).lower()
        if op == "conv2d":
            kernel_h, kernel_w = pair(layer.get("kernel", layer.get("kernel_size")))
            stride_h, stride_w = pair(layer.get("stride"), 1)
            padding_h, padding_w = pair(layer.get("padding"), 0)
            dilation_h, dilation_w = pair(layer.get("dilation"), 1)
            height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
            width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
            channels = layer.get("out_channels", channels)
        elif op in ("avg_pool2d", "avgpool2d", "max_pool2d", "maxpool2d"):
            kernel_h, kernel_w = pair(layer.get("kernel", layer.get("kernel_size")))
            stride_h, stride_w = pair(layer.get("stride"), (kernel_h, kernel_w))
            padding_h, padding_w = pair(layer.get("padding"), 0)
            height = (height + 2 * padding_h - kernel_h) // stride_h + 1
            width = (width + 2 * padding_w - kernel_w) // stride_w + 1
        elif op == "linear":
            widest = max(widest, int(layer.get("in_features", 0)),
                         int(layer.get("out_features", 0)))
        widest = max(widest, channels * height * width)
    ring = 2
    while ring // 2 < widest:
        ring *= 2
    return ring


def plaintext_model(name, device):
    cache_key = (name, device)
    with plaintext_models_lock:
        if cache_key in plaintext_models:
            model = plaintext_models.pop(cache_key)
            plaintext_models[cache_key] = model
            return model, False

        if is_trainer_model(name):
            export = load_export(model_file(name))
            # Import the trainer implementation only after resolving its export.
            # That keeps the dashboard usable without the optional sibling repo.
            lenet_source = str(LENET_ROOT)
            if lenet_source not in sys.path:
                sys.path.insert(0, lenet_source)
            from lenet5 import model_from_architecture
            model = model_from_architecture(
                export.manifest["architecture"], len(export.manifest["classes"]))
            model.load_state_dict(export.state_dict)
            model = model.eval().to(device)
            plaintext_models[cache_key] = model
            while len(plaintext_models) > MAX_PLAINTEXT_MODELS:
                plaintext_models.popitem(last=False)
            return model, True

        if is_native_model(name):
            artifact = native_manifest(name)
            tensors, modules, weight_index = artifact["tensors"], [], 0
            weight_keys = [key[:-7] for key in tensors if key.endswith(".weight")]
            for layer in artifact["model"]["layers"]:
                kind = layer["type"]
                if kind == "Flatten":
                    modules.append(torch.nn.Flatten())
                elif kind == "Linear":
                    prefix = weight_keys[weight_index]; weight_index += 1
                    module = torch.nn.Linear(layer["in_features"], layer["out_features"])
                    with torch.no_grad():
                        module.weight.copy_(torch.tensor(tensors[prefix + ".weight"]["data"], dtype=torch.float32))
                        module.bias.copy_(torch.tensor(tensors[prefix + ".bias"]["data"], dtype=torch.float32))
                    modules.append(module)
                elif kind == "Conv2d":
                    prefix = weight_keys[weight_index]; weight_index += 1
                    module = torch.nn.Conv2d(layer["in_channels"], layer["out_channels"], layer["kernel_size"])
                    with torch.no_grad():
                        module.weight.copy_(torch.tensor(tensors[prefix + ".weight"]["data"], dtype=torch.float32))
                        module.bias.copy_(torch.tensor(tensors[prefix + ".bias"]["data"], dtype=torch.float32))
                    modules.append(module)
                elif kind == "AvgPool2d":
                    modules.append(torch.nn.AvgPool2d(layer["kernel_size"], layer.get("stride", layer["kernel_size"])))
                elif kind == "Tanh": modules.append(torch.nn.Tanh())
                elif kind == "ReLU": modules.append(torch.nn.ReLU())
                elif "GELU" in kind: modules.append(torch.nn.GELU())
                else: raise ValueError(f"unsupported native model layer: {kind}")
            model = torch.nn.Sequential(*modules).eval().to(device)
            plaintext_models[cache_key] = model
            while len(plaintext_models) > MAX_PLAINTEXT_MODELS:
                plaintext_models.popitem(last=False)
            return model, True

        raise ValueError("unknown model")


def run_plaintext(pixels, name, requested_device, prepared_input=None):
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
    if not (is_trainer_model(name) or is_native_model(name)):
        raise ValueError("unknown model")
    value = (prepared_input.to(device) if prepared_input is not None else
             prepare_trainer_pixels(pixels, name, device) if is_trainer_model(name)
             else prepare_native_pixels(pixels, device))

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


def prepare_trainer_image(image, name, device="cpu"):
    """Apply an export's operations to a single-channel upright image tensor."""
    export = load_export(model_file(name))
    image = torch.as_tensor(image, dtype=torch.float32, device=device)
    if image.ndim != 2:
        raise ValueError("trainer input image must be two-dimensional")
    image = image.unsqueeze(0)
    operations = export.manifest.get("preprocessing", {}).get("operations", [])
    # Early EMNIST exports predate ``apply_to`` and describe both a transpose
    # and horizontal flip for their sideways source files.  A browser canvas is
    # already upright, so preserve only the flip needed by those legacy models.
    legacy_emnist = any(
        operation.get("reason") == "correct EMNIST storage orientation"
        and "apply_to" not in operation
        for operation in operations
    )
    for operation in operations:
        # EMNIST's on-disk images are sideways.  This correction was used only
        # while reading the dataset and must not be applied to canvas input.
        if operation.get("apply_to") == "dataset":
            continue
        kind = operation.get("op")
        if legacy_emnist and kind == "transpose":
            continue
        if kind == "transpose": image = image.transpose(-2, -1)
        elif kind == "flip_horizontal": image = image.flip(-1)
        elif kind == "normalize_foreground":
            image = normalize_foreground(
                image,
                canvas_size=operation.get("canvas_size", 28),
                foreground_size=operation.get("foreground_size", 20),
                threshold=operation.get("threshold", 20),
            )
        elif kind == "pad":
            image = torch.nn.functional.pad(
                image, (operation["left"], operation["right"], operation["top"], operation["bottom"]),
                value=operation.get("fill", 0))
        elif kind == "resize":
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=tuple(operation["size"]),
                mode="bilinear", align_corners=False).squeeze(0)
        elif kind == "to_tensor":
            continue
        elif kind == "normalize":
            return ((image - operation["mean"][0]) / operation["std"][0]).unsqueeze(0)
        elif kind == "invert": image = 1.0 - image
        else: raise ValueError(f"unsupported trainer preprocessing operation: {kind}")
    raise ValueError("trainer preprocessing must end with normalize")


def prepare_trainer_pixels(pixels, name, device="cpu"):
    """Apply an export's web-input operations to an upright 28×28 drawing."""
    return prepare_trainer_image(
        torch.tensor(pixels, dtype=torch.float32).reshape(28, 28), name, device)


def prepare_native_pixels(pixels, device="cpu"):
    """Match the bundled MNIST artifacts' crop, scale, and NCHW layout."""
    image = torch.tensor(pixels, dtype=torch.float32, device=device).reshape(1, 28, 28)
    return normalize_foreground(image).unsqueeze(0)


def prepare_native_validation_pixels(pixels, device="cpu"):
    """Prepare already-normalized MNIST validation images without re-cropping.

    Foreground centering is a canvas adaptation for hand drawings. Applying it
    again to canonical MNIST test images changes the dataset on which the
    bundled checkpoints report their accuracy.
    """
    return torch.as_tensor(
        pixels, dtype=torch.float32, device=device).reshape(1, 1, 28, 28) / 255.0


def prepare_validation_input(pixels, name, device="cpu"):
    """Prepare a raw validation sample, preserving its source image geometry."""
    if isinstance(pixels, torch.Tensor):
        return prepare_trainer_image(pixels, name, device)
    if is_trainer_model(name):
        return prepare_trainer_pixels([value / 255.0 for value in pixels], name, device)
    return prepare_native_validation_pixels(pixels, device)


def normalize_foreground(image, *, canvas_size=28, foreground_size=20, threshold=20):
    """Match the trainer's MNIST-style crop, scale, and centering operation."""
    if not 0 < foreground_size <= canvas_size:
        raise ValueError("foreground_size must be in (0, canvas_size]")
    coordinates = torch.nonzero(image[0] > threshold / 255.0, as_tuple=False)
    if not len(coordinates):
        return torch.zeros((1, canvas_size, canvas_size), dtype=image.dtype, device=image.device)
    top, left = coordinates.min(dim=0).values.tolist()
    bottom, right = coordinates.max(dim=0).values.tolist()
    glyph = image[:, top:bottom + 1, left:right + 1]
    height, width = glyph.shape[-2:]
    scale = foreground_size / max(width, height)
    resized = torch.nn.functional.interpolate(
        glyph.unsqueeze(0), size=(max(1, round(height * scale)), max(1, round(width * scale))),
        mode="bilinear", align_corners=False).squeeze(0)
    result = torch.zeros((1, canvas_size, canvas_size), dtype=image.dtype, device=image.device)
    y = (canvas_size - resized.shape[-2]) // 2
    x = (canvas_size - resized.shape[-1]) // 2
    result[:, y:y + resized.shape[-2], x:x + resized.shape[-1]] = resized
    return result


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
    model = request.get("model", "trainer:lenet5_mnist")
    if engine not in ("he", "pytorch"):
        raise ValueError("engine must be he or pytorch")
    if not (is_trainer_model(model) or is_native_model(model)):
        raise ValueError("unknown model")
    # Prepare once, before selecting an engine. This makes the export manifest
    # the single preprocessing contract for every model and every execution
    # route (single run, comparison, HE, and benchmark).
    prepared_input = request.get("_prepared_input")
    if prepared_input is None:
        validate_pixels(request.get("pixels"))
        prepared_input = (prepare_trainer_pixels(request["pixels"], model)
                          if is_trainer_model(model)
                          else prepare_native_pixels(request["pixels"]))
    before = telemetry_snapshot() if collect_telemetry else None
    started = time.monotonic()
    if engine == "pytorch":
        output, runtime_ms, device, setup_ms = run_plaintext(
            request["pixels"], model, config.get("device", "auto"),
            prepared_input=prepared_input)
        result = {
            "engine": "pytorch", "device": device, "output": output,
            "execution_ms": runtime_ms, "plain_ms": runtime_ms,
            "output_ciphertext_bytes": 0, "setup_ms": setup_ms,
        }
    else:
        binary = select_he_backend(config.get("device", "auto"))
        worker_request = dict(request)
        # ``_prepared_input`` is an in-process tensor used by benchmarks with
        # non-28×28 sources (not part of the native worker protocol).
        worker_request.pop("_prepared_input", None)
        worker_request["pixels"] = prepared_input.reshape(-1).tolist()
        worker_request["model_path"] = str(
            native_artifact(load_export(model_file(model)))
            if is_trainer_model(model) else model_file(model))
        result = he_worker.run(worker_request, binary)
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
    if "error" not in result:
        result["classes"] = model_classes(model)
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


class ValidationDataset:
    """A validation split whose source pixels can be read without preprocessing."""
    def __init__(self, dataset, labels, classes, pixels_at, location):
        self.dataset = dataset
        self.labels = labels
        self.classes = classes
        self.pixels_at = pixels_at
        self.location = location

    def status(self):
        counts = [0] * len(self.classes)
        for label in self.labels:
            if 0 <= label < len(counts):
                counts[label] += 1
        return {"loaded": True, "dataset": self.dataset, "samples": len(self.labels),
                "classes": self.classes, "label_counts": counts,
                "location": self.location}


def trainer_validation_set(name):
    """Return the selected export's validation split in upright source orientation."""
    if is_native_model(name):
        images, labels = mnist_validation_set()
        return ValidationDataset("mnist", labels, model_classes(name), lambda index: images[index],
                                 str(MNIST_DIR.relative_to(ROOT)))
    export = load_export(model_file(name))
    dataset = export.manifest.get("dataset", "")
    classes = export.manifest.get("classes", [])
    if dataset == "mnist":
        images, labels = mnist_validation_set()
        return ValidationDataset(dataset, labels, classes, lambda index: images[index],
                                 str(MNIST_DIR.relative_to(ROOT)))

    if dataset.startswith("emnist-"):
        root = LENET_ROOT / "data" / "EMNIST" / "raw"
        stem = f"{dataset}-test"
        image = root / f"{stem}-images-idx3-ubyte"
        label = root / f"{stem}-labels-idx1-ubyte"
        if not image.exists() or not label.exists():
            raise FileNotFoundError(
                f"{dataset} test IDX files were not found under {root}")
        images, labels = read_idx(image, 2051), read_idx(label, 2049)
        # EMNIST stores each glyph transposed.  Its manifest marks this transform
        # as dataset-only, whereas browser-like input must already be upright.
        def pixels_at(index):
            glyph = images[index]
            return [glyph[row + column * 28] for row in range(28) for column in range(28)]
        return ValidationDataset(dataset, labels, classes, pixels_at,
                                 str(root.relative_to(ROOT.parent)))

    if dataset == "nist19":
        source = str(LENET_ROOT)
        if source not in sys.path:
            sys.path.insert(0, source)
        from data import NIST19Letters
        root = NIST19_ROOT
        full = NIST19Letters(root, transform=None)
        training = export.manifest.get("training", {})
        fraction = float(training.get("val_fraction", 0.1))
        seed = int(training.get("seed", 42))
        validation_count = max(1, round(len(full) * fraction))
        order = torch.randperm(len(full), generator=torch.Generator().manual_seed(seed)).tolist()
        indices = order[len(full) - validation_count:]
        labels = [full.samples[index][1] for index in indices]
        def pixels_at(index):
            image, _ = full.raw_item(indices[index])
            # The trainer's foreground operation accepts raw NIST scan sizes.
            return torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(
                image.height, image.width) / 255.0
        return ValidationDataset(dataset, labels, classes, pixels_at,
                                 str(root.relative_to(ROOT.parent)))

    raise ValueError(f"benchmarking is not implemented for trainer dataset {dataset!r}")


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


def trainer_data_status(name):
    try:
        return trainer_validation_set(name).status()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        dataset = "mnist" if is_native_model(name) else load_export(model_file(name)).manifest.get("dataset", "")
        return {"loaded": False, "dataset": dataset,
                "error": str(error)}


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
        name = config.get("model", "trainer:lenet5_mnist")
        dataset = trainer_validation_set(name)
        labels, classes = dataset.labels, dataset.classes
        if not classes:
            raise ValueError("trainer export must declare its benchmark classes")
        indices = list(range(max(0, start_index), len(labels), max(1, stride)))
        if shuffle_seed is not None:
            import random
            random.Random(shuffle_seed).shuffle(indices)
        count = min(limit, len(indices)) if limit else len(indices)
        indices = indices[:count]
        job.update({"status": "running", "total": count, "completed": 0})
        latencies, setup_times, correct, predictions = [], [], 0, []
        confusion = [[0] * len(classes) for _ in classes]
        power_before, benchmark_started = telemetry_snapshot(), time.monotonic()
        for begin in range(0, count, batch_size):
            for index in range(begin, min(begin + batch_size, count)):
                dataset_index = indices[index]
                pixels = dataset.pixels_at(dataset_index)
                if isinstance(pixels, torch.Tensor):
                    prepared_input = prepare_validation_input(pixels, name)
                    pixels = pixels.reshape(-1).tolist() if pixels.numel() == 784 else [0.0] * 784
                else:
                    prepared_input = prepare_validation_input(pixels, name)
                    pixels = [value / 255.0 for value in pixels]
                result = execute(
                    {
                        "pixels": pixels,
                        "model": name,
                        "config": config,
                        "_prepared_input": prepared_input,
                    },
                    collect_telemetry=False,
                )
                if "error" in result:
                    raise RuntimeError(result["error"])
                prediction = max(
                    range(len(classes)),
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
        per_class_accuracy = [
            confusion[class_index][class_index] / label_totals[class_index]
            if label_totals[class_index]
            else None
            for class_index in range(len(classes))
        ]
        initialization_events = sum(
            1 for setup_ms in setup_times if setup_ms > 0)
        summary = {
            "id": job_id,
            "created_at": job["created_at"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "dataset": dataset.dataset,
            "classes": classes,
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
            "per_class_accuracy": per_class_accuracy,
            "latency_ms": latencies,
            "setup_ms": setup_times,
            "predictions": predictions,
            "telemetry": usage_telemetry(
                power_before, telemetry_snapshot(), elapsed),
        }
        RESULTS_DIR.mkdir(exist_ok=True)
        destination = RESULTS_DIR / f"{dataset.dataset}-benchmark-{job_id}.json"
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
        data = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path)
        if self.path == "/api/health":
            capabilities = he_capabilities()
            self.send_json({
                "ok": capabilities["cpu"],
                "he_backends": capabilities,
            })
            return
        if path.path == "/api/mnist/data":
            self.send_json(mnist_data_status())
            return
        if path.path == "/api/dataset/status":
            name = urllib.parse.parse_qs(path.query).get("model", ["trainer:lenet5_mnist"])[0]
            self.send_json(trainer_data_status(name))
            return
        if path.path == "/api/models":
            self.send_json({
                "models": [{"id": name, "label": (name.replace("trainer:", "LeNet trainer: ")
                                                      if is_trainer_model(name) else f"Bundled native: {name}"),
                            "classes": model_classes(name),
                            "preprocessing": model_preprocessing(name),
                            "minimum_ring_dimension": model_minimum_ring_dimension(name)}
                           for name in available_models()],
            })
            return
        if path.path.startswith("/api/benchmark/"):
            job_id = path.path.rsplit("/", 1)[-1]
            if job_id not in benchmark_jobs:
                self.send_json({"error": "benchmark not found"}, 404)
            else:
                self.send_json(benchmark_jobs[job_id])
            return
        if path.path not in ("/", "/index.html"):
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
                    "model": left_config.get("model", "trainer:lenet5_mnist"),
                    "config": left_config,
                })
                right = execute({
                    "pixels": pixels,
                    "model": right_config.get("model", "trainer:lenet5_mnist"),
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
