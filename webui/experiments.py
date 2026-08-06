"""Resumable plaintext/ciphertext sweep runner and research result exports."""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from .benchmarks import (ActivationRecorder, circuit_signature, generate_benchmark_plots,
                             weighted_approximation, wilson_interval, write_activation_samples)
    from .models import (load_data_splits, load_model, load_test_data, polynomial_model,
                         resolve_model, stratified_indices)
    from .polynomials import (ENCRYPTED_DEGREES, METHODS, PLAINTEXT_DEGREES, RANGES,
                              approximation_error, coefficients, fideslib_coefficients,
                              multiplicative_depth)
except ImportError:
    from benchmarks import (ActivationRecorder, circuit_signature, generate_benchmark_plots,
                            weighted_approximation, wilson_interval, write_activation_samples)
    from models import (load_data_splits, load_model, load_test_data, polynomial_model,
                        resolve_model, stratified_indices)
    from polynomials import (ENCRYPTED_DEGREES, METHODS, PLAINTEXT_DEGREES, RANGES,
                             approximation_error, coefficients, fideslib_coefficients,
                             multiplicative_depth)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("SEALTORCH_RESULTS", ROOT / "results"))


class RunCancelled(Exception):
    """Internal control flow for a user-requested cooperative stop."""


def percentile(values: list[float], fraction: float) -> float | None:
    return float(np.percentile(values, fraction * 100)) if values else None


def summarize(values: list[float], prefix: str) -> dict:
    return {
        f"{prefix}_mean_ms": statistics.fmean(values) if values else None,
        f"{prefix}_median_ms": statistics.median(values) if values else None,
        f"{prefix}_p95_ms": percentile(values, 0.95),
        f"{prefix}_q1_ms": percentile(values, 0.25),
        f"{prefix}_q3_ms": percentile(values, 0.75),
    }


def current_resident_memory_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def profile(document: dict, degree: int, overrides: dict | None = None) -> dict:
    layers = document["architecture"]["layers"]
    linear_count = sum(layer["op"] in ("linear", "conv2d", "avg_pool2d") for layer in layers)
    activation_count = sum(layer["op"] in ("relu", "gelu", "tanh") for layer in layers)
    # One final level is consumed while encoding/adding a following layer bias.
    # The former profile was one short for the degree-2 MLP circuit.
    # Degree 2 uses SEALTorch's explicit quadratic; higher degrees use
    # FIDESlib's series evaluator. These depths were verified by real-image
    # agreement probes and by confirming that one fewer level fails.
    activation_depth = multiplicative_depth(degree)
    depth = linear_count + activation_count * activation_depth + 2
    # HYBRID key switching's auxiliary modulus is part of the security bound.
    # OpenFHE requires 64K for the LeNet degree-2 profile at depth 17.
    ring = 16384 if depth <= 11 else (32768 if depth <= 13 else 65536)
    result = {"ring_dim": ring, "depth": depth, "scaling_mod_bits": 25,
              "first_mod_bits": 35, "device": 0, "security": "128-bit classic"}
    if overrides:
        result.update({key: value for key, value in overrides.items() if key in result and key != "security"})
    result["feasible"] = result["depth"] <= 28 and result["ring_dim"] <= 65536
    if not result["feasible"]:
        result["reason"] = "leveled CKKS depth exceeds the no-bootstrapping profile"
    return result


class NativeWorker:
    def __init__(self) -> None:
        binary = Path(os.environ.get("SEALTORCH_HE_BINARY", ROOT / "build" / "sealtorch_he"))
        if not binary.is_file():
            raise RuntimeError(f"encrypted worker is not built: {binary}")
        self.process = subprocess.Popen([str(binary), "--worker"], cwd=ROOT, text=True,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=1)

    def predict(self, request: dict) -> dict:
        if self.process.poll() is not None:
            error = self.process.stderr.read().strip() if self.process.stderr else ""
            raise RuntimeError(error or "encrypted worker stopped")
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        # FIDESlib prints a one-time GPU inventory to stdout while loading its
        # context. Ignore those banner lines; worker responses are JSON objects.
        while True:
            line = self.process.stdout.readline()
            if not line:
                error = self.process.stderr.read().strip() if self.process.stderr else ""
                raise RuntimeError(error or "encrypted worker closed its output")
            if line.lstrip().startswith("{"):
                response = json.loads(line)
                break
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream and not stream.closed:
                stream.close()


def accuracy_and_confusion(model, dataset, device: str, batch_size: int = 512,
                           progress=None):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    classes = model.classifier[-1].out_features if hasattr(model, "classifier") else model.network[-1].out_features
    confusion = torch.zeros(classes, classes, dtype=torch.long)
    correct = total = 0
    model.to(device).eval()
    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(loader, 1):
            predictions = model(images.to(device)).argmax(1).cpu()
            correct += int((predictions == labels).sum()); total += labels.numel()
            confusion += torch.bincount(labels * classes + predictions, minlength=classes * classes).reshape(classes, classes)
            if progress:
                progress(batch_number, len(loader))
    return correct / total, confusion.tolist()


def record_activation_inputs(model, dataset, indices: list[int], device: str,
                             maximum_values: int, progress=None) -> np.ndarray:
    recorder = ActivationRecorder(model, maximum_values)
    loader = DataLoader(Subset(dataset, indices), batch_size=512, num_workers=0)
    model.to(device).eval()
    with torch.inference_mode():
        for batch_number, (images, _) in enumerate(loader, 1):
            model(images.to(device))
            if progress:
                progress(batch_number, len(loader))
    return recorder.close()


def matched_plaintext(model, dataset, indices: list[int], device: str, progress=None,
                      timed_runs: int | None = None, warmup_runs: int = 3):
    model.to(device).eval(); timings, predictions, all_logits = [], [], []
    samples = [dataset[index] for index in indices]
    with torch.inference_mode():
        for image, _ in samples[:warmup_runs]:
            model(image.unsqueeze(0).to(device))
        if device.startswith("cuda"): torch.cuda.synchronize()
        idle_ram = current_resident_memory_bytes()
        idle_vram = int(torch.cuda.memory_allocated(device)) if device.startswith("cuda") else 0
        peak_ram = idle_ram
        if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(device)
        for sample_number, (image, _) in enumerate(samples, 1):
            start = time.perf_counter()
            logits = model(image.unsqueeze(0).to(device))
            if device.startswith("cuda"): torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
            cpu_logits = logits[0].detach().float().cpu().tolist()
            all_logits.append(cpu_logits); predictions.append(int(np.argmax(cpu_logits)))
            peak_ram = max(peak_ram, current_resident_memory_bytes())
            if progress:
                progress(sample_number, len(samples))
    labels = [int(dataset[index][1]) for index in indices]
    timed = timings[:timed_runs] if timed_runs else timings
    result = {
        **summarize(timed, "plaintext"),
        "plaintext_matched_accuracy": sum(a == b for a, b in zip(predictions, labels)) / len(labels),
        "plaintext_idle_ram_bytes": idle_ram,
        "plaintext_peak_ram_bytes": peak_ram,
        "plaintext_ram_bytes": max(0, peak_ram - idle_ram),
        "plaintext_peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0,
        "plaintext_timings_ms": timings, "plaintext_predictions": predictions,
        "plaintext_logits": all_logits,
        "sample_labels": labels, "plaintext_timed_runs": len(timed),
        "plaintext_warmup_runs": min(warmup_runs, len(samples)), "plaintext_batch_size": 1,
    }
    if device.startswith("cuda"):
        result["plaintext_idle_vram_bytes"] = idle_vram
        result["plaintext_vram_bytes"] = max(
            0, int(result["plaintext_peak_vram_bytes"]) - result["plaintext_idle_vram_bytes"])
    else:
        result.update({"plaintext_idle_vram_bytes": 0, "plaintext_vram_bytes": 0})
    return result


@dataclass
class Run:
    run_id: str
    directory: Path
    manifest: dict
    status: str = "queued"
    completed: int = 0
    total: int = 0
    error: str | None = None
    cancel: bool = False
    sweep_id: str | None = None
    gpu: int | None = None
    phase: str = "Waiting in queue"
    detail: str = ""
    phase_completed: int = 0
    phase_total: int = 0
    phase_eta_seconds: float | None = None
    started_at: float | None = None
    phase_started_at: float | None = None
    phase_updated_at: float | None = None
    point_started_at: float | None = None
    point_durations: list[float] = field(default_factory=list, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def public(self) -> dict:
        now = time.time()
        elapsed = now - self.started_at if self.started_at else 0.0
        phase_eta = self.phase_eta_seconds
        if phase_eta is not None and self.phase_updated_at is not None:
            phase_eta = max(0.0, phase_eta - (now - self.phase_updated_at))
        eta = phase_eta
        if self.point_durations and self.total > self.completed:
            remaining = self.total - self.completed
            if phase_eta is None:
                eta = statistics.fmean(self.point_durations) * remaining
            else:
                eta = phase_eta + statistics.fmean(self.point_durations) * max(0, remaining - 1)
        if self.status in ("complete", "partial", "cancelled", "failed"):
            eta = 0.0
        return {"id": self.run_id, "sweep_id": self.sweep_id, "status": self.status,
                "completed": self.completed, "total": self.total, "error": self.error,
                "gpu": self.gpu, "phase": self.phase, "detail": self.detail,
                "phase_completed": self.phase_completed, "phase_total": self.phase_total,
                "elapsed_seconds": elapsed, "eta_seconds": eta,
                "manifest": self.manifest}


@dataclass
class Sweep:
    sweep_id: str
    directory: Path
    run_ids: list[str]
    gpu_count: int
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    error: str | None = None
    cancel: bool = False
    kind: str = "sweep"
    label: str = "Model sweep"

    def public(self, runs: dict[str, Run]) -> dict:
        children = [runs[run_id] for run_id in self.run_ids]
        child_status = [run.public() for run in children]
        completed_runs = sum(run.status in ("complete", "partial") for run in children)
        completed_points = sum(run.completed for run in children)
        total_points = sum(run.total for run in children)
        active = [run for run in children if run.status == "running"]
        queued = [run for run in children if run.status == "queued"]
        elapsed = time.time() - self.created_at
        eta = None
        if completed_points and total_points > completed_points:
            eta = elapsed / completed_points * (total_points - completed_points)
        active_etas = [status["eta_seconds"] for status in child_status
                       if status["status"] == "running" and status["eta_seconds"] is not None]
        if active_etas:
            current_wave = max(active_etas)
            completed_times = [status["elapsed_seconds"] for status in child_status
                               if status["status"] == "complete"]
            typical_run = statistics.median(completed_times) if completed_times else current_wave
            queued_waves = math.ceil(len(queued) / max(1, self.gpu_count))
            eta = current_wave + queued_waves * typical_run
        if self.status in ("complete", "partial", "cancelled", "failed"):
            eta = 0.0
        figure_points = 0
        figure_status = self.directory / "benchmark_plots" / "status.json"
        if self.kind == "benchmark" and figure_status.is_file():
            try:
                figure_points = int(json.loads(figure_status.read_text()).get("points", 0))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return {"id": self.sweep_id, "kind": self.kind, "label": self.label,
                "status": self.status, "gpu_count": self.gpu_count,
                "completed": completed_runs, "total": len(children),
                "completed_points": completed_points, "total_points": total_points,
                "active": len(active), "queued": len(queued), "elapsed_seconds": elapsed,
                "eta_seconds": eta, "figure_points": figure_points,
                "error": self.error, "run_ids": self.run_ids}


class ExperimentManager:
    def __init__(self, results: Path = RESULTS) -> None:
        self.results = results; self.results.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, Run] = {}; self.sweeps: dict[str, Sweep] = {}
        self.lock = threading.Lock(); self.plot_lock = threading.Lock()
        self.sweep_export_lock = threading.Lock()
        self.recommendation_cache: dict[tuple, dict] = {}
        for manifest_path in sorted(self.results.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
                status_path = manifest_path.parent / "status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                run = Run(run_id=manifest["run_id"], directory=manifest_path.parent,
                          manifest=manifest, status=status.get("status", "interrupted"),
                          completed=status.get("completed", 0), total=status.get("total", 0),
                          error=status.get("error"), sweep_id=manifest.get("sweep_id"),
                          phase=status.get("phase", "Interrupted"), detail=status.get("detail", ""))
                if run.status in ("queued", "running"):
                    run.status = "interrupted"
                results_path = manifest_path.parent / "results.json"
                if manifest.get("benchmark") and results_path.exists():
                    legacy_points = json.loads(results_path.read_text())
                    if any(point.get("ciphertext_status") in ("failed", "infeasible", "invalid")
                           for point in legacy_points):
                        run.status = "partial"
                self.runs[run.run_id] = run
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        for manifest_path in sorted((self.results / "_sweeps").glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
                status_path = manifest_path.parent / "status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                sweep = Sweep(manifest["sweep_id"], manifest_path.parent, manifest["run_ids"],
                              int(manifest["gpu_count"]), manifest_path.stat().st_mtime,
                              status.get("status", "interrupted"), status.get("error"), False,
                              manifest.get("kind", "sweep"), manifest.get("label", "Model sweep"))
                if sweep.status in ("queued", "running"):
                    sweep.status = "interrupted"
                if sweep.kind == "benchmark" and any(
                        self.runs.get(run_id) and self.runs[run_id].status == "partial"
                        for run_id in sweep.run_ids):
                    sweep.status = "partial"
                self.sweeps[sweep.sweep_id] = sweep
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue

    def _active(self) -> bool:
        return any(sweep.status in ("queued", "running") for sweep in self.sweeps.values()) or any(
            run.status in ("queued", "running") for run in self.runs.values())

    def recommend_manual_parameters(self, request: dict) -> dict:
        """Measure a model and return an explicit, editable manual-job recommendation."""
        with self.lock:
            if self._active():
                raise RuntimeError(
                    "wait for active runs to finish before measuring an interval; calibration could skew memory data")
        model_id = request.get("model")
        method = request.get("method", "chebyshev")
        degree = int(request.get("degree", 0))
        per_class = int(request.get("activation_samples_per_class", 100))
        maximum_values = int(request.get("activation_sample_limit", 250_000))
        seed = int(request.get("seed", 42))
        if method not in METHODS:
            raise ValueError("method must be Taylor or Chebyshev")
        if not 1 <= degree <= 20:
            raise ValueError("degree must be in [1, 20]")
        if per_class < 1 or maximum_values < 1:
            raise ValueError("activation sampling values must be positive")
        path = resolve_model(model_id)
        cache_key = (str(path), path.stat().st_mtime_ns, method, degree,
                     per_class, maximum_values, seed)
        with self.lock:
            cached = self.recommendation_cache.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

        model, document = load_model(path)
        training, _ = load_data_splits(document)
        indices = stratified_indices(training, len(document["classes"]), per_class, seed=seed)
        inputs = record_activation_inputs(model, training, indices, "cpu", maximum_values)
        if not len(inputs):
            raise ValueError("the selected model produced no activation inputs")
        absolute = np.abs(inputs.astype(np.float64))
        p999 = float(np.quantile(absolute, 0.999))
        if method == "taylor":
            # A Taylor polynomial is invariant to the Chebyshev basis interval;
            # choose observed 99.9% coverage to limit normalized CKKS magnitude.
            interval = max(0.05, min(32.0, p999))
            selected_coefficients = coefficients(document["activation"], method, degree, interval)
            metric = weighted_approximation(
                document["activation"], selected_coefficients, interval, inputs, stored_samples=0)
            candidate_count = 1
            policy = "absolute activation p99.9 (Taylor polynomial is interval-invariant)"
        else:
            quantiles = [float(np.quantile(absolute, fraction)) for fraction in
                         (0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1.0)]
            lower = max(0.05, quantiles[0])
            upper = min(32.0, max(quantiles[-1], p999 * 1.25, lower * 1.01))
            candidates = set(float(value) for value in np.geomspace(lower, upper, 33))
            candidates.update(value for value in quantiles if 0.05 <= value <= 32.0)
            candidates.update(value for value in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
                                                   6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
                              if lower * 0.75 <= value <= upper * 1.25)
            scored = []
            for candidate in sorted(candidates):
                candidate_coefficients = coefficients(
                    document["activation"], method, degree, candidate)
                candidate_metric = weighted_approximation(
                    document["activation"], candidate_coefficients, candidate,
                    inputs, stored_samples=0)
                rmse = candidate_metric["weighted_approximation_rmse"]
                if math.isfinite(rmse):
                    scored.append((rmse, candidate, candidate_coefficients, candidate_metric))
            if not scored:
                raise ValueError("no finite symmetric interval candidate was found")
            _, interval, selected_coefficients, metric = min(scored, key=lambda item: (item[0], item[1]))
            candidate_count = len(scored)
            policy = "minimum activation-weighted RMSE over measured interval candidates"

        selected_profile = profile(document, degree)
        result = {"model": model_id, "method": method, "degree": degree,
            "range": float(interval),
            "he_parameters": {key: int(selected_profile[key]) for key in
                              ("ring_dim", "depth", "scaling_mod_bits", "first_mod_bits")},
            "polynomial_coefficients": selected_coefficients.astype(float).tolist(),
            "interval_policy": policy, "candidate_count": candidate_count,
            "weighted_rmse": metric["weighted_approximation_rmse"],
            "observed_maximum_absolute_error": metric["observed_approximation_max_error"],
            "outside_interval_fraction": metric["activation_outside_range_fraction"],
            "activation_abs_p99": metric["activation_abs_p99"],
            "activation_abs_p999": metric["activation_abs_p999"],
            "activation_input_count": int(len(inputs)), "activation_image_count": len(indices),
            "activation_indices": indices, "seed": seed, "cached": False,
            "profile_policy": "smallest architecture/degree profile currently verified by SEALTorch"}
        with self.lock:
            self.recommendation_cache[cache_key] = result
        return result

    def manual_degree_profiles(self, request: dict) -> dict:
        """Return fixed-interval coefficients and minimum profiles for several degrees."""
        model_id = request.get("model")
        method = request.get("method", "chebyshev")
        interval = float(request.get("range", 0))
        degrees = list(dict.fromkeys(int(value) for value in request.get("degrees", [])))
        if method not in METHODS:
            raise ValueError("method must be Taylor or Chebyshev")
        if not 0 < interval <= 32:
            raise ValueError("range must be in (0, 32]")
        if not degrees or any(not 1 <= degree <= 20 for degree in degrees):
            raise ValueError("degrees must be in [1, 20]")
        path = resolve_model(model_id)
        document = json.loads(path.read_text(encoding="utf-8"))
        profiles = []
        for degree in degrees:
            selected = profile(document, degree)
            polynomial = coefficients(document["activation"], method, degree, interval)
            profiles.append({"degree": degree, "range": interval,
                "polynomial_coefficients": polynomial.astype(float).tolist(),
                "he_parameters": {key: int(selected[key]) for key in
                                  ("ring_dim", "depth", "scaling_mod_bits", "first_mod_bits")}})
        return {"model": model_id, "method": method, "range": interval,
                "profiles": profiles,
                "profile_policy": "smallest architecture/degree profiles currently verified by SEALTorch"}

    @staticmethod
    def _selection(request: dict) -> tuple[list[str], list[float], list[int], list[int]]:
        methods = list(request.get("methods", METHODS))
        ranges = list(map(float, request.get("ranges", RANGES)))
        plaintext = list(map(int, request.get("plaintext_degrees", PLAINTEXT_DEGREES)))
        encrypted = list(map(int, request.get("encrypted_degrees", ENCRYPTED_DEGREES)))
        if not methods or not ranges or not (plaintext or encrypted):
            raise ValueError("select at least one method, range, and degree")
        if any(method not in METHODS for method in methods):
            raise ValueError("unknown approximation method")
        if any(not 0 < interval <= 32 for interval in ranges):
            raise ValueError("ranges must be in (0, 32]")
        if any(degree < 1 or degree > 20 for degree in (*plaintext, *encrypted)):
            raise ValueError("degrees must be in [1, 20]")
        return methods, ranges, plaintext, encrypted

    def start_sweep(self, request: dict) -> Sweep:
        with self.lock:
            if self._active():
                raise RuntimeError("a sweep is already running; its model runs are queued")
            model_ids = list(dict.fromkeys(request.get("models") or [request.get("model")]))
            if not model_ids or any(not model_id for model_id in model_ids):
                raise ValueError("select at least one model artifact")
            methods, ranges, plaintext, encrypted = self._selection(request)
            weight_cache = request.get("weight_cache", "cpu")
            if weight_cache not in ("cpu", "gpu"):
                raise ValueError("weight cache must be cpu or gpu")
            available = torch.cuda.device_count() if torch.cuda.is_available() else 1
            gpu_count = int(request.get("gpu_count", 1))
            if gpu_count < 1 or gpu_count > available:
                raise ValueError(f"GPU count must be between 1 and {available}")
            paths = [(model_id, resolve_model(model_id)) for model_id in model_ids]
            sweep_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
            sweep_directory = self.results / "_sweeps" / sweep_id
            sweep_directory.mkdir(parents=True)
            run_ids = []
            for model_id, path in paths:
                run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
                directory = self.results / run_id; (directory / "points").mkdir(parents=True)
                manifest = {
                    "run_id": run_id, "sweep_id": sweep_id, "model": model_id,
                    "model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "methods": methods, "ranges": ranges, "plaintext_degrees": plaintext,
                    "encrypted_degrees": encrypted, "samples_per_class": 10, "seed": 42,
                    "weight_cache": weight_cache,
                    "parameter_overrides": request.get("he_parameters", {}),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                total = len(methods) * len(ranges) * len(set(plaintext) | set(encrypted))
                run = Run(run_id, directory, manifest, total=total, sweep_id=sweep_id,
                          detail=f"Queued model {len(run_ids) + 1} of {len(paths)}")
                self.runs[run_id] = run; run_ids.append(run_id)
            sweep = Sweep(sweep_id, sweep_directory, run_ids, gpu_count)
            self.sweeps[sweep_id] = sweep
            sweep_manifest = {"sweep_id": sweep_id, "run_ids": run_ids, "models": model_ids,
                              "kind": "sweep", "label": "Model sweep",
                              "gpu_count": gpu_count, "weight_cache": weight_cache,
                              "created_at": time.strftime(
                                  "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (sweep_directory / "manifest.json").write_text(
                json.dumps(sweep_manifest, indent=2) + "\n", encoding="utf-8")
            threading.Thread(target=self._execute_sweep, args=(sweep,), daemon=True).start()
            return sweep

    def start(self, request: dict) -> Run:
        sweep = self.start_sweep({**request, "models": [request["model"]], "gpu_count": 1})
        return self.runs[sweep.run_ids[0]]

    def start_manual_batch(self, request: dict) -> Sweep:
        """Create independently configured jobs with fixed GPU assignments.

        Manual jobs deliberately have no parameter fitting, interval calibration, or
        CKKS retry policy.  The submitted manifest is the experiment contract.
        """
        with self.lock:
            if self._active():
                raise RuntimeError("a batch or sweep is already running")
            jobs = request.get("jobs")
            if not isinstance(jobs, list) or not jobs:
                raise ValueError("manual batch must contain at least one job")
            available = torch.cuda.device_count() if torch.cuda.is_available() else 0
            prepared = []
            required_he = ("ring_dim", "depth", "scaling_mod_bits", "first_mod_bits")
            for position, submitted in enumerate(jobs, 1):
                if not isinstance(submitted, dict):
                    raise ValueError(f"manual job {position} must be an object")
                job = dict(submitted)
                model_id = job.get("model")
                path = resolve_model(model_id)
                method = job.get("method")
                degree = int(job.get("degree", 0))
                interval = float(job.get("range", 0))
                gpu = int(job.get("gpu", -1))
                run_plaintext = bool(job.get("run_plaintext", True))
                run_encrypted = bool(job.get("run_encrypted", True))
                if method not in METHODS:
                    raise ValueError(f"manual job {position}: method must be Taylor or Chebyshev")
                if not 1 <= degree <= 20:
                    raise ValueError(f"manual job {position}: degree must be in [1, 20]")
                if not 0 < interval <= 32:
                    raise ValueError(f"manual job {position}: range must be in (0, 32]")
                if not (run_plaintext or run_encrypted):
                    raise ValueError(f"manual job {position}: enable plaintext and/or encrypted inference")
                if run_encrypted and not run_plaintext:
                    raise ValueError(
                        f"manual job {position}: plaintext is required for the encrypted numerical check")
                if run_encrypted and (available == 0 or gpu < 0 or gpu >= available):
                    raise ValueError(
                        f"manual job {position}: GPU must be between 0 and {max(0, available - 1)}")
                if not run_encrypted and gpu < -1:
                    raise ValueError(f"manual job {position}: plaintext GPU must be -1 (CPU) or a GPU index")
                if not run_encrypted and gpu >= available and gpu != -1:
                    raise ValueError(f"manual job {position}: GPU {gpu} is unavailable")
                he = job.get("he_parameters") or {}
                if run_encrypted:
                    missing = [key for key in required_he if key not in he]
                    if missing:
                        raise ValueError(
                            f"manual job {position}: missing HE parameters: {', '.join(missing)}")
                    he = {key: int(he[key]) for key in required_he}
                    if he["ring_dim"] < 1024 or he["ring_dim"] & (he["ring_dim"] - 1):
                        raise ValueError(f"manual job {position}: ring_dim must be a power of two >= 1024")
                    if he["depth"] < 1 or he["scaling_mod_bits"] < 1 or he["first_mod_bits"] < 1:
                        raise ValueError(f"manual job {position}: HE depth and modulus bits must be positive")
                else:
                    he = {key: int(he[key]) for key in required_he if key in he}
                for key, minimum in (("samples_per_class", 1), ("timed_runs", 1),
                                     ("warmup_runs", 0), ("activation_samples_per_class", 1),
                                     ("activation_sample_limit", 1)):
                    job[key] = int(job.get(key, {"samples_per_class": 100, "timed_runs": 30,
                        "warmup_runs": 5, "activation_samples_per_class": 100,
                        "activation_sample_limit": 1000000}[key]))
                    if job[key] < minimum:
                        raise ValueError(f"manual job {position}: {key} must be >= {minimum}")
                split = job.get("activation_split", "train")
                if split not in ("train", "test"):
                    raise ValueError(f"manual job {position}: activation_split must be train or test")
                weight_cache = job.get("weight_cache", "cpu")
                if weight_cache not in ("cpu", "gpu"):
                    raise ValueError(f"manual job {position}: weight_cache must be cpu or gpu")
                for key in ("polynomial_coefficients", "fideslib_coefficients"):
                    if job.get(key) is not None:
                        values = [float(value) for value in job[key]]
                        if len(values) != degree + 1:
                            raise ValueError(
                                f"manual job {position}: {key} has {len(values)} values; degree "
                                f"{degree} requires {degree + 1}. Regenerate it or set it to null")
                        job[key] = values
                for key in ("accuracy_indices", "activation_indices"):
                    if job.get(key) is not None:
                        values = list(dict.fromkeys(int(value) for value in job[key]))
                        if not values or any(value < 0 for value in values):
                            raise ValueError(f"manual job {position}: {key} must contain nonnegative indices")
                        job[key] = values
                class_count = len(json.loads(path.read_text()).get("classes", range(10)))
                available_timed = (len(job["accuracy_indices"]) if job.get("accuracy_indices") is not None
                                   else job["samples_per_class"] * class_count)
                if job["timed_runs"] > available_timed:
                    raise ValueError(
                        f"manual job {position}: timed_runs ({job['timed_runs']}) exceeds the "
                        f"{available_timed} selected accuracy samples")
                job.update({"name": str(job.get("name") or f"Manual job {position}"),
                            "model": model_id, "method": method, "degree": degree,
                            "range": interval, "gpu": gpu, "run_plaintext": run_plaintext,
                            "run_encrypted": run_encrypted, "he_parameters": he,
                            "activation_split": split, "weight_cache": weight_cache,
                            "seed": int(job.get("seed", 42))})
                prepared.append((job, path))

            batch_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
            batch_directory = self.results / "_sweeps" / batch_id
            batch_directory.mkdir(parents=True)
            run_ids = []
            for position, (job, path) in enumerate(prepared, 1):
                run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
                directory = self.results / run_id
                (directory / "points").mkdir(parents=True)
                degree = job["degree"]
                manifest = {"run_id": run_id, "sweep_id": batch_id, "model": job["model"],
                    "model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "methods": [job["method"]], "ranges": [job["range"]],
                    "plaintext_degrees": [degree] if job["run_plaintext"] else [],
                    "encrypted_degrees": [degree] if job["run_encrypted"] else [],
                    "samples_per_class": job["samples_per_class"], "timed_runs": job["timed_runs"],
                    "seed": job["seed"], "weight_cache": job["weight_cache"],
                    "parameter_overrides": job["he_parameters"], "assigned_gpu": job["gpu"],
                    "manual": job, "created_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                self._atomic_json(directory / "manifest.json", manifest)
                self.runs[run_id] = Run(run_id, directory, manifest, total=1, sweep_id=batch_id,
                    detail=f"Queued manual job {position} of {len(prepared)} on GPU {job['gpu']}")
                run_ids.append(run_id)
            assigned = sorted({job["gpu"] for job, _ in prepared})
            label = str(request.get("label") or "Manual HE batch")
            sweep = Sweep(batch_id, batch_directory, run_ids, len(assigned), kind="manual", label=label)
            self.sweeps[batch_id] = sweep
            self._atomic_json(batch_directory / "manifest.json", {
                "sweep_id": batch_id, "run_ids": run_ids, "kind": "manual", "label": label,
                "gpu_count": len(assigned), "assigned_gpus": assigned,
                "jobs": [job for job, _ in prepared],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            threading.Thread(target=self._execute_sweep, args=(sweep,), daemon=True).start()
            return sweep

    def start_benchmark(self, request: dict) -> Sweep:
        with self.lock:
            if self._active():
                raise RuntimeError("a benchmark or sweep is already running")
            model_id = request.get("model")
            path = resolve_model(model_id)
            document = json.loads(path.read_text())
            mode = request.get("mode", "full")
            if mode not in ("full", "smoke"):
                raise ValueError("benchmark mode must be full or smoke")
            methods = list(request.get("methods", METHODS))
            degrees = list(map(int, request.get("degrees", (2, 4) if mode == "smoke" else (2, 4, 6, 8))))
            interval = float(request.get("range", 4.0))
            if not methods or any(method not in METHODS for method in methods):
                raise ValueError("select Taylor and/or Chebyshev")
            if not degrees or any(degree < 1 or degree > 20 for degree in degrees):
                raise ValueError("benchmark degrees must be in [1, 20]")
            if not 0 < interval <= 32:
                raise ValueError("benchmark range must be in (0, 32]")
            jobs = [(method, degree) for method in methods for degree in dict.fromkeys(degrees)]
            available = torch.cuda.device_count() if torch.cuda.is_available() else 1
            requested_gpus = int(request.get("gpu_count", min(available, len(jobs))))
            if requested_gpus < 1 or requested_gpus > available:
                raise ValueError(f"GPU count must be between 1 and {available}")
            gpu_count = min(requested_gpus, len(jobs))
            class_count = len(document.get("classes", range(10)))
            samples_per_class = int(request.get("samples_per_class", 2 if mode == "smoke" else 100))
            timed_runs = (int(request.get("timed_runs", 5)) if mode == "smoke"
                          else samples_per_class * class_count)
            warmup_runs = int(request.get("warmup_runs", 5))
            calibration_images = int(request.get("calibration_images", 1000 if mode == "smoke" else 5000))
            requested_weight_cache = request.get("weight_cache", "cpu")
            if requested_weight_cache not in ("cpu", "gpu"):
                raise ValueError("weight cache must be cpu or gpu")
            # Publication jobs prioritize bounded VRAM and reproducibility.
            weight_cache = "cpu"
            if not 1 <= samples_per_class <= 100 or not 1 <= timed_runs <= 1000:
                raise ValueError("samples per class must be 1–100")
            if warmup_runs < 1 or calibration_images < class_count:
                raise ValueError("warmups and calibration size must be positive")

            benchmark_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
            directory = self.results / "_sweeps" / benchmark_id; directory.mkdir(parents=True)
            run_ids = []
            benchmark = {"schema_version": 2, "mode": mode, "model": model_id, "dataset": document["dataset"],
                "architecture": document["model"], "activation": document["activation"],
                "methods": methods, "degrees": degrees, "requested_range": interval,
                "range_policy": "training_abs_p999", "interval_percentile": 0.999,
                "samples_per_class": samples_per_class, "timed_runs": timed_runs,
                "warmup_runs": warmup_runs, "calibration_images": calibration_images,
                "calibration_split": "train", "seed": 42,
                "gpu_count": gpu_count, "weight_cache": weight_cache,
                "requested_weight_cache": requested_weight_cache,
                "activation_sample_limit": calibration_images * sum(
                    int(layer.get("out", 0)) for layer in document.get("architecture", {}).get("layers", [])
                    if layer.get("op") == "linear" and int(layer.get("out", 0)) != class_count),
                "failure_policy": "publish_partial", "retry_count": 1}
            for position, (method, degree) in enumerate(jobs, 1):
                run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
                run_directory = self.results / run_id; (run_directory / "points").mkdir(parents=True)
                manifest = {"run_id": run_id, "sweep_id": benchmark_id, "model": model_id,
                    "model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "methods": [method], "ranges": [interval], "plaintext_degrees": [degree],
                    "encrypted_degrees": [degree], "samples_per_class": samples_per_class,
                    "timed_runs": timed_runs, "seed": 42,
                    "weight_cache": weight_cache,
                    "parameter_overrides": request.get("he_parameters", {}), "benchmark": benchmark,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                self._atomic_json(run_directory / "manifest.json", manifest)
                run = Run(run_id, run_directory, manifest, total=1, sweep_id=benchmark_id,
                          detail=f"Queued benchmark point {position} of {len(jobs)}")
                self.runs[run_id] = run; run_ids.append(run_id)
            label = "10-minute smoke benchmark" if mode == "smoke" else "Publication benchmark"
            sweep = Sweep(benchmark_id, directory, run_ids, gpu_count, kind="benchmark", label=label)
            self.sweeps[benchmark_id] = sweep
            sweep_manifest = {"sweep_id": benchmark_id, "run_ids": run_ids, "models": [model_id],
                              "gpu_count": gpu_count, "kind": "benchmark", "label": label,
                              "benchmark": benchmark, "created_at": time.strftime(
                                  "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            self._atomic_json(directory / "manifest.json", sweep_manifest)
            self._atomic_json(directory / "benchmark.json", benchmark)
            threading.Thread(target=self._execute_sweep, args=(sweep,), daemon=True).start()
            return sweep

    def resume(self, run_id: str) -> Run:
        with self.lock:
            if self._active():
                raise RuntimeError("a sweep is already running")
            run = self.runs[run_id]
            if run.status == "complete":
                return run
            run.status, run.error, run.cancel = "queued", None, False
            run.completed = len(list((run.directory / "points").glob("*.json")))
            assigned_gpu = run.manifest.get("assigned_gpu", 0)
            threading.Thread(target=self._execute,
                             args=(run, None if assigned_gpu == -1 else int(assigned_gpu)),
                             daemon=True).start()
            return run

    def cancel(self, run_id: str) -> None:
        with self.lock:
            run = self.runs[run_id]
            if run.status in ("complete", "failed", "cancelled"):
                return
            run.cancel = True
            if run.status == "queued":
                run.status = "cancelled"; run.phase = "Cancelled"
                run.detail = "Removed from the queue by request"
                self._record_event(run, "run_cancelled")

    def cancel_sweep(self, sweep_id: str) -> None:
        with self.lock:
            sweep = self.sweeps[sweep_id]; sweep.cancel = True
            for run_id in sweep.run_ids:
                run = self.runs[run_id]; run.cancel = True
                if run.status == "queued":
                    run.status = "cancelled"; run.phase = "Cancelled"
                    run.detail = "Sweep cancelled before this run started"
                    self._record_event(run, "run_cancelled")

    def delete_run(self, run_id: str) -> None:
        with self.lock:
            run = self.runs[run_id]
            if run.status in ("queued", "running"):
                raise RuntimeError("cancel the run and wait for it to stop before deleting it")
            sweep = self.sweeps.get(run.sweep_id) if run.sweep_id else None
            if sweep and sweep.status in ("queued", "running"):
                raise RuntimeError("cancel the parent sweep and wait for it to stop before deleting this run")
            directory = run.directory.resolve()
            if directory.parent != self.results.resolve():
                raise RuntimeError("refusing to delete a run outside the results directory")
            del self.runs[run_id]
            if sweep:
                sweep.run_ids.remove(run_id)
                if sweep.run_ids:
                    manifest_path = sweep.directory / "manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["run_ids"] = sweep.run_ids
                    self._atomic_json(manifest_path, manifest)
                else:
                    del self.sweeps[sweep.sweep_id]
        shutil.rmtree(directory)
        if sweep:
            if sweep.run_ids:
                with self.sweep_export_lock:
                    self._consolidate_sweep(sweep)
                self._atomic_json(sweep.directory / "status.json", sweep.public(self.runs))
            else:
                sweep_directory = sweep.directory.resolve()
                expected_parent = (self.results / "_sweeps").resolve()
                if sweep_directory.parent != expected_parent:
                    raise RuntimeError("refusing to delete a sweep outside the sweep results directory")
                shutil.rmtree(sweep_directory)

    def delete_sweep(self, sweep_id: str) -> None:
        with self.lock:
            sweep = self.sweeps[sweep_id]
            children = [self.runs[run_id] for run_id in sweep.run_ids]
            if sweep.status in ("queued", "running") or any(
                    run.status in ("queued", "running") for run in children):
                raise RuntimeError("cancel the sweep and wait for every run to stop before deleting it")
            run_directories = [run.directory.resolve() for run in children]
            sweep_directory = sweep.directory.resolve()
            if any(path.parent != self.results.resolve() for path in run_directories):
                raise RuntimeError("refusing to delete a run outside the results directory")
            if sweep_directory.parent != (self.results / "_sweeps").resolve():
                raise RuntimeError("refusing to delete a sweep outside the sweep results directory")
            for run_id in sweep.run_ids:
                del self.runs[run_id]
            del self.sweeps[sweep_id]
        for directory in run_directories:
            shutil.rmtree(directory)
        shutil.rmtree(sweep_directory)

    @staticmethod
    def _atomic_json(path: Path, value) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _record_event(self, run: Run, event: str) -> None:
        snapshot = run.public()
        self._atomic_json(run.directory / "status.json",
                          {key: value for key, value in snapshot.items() if key != "manifest"})
        record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) +
                  f".{int(time.time_ns() % 1_000_000_000):09d}Z", "event": event,
                  **{key: snapshot[key] for key in ("status", "completed", "total", "gpu", "phase",
                     "detail", "phase_completed", "phase_total", "elapsed_seconds", "eta_seconds")}}
        with (run.directory / "events.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
        event_csv = run.directory / "events.csv"
        with event_csv.open("a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, record.keys())
            if event_csv.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(record)

    @staticmethod
    def _environment(device: str) -> dict:
        result = {"python": sys.version, "platform": platform.platform(), "pid": os.getpid(),
                  "torch": torch.__version__, "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available(), "device": device,
                  "cpu_count": os.cpu_count(), "recorded_at": time.strftime(
                      "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if device.startswith("cuda"):
            index = torch.device(device).index or 0
            properties = torch.cuda.get_device_properties(index)
            result["gpu"] = {"index": index, "name": properties.name,
                             "total_memory_bytes": properties.total_memory,
                             "compute_capability": f"{properties.major}.{properties.minor}"}
        return result

    def _phase(self, run: Run, phase: str, detail: str = "",
               completed: int = 0, total: int = 0) -> None:
        with run.lock:
            if run.phase != phase:
                run.phase_started_at = time.time()
            run.phase, run.detail = phase, detail
            run.phase_completed, run.phase_total = completed, total
            run.phase_eta_seconds = None; run.phase_updated_at = time.time()
        self._record_event(run, "phase")

    def _phase_progress(self, run: Run, phase: str, completed: int, total: int,
                        detail: str = "") -> None:
        if run.cancel:
            raise RunCancelled
        with run.lock:
            if run.phase != phase or run.phase_started_at is None:
                run.phase_started_at = time.time()
            run.phase, run.detail = phase, detail
            run.phase_completed, run.phase_total = completed, total
            elapsed = time.time() - run.phase_started_at
            run.phase_eta_seconds = elapsed / completed * (total - completed) if completed else None
            run.phase_updated_at = time.time()
        self._record_event(run, "progress")

    def _execute_sweep(self, sweep: Sweep) -> None:
        sweep.status = "running"
        if sweep.kind == "benchmark":
            try:
                first_run = self.runs[sweep.run_ids[0]]
                benchmark = first_run.manifest["benchmark"]
                model, document = load_model(first_run.manifest["model_path"])
                training, _ = load_data_splits(document)
                class_count = len(document["classes"])
                per_class = int(benchmark["calibration_images"]) // class_count
                calibration_indices = stratified_indices(
                    training, class_count, per_class, seed=int(benchmark.get("seed", 42)))
                inputs = record_activation_inputs(
                    model, training, calibration_indices, "cpu",
                    int(benchmark["activation_sample_limit"]))
                interval = float(np.quantile(np.abs(inputs), benchmark["interval_percentile"]))
                input_path = sweep.directory / "calibration_inputs.npy"
                np.save(input_path, inputs, allow_pickle=False)
                calibration = {"split": "train", "image_count": len(calibration_indices),
                    "activation_input_count": len(inputs), "indices": calibration_indices,
                    "absolute_percentile": benchmark["interval_percentile"],
                    "calibrated_interval": interval, "input_path": str(input_path)}
                self._atomic_json(sweep.directory / "calibration.json", calibration)
                for run_id in sweep.run_ids:
                    run = self.runs[run_id]
                    run.manifest["calibration"] = calibration
                    run.manifest["calibrated_interval"] = interval
                    run.manifest["ranges"] = [interval]
                    self._atomic_json(run.directory / "manifest.json", run.manifest)
            except Exception as error:
                sweep.status = "failed"; sweep.error = f"calibration failed: {error}"
                self._atomic_json(sweep.directory / "status.json", sweep.public(self.runs))
                return
        if sweep.kind == "manual":
            queues = {}
            for run_id in sweep.run_ids:
                assigned_gpu = int(self.runs[run_id].manifest["assigned_gpu"])
                queues.setdefault(assigned_gpu, deque()).append(run_id)
        else:
            queues = {gpu: deque() for gpu in range(sweep.gpu_count)}
            for position, run_id in enumerate(sweep.run_ids):
                queues[position % sweep.gpu_count].append(run_id)

        def gpu_worker(gpu: int) -> None:
            queue = queues[gpu]
            while True:
                if not queue or sweep.cancel:
                    return
                run_id = queue.popleft()
                for position, queued_id in enumerate(queue, 1):
                    self.runs[queued_id].detail = f"GPU {gpu} queue position {position}"
                if self.runs[run_id].cancel:
                    continue
                self._execute(self.runs[run_id], None if gpu == -1 else gpu)
                if sweep.kind in ("benchmark", "manual"):
                    try:
                        with self.sweep_export_lock:
                            self._consolidate_sweep(sweep)
                    except Exception as error:
                        sweep.error = f"live figure update failed: {error}"

        workers = [threading.Thread(target=gpu_worker, args=(gpu,), daemon=True)
                   for gpu in queues]
        for worker in workers: worker.start()
        for worker in workers: worker.join()
        if sweep.cancel:
            for queue in queues.values():
                for run_id in list(queue):
                    self.runs[run_id].status = "cancelled"
            sweep.status = "cancelled"
        else:
            statuses = [self.runs[run_id].status for run_id in sweep.run_ids]
            sweep.status = "failed" if "failed" in statuses else (
                "cancelled" if "cancelled" in statuses else (
                "partial" if "partial" in statuses else "complete"))
        try:
            with self.sweep_export_lock:
                self._consolidate_sweep(sweep)
        except Exception as error:
            sweep.status = "failed"; sweep.error = f"final export failed: {error}"
        (sweep.directory / "status.json").write_text(
            json.dumps(sweep.public(self.runs), indent=2) + "\n", encoding="utf-8")

    def _write_point(self, run: Run, point: dict) -> None:
        key = f'{point["method"]}-r{point["range"]:g}-d{point["degree"]}'
        self._atomic_json(run.directory / "points" / f"{key}.json", point)
        run.completed += 1
        self._consolidate(run)
        self._record_event(run, "point_complete")

    def _consolidate(self, run: Run, full_exports: bool = False) -> None:
        points = [json.loads(path.read_text()) for path in sorted((run.directory / "points").glob("*.json"))]
        self._write_tables(run.directory, points, run.manifest.get("sample_indices", []), full_exports)

    @staticmethod
    def _write_tables(directory: Path, points: list[dict], sample_indices: list[int] | None = None,
                      full_exports: bool = True) -> None:
        summaries = [{key: value for key, value in point.items() if not isinstance(value, (dict, list))}
                     for point in points]
        (directory / "results.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        if full_exports:
            (directory / "raw_results.json").write_text(
                json.dumps(points, indent=2) + "\n", encoding="utf-8")
        fields = sorted({key for point in points for key, value in point.items() if not isinstance(value, (dict, list))})
        with (directory / "results.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fields); writer.writeheader()
            for point in points: writer.writerow({key: point.get(key) for key in fields})

        accuracy_fields = ["run_id", "dataset", "model", "activation", "method", "degree",
                           "encryption_mode", "correct_predictions", "total_predictions",
                           "accuracy", "wilson_95_low", "wilson_95_high"]
        with (directory / "accuracy.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, accuracy_fields); writer.writeheader()
            for point in points:
                common = {key: point.get(key) for key in accuracy_fields[:6]}
                for mode, prefix, accuracy_key in (
                        ("original", "baseline", "baseline_accuracy"),
                        ("plaintext", "plaintext", "plaintext_polynomial_accuracy"),
                        ("encrypted", "ciphertext", "ciphertext_accuracy")):
                    if point.get(accuracy_key) is None:
                        continue
                    writer.writerow({**common, "encryption_mode": mode,
                        "correct_predictions": point.get(f"{prefix}_correct_predictions"),
                        "total_predictions": point.get(f"{prefix}_total_predictions"),
                        "accuracy": point[accuracy_key],
                        "wilson_95_low": point.get(f"{prefix}_wilson_95_low"),
                        "wilson_95_high": point.get(f"{prefix}_wilson_95_high")})

        def context(point: dict) -> dict:
            he = point.get("he_profile") or point.get("he_request") or {}
            return {"run_id": point.get("run_id"), "manual_job_name": point.get("manual_job_name"),
                "dataset": point.get("dataset"), "model": point.get("model"),
                "activation": point.get("activation"), "method": point.get("method"),
                "degree": point.get("degree"), "range": point.get("range"),
                "assigned_gpu": point.get("assigned_gpu"), "weight_cache": point.get("weight_cache"),
                "ring_dim": he.get("ring_dim"), "depth": he.get("depth"),
                "scaling_mod_bits": he.get("scaling_mod_bits"),
                "first_mod_bits": he.get("first_mod_bits")}

        def write_rows(name: str, rows: list[dict]) -> None:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (directory / name).open("w", newline="", encoding="utf-8") as output:
                if not fields:
                    output.write("")
                    return
                writer = csv.DictWriter(output, fields); writer.writeheader(); writer.writerows(rows)

        accuracy_rows = []
        for point in points:
            for mode, prefix, key in (("original", "baseline", "baseline_accuracy"),
                                      ("plaintext", "plaintext", "plaintext_polynomial_accuracy"),
                                      ("encrypted", "ciphertext", "ciphertext_accuracy")):
                if point.get(key) is not None:
                    accuracy_rows.append({**context(point), "encryption_mode": mode,
                        "correct_predictions": point.get(f"{prefix}_correct_predictions"),
                        "total_predictions": point.get(f"{prefix}_total_predictions"),
                        "accuracy_fraction": point[key], "accuracy_percent": 100 * point[key],
                        "wilson_95_low_fraction": point.get(f"{prefix}_wilson_95_low"),
                        "wilson_95_high_fraction": point.get(f"{prefix}_wilson_95_high")})
        write_rows("accuracy_data.csv", accuracy_rows)

        error_rows = [{**context(point),
            "weighted_rmse": point.get("weighted_approximation_rmse"),
            "maximum_absolute_error_observed": point.get("observed_approximation_max_error"),
            "uniform_grid_rmse": point.get("approximation_rmse"),
            "uniform_grid_maximum_absolute_error": point.get("approximation_max_error"),
            "activation_input_count": point.get("activation_input_count"),
            "activation_outside_range_fraction": point.get("activation_outside_range_fraction"),
            "activation_abs_p99": point.get("activation_abs_p99"),
            "activation_abs_p999": point.get("activation_abs_p999")}
            for point in points]
        write_rows("approximation_error_data.csv", error_rows)

        timing_rows = []
        for point in points:
            for mode, prefix in (("plaintext", "plaintext"), ("encrypted", "encrypted")):
                if point.get(f"{prefix}_median_ms") is not None:
                    timing_rows.append({**context(point), "encryption_mode": mode,
                        "batch_size": point.get(f"{prefix}_batch_size", 1),
                        "warmup_runs": point.get(f"{prefix}_warmup_runs"),
                        "timed_runs": point.get(f"{prefix}_timed_runs"),
                        "mean_ms": point.get(f"{prefix}_mean_ms"),
                        "median_ms": point.get(f"{prefix}_median_ms"),
                        "q1_ms": point.get(f"{prefix}_q1_ms"),
                        "q3_ms": point.get(f"{prefix}_q3_ms"),
                        "p95_ms": point.get(f"{prefix}_p95_ms"),
                        "encrypt_median_ms": point.get("encrypt_median_ms") if mode == "encrypted" else None,
                        "evaluate_median_ms": point.get("evaluate_median_ms") if mode == "encrypted" else None,
                        "decrypt_median_ms": point.get("decrypt_median_ms") if mode == "encrypted" else None})
        write_rows("inference_time_data.csv", timing_rows)

        memory_rows = []
        for point in points:
            for mode, prefix in (("plaintext", "plaintext"), ("encrypted", "encrypted")):
                if point.get(f"{prefix}_peak_ram_bytes") is not None:
                    memory_rows.append({**context(point), "encryption_mode": mode,
                        "batch_size": point.get(f"{prefix}_batch_size", 1),
                        "idle_ram_bytes": point.get(f"{prefix}_idle_ram_bytes"),
                        "peak_ram_bytes": point.get(f"{prefix}_peak_ram_bytes"),
                        "workload_ram_bytes": point.get(f"{prefix}_ram_bytes"),
                        "idle_ram_gib": (point.get(f"{prefix}_idle_ram_bytes") or 0) / 2**30,
                        "peak_ram_gib": (point.get(f"{prefix}_peak_ram_bytes") or 0) / 2**30,
                        "workload_ram_gib": (point.get(f"{prefix}_ram_bytes") or 0) / 2**30,
                        "idle_vram_bytes": point.get(f"{prefix}_idle_vram_bytes"),
                        "peak_vram_bytes": point.get(f"{prefix}_peak_vram_bytes"),
                        "workload_vram_bytes": point.get(f"{prefix}_vram_bytes"),
                        "idle_vram_gib": (point.get(f"{prefix}_idle_vram_bytes") or 0) / 2**30,
                        "peak_vram_gib": (point.get(f"{prefix}_peak_vram_bytes") or 0) / 2**30,
                        "workload_vram_gib": (point.get(f"{prefix}_vram_bytes") or 0) / 2**30})
        write_rows("memory_data.csv", memory_rows)

        metrics = ("baseline_accuracy", "plaintext_polynomial_accuracy", "ciphertext_accuracy",
                   "baseline_wilson_95_low", "baseline_wilson_95_high",
                   "plaintext_wilson_95_low", "plaintext_wilson_95_high",
                   "ciphertext_wilson_95_low", "ciphertext_wilson_95_high",
                   "plaintext_mean_ms", "plaintext_median_ms", "plaintext_q1_ms", "plaintext_q3_ms",
                   "encrypted_mean_ms", "encrypted_median_ms", "encrypted_q1_ms", "encrypted_q3_ms",
                   "encrypt_median_ms", "evaluate_median_ms", "decrypt_median_ms", "slowdown_ratio",
                   "approximation_rmse", "approximation_max_error", "plaintext_ram_bytes",
                   "weighted_approximation_rmse", "observed_approximation_max_error",
                   "activation_input_count", "activation_outside_range_fraction",
                   "activation_abs_p99", "activation_abs_p999",
                   "encrypted_ram_bytes", "plaintext_vram_bytes", "encrypted_vram_bytes",
                   "encrypted_peak_vram_bytes", "encrypted_peak_ram_bytes",
                   "plaintext_peak_ram_bytes", "plaintext_peak_vram_bytes", "plaintext_idle_ram_bytes",
                   "plaintext_idle_vram_bytes", "encrypted_idle_ram_bytes", "encrypted_idle_vram_bytes")
        figure_fields = ["dataset", "model", "activation", "method", "range", "degree", "metric", "value", "run_id"]
        with (directory / "figure_data.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, figure_fields); writer.writeheader()
            for point in points:
                for metric in metrics:
                    if point.get(metric) is not None:
                        writer.writerow({**{key: point.get(key) for key in figure_fields[:-3]},
                                         "metric": metric, "value": point[metric],
                                         "run_id": point.get("run_id")})

        if not full_exports:
            return

        activation_fields = ["run_id", "dataset", "model", "activation", "method", "range",
                             "degree", "sample", "activation_input", "original_output",
                             "polynomial_output", "error"]
        with (directory / "activation_samples.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, activation_fields); writer.writeheader()
            for point in points:
                inputs = point.get("sampled_activation_inputs", [])
                originals = point.get("sampled_original_activation_outputs", [])
                approximations = point.get("sampled_polynomial_outputs", [])
                for position, (value, original, approximation) in enumerate(
                        zip(inputs, originals, approximations)):
                    writer.writerow({"run_id": point.get("run_id"), "dataset": point.get("dataset"),
                        "model": point.get("model"), "activation": point.get("activation"),
                        "method": point.get("method"), "range": point.get("range"),
                        "degree": point.get("degree"), "sample": position,
                        "activation_input": value, "original_output": original,
                        "polynomial_output": approximation, "error": approximation - original})

        sample_fields = ["run_id", "dataset", "model", "activation", "method", "range", "degree",
                         "batch_size",
                         "sample_position", "dataset_index", "label", "plaintext_prediction",
                         "plaintext_ms", "plaintext_logits", "ciphertext_prediction",
                         "ciphertext_logits", "encrypted_ms", "encrypt_ms",
                         "evaluate_ms", "decrypt_ms", "encrypted_response_json"]
        with (directory / "samples.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, sample_fields); writer.writeheader()
            for point in points:
                arrays = [point.get(key, []) for key in ("sample_labels", "plaintext_timings_ms",
                    "plaintext_predictions", "ciphertext_predictions", "encrypted_timings_ms",
                    "encrypt_timings_ms", "evaluate_timings_ms", "decrypt_timings_ms",
                    "plaintext_logits", "ciphertext_logits", "encrypted_responses")]
                count = max(map(len, arrays), default=0)
                for position in range(count):
                    def item(values): return values[position] if position < len(values) else None
                    writer.writerow({"run_id": point.get("run_id"), "dataset": point.get("dataset"),
                        "model": point.get("model"), "activation": point.get("activation"),
                        "method": point.get("method"), "range": point.get("range"),
                        "degree": point.get("degree"), "batch_size": 1, "sample_position": position,
                        "dataset_index": (point.get("sample_indices") or sample_indices or [None] * count)[position],
                        "label": item(arrays[0]), "plaintext_ms": item(arrays[1]),
                        "plaintext_prediction": item(arrays[2]), "ciphertext_prediction": item(arrays[3]),
                        "encrypted_ms": item(arrays[4]), "encrypt_ms": item(arrays[5]),
                        "evaluate_ms": item(arrays[6]), "decrypt_ms": item(arrays[7]),
                        "plaintext_logits": json.dumps(item(arrays[8])) if item(arrays[8]) is not None else None,
                        "ciphertext_logits": json.dumps(item(arrays[9])) if item(arrays[9]) is not None else None,
                        "encrypted_response_json": json.dumps(item(arrays[10])) if item(arrays[10]) is not None else None})

        logit_fields = ["run_id", "dataset", "model", "activation", "method", "range", "degree",
                        "sample_position", "dataset_index", "label", "class_index",
                        "plaintext_logit", "ciphertext_logit", "absolute_logit_error"]
        with (directory / "logits.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, logit_fields); writer.writeheader()
            for point in points:
                plaintext = point.get("plaintext_logits", []); ciphertext = point.get("ciphertext_logits", [])
                labels = point.get("sample_labels", []); indices = point.get("sample_indices") or sample_indices or []
                for sample_position in range(max(len(plaintext), len(ciphertext))):
                    plain = plaintext[sample_position] if sample_position < len(plaintext) else []
                    cipher = ciphertext[sample_position] if sample_position < len(ciphertext) else []
                    for class_index in range(max(len(plain), len(cipher))):
                        plain_value = plain[class_index] if class_index < len(plain) else None
                        cipher_value = cipher[class_index] if class_index < len(cipher) else None
                        writer.writerow({"run_id": point.get("run_id"), "dataset": point.get("dataset"),
                            "model": point.get("model"), "activation": point.get("activation"),
                            "method": point.get("method"), "range": point.get("range"),
                            "degree": point.get("degree"), "sample_position": sample_position,
                            "dataset_index": indices[sample_position] if sample_position < len(indices) else None,
                            "label": labels[sample_position] if sample_position < len(labels) else None,
                            "class_index": class_index, "plaintext_logit": plain_value,
                            "ciphertext_logit": cipher_value,
                            "absolute_logit_error": (abs(plain_value - cipher_value)
                                if plain_value is not None and cipher_value is not None else None)})

    def _consolidate_sweep(self, sweep: Sweep) -> None:
        points = []
        indices = []
        for run_id in sweep.run_ids:
            run = self.runs[run_id]
            path = run.directory / "raw_results.json"
            if not path.exists(): continue
            for point in json.loads(path.read_text()):
                points.append({**point, "run_id": run_id})
            if not indices:
                indices = run.manifest.get("sample_indices", [])
        self._write_tables(sweep.directory, points, indices, True)
        if points:
            with self.plot_lock:
                if sweep.kind == "benchmark":
                    generate_benchmark_plots(sweep.directory)
                generate_plots(sweep.directory)

    def _execute(self, run: Run, gpu: int | None = None) -> None:
        worker = None
        try:
            if run.cancel:
                raise RunCancelled
            run.status = "running"; run.gpu = gpu; run.started_at = time.time()
            run.completed = 0
            self._phase(run, "Loading model", "Reading the trained artifact")
            base_model, document = load_model(run.manifest["model_path"])
            self._phase(run, "Loading dataset", f'Preparing the {document["dataset"]} data splits')
            benchmark = run.manifest.get("benchmark")
            manual = run.manifest.get("manual")
            if benchmark or manual:
                training_dataset, dataset = load_data_splits(document)
            else:
                training_dataset, dataset = None, load_test_data(document)
            samples_per_class = int(run.manifest.get("samples_per_class", 10))
            self._phase(run, "Selecting validation samples",
                        f"Taking {samples_per_class} deterministic samples per class")
            seed = int(run.manifest.get("seed", 42))
            if manual and manual.get("accuracy_indices") is not None:
                indices = list(manual["accuracy_indices"])
                if any(index >= len(dataset) for index in indices):
                    raise ValueError("an accuracy index is outside the test dataset")
            else:
                indices = stratified_indices(dataset, len(document["classes"]), samples_per_class,
                                              seed=seed if benchmark or manual else None)
            run.manifest["sample_indices"] = indices
            activation_inputs = None
            if benchmark and run.manifest.get("calibration"):
                calibration = run.manifest["calibration"]
                activation_inputs = np.load(calibration["input_path"], allow_pickle=False)
                interval = float(calibration["calibrated_interval"])
                run.manifest["calibration_indices"] = calibration["indices"]
                run.manifest["calibrated_interval"] = interval
                run.manifest["ranges"] = [interval]
            elif benchmark and training_dataset is not None:
                per_class = int(benchmark["calibration_images"]) // len(document["classes"])
                calibration_indices = stratified_indices(
                    training_dataset, len(document["classes"]), per_class, seed=seed)
                self._phase(run, "Activation calibration",
                            f"Recording {len(calibration_indices)} stratified training images")
                activation_inputs = record_activation_inputs(
                    base_model, training_dataset, calibration_indices,
                    f"cuda:{gpu}" if torch.cuda.is_available() and gpu is not None else "cpu",
                    int(benchmark["activation_sample_limit"]),
                    progress=lambda done, total: self._phase_progress(
                        run, "Activation calibration", done, total,
                        f"Calibration batch {done} of {total}"))
                interval = float(np.quantile(np.abs(activation_inputs), benchmark["interval_percentile"]))
                run.manifest["calibration_indices"] = calibration_indices
                run.manifest["calibrated_interval"] = interval
                run.manifest["ranges"] = [interval]
            elif manual:
                activation_dataset = training_dataset if manual["activation_split"] == "train" else dataset
                if manual.get("activation_indices") is not None:
                    activation_indices = list(manual["activation_indices"])
                    if any(index >= len(activation_dataset) for index in activation_indices):
                        raise ValueError("an activation index is outside the selected dataset split")
                else:
                    activation_indices = stratified_indices(
                        activation_dataset, len(document["classes"]),
                        int(manual["activation_samples_per_class"]), seed=seed)
                self._phase(run, "Activation sampling",
                            f"Recording fixed {manual['activation_split']} activation inputs from "
                            f"{len(activation_indices)} images")
                activation_inputs = record_activation_inputs(
                    base_model, activation_dataset, activation_indices,
                    f"cuda:{gpu}" if torch.cuda.is_available() and gpu is not None else "cpu",
                    int(manual["activation_sample_limit"]),
                    progress=lambda done, total: self._phase_progress(
                        run, "Activation sampling", done, total,
                        f"Activation batch {done} of {total}"))
                activation_path = run.directory / "activation_inputs.npy"
                np.save(activation_path, activation_inputs, allow_pickle=False)
                run.manifest["activation_indices"] = activation_indices
                run.manifest["activation_input_path"] = str(activation_path)
                run.manifest["activation_input_count"] = len(activation_inputs)
            self._atomic_json(run.directory / "manifest.json", run.manifest)
            plain_degrees = set(map(int, run.manifest["plaintext_degrees"])); encrypted_degrees = set(map(int, run.manifest["encrypted_degrees"]))
            degrees = sorted(plain_degrees | encrypted_degrees)
            run.total = len(run.manifest["methods"]) * len(run.manifest["ranges"]) * len(degrees)
            device = f"cuda:{gpu}" if torch.cuda.is_available() and gpu is not None else "cpu"
            self._atomic_json(run.directory / "environment.json", self._environment(device))
            self._phase(run, "Baseline inference", "Evaluating the original activation on the shared test subset")
            baseline_started = time.perf_counter()
            evaluation_dataset = Subset(dataset, indices) if benchmark or manual else dataset
            baseline_accuracy, baseline_confusion = accuracy_and_confusion(base_model, evaluation_dataset, device,
                progress=lambda done, total: self._phase_progress(run, "Baseline inference", done, total,
                    f"Original model batch {done} of {total}"))
            baseline_correct = sum(baseline_confusion[index][index] for index in range(len(baseline_confusion)))
            baseline_total = len(evaluation_dataset)
            baseline_ci = wilson_interval(baseline_correct, baseline_total)
            (run.directory / "baseline.json").write_text(json.dumps({
                "accuracy": baseline_accuracy, "confusion_matrix": baseline_confusion,
                "correct_predictions": baseline_correct, "total_predictions": baseline_total,
                "wilson_95_low": baseline_ci[0], "wilson_95_high": baseline_ci[1],
                "inference_seconds": time.perf_counter() - baseline_started,
                "dataset": document["dataset"], "model": document["model"],
                "activation": document["activation"]}, indent=2) + "\n", encoding="utf-8")

            for method in run.manifest["methods"]:
                for interval in map(float, run.manifest["ranges"]):
                    for degree in degrees:
                        if run.cancel:
                            run.status = "cancelled"; self._phase(run, "Cancelled", "Stopped by request"); return
                        point_path = run.directory / "points" / f"{method}-r{interval:g}-d{degree}.json"
                        if point_path.exists():
                            existing_point = json.loads(point_path.read_text())
                            if existing_point.get("status") == "complete" and not (
                                    degree in encrypted_degrees and
                                    existing_point.get("ciphertext_status") != "complete"):
                                run.completed += 1
                                continue
                        run.point_started_at = time.time()
                        point_label = f"{method.title()} · ±{interval:g} · degree {degree}"
                        point = {"dataset": document["dataset"], "model": document["model"],
                                 "activation": document["activation"], "method": method,
                                 "range": interval, "degree": degree, "sample_count": len(indices),
                                 "sample_indices": indices,
                                 "run_id": run.run_id,
                                 "manual_job_name": manual.get("name") if manual else None,
                                 "assigned_gpu": manual.get("gpu") if manual else gpu,
                                 "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "baseline_accuracy": baseline_accuracy,
                                 "baseline_correct_predictions": baseline_correct,
                                 "baseline_total_predictions": baseline_total,
                                 "baseline_wilson_95_low": baseline_ci[0],
                                 "baseline_wilson_95_high": baseline_ci[1],
                                 "weight_cache": run.manifest.get("weight_cache", "cpu"),
                                 "status": "complete"}
                        if method == "taylor" and document["activation"] == "relu":
                            self._phase(run, "Skipping unsupported point",
                                point_label)
                            point.update({"status": "unsupported", "reason": "Taylor series is undefined for ReLU at zero"})
                            point["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                            point["point_seconds"] = time.time() - run.point_started_at
                            self._write_point(run, point)
                            continue
                        self._phase(run, "Polynomial approximation",
                            f"{point_label} · point {run.completed + 1}/{run.total}")
                        if manual and manual.get("polynomial_coefficients") is not None:
                            coeff = np.asarray(manual["polynomial_coefficients"], dtype=float)
                            point["polynomial_coefficient_source"] = "manual"
                        else:
                            coeff = coefficients(document["activation"], method, degree, interval)
                            point["polynomial_coefficient_source"] = "generated"
                        point["polynomial_coefficients"] = coeff.tolist()
                        if manual and manual.get("fideslib_coefficients") is not None:
                            he_coeff = np.asarray(manual["fideslib_coefficients"], dtype=float)
                            point["fideslib_coefficient_source"] = "manual"
                        else:
                            he_coeff = fideslib_coefficients(coeff)
                            point["fideslib_coefficient_source"] = "converted"
                        point["fideslib_polynomial_coefficients"] = he_coeff.tolist()
                        point.update(approximation_error(document["activation"], coeff, interval))
                        if activation_inputs is not None:
                            point.update(weighted_approximation(
                                document["activation"], coeff, interval, activation_inputs))
                        if degree in plain_degrees:
                            self._phase(run, "Plaintext polynomial inference",
                                f"{point_label} · shared {len(indices)}-image test subset")
                            poly = polynomial_model(base_model, coeff, interval)
                            point["plaintext_idle_ram_bytes"] = current_resident_memory_bytes()
                            point["plaintext_idle_vram_bytes"] = (
                                torch.cuda.memory_allocated(device) if device.startswith("cuda") else 0)
                            plaintext_accuracy_started = time.perf_counter()
                            point["plaintext_polynomial_accuracy"], point["plaintext_confusion_matrix"] = accuracy_and_confusion(
                                poly, evaluation_dataset, device, progress=lambda done, total: self._phase_progress(
                                    run, "Plaintext polynomial inference", done, total,
                                    f"{point_label} · full test-set batch {done} of {total}"))
                            point["plaintext_correct_predictions"] = sum(
                                point["plaintext_confusion_matrix"][index][index]
                                for index in range(len(point["plaintext_confusion_matrix"])))
                            point["plaintext_total_predictions"] = len(indices)
                            plain_ci = wilson_interval(point["plaintext_correct_predictions"], len(indices))
                            point["plaintext_wilson_95_low"], point["plaintext_wilson_95_high"] = plain_ci
                            point["plaintext_accuracy_seconds"] = time.perf_counter() - plaintext_accuracy_started
                            self._phase(run, "Plaintext timing", f"{point_label} · {len(indices)} matched samples")
                            point.update(matched_plaintext(poly, dataset, indices, device,
                                progress=lambda done, total: self._phase_progress(run, "Plaintext timing",
                                    done, total, f"{point_label} · matched sample {done} of {total}"),
                                timed_runs=run.manifest.get("timed_runs"),
                                warmup_runs=int((manual or benchmark or {}).get("warmup_runs", 3))))
                            point["circuit_signature"] = circuit_signature(
                                document["model"], degree, coeff)
                        if degree in encrypted_degrees:
                            if manual:
                                he_profile = {**manual["he_parameters"], "device": gpu,
                                    "security": "validated by OpenFHE at context creation", "feasible": True}
                            else:
                                he_profile = profile(document, degree, run.manifest["parameter_overrides"])
                            if gpu is not None: he_profile["device"] = gpu
                            point["he_profile"] = he_profile
                            if not he_profile["feasible"]:
                                point.update({"ciphertext_status": "infeasible", "ciphertext_reason": he_profile["reason"]})
                            else:
                                try:
                                    self._phase(run, "Creating CKKS context",
                                        f"{point_label} · GPU {gpu} · ring {he_profile['ring_dim']} · depth {he_profile['depth']}")
                                    request = {"model_path": run.manifest["model_path"], "coefficients": he_coeff.tolist(),
                                               "lower_bound": -interval, "upper_bound": interval,
                                               "stream_weights": 1 if run.manifest.get("weight_cache", "cpu") == "cpu" else 0,
                                               **{key: he_profile[key] for key in ("ring_dim", "depth", "scaling_mod_bits", "first_mod_bits", "device")}}
                                    point["he_request"] = request
                                    first_image = dataset[indices[0]][0].reshape(-1).tolist()
                                    retry_history = []
                                    for attempt in range(1 if manual else 2):
                                        try:
                                            worker = worker or NativeWorker()
                                            warmup = worker.predict({**request, "pixels": first_image})
                                            break
                                        except Exception as preflight_error:
                                            retry_history.append({"attempt": attempt + 1,
                                                "depth": request["depth"], "error": str(preflight_error)})
                                            if worker:
                                                worker.close(); worker = None
                                            error_text = str(preflight_error).lower()
                                            level_error = ("level" in error_text or
                                                           "too few towers" in error_text)
                                            if not manual and attempt == 0 and level_error:
                                                request["depth"] += 1
                                                he_profile["depth"] = request["depth"]
                                                if request["depth"] > 11 and request["ring_dim"] < 32768:
                                                    request["ring_dim"] = 32768
                                                    he_profile["ring_dim"] = 32768
                                                continue
                                            raise
                                    point["he_retry_history"] = retry_history
                                    point["he_request"] = request
                                    expected_preflight = np.asarray(point.get("plaintext_logits", [[]])[0], dtype=float)
                                    actual_preflight = np.asarray(warmup.get("logits", []), dtype=float)
                                    if expected_preflight.size != actual_preflight.size:
                                        raise RuntimeError("encrypted numerical preflight returned the wrong logit count")
                                    preflight_max_error = float(np.max(np.abs(
                                        expected_preflight - actual_preflight)))
                                    point["encrypted_preflight_max_logit_error"] = preflight_max_error
                                    if (preflight_max_error > 0.02 or
                                            int(np.argmax(expected_preflight)) != int(np.argmax(actual_preflight))):
                                        raise RuntimeError(
                                            f"encrypted numerical preflight failed: max logit error {preflight_max_error:.6g}")
                                    warmup_count = int((manual or benchmark or {}).get("warmup_runs", 3))
                                    point["encrypted_setup_ms"] = warmup.get("setup_ms", 0.0)
                                    warmup_responses = [warmup] if warmup_count else []
                                    for warmup_number in range(1, warmup_count):
                                        self._phase_progress(run, "Warming encrypted inference", warmup_number, warmup_count,
                                            f"{point_label} · warmup {warmup_number + 1} of {warmup_count}")
                                        warmup_responses.append(worker.predict({**request, "pixels": first_image}))
                                    checkpoint_path = run.directory / "points" / f".{method}-d{degree}.checkpoint.json"
                                    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
                                    timings = checkpoint.get("encrypted_timings_ms", [])
                                    encrypt = checkpoint.get("encrypt_timings_ms", [])
                                    evaluate_times = checkpoint.get("evaluate_timings_ms", [])
                                    decrypt = checkpoint.get("decrypt_timings_ms", [])
                                    predictions = checkpoint.get("ciphertext_predictions", [])
                                    ciphertext_logits = checkpoint.get("ciphertext_logits", [])
                                    native_responses = checkpoint.get("encrypted_responses", [])
                                    ram = int(checkpoint.get("encrypted_peak_ram_bytes", 0))
                                    vram = int(checkpoint.get("encrypted_vram_bytes", 0))
                                    idle_ram = int(warmup.get("idle_ram_bytes", 0))
                                    idle_vram = int(warmup.get("idle_vram_bytes", 0))
                                    labels = [int(dataset[index][1]) for index in indices]
                                    self._phase(run, "Encrypted inference", f"{point_label} · {len(indices)} validation samples")
                                    for sample_number, index in enumerate(indices[len(predictions):], len(predictions) + 1):
                                        response = worker.predict({**request, "pixels": dataset[index][0].reshape(-1).tolist()})
                                        native_responses.append(response); ciphertext_logits.append(response["logits"])
                                        predictions.append(int(np.argmax(response["logits"])))
                                        timings.append(response["total_ms"]); encrypt.append(response["encrypt_ms"])
                                        evaluate_times.append(response["evaluate_ms"]); decrypt.append(response["decrypt_ms"])
                                        ram = max(ram, int(response["ram_bytes"])); vram = max(vram, int(response["vram_bytes"]))
                                        idle_ram = max(idle_ram, int(response.get("idle_ram_bytes", 0)))
                                        idle_vram = max(idle_vram, int(response.get("idle_vram_bytes", 0)))
                                        if sample_number % 10 == 0 or sample_number == len(indices):
                                            self._atomic_json(checkpoint_path, {
                                                "ciphertext_predictions": predictions,
                                                "ciphertext_logits": ciphertext_logits,
                                                "encrypted_responses": native_responses,
                                                "encrypted_timings_ms": timings,
                                                "encrypt_timings_ms": encrypt,
                                                "evaluate_timings_ms": evaluate_times,
                                                "decrypt_timings_ms": decrypt,
                                                "encrypted_peak_ram_bytes": ram,
                                                "encrypted_vram_bytes": vram})
                                        self._phase_progress(run, "Encrypted inference", sample_number, len(indices),
                                            f"{point_label} · encrypted sample {sample_number} of {len(indices)}")
                                    timed_runs = min(int(run.manifest.get("timed_runs", len(timings))), len(timings))
                                    point.update(summarize(timings[:timed_runs], "encrypted"))
                                    point.update(summarize(encrypt[:timed_runs], "encrypt"))
                                    point.update(summarize(evaluate_times[:timed_runs], "evaluate"))
                                    point.update(summarize(decrypt[:timed_runs], "decrypt"))
                                    correct_predictions = sum(a == b for a, b in zip(predictions, labels))
                                    cipher_ci = wilson_interval(correct_predictions, len(labels))
                                    plaintext_predictions = point.get("plaintext_predictions", [])
                                    agreement = (sum(a == b for a, b in zip(plaintext_predictions, predictions)) /
                                                 len(predictions)) if predictions and plaintext_predictions else None
                                    point.update({"ciphertext_status": "complete",
                                                  "ciphertext_accuracy": correct_predictions / len(labels),
                                                  "ciphertext_correct_predictions": correct_predictions,
                                                  "ciphertext_total_predictions": len(labels),
                                                  "ciphertext_wilson_95_low": cipher_ci[0],
                                                  "ciphertext_wilson_95_high": cipher_ci[1],
                                                  "plaintext_ciphertext_prediction_agreement": agreement,
                                                  "encrypted_timed_runs": timed_runs,
                                                  "encrypted_warmup_runs": len(warmup_responses),
                                                  "encrypted_batch_size": 1,
                                                  "encrypted_ram_bytes": max(0, ram - idle_ram),
                                                  "encrypted_vram_bytes": vram,
                                                  "encrypted_peak_ram_bytes": ram,
                                                  "encrypted_idle_ram_bytes": idle_ram,
                                                  "encrypted_idle_vram_bytes": idle_vram,
                                                  "encrypted_peak_vram_bytes": idle_vram + vram,
                                                  "sample_labels": labels,
                                                  "ciphertext_predictions": predictions,
                                                  "ciphertext_logits": ciphertext_logits,
                                                  "encrypted_warmup_responses": warmup_responses,
                                                  "encrypted_responses": native_responses,
                                                  "encrypted_timings_ms": timings, "encrypt_timings_ms": encrypt,
                                                  "evaluate_timings_ms": evaluate_times, "decrypt_timings_ms": decrypt})
                                    if agreement is None or agreement < 0.99:
                                        point.update({"status": "partial", "ciphertext_status": "invalid",
                                            "ciphertext_reason":
                                                f"plaintext/encrypted prediction agreement {agreement:.3%} is below 99%"})
                                    if checkpoint_path.exists():
                                        checkpoint_path.unlink()
                                    if point.get("plaintext_mean_ms"):
                                        point["slowdown_ratio"] = point["encrypted_mean_ms"] / point["plaintext_mean_ms"]
                                except Exception as error:
                                    point.update({"status": "partial", "ciphertext_status": "failed",
                                                  "ciphertext_reason": str(error)})
                                    if worker:
                                        worker.close(); worker = None
                        point["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        point["point_seconds"] = time.time() - run.point_started_at
                        if benchmark or manual:
                            write_activation_samples(run.directory, point)
                        self._write_point(run, point)
                        run.point_durations.append(point["point_seconds"])
            self._phase(run, "Generating figures", "Writing PNG, SVG, and editable figure-data CSV")
            self._consolidate(run, True)
            with self.plot_lock:
                generate_plots(run.directory)
            point_documents = [json.loads(path.read_text())
                               for path in (run.directory / "points").glob("*.json")]
            run.status = "partial" if any(point.get("status") != "complete" or
                point.get("ciphertext_status") in ("failed", "infeasible", "invalid")
                for point in point_documents) else "complete"
            self._phase(run, "Partial" if run.status == "partial" else "Complete",
                        "Figures include explicit failed points" if run.status == "partial"
                        else "All result tables and figures are ready")
        except RunCancelled:
            run.status = "cancelled"; run.error = None
            self._phase(run, "Cancelled", "Stopped by request")
        except Exception as error:
            run.status = "failed"; run.error = str(error); self._phase(run, "Failed", str(error))
        finally:
            if worker: worker.close()
            if run.status != "complete":
                try:
                    self._consolidate(run, True)
                except Exception as export_error:
                    run.error = f"{run.error or run.status}; export failed: {export_error}"
            self._record_event(run, "run_finished")


def generate_plots(directory: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    points = json.loads((directory / "results.json").read_text())
    usable = [point for point in points if point.get("status") == "complete"]
    plots = directory / "plots"; plots.mkdir(exist_ok=True)
    model_keys = {(point.get("dataset"), point.get("model"), point.get("activation"))
                  for point in usable}
    include_model = len(model_keys) > 1

    def series(key):
        groups = {}
        for point in usable:
            if point.get(key) is not None:
                model = (f'{point.get("dataset")}/{point.get("model")}/{point.get("activation")} · '
                         if include_model else "")
                label = f'{model}{point["method"]} ±{point["range"]:g}'
                groups.setdefault(label, []).append((point["degree"], point[key]))
        return groups

    def save(fig, name):
        fig.tight_layout(); fig.savefig(plots / f"{name}.png", dpi=180); fig.savefig(plots / f"{name}.svg"); plt.close(fig)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    for label, values in series("plaintext_polynomial_accuracy").items(): axes[0].plot(*zip(*sorted(values)), marker="o", label=label)
    for label, values in series("encrypted_mean_ms").items(): axes[1].plot(*zip(*sorted(values)), marker="o", label=label)
    axes[0].set_ylabel("Accuracy"); axes[1].set_ylabel("Warm HE latency (ms)"); axes[1].set_xlabel("Polynomial degree")
    axes[0].legend(fontsize=7); save(fig, "accuracy_and_latency_vs_degree")

    for key, name, ylabel in (("slowdown_ratio", "encrypted_plaintext_slowdown", "HE / plaintext time"),
                              ("approximation_rmse", "approximation_rmse", "RMSE"),
                              ("approximation_max_error", "approximation_max_error", "Maximum absolute error")):
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for label, values in series(key).items(): axis.plot(*zip(*sorted(values)), marker="o", label=label)
        axis.set(xlabel="Polynomial degree", ylabel=ylabel); axis.legend(fontsize=7)
        if key == "slowdown_ratio": axis.axhspan(1000, 10000, alpha=0.12, color="green", label="ideal band")
        save(fig, name)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    baselines = {}
    for point in usable:
        if point.get("baseline_accuracy") is not None:
            label = (f'{point.get("dataset")}/{point.get("model")}/{point.get("activation")}'
                     if include_model else "original")
            baselines[label] = point["baseline_accuracy"]
    for label, baseline in baselines.items():
        axis.axhline(baseline, linestyle="--", label=f"original {label}" if include_model else label)
    for key, style in (("plaintext_polynomial_accuracy", "-"), ("ciphertext_accuracy", ":")):
        for label, values in series(key).items(): axis.plot(*zip(*sorted(values)), linestyle=style, marker="o", label=f"{key.split('_')[0]} {label}")
    axis.set(xlabel="Polynomial degree", ylabel="Accuracy"); axis.legend(fontsize=6); save(fig, "plaintext_ciphertext_accuracy")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    for key, axis, label in (("plaintext_ram_bytes", axes[0], "Plaintext RAM"), ("encrypted_ram_bytes", axes[0], "Encrypted RAM"),
                             ("plaintext_vram_bytes", axes[1], "Plaintext VRAM"), ("encrypted_vram_bytes", axes[1], "Encrypted VRAM")):
        for group, values in series(key).items():
            scaled = [(degree, value / 2**30) for degree, value in values]
            axis.plot(*zip(*sorted(scaled)), marker="o", label=f"{label} · {group}")
    axes[0].set_ylabel("Peak RAM (GiB)"); axes[1].set_ylabel("Peak VRAM (GiB)"); axes[1].set_xlabel("Polynomial degree")
    axes[0].legend(); axes[1].legend(); save(fig, "memory_vs_degree")

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    summary = axes.reshape(-1)
    for label, baseline in baselines.items():
        summary[0].axhline(baseline, linestyle="--", label=f"original {label}" if include_model else label)
    for key, style in (("plaintext_polynomial_accuracy", "-"), ("ciphertext_accuracy", ":")):
        for label, values in series(key).items():
            summary[0].plot(*zip(*sorted(values)), linestyle=style, marker="o",
                            label=f"{key.split('_')[0]} {label}")
    summary[0].set(xlabel="Degree", ylabel="Accuracy", title="Accuracy")
    for label, values in series("encrypted_mean_ms").items():
        summary[1].plot(*zip(*sorted(values)), marker="o", label=label)
    summary[1].set(xlabel="Degree", ylabel="Milliseconds", title="Encrypted inference time")
    for label, values in series("slowdown_ratio").items():
        summary[2].plot(*zip(*sorted(values)), marker="o", label=label)
    summary[2].axhspan(1000, 10000, alpha=0.12, color="green")
    summary[2].set(xlabel="Degree", ylabel="HE / plaintext", title="Encrypted slowdown")
    for key, style in (("approximation_rmse", "-"), ("approximation_max_error", ":")):
        for label, values in series(key).items():
            summary[3].plot(*zip(*sorted(values)), linestyle=style, marker="o",
                            label=f"{key.replace('approximation_', '')} {label}")
    summary[3].set(xlabel="Degree", ylabel="Error", title="Approximation quality")
    for axis, suffix, title in ((summary[4], "ram_bytes", "Peak RAM"),
                                (summary[5], "vram_bytes", "Peak GPU VRAM")):
        for prefix in ("plaintext", "encrypted"):
            for label, values in series(f"{prefix}_{suffix}").items():
                scaled = [(degree, value / 2**30) for degree, value in values]
                axis.plot(*zip(*sorted(scaled)), marker="o", label=f"{prefix} {label}")
        axis.set(xlabel="Degree", ylabel="GiB", title=title)
    for axis in summary:
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=5)
        axis.grid(alpha=0.2)
    save(fig, "research_summary")
