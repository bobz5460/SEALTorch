#!/usr/bin/env python3
"""Create a fast, controlled plaintext accuracy/latency trend figure."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webui.models import load_data_splits, load_model, polynomial_model, resolve_model
from webui.polynomials import coefficients

MODEL = "mnist/mlp-gelu.json"
METHOD = "chebyshev"
DEGREES = (2, 4, 6)
INTERVAL = 17.02488899230957
SEED = 42
SAMPLES_PER_CLASS = 100
TIMING_BATCH_SIZE = 128
WARMUPS = 10
TIMED_RUNS = 50


def selected_indices(dataset, class_count: int) -> list[int]:
    targets = np.asarray(dataset.targets)
    generator = np.random.default_rng(SEED)
    result = []
    for label in range(class_count):
        candidates = np.flatnonzero(targets == label)
        result.extend(generator.choice(candidates, SAMPLES_PER_CLASS, replace=False).tolist())
    return result


def accuracy(model, loader, device: str) -> tuple[int, int]:
    correct = total = 0
    model.to(device).eval()
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return correct, total


def latency(model, images, device: str) -> list[float]:
    model.to(device).eval(); images = images.to(device)
    with torch.inference_mode():
        for _ in range(WARMUPS):
            model(images)
        torch.cuda.synchronize(device)
        values = []
        for _ in range(TIMED_RUNS):
            started = time.perf_counter()
            model(images)
            torch.cuda.synchronize(device)
            values.append((time.perf_counter() - started) * 1000)
    return values


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this quick benchmark requires CUDA")
    output = ROOT / "results" / "quick_accuracy_latency_20260806"
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    base, document = load_model(resolve_model(MODEL))
    _, test = load_data_splits(document)
    indices = selected_indices(test, len(document["classes"]))
    subset = Subset(test, indices)
    accuracy_loader = DataLoader(subset, batch_size=512, num_workers=0)
    timing_images, _ = next(iter(DataLoader(
        subset, batch_size=TIMING_BATCH_SIZE, shuffle=False, num_workers=0)))

    baseline_correct, total = accuracy(base, accuracy_loader, device)
    rows = []
    for degree in DEGREES:
        polynomial = polynomial_model(
            base, coefficients(document["activation"], METHOD, degree, INTERVAL), INTERVAL)
        correct, _ = accuracy(polynomial, accuracy_loader, device)
        timings = latency(polynomial, timing_images, device)
        rows.append({"degree": degree, "correct_predictions": correct,
            "total_predictions": total, "accuracy_percent": 100 * correct / total,
            "median_batch_latency_ms": statistics.median(timings),
            "median_latency_ms_per_image": statistics.median(timings) / TIMING_BATCH_SIZE,
            "timing_batch_size": TIMING_BATCH_SIZE, "warmup_runs": WARMUPS,
            "timed_runs": TIMED_RUNS, "method": METHOD, "interval": INTERVAL,
            "model": document["model"], "dataset": document["dataset"],
            "activation": document["activation"], "device": device})
        del polynomial
        torch.cuda.empty_cache()

    with (output / "accuracy_latency_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({"purpose": "quick exploratory trend",
        "model": MODEL, "method": METHOD, "degrees": list(DEGREES), "interval": INTERVAL,
        "seed": SEED, "sample_indices": indices, "samples_per_class": SAMPLES_PER_CLASS,
        "baseline_correct_predictions": baseline_correct, "total_predictions": total,
        "baseline_accuracy_percent": 100 * baseline_correct / total,
        "timing_batch_size": TIMING_BATCH_SIZE, "warmup_runs": WARMUPS,
        "timed_runs": TIMED_RUNS, "device": device}, indent=2) + "\n")

    degrees = [row["degree"] for row in rows]
    accuracies = [row["accuracy_percent"] for row in rows]
    latencies = [row["median_batch_latency_ms"] for row in rows]
    figure, accuracy_axis = plt.subplots(figsize=(8, 5))
    latency_axis = accuracy_axis.twinx()
    accuracy_line = accuracy_axis.plot(degrees, accuracies, "o-", linewidth=2.3,
                                       color="#2563eb", label="Polynomial accuracy")[0]
    baseline = accuracy_axis.axhline(100 * baseline_correct / total, linestyle="--",
                                     color="#64748b", label="Original GELU accuracy")
    latency_line = latency_axis.plot(degrees, latencies, "s-", linewidth=2.3,
                                     color="#ea580c", label="Median batch latency")[0]
    accuracy_axis.set(xlabel="Polynomial degree", ylabel="Test accuracy (%)",
                      xticks=degrees, title="Accuracy and latency vs. polynomial degree")
    latency_axis.set_ylabel(f"Median latency per {TIMING_BATCH_SIZE}-image batch (ms)")
    accuracy_axis.grid(alpha=.25)
    accuracy_axis.legend([accuracy_line, baseline, latency_line],
                         [line.get_label() for line in (accuracy_line, baseline, latency_line)],
                         loc="best")
    figure.tight_layout()
    figure.savefig(output / "accuracy_latency_vs_degree.png", dpi=200)
    figure.savefig(output / "accuracy_latency_vs_degree.svg")
    print(json.dumps({"output": str(output), "baseline_accuracy_percent":
                      100 * baseline_correct / total, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
