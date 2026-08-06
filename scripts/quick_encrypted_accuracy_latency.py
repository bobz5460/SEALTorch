#!/usr/bin/env python3
"""Matched plaintext/encrypted trend benchmark selected on training data."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from webui.experiments import NativeWorker, profile
from webui.models import load_data_splits, load_model, polynomial_model, resolve_model
from webui.polynomials import coefficients, fideslib_coefficients

MODEL = "mnist/mlp-gelu.json"
METHOD = "chebyshev"
DEGREES = (2, 4, 6, 8)
INTERVAL = 20.0
SEED = 42
SAMPLES_PER_CLASS = 10
WARMUPS = 5


def selected_indices(dataset, class_count: int) -> list[int]:
    targets = np.asarray(dataset.targets)
    generator = np.random.default_rng(SEED)
    result = []
    for label in range(class_count):
        choices = np.flatnonzero(targets == label)
        result.extend(generator.choice(choices, SAMPLES_PER_CLASS, replace=False).tolist())
    return result


def plaintext_point(model_path: str, document, dataset, indices, degree: int, device: str) -> dict:
    # Load independently so moving one benchmark model between CUDA devices
    # cannot contaminate another degree's reference logits.
    base, _ = load_model(model_path)
    polynomial_coefficients = coefficients(document["activation"], METHOD, degree, INTERVAL)
    model = polynomial_model(base, polynomial_coefficients, INTERVAL).to(device).eval()
    images = torch.stack([dataset[index][0] for index in indices])
    labels = [int(dataset[index][1]) for index in indices]
    with torch.inference_mode():
        logits = model(images.to(device)).float().cpu().numpy()
        for _ in range(WARMUPS):
            model(images.to(device))
        torch.cuda.synchronize(device)
        timings = []
        for _ in range(50):
            started = time.perf_counter(); model(images.to(device)); torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)
    predictions = logits.argmax(axis=1).astype(int).tolist()
    return {"degree": degree, "coefficients": polynomial_coefficients.tolist(),
        "images": [dataset[index][0].reshape(-1).tolist() for index in indices],
        "labels": labels, "plaintext_logits": logits.tolist(),
        "plaintext_predictions": predictions,
        "plaintext_correct": sum(a == b for a, b in zip(predictions, labels)),
        "plaintext_median_batch_ms": statistics.median(timings)}


def encrypted_point(model_path: str, document: dict, point: dict, gpu: int) -> dict:
    degree = point["degree"]
    selected = profile(document, degree)
    selected["device"] = gpu
    request = {"model_path": model_path,
        "coefficients": fideslib_coefficients(point["coefficients"]).tolist(),
        "lower_bound": -INTERVAL, "upper_bound": INTERVAL, "stream_weights": 0,
        **{key: selected[key] for key in
           ("ring_dim", "depth", "scaling_mod_bits", "first_mod_bits", "device")}}
    worker = NativeWorker()
    try:
        warmups = []
        for _ in range(WARMUPS):
            warmups.append(worker.predict({**request, "pixels": point["images"][0]}))
        responses = [worker.predict({**request, "pixels": image}) for image in point["images"]]
    finally:
        worker.close()
    logits = [response["logits"] for response in responses]
    predictions = [int(np.argmax(values)) for values in logits]
    total_times = [float(response["total_ms"]) for response in responses]
    max_logit_error = max(float(np.max(np.abs(np.asarray(plain) - np.asarray(cipher))))
                          for plain, cipher in zip(point["plaintext_logits"], logits))
    return {**point, "gpu": gpu, "he_parameters": {key: selected[key] for key in
        ("ring_dim", "depth", "scaling_mod_bits", "first_mod_bits")},
        "encrypted_logits": logits, "encrypted_predictions": predictions,
        "encrypted_correct": sum(a == b for a, b in zip(predictions, point["labels"])),
        "encrypted_median_ms": statistics.median(total_times),
        "encrypt_median_ms": statistics.median(response["encrypt_ms"] for response in responses),
        "evaluate_median_ms": statistics.median(response["evaluate_ms"] for response in responses),
        "decrypt_median_ms": statistics.median(response["decrypt_ms"] for response in responses),
        "prediction_agreement": sum(a == b for a, b in zip(
            predictions, point["plaintext_predictions"])) / len(predictions),
        "maximum_logit_error": max_logit_error,
        "encrypted_warmups": len(warmups), "encrypted_timed_runs": len(total_times)}


def main() -> None:
    if torch.cuda.device_count() < len(DEGREES):
        raise RuntimeError(f"benchmark requires {len(DEGREES)} GPUs")
    output = ROOT / "results" / "quick_encrypted_accuracy_latency_20260806"
    output.mkdir(parents=True, exist_ok=True)
    path = resolve_model(MODEL)
    base, document = load_model(path)
    _, test = load_data_splits(document)
    indices = selected_indices(test, len(document["classes"]))
    baseline_model = base.to("cuda:0").eval()
    labels = [int(test[index][1]) for index in indices]
    with torch.inference_mode():
        images = torch.stack([test[index][0] for index in indices]).to("cuda:0")
        baseline_predictions = baseline_model(images).argmax(1).cpu().tolist()
    baseline_correct = sum(a == b for a, b in zip(baseline_predictions, labels))

    plaintext = [plaintext_point(str(path), document, test, indices, degree, f"cuda:{position}")
                 for position, degree in enumerate(DEGREES)]
    results = []
    with ThreadPoolExecutor(max_workers=len(DEGREES)) as executor:
        futures = {executor.submit(encrypted_point, str(path), document, point, gpu): point["degree"]
                   for gpu, point in enumerate(plaintext)}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["degree"])

    rows = []
    for result in results:
        he = result["he_parameters"]
        rows.append({"degree": result["degree"], "plaintext_correct": result["plaintext_correct"],
            "encrypted_correct": result["encrypted_correct"], "total_predictions": len(indices),
            "plaintext_accuracy_percent": 100 * result["plaintext_correct"] / len(indices),
            "encrypted_accuracy_percent": 100 * result["encrypted_correct"] / len(indices),
            "plaintext_median_100_image_batch_ms": result["plaintext_median_batch_ms"],
            "encrypted_median_ms_per_image": result["encrypted_median_ms"],
            "encrypt_median_ms": result["encrypt_median_ms"],
            "evaluate_median_ms": result["evaluate_median_ms"],
            "decrypt_median_ms": result["decrypt_median_ms"],
            "prediction_agreement": result["prediction_agreement"],
            "maximum_logit_error": result["maximum_logit_error"], "gpu": result["gpu"],
            "ring_dim": he["ring_dim"], "depth": he["depth"],
            "scaling_mod_bits": he["scaling_mod_bits"], "first_mod_bits": he["first_mod_bits"],
            "interval": INTERVAL, "method": METHOD, "model": document["model"],
            "dataset": document["dataset"], "activation": document["activation"]})
    with (output / "accuracy_latency_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (output / "raw_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"purpose": "validation-selected quick trend",
        "selection": "interval selected on a separate seeded training subset before test evaluation",
        "model": MODEL, "method": METHOD, "degrees": list(DEGREES), "interval": INTERVAL,
        "seed": SEED, "sample_indices": indices, "samples_per_class": SAMPLES_PER_CLASS,
        "baseline_correct": baseline_correct, "total_predictions": len(indices),
        "warmups": WARMUPS, "encrypted_runs_per_degree": len(indices)}, indent=2) + "\n")

    degrees = [row["degree"] for row in rows]
    plain_accuracy = [row["plaintext_accuracy_percent"] for row in rows]
    encrypted_accuracy = [row["encrypted_accuracy_percent"] for row in rows]
    latency = [row["encrypted_median_ms_per_image"] for row in rows]
    figure, accuracy_axis = plt.subplots(figsize=(8.5, 5.3)); latency_axis = accuracy_axis.twinx()
    plain_line = accuracy_axis.plot(degrees, plain_accuracy, "o-", color="#2563eb",
                                    linewidth=2.2, label="Plaintext polynomial accuracy")[0]
    encrypted_line = accuracy_axis.plot(degrees, encrypted_accuracy, "D--", color="#059669",
                                        linewidth=2.2, label="Encrypted polynomial accuracy")[0]
    original = accuracy_axis.axhline(100 * baseline_correct / len(indices), color="#64748b",
                                     linestyle=":", linewidth=2, label="Original GELU accuracy")
    latency_line = latency_axis.plot(degrees, latency, "s-", color="#ea580c", linewidth=2.2,
                                     label="Encrypted median latency")[0]
    accuracy_axis.set(xlabel="Polynomial degree", ylabel="Test accuracy (%)", xticks=degrees,
                      title="Plaintext/encrypted accuracy and latency vs. degree")
    latency_axis.set_ylabel("Encrypted median latency per image (ms)")
    accuracy_axis.grid(alpha=.25)
    lines = (plain_line, encrypted_line, original, latency_line)
    accuracy_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout(); figure.savefig(output / "accuracy_latency_vs_degree.png", dpi=200)
    figure.savefig(output / "accuracy_latency_vs_degree.svg")
    print(json.dumps({"output": str(output), "baseline_accuracy_percent":
                      100 * baseline_correct / len(indices), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
