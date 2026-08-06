"""Publication benchmark metrics and automatic figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .polynomials import activation_value, evaluate
except ImportError:
    from polynomials import activation_value, evaluate


class ActivationRecorder:
    """Occurrence-weighted sample of real pre-activation scalar values."""

    def __init__(self, model: nn.Module, maximum_values: int = 200_000) -> None:
        modules = [module for module in model.modules() if isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh))]
        self.maximum_values = maximum_values
        self.values: dict[int, list[np.ndarray]] = {id(module): [] for module in modules}
        self.hooks = [module.register_forward_pre_hook(self._hook(id(module))) for module in modules]

    def _hook(self, key: int):
        def record(_module, inputs) -> None:
            values = inputs[0].detach().reshape(-1)
            sample = values.float().cpu().numpy()
            self.values[key].append(sample)
        return record

    def close(self) -> np.ndarray:
        for hook in self.hooks:
            hook.remove()
        arrays = [array for items in self.values.values() for array in items]
        result = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
        if len(result) > self.maximum_values:
            positions = np.linspace(0, len(result) - 1, self.maximum_values, dtype=np.int64)
            result = result[positions]
        return result


def weighted_approximation(name: str, polynomial_coefficients, interval: float,
                           inputs: np.ndarray, stored_samples: int = 10_000) -> dict:
    original = activation_value(name, inputs)
    polynomial = evaluate(inputs, polynomial_coefficients, interval)
    errors = polynomial - original
    count = min(stored_samples, len(inputs))
    positions = np.linspace(0, len(inputs) - 1, count, dtype=np.int64) if count else np.empty(0, dtype=np.int64)
    return {
        "weighted_approximation_rmse": float(np.sqrt(np.mean(errors * errors))),
        "observed_approximation_max_error": float(np.max(np.abs(errors))),
        "activation_input_count": int(len(inputs)),
        "activation_outside_range_fraction": float(np.mean(np.abs(inputs) > interval)),
        "activation_abs_p99": float(np.quantile(np.abs(inputs), 0.99)),
        "activation_abs_p999": float(np.quantile(np.abs(inputs), 0.999)),
        "sampled_activation_inputs": inputs[positions].astype(float).tolist(),
        "sampled_original_activation_outputs": original[positions].astype(float).tolist(),
        "sampled_polynomial_outputs": polynomial[positions].astype(float).tolist(),
    }


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    proportion = correct / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def circuit_signature(model: str, degree: int, coefficients_) -> str:
    realized = len(coefficients_)
    while realized > 1 and abs(float(coefficients_[realized - 1])) < 1e-15:
        realized -= 1
    strategy = "constant" if realized == 1 else "linear" if realized == 2 else "chebyshev-series"
    return f"{model}:{strategy}:degree-{realized - 1}"


def write_activation_samples(directory: Path, point: dict) -> None:
    inputs = point.get("sampled_activation_inputs", [])
    original = point.get("sampled_original_activation_outputs", [])
    polynomial = point.get("sampled_polynomial_outputs", [])
    path = directory / "activation_samples.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("sample", "activation_input", "original_output", "polynomial_output", "error"))
        for index, (value, expected, actual) in enumerate(zip(inputs, original, polynomial)):
            writer.writerow((index, value, expected, actual, actual - expected))


def generate_benchmark_plots(directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    source = directory / "raw_results.json"
    points = json.loads(source.read_text() if source.exists()
                        else (directory / "results.json").read_text())
    if not points:
        return
    plots = directory / "benchmark_plots"; plots.mkdir(exist_ok=True)
    sns.set_theme(style="ticks", context="notebook", font_scale=1.08,
                  rc={"axes.facecolor": "#f8fafc", "figure.facecolor": "white",
                      "axes.titleweight": "bold", "axes.edgecolor": "#cbd5e1",
                      "axes.titlesize": 18, "axes.labelsize": 13,
                      "xtick.labelsize": 11, "ytick.labelsize": 11,
                      "grid.color": "#dbe4ee", "grid.linewidth": 0.8})

    def save(figure, name):
        figure.tight_layout()
        temporary_png = plots / f".{name}.tmp.png"
        temporary_svg = plots / f".{name}.tmp.svg"
        figure.savefig(temporary_png, dpi=200, bbox_inches="tight")
        figure.savefig(temporary_svg, bbox_inches="tight")
        temporary_png.replace(plots / f"{name}.png")
        temporary_svg.replace(plots / f"{name}.svg")
        plt.close(figure)

    methods = ("taylor", "chebyshev")
    degrees = [2, 4, 6, 8]
    colors = {"taylor": "#2563eb", "chebyshev": "#f97316"}
    styles = {"plaintext": "-", "encrypted": "--"}
    markers = {"plaintext": "o", "encrypted": "s"}
    activation = str(next((point.get("activation") for point in points
                           if point.get("activation")), "gelu"))
    activation_label = activation.upper() if activation.lower() == "gelu" else activation.title()
    baseline_label = f"Original {activation_label}"

    def failures_for(method, degree):
        return [point for point in points if point.get("method") == method and
                point.get("degree") == degree and
                point.get("ciphertext_status") in ("failed", "infeasible", "invalid")]

    def mark_failure(axis, degree, reason, log=False):
        axis.text(degree, 0.025 if not log else 0.035, "×", color="#dc2626",
                  fontsize=16, fontweight="bold", ha="center", va="bottom",
                  transform=axis.get_xaxis_transform())
        axis.annotate(reason[:42], (degree, 0), xycoords=("data", "axes fraction"),
                      xytext=(0, -25), textcoords="offset points", ha="center",
                      fontsize=6.5, color="#991b1b")

    figure, axis = plt.subplots(figsize=(9, 6.0))
    for method in methods:
        selected = sorted((point for point in points if point.get("method") == method),
                          key=lambda point: point["degree"])
        for mode, key in (("plaintext", "plaintext_polynomial_accuracy"),
                          ("encrypted", "ciphertext_accuracy")):
            values = [(point["degree"], point[key] * 100, point) for point in selected if point.get(key) is not None]
            if values:
                axis.plot([v[0] for v in values], [v[1] for v in values],
                          linestyle=styles[mode], color=colors[method],
                          linewidth=2.6, markersize=7, marker=markers[mode],
                          label=f"{method.title()} · {mode.title()}")
                prefix = "plaintext" if mode == "plaintext" else "ciphertext"
                lows = [(value - item.get(f"{prefix}_wilson_95_low", value / 100) * 100)
                        for _, value, item in values]
                highs = [(item.get(f"{prefix}_wilson_95_high", value / 100) * 100 - value)
                         for _, value, item in values]
                axis.errorbar([v[0] for v in values], [v[1] for v in values],
                              yerr=[lows, highs], fmt="none", ecolor=colors[method],
                              alpha=0.35, capsize=2)
            if mode == "encrypted":
                for degree in degrees:
                    failed = failures_for(method, degree)
                    if failed:
                        mark_failure(axis, degree, failed[0].get("ciphertext_reason", "failed"))
    baseline = next((point.get("baseline_accuracy") for point in points if point.get("baseline_accuracy") is not None), None)
    if baseline is not None:
        axis.axhline(baseline * 100, color="black", linestyle="--", label=baseline_label)
    outside = max((point.get("activation_outside_range_fraction", 0.0) for point in points), default=0.0)
    interval = next((point.get("range") for point in points if point.get("range") is not None), None)
    recommended = max((point.get("activation_abs_p999", 0.0) for point in points), default=0.0)
    axis.set(xlabel="Polynomial degree", ylabel="Test accuracy (%)",
             title="Accuracy vs. polynomial degree")
    # Accuracy is intentionally not auto-zoomed: a failed approximation should
    # look like a failed approximation, rather than an exaggerated trend.
    axis.set_ylim(0, 101); axis.set_xticks(degrees)
    axis.grid(True, axis="y"); axis.grid(False, axis="x")
    axis.legend(ncol=2, frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)
    if interval is not None:
        note = f"Range ±{interval:g} contains {(1-outside)*100:.1f}% of observed inputs"
        if recommended:
            note += f"  •  99.9% coverage needs about ±{recommended:.1f}"
        warning = outside > 0.01
        axis.text(0.01, 0.02, note, transform=axis.transAxes, fontsize=8.5,
                  color="#9f1239" if warning else "#475569",
                  bbox={"facecolor": "#fff1f2" if warning else "white",
                        "edgecolor": "#fda4af" if warning else "#cbd5e1",
                                           "boxstyle": "round,pad=0.35", "alpha": 0.92})
    sns.despine(ax=axis)
    save(figure, "accuracy_vs_degree")

    figure, axis = plt.subplots(figsize=(9, 6.8))
    table_rows = []
    for method in methods:
        selected = sorted((point for point in points if point.get("method") == method and
                           point.get("weighted_approximation_rmse") is not None),
                          key=lambda point: point["degree"])
        if selected:
            axis.plot([point["degree"] for point in selected],
                      [point["weighted_approximation_rmse"] for point in selected],
                      marker="o", linewidth=2.6, markersize=7,
                      color=colors[method], label=method.title())
            table_rows.extend([[method.title(), point["degree"],
                                f'{point["observed_approximation_max_error"]:.3g}'] for point in selected])
    axis.set_yscale("log"); axis.set(xlabel="Polynomial degree", ylabel="Weighted RMSE (log scale)",
                                      title=f"Approximation error on observed {activation_label} inputs")
    axis.set_xticks(degrees)
    handles, labels = axis.get_legend_handles_labels()
    if handles: axis.legend(handles, labels, frameon=True)
    axis.grid(True, which="both", axis="y"); axis.grid(False, axis="x"); sns.despine(ax=axis)
    if table_rows:
        table = axis.table(cellText=table_rows, colLabels=("Method", "Degree", "Max |error|"),
                           loc="bottom", bbox=(0, -0.62, 1, 0.45))
        table.auto_set_font_size(False); table.set_fontsize(8)
        figure.subplots_adjust(bottom=0.36)
    save(figure, "approximation_error_vs_degree")

    figure, axis = plt.subplots(figsize=(9, 6.0))
    for mode, raw_key, color in (("Plaintext", "plaintext_timings_ms", "#1b9e77"),
                                 ("Encrypted", "encrypted_timings_ms", "#7570b3")):
        values = []
        for degree in degrees:
            signatures = {point.get("circuit_signature") for point in points
                          if point.get("degree") == degree and point.get(raw_key)}
            timings = [timing for point in points if point.get("degree") == degree
                       and point.get(raw_key) and len(signatures) <= 1
                       for timing in point[raw_key]]
            if timings:
                values.append((degree, float(np.median(timings)),
                               float(np.quantile(timings, .25)), float(np.quantile(timings, .75))))
        if values:
            axis.plot([v[0] for v in values], [v[1] for v in values], marker="o", linewidth=2.8, markersize=7,
                      color=color, label=mode)
            axis.fill_between([v[0] for v in values], [v[2] for v in values],
                              [v[3] for v in values], color=color, alpha=.18)
        if mode == "Encrypted":
            for degree in degrees:
                failed = [item for method in methods for item in failures_for(method, degree)]
                if failed and not any(v[0] == degree for v in values):
                    mark_failure(axis, degree, failed[0].get("ciphertext_reason", "failed"), log=True)
    axis.set_yscale("log"); axis.set(xlabel="Polynomial degree",
        ylabel="Median inference time per image (ms, log scale)",
        title="Inference time vs. polynomial degree")
    axis.set_xticks(degrees)
    handles, labels = axis.get_legend_handles_labels()
    if handles: axis.legend(handles, labels, frameon=True)
    axis.grid(True, which="both", axis="y"); axis.grid(False, axis="x"); sns.despine(ax=axis)
    save(figure, "inference_time_vs_degree")

    memory_keys = (("plaintext_idle_ram_bytes", "plaintext_ram_bytes", "Plaintext RAM"),
                   ("encrypted_idle_ram_bytes", "encrypted_ram_bytes", "Encrypted RAM"),
                   ("encrypted_idle_vram_bytes", "encrypted_vram_bytes", "Encrypted GPU VRAM"))
    memory = {label: ([], []) for _, _, label in memory_keys}
    for degree in degrees:
        same_degree = [point for point in points if point["degree"] == degree]
        for idle_key, workload_key, label in memory_keys:
            idle = [point[idle_key] / 2**30 for point in same_degree if point.get(idle_key) is not None]
            workload = [point[workload_key] / 2**30 for point in same_degree if point.get(workload_key) is not None]
            memory[label][0].append(float(np.median(idle)) if idle else 0.0)
            memory[label][1].append(float(np.median(workload)) if workload else 0.0)
    figure, axis = plt.subplots(figsize=(9, 5.5)); width = 0.24; positions = np.arange(len(degrees))
    for offset, (_, _, label) in enumerate(memory_keys):
        idle, workload = memory[label]
        axis.bar(positions + (offset - 1) * width, idle, width, label=f"{label} idle",
                 color=("#bfdbfe", "#fed7aa", "#ddd6fe")[offset], edgecolor="white")
        bars = axis.bar(positions + (offset - 1) * width, workload, width, bottom=idle,
                        label=f"{label} workload",
                        color=("#60a5fa", "#fb923c", "#a78bfa")[offset],
                        edgecolor="white", linewidth=0.8)
        totals = [a + b for a, b in zip(idle, workload)]
        for x, total in zip(positions + (offset - 1) * width, totals):
            if total: axis.text(x, total, f"{total:.1f}", ha="center", va="bottom", fontsize=7)
    axis.set_xticks(positions, degrees); axis.set(xlabel="Polynomial degree", ylabel="Peak memory (GiB)",
                                                   title="Memory usage vs. polynomial degree")
    axis.legend(frameon=True, ncol=2, fontsize=8); axis.grid(True, axis="y"); axis.grid(False, axis="x")
    sns.despine(ax=axis); save(figure, "memory_vs_degree")

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for method in methods:
        selected = sorted((point for point in points if point.get("method") == method), key=lambda point: point["degree"])
        plain = [(p["degree"], p["plaintext_polynomial_accuracy"] * 100) for p in selected if p.get("plaintext_polynomial_accuracy") is not None]
        encrypted = [(p["degree"], p["ciphertext_accuracy"] * 100) for p in selected if p.get("ciphertext_accuracy") is not None]
        error = [(p["degree"], p["weighted_approximation_rmse"]) for p in selected if p.get("weighted_approximation_rmse") is not None]
        latency = [(p["degree"], p["encrypted_median_ms"]) for p in selected if p.get("encrypted_median_ms") is not None]
        for values, label, style in ((plain, f"{method.title()} plaintext", "-"),
                                     (encrypted, f"{method.title()} encrypted", ":")):
            if values: axes[0].plot(*zip(*values), marker="o", color=colors[method], linestyle=style, label=label)
        if error: axes[1].plot(*zip(*error), marker="o", color=colors[method], label=method.title())
        if latency: axes[2].plot(*zip(*latency), marker="o", color=colors[method], label=method.title())
    if baseline is not None: axes[0].axhline(baseline * 100, color="black", linestyle="--", label=baseline_label)
    axes[0].set(xlabel="Degree", ylabel="Accuracy (%)", title="Accuracy"); axes[0].legend(fontsize=8)
    axes[1].set_yscale("log"); axes[1].set(xlabel="Degree", ylabel="Weighted RMSE", title="Approximation error"); axes[1].legend(fontsize=8)
    axes[2].set_yscale("log"); axes[2].set(xlabel="Degree", ylabel="Encrypted median ms", title="Encrypted latency")
    handles, labels = axes[2].get_legend_handles_labels()
    if handles: axes[2].legend(handles, labels, fontsize=8)
    save(figure, "publication_summary")
    status = plots / "status.json"
    temporary_status = plots / ".status.tmp.json"
    failures = sum(point.get("ciphertext_status") in ("failed", "infeasible", "invalid") for point in points)
    temporary_status.write_text(json.dumps({"points": len(points), "failures": failures,
        "status": "partial" if failures else "complete"}) + "\n", encoding="utf-8")
    temporary_status.replace(status)
