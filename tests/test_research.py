import csv
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LeNet-5"))

from export_model import save_artifact
from lenet5 import MLP
from webui.benchmarks import generate_benchmark_plots, weighted_approximation, wilson_interval
from webui.experiments import ExperimentManager, NativeWorker, Run, Sweep, generate_plots, profile
from webui.graph_lab import available_runs, render as render_graph, render_saved, schema as graph_schema
from webui.models import PolynomialActivation, polynomial_model, stratified_indices
from webui.polynomials import (approximation_error, coefficients, evaluate,
                               fideslib_coefficients, taylor_power_coefficients)


class PolynomialTests(unittest.TestCase):
    def test_taylor_relu_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            coefficients("relu", "taylor", 3, 2)

    def test_tanh_taylor_coefficients(self):
        power = taylor_power_coefficients("tanh", 5)
        np.testing.assert_allclose(power[[1, 3, 5]], [1, -1 / 3, 2 / 15])
        cheb = coefficients("tanh", "taylor", 5, 1)
        values = np.linspace(-0.5, 0.5, 11)
        np.testing.assert_allclose(evaluate(values, cheb, 1), np.polynomial.polynomial.polyval(values, power), atol=1e-12)

    def test_chebyshev_error_improves_for_gelu(self):
        low = approximation_error("gelu", coefficients("gelu", "chebyshev", 2, 2), 2)
        high = approximation_error("gelu", coefficients("gelu", "chebyshev", 8, 2), 2)
        self.assertLess(high["approximation_rmse"], low["approximation_rmse"])

    def test_fideslib_coefficient_wire_format_doubles_constant_term(self):
        numpy_coefficients = coefficients("gelu", "chebyshev", 4, 4)
        wire = fideslib_coefficients(numpy_coefficients)
        self.assertAlmostEqual(wire[0], 2 * numpy_coefficients[0])
        np.testing.assert_allclose(wire[1:], numpy_coefficients[1:])

    def test_polynomial_model_replaces_activations(self):
        model = polynomial_model(MLP(10, "relu"), coefficients("relu", "chebyshev", 3, 2), 2)
        self.assertEqual(sum(isinstance(module, PolynomialActivation) for module in model.modules()), 2)
        self.assertEqual(model(torch.zeros(2, 1, 28, 28)).shape, (2, 10))


class ExperimentTests(unittest.TestCase):
    def test_automatic_profile_is_single_gpu_and_bounded(self):
        document = {"architecture": {"layers": [
            {"op": "linear"}, {"op": "tanh"}, {"op": "linear"}
        ]}}
        selected = profile(document, 5)
        self.assertEqual(selected["device"], 0)
        self.assertEqual(selected["security"], "128-bit classic")
        self.assertTrue(selected["feasible"])

    def test_mlp_degree_two_profile_includes_bias_level(self):
        document = {"architecture": {"layers": [
            {"op": "linear"}, {"op": "gelu"}, {"op": "linear"},
            {"op": "gelu"}, {"op": "linear"}
        ]}}
        selected = profile(document, 2)
        self.assertEqual(selected["depth"], 9)
        self.assertEqual(selected["ring_dim"], 16384)
        self.assertEqual(selected["scaling_mod_bits"], 25)
        self.assertEqual(selected["first_mod_bits"], 35)
        self.assertEqual(profile(document, 4)["depth"], 11)
        self.assertEqual(profile(document, 4)["ring_dim"], 16384)
        self.assertEqual(profile(document, 8)["depth"], 13)
        self.assertEqual(profile(document, 8)["ring_dim"], 32768)

    def test_lenet_degree_two_uses_security_compliant_ring(self):
        document = {"architecture": {"layers": [
            {"op": "conv2d"}, {"op": "gelu"}, {"op": "avg_pool2d"},
            {"op": "conv2d"}, {"op": "gelu"}, {"op": "avg_pool2d"},
            {"op": "conv2d"}, {"op": "gelu"}, {"op": "linear"},
            {"op": "gelu"}, {"op": "linear"},
        ]}}
        selected = profile(document, 2)
        self.assertGreater(selected["depth"], 12)
        self.assertEqual(selected["ring_dim"], 65536)

    def test_stratified_selection(self):
        dataset = [(torch.zeros(1), label) for label in range(3) for _ in range(12)]
        selected = stratified_indices(dataset, 3, 10)
        self.assertEqual(len(selected), 30)
        self.assertEqual([dataset[index][1] for index in selected].count(2), 10)

    def test_seeded_stratified_selection_is_balanced_and_reproducible(self):
        dataset = [(torch.zeros(1), label) for label in range(3) for _ in range(30)]
        first = stratified_indices(dataset, 3, 10, seed=42)
        second = stratified_indices(dataset, 3, 10, seed=42)
        self.assertEqual(first, second)
        self.assertEqual([int(dataset[index][1]) for index in first].count(1), 10)

    def test_wilson_interval_contains_observed_accuracy(self):
        low, high = wilson_interval(950, 1000)
        self.assertLess(low, .95)
        self.assertGreater(high, .95)

    def test_manager_recovers_interrupted_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"; run.mkdir()
            (run / "manifest.json").write_text(json.dumps({"run_id": "run"}))
            manager = ExperimentManager(Path(directory))
            self.assertEqual(manager.runs["run"].status, "interrupted")

    def test_cpp_validates_shared_artifact(self):
        binary = Path(__file__).resolve().parents[1] / "build" / "sealtorch_he"
        if not binary.is_file():
            self.skipTest("C++ worker is not built")
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "mlp.json"
            save_artifact(artifact, model=MLP(10, "tanh"), model_name="mlp",
                          activation="tanh", dataset="mnist", classes=list("0123456789"),
                          preprocessing={"input_shape": [1, 28, 28], "pad": 0,
                                         "mean": 0.1307, "std": 0.3081},
                          training={"seed": 42}, baseline_metrics={})
            output = subprocess.run([str(binary), "--validate-model", str(artifact)],
                                    check=True, text=True, capture_output=True)
            metadata = json.loads(output.stdout)
            self.assertEqual(metadata["input_size"], 784)
            self.assertEqual(metadata["output_size"], 10)
            self.assertEqual(metadata["activations"], 2)

    def test_required_plot_exports_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = []
            for degree in (1, 3):
                points.append({"status": "complete", "method": "chebyshev", "range": 2.0,
                    "degree": degree, "baseline_accuracy": 0.95,
                    "plaintext_polynomial_accuracy": 0.9 + degree / 100,
                    "ciphertext_accuracy": 0.89 + degree / 100,
                    "encrypted_mean_ms": 100 * degree, "plaintext_mean_ms": 0.1,
                    "slowdown_ratio": 1000 * degree, "approximation_rmse": 0.1 / degree,
                    "approximation_max_error": 0.2 / degree,
                    "plaintext_ram_bytes": 2**28, "encrypted_ram_bytes": 2**30,
                    "plaintext_vram_bytes": 2**27, "encrypted_vram_bytes": 2**31})
            (root / "results.json").write_text(json.dumps(points))
            generate_plots(root)
            expected = {"accuracy_and_latency_vs_degree", "encrypted_plaintext_slowdown",
                        "approximation_rmse", "approximation_max_error",
                        "plaintext_ciphertext_accuracy", "memory_vs_degree", "research_summary"}
            self.assertEqual({path.stem for path in (root / "plots").glob("*.svg")}, expected)

    def test_editable_summary_and_sample_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point = {"status": "complete", "run_id": "run", "dataset": "mnist",
                     "model": "mlp", "activation": "tanh", "method": "chebyshev",
                     "range": 2.0, "degree": 3, "baseline_accuracy": 0.95,
                     "plaintext_polynomial_accuracy": 0.94, "sample_indices": [7],
                     "sample_labels": [2], "plaintext_predictions": [2],
                     "plaintext_timings_ms": [0.1], "ciphertext_predictions": [2],
                     "plaintext_logits": [[0.1, 0.9]], "ciphertext_logits": [[0.11, 0.89]],
                     "encrypted_timings_ms": [1000.0], "encrypt_timings_ms": [10.0],
                     "evaluate_timings_ms": [980.0], "decrypt_timings_ms": [10.0]}
            point.update({"sampled_activation_inputs": [-1.0, 1.0],
                          "sampled_original_activation_outputs": [-0.1, 0.9],
                          "sampled_polynomial_outputs": [-0.09, 0.88]})
            ExperimentManager._write_tables(root, [point])
            self.assertTrue((root / "results.csv").is_file())
            self.assertTrue((root / "accuracy.csv").is_file())
            self.assertTrue((root / "accuracy_data.csv").is_file())
            self.assertTrue((root / "approximation_error_data.csv").is_file())
            self.assertTrue((root / "inference_time_data.csv").is_file())
            self.assertTrue((root / "memory_data.csv").is_file())
            self.assertIn("plaintext_polynomial_accuracy", (root / "figure_data.csv").read_text())
            with (root / "samples.csv").open() as source:
                sample = next(csv.DictReader(source))
            self.assertEqual(sample["dataset_index"], "7")
            self.assertEqual(sample["encrypted_ms"], "1000.0")
            with (root / "logits.csv").open() as source:
                logits = list(csv.DictReader(source))
            self.assertEqual(len(logits), 2)
            self.assertAlmostEqual(float(logits[0]["absolute_logit_error"]), 0.01)
            with (root / "activation_samples.csv").open() as source:
                activations = list(csv.DictReader(source))
            self.assertEqual(len(activations), 2)
            self.assertAlmostEqual(float(activations[0]["error"]), 0.01)

    def test_run_and_sweep_progress_include_etas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = Run("run", root, {"model": "model.json"}, status="running",
                      completed=1, total=3, started_at=time.time() - 5,
                      point_durations=[10.0], phase="Encrypted inference",
                      phase_completed=2, phase_total=10)
            public = run.public()
            self.assertEqual(public["phase"], "Encrypted inference")
            self.assertEqual(public["eta_seconds"], 20.0)
            manager = ExperimentManager(root / "results")
            manager._phase(run, "CKKS setup", "Creating context")
            self.assertTrue((root / "status.json").is_file())
            self.assertIn('"phase":"CKKS setup"', (root / "events.jsonl").read_text())
            self.assertIn("eta_seconds", (root / "events.csv").read_text())
            sweep = Sweep("sweep", root, ["run"], 1, created_at=time.time() - 5,
                          status="running")
            aggregate = sweep.public({"run": run})
            self.assertEqual(aggregate["active"], 1)
            self.assertGreater(aggregate["eta_seconds"], 0)

    def test_queued_run_can_be_cancelled_without_starting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run_directory = root / "run"; run_directory.mkdir()
            manager = ExperimentManager(root / "results")
            run = Run("run", run_directory, {"model": "model.json"}, status="queued")
            manager.runs[run.run_id] = run
            manager.cancel(run.run_id)
            self.assertEqual(run.status, "cancelled")
            self.assertTrue(run.cancel)
            self.assertTrue((run_directory / "status.json").is_file())

    def test_completed_sweep_and_children_can_be_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"; results.mkdir()
            sweep_directory = results / "_sweeps" / "sweep"; sweep_directory.mkdir(parents=True)
            run_directories = [results / "run-a", results / "run-b"]
            for path in run_directories: path.mkdir()
            manager = ExperimentManager(results)
            for path in run_directories:
                run = Run(path.name, path, {"model": "model.json"}, status="complete",
                          sweep_id="sweep")
                manager.runs[run.run_id] = run
            manager.sweeps["sweep"] = Sweep(
                "sweep", sweep_directory, ["run-a", "run-b"], 1, status="complete")
            manager.delete_sweep("sweep")
            self.assertNotIn("sweep", manager.sweeps)
            self.assertFalse(sweep_directory.exists())
            self.assertTrue(all(not path.exists() for path in run_directories))

    def test_all_model_sweep_creates_gpu_bounded_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); artifact = root / "model.json"
            artifact.write_text("artifact")
            manager = ExperimentManager(root / "results")
            request = {"models": ["a.json", "b.json", "c.json"], "gpu_count": 2,
                       "methods": ["chebyshev"], "ranges": [2],
                       "plaintext_degrees": [1, 2], "encrypted_degrees": [1]}
            with patch("webui.experiments.resolve_model", return_value=artifact), \
                 patch("webui.experiments.torch.cuda.is_available", return_value=True), \
                 patch("webui.experiments.torch.cuda.device_count", return_value=8), \
                 patch("webui.experiments.threading.Thread.start"):
                sweep = manager.start_sweep(request)
            self.assertEqual(sweep.gpu_count, 2)
            self.assertEqual(len(sweep.run_ids), 3)
            self.assertEqual([manager.runs[run_id].total for run_id in sweep.run_ids], [2, 2, 2])

    def test_smoke_benchmark_creates_four_gpu_point_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); artifact = root / "model.json"
            artifact.write_text(json.dumps({"dataset": "mnist", "model": "lenet5",
                                             "activation": "gelu"}))
            manager = ExperimentManager(root / "results")
            with patch("webui.experiments.resolve_model", return_value=artifact), \
                 patch("webui.experiments.torch.cuda.is_available", return_value=True), \
                 patch("webui.experiments.torch.cuda.device_count", return_value=8), \
                 patch("webui.experiments.threading.Thread.start"):
                benchmark = manager.start_benchmark({"mode": "smoke", "model": "model.json",
                    "methods": ["taylor", "chebyshev"], "degrees": [2, 4], "gpu_count": 8,
                    "weight_cache": "gpu"})
            self.assertEqual(benchmark.kind, "benchmark")
            self.assertEqual(benchmark.gpu_count, 4)
            self.assertEqual(len(benchmark.run_ids), 4)
            manifests = [manager.runs[run_id].manifest for run_id in benchmark.run_ids]
            self.assertEqual({manifest["samples_per_class"] for manifest in manifests}, {2})
            self.assertEqual({manifest["timed_runs"] for manifest in manifests}, {5})
            self.assertEqual({manifest["weight_cache"] for manifest in manifests}, {"cpu"})

    def test_manual_batch_preserves_exact_parameters_and_gpu_assignments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); artifact = root / "model.json"
            artifact.write_text(json.dumps({"classes": list("0123456789")}))
            manager = ExperimentManager(root / "results")
            base = {"model": "mnist/mlp-gelu.json", "method": "chebyshev", "degree": 2,
                    "range": 4, "gpu": 0, "run_plaintext": True, "run_encrypted": True,
                    "samples_per_class": 1, "timed_runs": 10, "warmup_runs": 0,
                    "activation_samples_per_class": 1, "activation_sample_limit": 100,
                    "he_parameters": {"ring_dim": 8192, "depth": 7,
                                      "scaling_mod_bits": 23, "first_mod_bits": 31}}
            with patch("webui.experiments.resolve_model", return_value=artifact), \
                 patch("webui.experiments.torch.cuda.is_available", return_value=True), \
                 patch("webui.experiments.torch.cuda.device_count", return_value=4), \
                 patch("webui.experiments.threading.Thread.start"):
                batch = manager.start_manual_batch({"label": "Exact runs", "jobs": [
                    {**base, "name": "first"},
                    {**base, "name": "second", "degree": 4, "gpu": 2,
                     "he_parameters": {**base["he_parameters"], "depth": 11}}]})
            self.assertEqual(batch.kind, "manual")
            self.assertEqual(batch.gpu_count, 2)
            manifests = [manager.runs[run_id].manifest for run_id in batch.run_ids]
            self.assertEqual([manifest["assigned_gpu"] for manifest in manifests], [0, 2])
            self.assertEqual(manifests[0]["parameter_overrides"], base["he_parameters"])
            self.assertEqual(manifests[1]["parameter_overrides"]["depth"], 11)
            saved = json.loads((batch.directory / "manifest.json").read_text())
            self.assertEqual(saved["assigned_gpus"], [0, 2])

    def test_manual_batch_requires_complete_he_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); artifact = root / "model.json"
            artifact.write_text(json.dumps({"classes": list("0123456789")}))
            manager = ExperimentManager(root / "results")
            with patch("webui.experiments.resolve_model", return_value=artifact), \
                 patch("webui.experiments.torch.cuda.is_available", return_value=True), \
                 patch("webui.experiments.torch.cuda.device_count", return_value=1):
                with self.assertRaisesRegex(ValueError, "missing HE parameters"):
                    manager.start_manual_batch({"jobs": [{"model": "model.json",
                        "method": "chebyshev", "degree": 2, "range": 4, "gpu": 0,
                        "samples_per_class": 1, "timed_runs": 1,
                        "activation_samples_per_class": 1, "activation_sample_limit": 1,
                        "he_parameters": {"ring_dim": 8192}}]})

    def test_manual_recommendation_measures_interval_and_returns_minimum_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); artifact = root / "model.json"; artifact.write_text("artifact")
            manager = ExperimentManager(root / "results")
            document = {"activation": "gelu", "classes": list("01"),
                        "architecture": {"layers": [
                            {"op": "linear"}, {"op": "gelu"}, {"op": "linear"},
                            {"op": "gelu"}, {"op": "linear"}]}}
            dataset = [(torch.zeros(1), label) for label in range(2) for _ in range(4)]
            inputs = np.linspace(-3, 3, 2000, dtype=np.float32)
            with patch("webui.experiments.resolve_model", return_value=artifact), \
                 patch("webui.experiments.load_model", return_value=(object(), document)), \
                 patch("webui.experiments.load_data_splits", return_value=(dataset, dataset)), \
                 patch("webui.experiments.record_activation_inputs", return_value=inputs):
                result = manager.recommend_manual_parameters({"model": "model.json",
                    "method": "chebyshev", "degree": 2,
                    "activation_samples_per_class": 2, "activation_sample_limit": 2000})
                cached = manager.recommend_manual_parameters({"model": "model.json",
                    "method": "chebyshev", "degree": 2,
                    "activation_samples_per_class": 2, "activation_sample_limit": 2000})
            self.assertEqual(result["he_parameters"], {"ring_dim": 16384, "depth": 9,
                "scaling_mod_bits": 25, "first_mod_bits": 35})
            self.assertGreater(result["range"], 0)
            self.assertEqual(len(result["polynomial_coefficients"]), 3)
            self.assertEqual(result["activation_input_count"], 2000)
            self.assertTrue(cached["cached"])

    def test_manual_degree_profiles_regenerate_coefficients_for_each_degree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); artifact = root / "model.json"
            artifact.write_text(json.dumps({"activation": "gelu", "architecture": {"layers": [
                {"op": "linear"}, {"op": "gelu"}, {"op": "linear"},
                {"op": "gelu"}, {"op": "linear"}]}}))
            manager = ExperimentManager(root / "results")
            with patch("webui.experiments.resolve_model", return_value=artifact):
                result = manager.manual_degree_profiles({"model": "model.json",
                    "method": "taylor", "range": 10, "degrees": [2, 4, 6, 8]})
            self.assertEqual([item["degree"] for item in result["profiles"]], [2, 4, 6, 8])
            self.assertEqual([len(item["polynomial_coefficients"])
                              for item in result["profiles"]], [3, 5, 7, 9])
            self.assertEqual({item["range"] for item in result["profiles"]}, {10})
            self.assertEqual([item["he_parameters"]["depth"]
                              for item in result["profiles"]], [9, 11, 13, 13])

    def test_weighted_activation_error_and_publication_plots(self):
        inputs = np.linspace(-2, 2, 1000)
        approximation = weighted_approximation(
            "gelu", coefficients("gelu", "chebyshev", 4, 2), 2, inputs, stored_samples=50)
        self.assertEqual(approximation["activation_input_count"], 1000)
        self.assertEqual(len(approximation["sampled_activation_inputs"]), 50)
        self.assertGreater(approximation["weighted_approximation_rmse"], 0)
        self.assertAlmostEqual(approximation["activation_abs_p999"], 1.998, places=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); points = []
            for method in ("taylor", "chebyshev"):
                for degree in (2, 4, 6, 8):
                    points.append({"status": "complete", "method": method, "degree": degree,
                        "baseline_accuracy": 0.98, "plaintext_polynomial_accuracy": 0.95,
                        "ciphertext_accuracy": 0.94, "weighted_approximation_rmse": 0.1 / degree,
                        "observed_approximation_max_error": 0.2 / degree,
                        "plaintext_median_ms": 0.1, "encrypted_median_ms": 1000 * degree,
                        "plaintext_ram_bytes": 2**28, "encrypted_ram_bytes": 2**30,
                        "encrypted_peak_vram_bytes": 2**31})
            (root / "results.json").write_text(json.dumps(points))
            generate_benchmark_plots(root)
            expected = {"accuracy_vs_degree", "approximation_error_vs_degree",
                        "inference_time_vs_degree", "memory_vs_degree", "publication_summary"}
            self.assertEqual({path.stem for path in (root / "benchmark_plots").glob("*.svg")}, expected)
            status = json.loads((root / "benchmark_plots" / "status.json").read_text())
            self.assertEqual(status["points"], 8)

    def test_failed_ciphertext_point_produces_partial_publication_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = [{"status": "partial", "method": "taylor", "degree": 2,
                       "baseline_accuracy": .95, "plaintext_polynomial_accuracy": .9,
                       "weighted_approximation_rmse": .2,
                       "observed_approximation_max_error": .4,
                       "ciphertext_status": "failed", "ciphertext_reason": "level budget"}]
            (root / "results.json").write_text(json.dumps(points))
            generate_benchmark_plots(root)
            status = json.loads((root / "benchmark_plots" / "status.json").read_text())
            self.assertEqual(status["status"], "partial")
            self.assertEqual(status["failures"], 1)

    def test_graph_lab_combines_runs_with_dual_axes_and_saved_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number, activation in enumerate(("tanh", "gelu"), 1):
                run = root / f"run-{number}"; run.mkdir()
                (run / "manifest.json").write_text(json.dumps(
                    {"run_id": f"run-{number}", "model": f"mnist/mlp-{activation}.json"}))
                with (run / "results.csv").open("w", newline="") as output:
                    writer = csv.DictWriter(output, ["degree", "method", "activation",
                        "plaintext_polynomial_accuracy", "encrypted_mean_ms"])
                    writer.writeheader()
                    for degree in (1, 3, 5):
                        writer.writerow({"degree": degree, "method": "chebyshev",
                            "activation": activation,
                            "plaintext_polynomial_accuracy": 0.8 + number / 100 + degree / 100,
                            "encrypted_mean_ms": degree * 1000 * number})
            runs = available_runs(root)
            self.assertEqual(len(runs), 2)
            selected = [run["id"] for run in runs]
            description = graph_schema(root, {"runs": selected, "source": "summary"})
            self.assertEqual(description["rows"], 6)
            self.assertIn("encrypted_mean_ms", description["numeric"])
            graph = render_graph(root, {"runs": selected, "source": "summary",
                "x": "degree", "left_y": ["plaintext_polynomial_accuracy"],
                "right_y": ["encrypted_mean_ms"], "left_type": "line",
                "right_type": "scatter", "hue": "run_label", "style": "activation",
                "filters": {"degree": {"min": 3}}, "theme": "whitegrid",
                "context": "paper", "palette": "colorblind", "aggregation": "none",
                "error_bar": "none", "title": "Combined run comparison"})
            output = root / graph["base"]
            self.assertTrue((output / "graph.png").is_file())
            self.assertTrue((output / "graph.svg").is_file())
            with (output / "data.csv").open() as source:
                self.assertEqual(len(list(csv.DictReader(source))), 4)
            render_saved(output)

    @unittest.skipUnless(os.environ.get("SEALTORCH_GPU_TESTS") == "1",
                         "set SEALTORCH_GPU_TESTS=1 for the FIDESlib numerical smoke test")
    def test_fideslib_logits_match_plaintext_polynomial(self):
        torch.manual_seed(42)
        base = MLP(10, "tanh").eval()
        coeff = coefficients("tanh", "chebyshev", 3, 1)
        expected = polynomial_model(base, coeff, 1)(torch.zeros(1, 1, 28, 28))[0]
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "mlp.json"
            save_artifact(artifact, model=base, model_name="mlp", activation="tanh",
                          dataset="mnist", classes=list("0123456789"),
                          preprocessing={"input_shape": [1, 28, 28], "pad": 0,
                                         "mean": 0.1307, "std": 0.3081},
                          training={"seed": 42}, baseline_metrics={})
            selected = profile({"architecture": {"layers": [
                {"op": "linear"}, {"op": "tanh"}, {"op": "linear"},
                {"op": "tanh"}, {"op": "linear"}
            ]}}, 3)
            worker = NativeWorker()
            try:
                response = worker.predict({"model_path": str(artifact), "pixels": [0.0] * 784,
                    "coefficients": coeff.tolist(), "lower_bound": -1.0, "upper_bound": 1.0,
                    **{key: selected[key] for key in ("ring_dim", "depth", "scaling_mod_bits", "first_mod_bits", "device")}})
            finally:
                worker.close()
            torch.testing.assert_close(torch.tensor(response["logits"]), expected, atol=2e-3, rtol=2e-3)

    @unittest.skipUnless(os.environ.get("SEALTORCH_GPU_TESTS") == "1",
                         "set SEALTORCH_GPU_TESTS=1 for the FIDESlib numerical smoke test")
    def test_fideslib_gelu_quadratic_matches_nonzero_plaintext(self):
        torch.manual_seed(42)
        base = MLP(10, "gelu").eval()
        coeff = coefficients("gelu", "chebyshev", 2, 4)
        image = torch.linspace(-1, 1, 784).reshape(1, 1, 28, 28)
        expected = polynomial_model(base, coeff, 4)(image)[0]
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "mlp.json"
            save_artifact(artifact, model=base, model_name="mlp", activation="gelu",
                          dataset="mnist", classes=list("0123456789"),
                          preprocessing={"input_shape": [1, 28, 28], "pad": 0,
                                         "mean": 0.1307, "std": 0.3081},
                          training={"seed": 42}, baseline_metrics={})
            selected = profile({"architecture": {"layers": [
                {"op": "linear"}, {"op": "gelu"}, {"op": "linear"},
                {"op": "gelu"}, {"op": "linear"}
            ]}}, 2)
            worker = NativeWorker()
            try:
                response = worker.predict({"model_path": str(artifact),
                    "pixels": image.reshape(-1).tolist(),
                    "coefficients": fideslib_coefficients(coeff).tolist(),
                    "lower_bound": -4.0, "upper_bound": 4.0, "stream_weights": 1,
                    **{key: selected[key] for key in ("ring_dim", "depth", "scaling_mod_bits",
                                                       "first_mod_bits", "device")}})
            finally:
                worker.close()
            torch.testing.assert_close(torch.tensor(response["logits"]), expected,
                                       atol=2e-2, rtol=2e-2)


if __name__ == "__main__": unittest.main()
