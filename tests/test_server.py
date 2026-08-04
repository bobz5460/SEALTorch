import importlib.util
import math
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sealtorch_server", ROOT / "webui" / "server.py")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class ServerTests(unittest.TestCase):
    def test_duplicate_export_names_get_distinct_model_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for group in ("original", "clamped"):
                path = root / "exports" / group / "same_name.pt"
                path.parent.mkdir(parents=True)
                path.touch()
            previous = server.LENET_ROOT
            try:
                server.LENET_ROOT = root
                models = server.available_models()
            finally:
                server.LENET_ROOT = previous
        self.assertIn("trainer:original/same_name", models)
        self.assertIn("trainer:clamped/same_name", models)

    def test_validate_pixels(self):
        server.validate_pixels([0.5] * 784)
        with self.assertRaises(ValueError):
            server.validate_pixels([0.5] * 783)
        with self.assertRaises(ValueError):
            server.validate_pixels([2.0] * 784)

    def test_trainer_model_produces_exported_number_of_outputs(self):
        pixels = [0.0] * 784
        self.assertIn(server.model_activation("trainer:lenet5_mnist"),
                      ("relu", "gelu", "tanh"))
        output, _, device, _ = server.run_plaintext(
            pixels, "trainer:lenet5_mnist", "cpu")
        self.assertEqual(device, "cpu")
        self.assertEqual(len(output), 10)
        self.assertTrue(all(math.isfinite(value) for value in output))

    def test_emnist_web_input_skips_dataset_orientation_and_centers_ink(self):
        pixels = [0.0] * 784
        row, column = 3, 17
        pixels[row * 28 + column] = 1.0

        image = server.prepare_trainer_pixels(
            pixels, "trainer:emnist-lenet5-gelu")

        # Dataset-only orientation correction must not rotate browser drawings.
        # The trainer's foreground-normalization operation then puts the glyph
        # into the centered 20×20 MNIST content box, before its two-pixel pad.
        self.assertEqual(image.shape, (1, 1, 32, 32))
        center = 2 + (28 - 20) // 2 + 10
        self.assertGreater(float(image[0, 0, center, center]), 0)
        self.assertLessEqual(float(image[0, 0, row + 2, column + 2]), 0)

    def test_benchmark_uses_selected_model_dataset_and_class_count(self):
        job_id = "emnist-test"
        server.benchmark_jobs[job_id] = {
            "id": job_id,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            old_results = server.RESULTS_DIR
            try:
                server.RESULTS_DIR = pathlib.Path(directory)
                server.benchmark_runner(
                    job_id,
                    {
                        "model": "trainer:emnist-lenet5-gelu",
                        "engine": "pytorch",
                        "device": "cpu",
                        "activation_range": "2",
                    },
                    limit=1,
                    batch_size=1,
                )
            finally:
                server.RESULTS_DIR = old_results
        result = server.benchmark_jobs.pop(job_id)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["dataset"], "emnist-byclass")
        self.assertEqual(len(result["classes"]), 62)
        self.assertEqual(len(result["confusion_matrix"]), 62)
        self.assertEqual(result["config"]["activation_range"], "2")

    def test_saved_result_loader_stays_inside_results_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            previous = server.RESULTS_DIR
            server.RESULTS_DIR = pathlib.Path(directory)
            try:
                path = server.RESULTS_DIR / "mnist-benchmark-example.json"
                path.write_text('{"id":"example","confusion_matrix":[[1]]}')
                entries = server.saved_benchmark_results()
                self.assertEqual(entries[0]["file"], path.name)
                self.assertEqual(server.saved_benchmark_result(path.name)["id"], "example")
                with self.assertRaises(ValueError):
                    server.saved_benchmark_result("../outside.json")
            finally:
                server.RESULTS_DIR = previous

    def test_ckks_decode_failure_has_actionable_benchmark_message(self):
        message = server.benchmark_error_message(
            RuntimeError("Decode(): The decryption failed because the approximation error is too high."),
            {"activation_degree": "3"})
        self.assertIn("no benchmark samples were recorded", message)
        self.assertIn("ring 65536", message)
        self.assertIn("degree is 3", message)

    def test_he_native_emnist_export_advertises_its_exact_cuda_profile(self):
        name = "trainer:emnist-he/lenet_he_emnist-byclass"
        if name not in server.available_models():
            self.skipTest("HE-native EMNIST export is not installed")
        self.assertEqual(server.model_activation(name), "poly_gelu2")
        self.assertEqual(server.model_he_activation_degree(name), 2)
        self.assertEqual(server.model_he_profile(name)["ring_dim"], 8192)


if __name__ == "__main__":
    unittest.main()
