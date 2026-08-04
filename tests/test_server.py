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
    def test_validate_pixels(self):
        server.validate_pixels([0.5] * 784)
        with self.assertRaises(ValueError):
            server.validate_pixels([0.5] * 783)
        with self.assertRaises(ValueError):
            server.validate_pixels([2.0] * 784)

    def test_trainer_model_produces_exported_number_of_outputs(self):
        pixels = [0.0] * 784
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
                    {"model": "trainer:emnist-lenet5-gelu", "engine": "pytorch", "device": "cpu"},
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


if __name__ == "__main__":
    unittest.main()
