import gzip
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
    def test_percentile_interpolates(self):
        self.assertEqual(server.percentile([1, 2, 3], 0.5), 2)
        self.assertEqual(server.percentile([], 0.5), 0)

    def test_validate_pixels(self):
        server.validate_pixels([0.5] * 784)
        with self.assertRaises(ValueError):
            server.validate_pixels([0.5] * 783)
        with self.assertRaises(ValueError):
            server.validate_pixels([2.0] * 784)

    def test_read_idx_rejects_truncated_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "images.gz"
            with gzip.open(path, "wb") as output:
                output.write((2051).to_bytes(4, "big"))
                output.write((1).to_bytes(4, "big"))
                output.write((28).to_bytes(4, "big"))
                output.write((28).to_bytes(4, "big"))
                output.write(b"\0" * 10)
            with self.assertRaises(ValueError):
                server.read_idx(path, 2051)

    def test_plaintext_models_produce_ten_outputs(self):
        pixels = [0.0] * 784
        for name in ("relu", "gelu", "lenet"):
            output, _, device, _ = server.run_plaintext(
                pixels, name, "cpu")
            self.assertEqual(device, "cpu")
            self.assertEqual(len(output), 10)
            self.assertTrue(all(math.isfinite(value) for value in output))


if __name__ == "__main__":
    unittest.main()
