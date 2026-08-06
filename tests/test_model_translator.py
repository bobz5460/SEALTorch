import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "model_translator", ROOT / "webui" / "model_translator.py")
translator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = translator
SPEC.loader.exec_module(translator)


def manifest(out_features=2):
    return {
        "format_version": 1,
        "architecture": {
            "input": {"shape": [1, 1, 1, 2]},
            "layers": [{
                "name": "classifier.0", "op": "linear",
                "weight_key": "classifier.0.weight", "bias_key": "classifier.0.bias",
                "in_features": 2, "out_features": out_features,
            }],
        },
        "classes": list(range(out_features)),
    }


class ModelTranslatorTests(unittest.TestCase):
    def test_paired_pt_uses_json_topology_and_pt_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "model"
            paired_manifest = manifest(2)
            root.with_suffix(".json").write_text(json.dumps(paired_manifest))
            # Deliberately put incompatible metadata in the checkpoint.  The
            # manifest is the layer/dimension authority for the native graph.
            torch.save({"architecture": manifest(3)["architecture"], "state_dict": {
                "classifier.0.weight": torch.tensor([[1., 2.], [3., 4.]]),
                "classifier.0.bias": torch.tensor([5., 6.]),
            }}, root.with_suffix(".pt"))

            export = translator.load_export(root.with_suffix(".pt"))
            self.assertEqual(export.manifest, paired_manifest)
            self.assertEqual(export.manifest_source, root.with_suffix(".json"))
            self.assertEqual(export.weights_source, root.with_suffix(".pt"))
            artifact = json.loads(translator.native_artifact(export, directory).read_text())
            self.assertEqual(artifact["architecture"], paired_manifest["architecture"])
            self.assertEqual(artifact["tensors"]["classifier.0.weight"], [[1., 2.], [3., 4.]])

    def test_paired_manifest_and_checkpoint_must_agree_on_tensor_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "model"
            root.with_suffix(".json").write_text(json.dumps(manifest(3)))
            torch.save({"state_dict": {
                "classifier.0.weight": torch.zeros(2, 2),
                "classifier.0.bias": torch.zeros(2),
            }}, root.with_suffix(".pt"))
            with self.assertRaisesRegex(ValueError, "classifier.0.weight"):
                translator.load_export(root.with_suffix(".pt"))


if __name__ == "__main__":
    unittest.main()
