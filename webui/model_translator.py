"""Load LeNet-5 trainer exports and produce the native worker representation.

The trainer's JSON is intentionally a manifest: its tensors live in the
neighbouring ``*.weights.npz`` file.  The native worker only has a small JSON
reader, so this module joins the manifest and tensors (or a self-describing
PyTorch bundle) into one cached JSON artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


@dataclass(frozen=True)
class LeNetExport:
    source: Path
    manifest: dict[str, Any]
    state_dict: dict[str, torch.Tensor]


def load_export(path: str | Path) -> LeNetExport:
    """Read either trainer artifact, validating the paired portable weights."""
    source = Path(path).expanduser().resolve()
    if source.suffix == ".pt":
        bundle = torch.load(source, map_location="cpu", weights_only=False)
        manifest = {key: value for key, value in bundle.items() if key != "state_dict"}
        state = bundle.get("state_dict")
    elif source.suffix == ".json":
        import numpy as np
        manifest = json.loads(source.read_text(encoding="utf-8"))
        weights = manifest.get("weights", {})
        if weights.get("format") != "npz" or not weights.get("file"):
            raise ValueError("LeNet JSON exports must reference a .weights.npz file")
        with np.load(source.parent / weights["file"], allow_pickle=False) as archive:
            state = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
    else:
        raise ValueError("model must be a LeNet trainer .pt or .json export")
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("architecture"), dict):
        raise ValueError("not a supported self-describing LeNet trainer export")
    if not isinstance(state, dict):
        raise ValueError("LeNet export has no state_dict")
    return LeNetExport(source, manifest, state)


def native_artifact(export: LeNetExport, cache_dir: str | Path | None = None) -> Path:
    """Materialize the worker's self-contained artifact and return its path."""
    fingerprint = hashlib.sha256()
    fingerprint.update(export.source.read_bytes())
    if export.source.suffix == ".json":
        fingerprint.update((export.source.parent / export.manifest["weights"]["file"]).read_bytes())
    key = fingerprint.hexdigest()[:24]
    directory = Path(cache_dir or Path(tempfile.gettempdir()) / "sealtorch-models")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{key}.json"
    if destination.exists():
        return destination
    architecture = json.loads(json.dumps(export.manifest["architecture"]))
    state = {key: value.detach().cpu().float().clone()
             for key, value in export.state_dict.items()}
    # Batch norm is affine in eval mode. Fold it into its preceding Conv/Linear
    # so encrypted inference does not need a dense per-pixel diagonal layer.
    layers = []
    for layer in architecture["layers"]:
        if layer["op"] in ("batch_norm1d", "batch_norm2d"):
            if not layers or layers[-1]["op"] not in ("conv2d", "linear"):
                raise ValueError("batch norm must follow a Conv2d or Linear layer")
            previous = layers[-1]
            prefix = layer["name"]
            factor = state[prefix + ".weight"] / torch.sqrt(
                state[prefix + ".running_var"] + layer.get("eps", 1e-5))
            weight = state[previous["weight_key"]]
            state[previous["weight_key"]] = weight * factor.reshape(
                (-1,) + (1,) * (weight.ndim - 1))
            state[previous["bias_key"]] = (
                (state[previous["bias_key"]] - state[prefix + ".running_mean"]) * factor
                + state[prefix + ".bias"])
        elif layer["op"] != "dropout":  # inference behavior is identity
            layers.append(layer)
    architecture["layers"] = layers
    artifact = {
        "format": "sealtorch-lenet-trainer-v1",
        "architecture": architecture,
        "preprocessing": export.manifest.get("preprocessing", {}),
        "classes": export.manifest.get("classes", []),
        "tensors": {key: value.tolist() for key, value in state.items()},
    }
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def export_info(path: str | Path) -> dict[str, Any]:
    export = load_export(path)
    architecture = export.manifest["architecture"]
    return {
        "source": str(export.source),
        "classes": export.manifest.get("classes", []),
        "input_shape": architecture.get("input", {}).get("shape"),
        "config": architecture.get("config", {}),
        "operations": [layer.get("op") for layer in architecture.get("layers", [])],
        "preprocessing": export.manifest.get("preprocessing", {}),
    }
