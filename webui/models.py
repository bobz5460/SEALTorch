"""Model discovery, loading, polynomial substitution, and dataset access."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
TRAINER_ROOT = Path(os.environ.get("SEALTORCH_TRAINER_ROOT", ROOT.parent / "LeNet-5")).resolve()


def _trainer_imports():
    if str(TRAINER_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAINER_ROOT))
    from data import load_datasets
    from export_model import load_artifact
    return load_datasets, load_artifact


def discover_models() -> list[dict]:
    models = []
    export_root = TRAINER_ROOT / "exports"
    if not export_root.is_dir():
        return models
    for path in sorted(export_root.glob("*/*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("format") != "sealtorch-model-v1":
                continue
            models.append({
                "id": str(path.relative_to(export_root)), "path": str(path),
                "dataset": document["dataset"], "model": document["model"],
                "activation": document["activation"], "classes": document["classes"],
                "baseline_accuracy": document.get("baseline_metrics", {}).get("test_accuracy"),
                "label": f'{document["dataset"].upper()} · {document["model"]} · {document["activation"]}',
            })
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return models


def resolve_model(model_id: str) -> Path:
    export_root = (TRAINER_ROOT / "exports").resolve()
    path = (export_root / model_id).resolve()
    if export_root not in path.parents or not path.is_file():
        raise ValueError("unknown model artifact")
    return path


def load_model(path: str | Path, device="cpu"):
    _, loader = _trainer_imports()
    return loader(path, device)


def load_test_data(document: dict, root: str | Path | None = None):
    dataset_loader, _ = _trainer_imports()
    _, test, _ = dataset_loader(document["dataset"], document["model"], root or TRAINER_ROOT / "data")
    return test


def load_data_splits(document: dict, root: str | Path | None = None):
    """Return the trainer's canonical train/test splits with identical preprocessing."""
    dataset_loader, _ = _trainer_imports()
    train, test, _ = dataset_loader(
        document["dataset"], document["model"], root or TRAINER_ROOT / "data")
    return train, test


class PolynomialActivation(nn.Module):
    def __init__(self, coefficients, interval: float) -> None:
        super().__init__()
        self.register_buffer("coefficients", torch.as_tensor(coefficients, dtype=torch.float32))
        self.interval = interval

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = values / self.interval
        if len(self.coefficients) == 1:
            return torch.full_like(values, self.coefficients[0])
        previous = torch.ones_like(values)
        current = normalized
        result = self.coefficients[0] * previous + self.coefficients[1] * current
        for order in range(2, len(self.coefficients)):
            following = 2 * normalized * current - previous
            result = result + self.coefficients[order] * following
            previous, current = current, following
        return result


def polynomial_model(model: nn.Module, coefficients, interval: float) -> nn.Module:
    result = copy.deepcopy(model)
    activation_types = (nn.ReLU, nn.GELU, nn.Tanh)
    for parent in result.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, activation_types):
                setattr(parent, name, PolynomialActivation(coefficients, interval))
    return result


def stratified_indices(dataset, class_count: int, per_class: int = 10,
                       seed: int | None = None) -> list[int]:
    """Select an exactly class-balanced subset, optionally shuffled reproducibly."""
    candidates = [[] for _ in range(class_count)]
    for index in range(len(dataset)):
        target = int(dataset[index][1])
        candidates[target].append(index)
    if any(len(items) < per_class for items in candidates):
        raise ValueError("dataset does not contain enough examples for a stratified subset")
    if seed is None:
        selected = [items[:per_class] for items in candidates]
    else:
        generator = np.random.default_rng(seed)
        selected = [generator.choice(items, per_class, replace=False).tolist()
                    for items in candidates]
    return [index for items in selected for index in items]


def prepare_drawing(pixels: list[float], document: dict) -> torch.Tensor:
    if len(pixels) != 28 * 28:
        raise ValueError("drawing must contain 784 pixels")
    metadata = document["preprocessing"]
    values = torch.tensor(pixels, dtype=torch.float32).reshape(1, 28, 28)
    if metadata.get("pad"):
        values = torch.nn.functional.pad(values, (2, 2, 2, 2))
    return (values - float(metadata["mean"])) / float(metadata["std"])
