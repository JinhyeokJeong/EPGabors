"""Preprocessing and feature extraction utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def make_deterministic_transform(input_size: int = 224) -> transforms.Compose:
    """Deterministic ImageNet-style preprocessing without augmentation."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _flatten_activation(activation: torch.Tensor, feature_mode: str = "flatten") -> torch.Tensor:
    if feature_mode != "flatten":
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    if activation.ndim < 2:
        raise ValueError(f"Expected activation with ndim >= 2, got {activation.ndim}")
    return activation.flatten(start_dim=1)


def _labels_to_dataframe(label_batches: List[Dict]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for labels in label_batches:
        data = {}
        for key, value in labels.items():
            if torch.is_tensor(value):
                data[key] = value.detach().cpu().numpy()
            else:
                data[key] = np.asarray(value)
        rows.append(pd.DataFrame(data))
    return pd.concat(rows, ignore_index=True)


def extract_layer_features(
    model: torch.nn.Module,
    layer_map: Dict[str, torch.nn.Module],
    dataloader: DataLoader,
    device: str = "cpu",
    feature_mode: str = "flatten",
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Extract endpoint activations for all samples in a dataloader."""
    batch_features = {layer_name: [] for layer_name in layer_map}
    label_batches: List[Dict] = []
    activations: Dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(layer_name: str):
        def hook(_, __, output):
            activations[layer_name] = output

        return hook

    for layer_name, module in layer_map.items():
        hooks.append(module.register_forward_hook(_make_hook(layer_name)))

    try:
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device=device, dtype=torch.float32)
                activations.clear()
                _ = model(images)

                for layer_name in layer_map:
                    flattened = _flatten_activation(activations[layer_name], feature_mode=feature_mode)
                    batch_features[layer_name].append(flattened.detach().cpu().numpy())

                label_batches.append(labels)
    finally:
        for hook in hooks:
            hook.remove()

    layer_features = {
        layer_name: np.concatenate(feature_list, axis=0)
        for layer_name, feature_list in batch_features.items()
    }
    metadata = _labels_to_dataframe(label_batches)
    return layer_features, metadata
