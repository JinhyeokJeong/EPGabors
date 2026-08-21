"""Model construction and layer maps."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import timm
import torch


SPARSE_LAYER_PATHS = {
    "stem": "maxpool",
    "layer1_last": "layer1.2",
    "layer2_last": "layer2.3",
    "layer3_last": "layer3.5",
    "layer4_last": "layer4.2",
}


def list_available_layers(model_name: str = "resnet50") -> pd.DataFrame:
    """List the sparse ResNet50 endpoints exposed by the pipeline."""
    if model_name != "resnet50":
        raise ValueError("Only 'resnet50' is supported in this pipeline.")

    model = timm.create_model("resnet50", pretrained=False)
    rows = []
    for layer_name, module_path in SPARSE_LAYER_PATHS.items():
        module = model.get_submodule(module_path)
        rows.append(
            {
                "layer_name": layer_name,
                "module_path": module_path,
                "module_type": module.__class__.__name__,
            }
        )
    return pd.DataFrame(rows)


def get_resnet50_sparse_layer_map(
    pretrained: bool = True,
    device: str = "cpu",
    model_seed: int | None = None,
) -> Tuple[torch.nn.Module, Dict[str, torch.nn.Module]]:
    """Return a ResNet50 model and sparse endpoint module map."""
    if model_seed is not None:
        torch.manual_seed(int(model_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(model_seed))
    model = timm.create_model("resnet50", pretrained=pretrained)
    model.eval().to(device)

    layer_map: Dict[str, torch.nn.Module] = {}
    for layer_name, module_path in SPARSE_LAYER_PATHS.items():
        layer_map[layer_name] = model.get_submodule(module_path)
    return model, layer_map


def get_model_and_layer_map(
    model_name: str,
    pretrained: bool = True,
    source: str = "timm",
    device: str = "cpu",
    model_seed: int | None = None,
) -> Tuple[torch.nn.Module, Dict[str, torch.nn.Module]]:
    """Backward-compatible model/layer factory."""
    if source != "timm":
        raise ValueError("Only 'timm' is supported.")
    if model_name != "resnet50":
        raise ValueError("Only 'resnet50' is supported in this pipeline.")
    return get_resnet50_sparse_layer_map(pretrained=pretrained, device=device, model_seed=model_seed)


def select_layer_map(
    layer_map: Dict[str, torch.nn.Module],
    selected_layer: str = "all",
) -> Dict[str, torch.nn.Module]:
    """Select one sparse endpoint or all available endpoints."""
    if selected_layer == "all":
        return dict(layer_map)
    if selected_layer not in layer_map:
        valid = ", ".join(layer_map.keys())
        raise ValueError(f"Invalid layer '{selected_layer}'. Valid layers: {valid}")
    return {selected_layer: layer_map[selected_layer]}
