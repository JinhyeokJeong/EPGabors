"""v1.1 pipeline for layerwise mean-orientation decoding in pretrained CNNs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torchvision import transforms

from EPOriGabors import EPGabors


DEFAULT_MODEL_NAMES = ["resnet50", "convnext_tiny", "efficientnet_b0", "vgg16_bn"]


@dataclass
class SplitSpec:
    split_id: str
    train_mask: np.ndarray
    test_nonzero_mask: np.ndarray
    boundary_mask: np.ndarray
    held_out_axis: str
    held_out_value: Any


@dataclass
class ConditionFilter:
    means: Optional[List[int]] = None
    sds: Optional[List[int]] = None
    sss: Optional[List[int]] = None
    instances: Optional[List[int]] = None
    exclude_means: Optional[List[int]] = None
    exclude_sds: Optional[List[int]] = None
    exclude_sss: Optional[List[int]] = None
    exclude_instances: Optional[List[int]] = None


def make_deterministic_transform(input_size: int = 224) -> transforms.Compose:
    """Deterministic preprocessing without crop/rotation/augmentation."""
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


def _get_layer_path_map(model_name: str) -> Dict[str, str]:
    if model_name == "resnet50":
        return {
            "stem": "maxpool",
            "stage1": "layer1",
            "stage2": "layer2",
            "stage3": "layer3",
            "stage4": "layer4",
        }
    if model_name == "convnext_tiny":
        return {
            "stem": "stem",
            "stage1": "stages.0",
            "stage2": "stages.1",
            "stage3": "stages.2",
            "stage4": "stages.3",
        }
    if model_name == "efficientnet_b0":
        return {
            "stem": "bn1",
            "stage1": "blocks.1",
            "stage2": "blocks.2",
            "stage3": "blocks.4",
            "stage4": "blocks.6",
            "head": "bn2",
        }
    if model_name == "vgg16_bn":
        return {
            "stage1": "features.6",
            "stage2": "features.13",
            "stage3": "features.23",
            "stage4": "features.33",
            "stage5": "features.43",
        }
    raise ValueError(f"Unsupported model_name: {model_name}")


def list_available_layers(model_name: str, source: str = "timm") -> pd.DataFrame:
    """List pipeline-facing layer names and the underlying module paths."""
    if source != "timm":
        raise ValueError(f"Unsupported source: {source}")

    model = timm.create_model(model_name, pretrained=False)
    rows = []
    for layer_name, module_path in _get_layer_path_map(model_name).items():
        module = model.get_submodule(module_path)
        rows.append(
            {
                "layer_name": layer_name,
                "module_path": module_path,
                "module_type": module.__class__.__name__,
            }
        )
    return pd.DataFrame(rows)


def get_model_and_layer_map(
    model_name: str,
    pretrained: bool = True,
    source: str = "timm",
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, torch.nn.Module]]:
    """Return model and stage-end layer modules for feature extraction."""
    if source != "timm":
        raise ValueError(f"Unsupported source: {source}")

    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval().to(device)

    path_map = _get_layer_path_map(model_name)
    layer_map: Dict[str, torch.nn.Module] = {}
    for layer_name, module_path in path_map.items():
        try:
            layer_map[layer_name] = model.get_submodule(module_path)
        except AttributeError as exc:
            raise ValueError(
                f"Could not resolve module path '{module_path}' for '{model_name}'."
            ) from exc
    return model, layer_map


def select_layer_map(
    layer_map: Dict[str, torch.nn.Module],
    selected_layer: str = "all",
) -> Dict[str, torch.nn.Module]:
    if selected_layer == "all":
        return dict(layer_map)
    if selected_layer not in layer_map:
        valid = ", ".join(layer_map.keys())
        raise ValueError(f"Invalid layer '{selected_layer}'. Valid layers: {valid}")
    return {selected_layer: layer_map[selected_layer]}


def _pool_activation(activation: torch.Tensor, pooling: str = "gap") -> torch.Tensor:
    if pooling != "gap":
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    if activation.ndim == 2:
        return activation
    if activation.ndim == 3:
        return activation.mean(dim=1)
    if activation.ndim == 4:
        return activation.mean(dim=(2, 3))
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
    pooling: str = "gap",
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Extract pooled layer features for all samples in a dataloader."""
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
                    pooled = _pool_activation(activations[layer_name], pooling=pooling)
                    batch_features[layer_name].append(pooled.detach().cpu().numpy())

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


def _eligible_mask(metadata: pd.DataFrame, condition_filter: Optional[ConditionFilter]) -> np.ndarray:
    mask = np.ones(len(metadata), dtype=bool)
    if condition_filter is None:
        return mask

    if condition_filter.means is not None:
        mask &= metadata["mean"].isin(condition_filter.means).to_numpy()
    if condition_filter.sds is not None:
        mask &= metadata["sd"].isin(condition_filter.sds).to_numpy()
    if condition_filter.sss is not None:
        mask &= metadata["ss"].isin(condition_filter.sss).to_numpy()
    if condition_filter.instances is not None:
        mask &= metadata["instance"].isin(condition_filter.instances).to_numpy()

    if condition_filter.exclude_means is not None:
        mask &= ~metadata["mean"].isin(condition_filter.exclude_means).to_numpy()
    if condition_filter.exclude_sds is not None:
        mask &= ~metadata["sd"].isin(condition_filter.exclude_sds).to_numpy()
    if condition_filter.exclude_sss is not None:
        mask &= ~metadata["ss"].isin(condition_filter.exclude_sss).to_numpy()
    if condition_filter.exclude_instances is not None:
        mask &= ~metadata["instance"].isin(condition_filter.exclude_instances).to_numpy()

    return mask


def build_controlled_splits(
    metadata: pd.DataFrame,
    split_mode: str = "leave_one_sd_out",
    holdout_values: Optional[Iterable[int]] = None,
    train_filter: Optional[ConditionFilter] = None,
    test_filter: Optional[ConditionFilter] = None,
) -> List[SplitSpec]:
    """Build train/test masks with explicit eligibility control."""
    train_eligible = _eligible_mask(metadata, train_filter)
    test_eligible = _eligible_mask(metadata, test_filter)

    mean_values = metadata["mean"].to_numpy()

    if split_mode == "explicit":
        train_mask = train_eligible & (mean_values != 0)
        test_nonzero_mask = test_eligible & (mean_values != 0)
        boundary_mask = test_eligible & (mean_values == 0)
        if np.any(train_mask & test_nonzero_mask) or np.any(train_mask & boundary_mask):
            raise ValueError("Explicit train/test filters overlap. Make them disjoint.")
        return [
            SplitSpec(
                split_id="explicit",
                train_mask=train_mask,
                test_nonzero_mask=test_nonzero_mask,
                boundary_mask=boundary_mask,
                held_out_axis="explicit",
                held_out_value="explicit",
            )
        ]

    if split_mode not in {"leave_one_sd_out", "leave_one_ss_out"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    axis = "sd" if split_mode == "leave_one_sd_out" else "ss"
    if holdout_values is None:
        holdout_values = sorted(int(v) for v in metadata[axis].unique())
    else:
        holdout_values = [int(v) for v in holdout_values]

    axis_values = metadata[axis].to_numpy()
    splits: List[SplitSpec] = []
    for value in holdout_values:
        in_holdout = axis_values == value
        train_mask = train_eligible & (~in_holdout) & (mean_values != 0)
        test_nonzero_mask = test_eligible & in_holdout & (mean_values != 0)
        boundary_mask = test_eligible & in_holdout & (mean_values == 0)

        if np.any(train_mask & test_nonzero_mask) or np.any(train_mask & boundary_mask):
            raise ValueError(
                f"Train/test overlap detected for split {axis}{value}. Check filters."
            )

        splits.append(
            SplitSpec(
                split_id=f"{axis}{value}",
                train_mask=train_mask,
                test_nonzero_mask=test_nonzero_mask,
                boundary_mask=boundary_mask,
                held_out_axis=axis,
                held_out_value=value,
            )
        )
    return splits


def build_cross_condition_splits(
    metadata: pd.DataFrame,
    split_mode: str = "leave_one_sd_out",
    holdout_values: Optional[Iterable[int]] = None,
) -> List[SplitSpec]:
    """Backward-compatible wrapper around the controlled split builder."""
    return build_controlled_splits(
        metadata=metadata,
        split_mode=split_mode,
        holdout_values=holdout_values,
        train_filter=None,
        test_filter=None,
    )


def fit_linear_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
    random_state: int = 0,
    max_iter: int = 10000,
) -> Dict:
    """Train/evaluate a LinearSVC with train-only feature scaling."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = LinearSVC(
        C=C,
        class_weight="balanced",
        random_state=random_state,
        max_iter=max_iter,
        dual="auto",
    )
    svm.fit(X_train_scaled, y_train)

    y_pred = svm.predict(X_test_scaled)
    margins = svm.decision_function(X_test_scaled)

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_test)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_test, margins))
    else:
        metrics["auroc"] = np.nan

    return {
        "estimator": svm,
        "scaler": scaler,
        "y_pred": y_pred,
        "margins": margins,
        "metrics": metrics,
    }


def evaluate_boundary_m0(
    estimator: LinearSVC,
    scaler: StandardScaler,
    X_boundary: np.ndarray,
    metadata_boundary: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate m=0 boundary cases using SVM hyperplane distance."""
    if len(X_boundary) == 0:
        empty_cols = list(metadata_boundary.columns) + ["pred_class", "margin", "choice_cw"]
        return pd.DataFrame(columns=empty_cols), {
            "boundary_n": 0.0,
            "boundary_cw_rate": np.nan,
            "boundary_margin_mean": np.nan,
            "boundary_margin_std": np.nan,
        }

    X_boundary_scaled = scaler.transform(X_boundary)
    pred = estimator.predict(X_boundary_scaled)
    margins = estimator.decision_function(X_boundary_scaled)

    boundary_trials = metadata_boundary.copy()
    boundary_trials["pred_class"] = pred.astype(int)
    boundary_trials["margin"] = margins.astype(float)
    boundary_trials["choice_cw"] = (boundary_trials["pred_class"] == 1).astype(int)

    summary = {
        "boundary_n": float(len(boundary_trials)),
        "boundary_cw_rate": float(boundary_trials["choice_cw"].mean()),
        "boundary_margin_mean": float(boundary_trials["margin"].mean()),
        "boundary_margin_std": float(boundary_trials["margin"].std(ddof=1))
        if len(boundary_trials) > 1
        else 0.0,
    }
    return boundary_trials, summary


def save_trial_outputs(trial_df: pd.DataFrame, output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    trial_df.to_csv(output_path, index=False)


def save_condition_summaries(condition_df: pd.DataFrame, output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    condition_df.to_csv(output_path, index=False)


def save_decoder_artifacts(
    artifact_dir: str,
    estimator: LinearSVC,
    scaler: StandardScaler,
    split_manifest: pd.DataFrame,
    run_config: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(
        {
            "estimator": estimator,
            "scaler": scaler,
            "metadata": metadata,
        },
        os.path.join(artifact_dir, "decoder_bundle.joblib"),
    )
    split_manifest.to_csv(os.path.join(artifact_dir, "split_manifest.csv"), index=False)
    with open(os.path.join(artifact_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")


def _aggregate_condition_summary(trial_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_name",
        "layer_name",
        "split_id",
        "subset",
        "mean",
        "sd",
        "ss",
    ]
    out = (
        trial_df.groupby(group_cols, dropna=False)
        .agg(
            n=("choice_cw", "size"),
            cw_rate=("choice_cw", "mean"),
            margin_mean=("margin", "mean"),
            margin_std=("margin", "std"),
        )
        .reset_index()
    )
    return out


def _build_split_manifest(metadata: pd.DataFrame, split: SplitSpec) -> pd.DataFrame:
    manifest = metadata[["file_name", "mean", "sd", "ss", "instance"]].copy()
    manifest["subset"] = "excluded"
    manifest.loc[split.train_mask, "subset"] = "train"
    manifest.loc[split.test_nonzero_mask, "subset"] = "test_nonzero"
    manifest.loc[split.boundary_mask, "subset"] = "test_boundary_m0"
    manifest["split_id"] = split.split_id
    manifest["held_out_axis"] = split.held_out_axis
    manifest["held_out_value"] = split.held_out_value
    return manifest


def _condition_filter_to_dict(condition_filter: Optional[ConditionFilter]) -> Optional[Dict[str, Any]]:
    if condition_filter is None:
        return None
    return asdict(condition_filter)


def run_layerwise_binary_decoding(
    dataset: EPGabors,
    model_name: str,
    output_dir: str,
    split_mode: str = "leave_one_sd_out",
    holdout_values: Optional[Iterable[int]] = None,
    pretrained: bool = True,
    device: str = "cpu",
    batch_size: int = 64,
    num_workers: int = 0,
    svm_c: float = 1.0,
    random_state: int = 0,
    selected_layer: str = "all",
    train_filter: Optional[ConditionFilter] = None,
    test_filter: Optional[ConditionFilter] = None,
    save_artifacts: bool = True,
    run_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run v1.1 decoding for one model and one layer or all mapped layers."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model, full_layer_map = get_model_and_layer_map(
        model_name=model_name,
        pretrained=pretrained,
        source="timm",
        device=device,
    )
    layer_map = select_layer_map(full_layer_map, selected_layer=selected_layer)
    layer_features, metadata = extract_layer_features(
        model=model,
        layer_map=layer_map,
        dataloader=dataloader,
        device=device,
        pooling="gap",
    )

    required_cols = {"mean", "sd", "ss", "instance"}
    missing = required_cols - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")
    if "file_name" not in metadata.columns:
        metadata["file_name"] = np.arange(len(metadata)).astype(str)

    metadata["target"] = (metadata["mean"] > 0).astype(int)
    splits = build_controlled_splits(
        metadata=metadata,
        split_mode=split_mode,
        holdout_values=holdout_values,
        train_filter=train_filter,
        test_filter=test_filter,
    )

    all_trials: List[pd.DataFrame] = []
    metric_rows: List[Dict] = []

    for split in splits:
        for layer_name, X in layer_features.items():
            train_idx = split.train_mask
            test_idx = split.test_nonzero_mask
            boundary_idx = split.boundary_mask

            y_train = metadata.loc[train_idx, "target"].to_numpy()
            y_test = metadata.loc[test_idx, "target"].to_numpy()

            if train_idx.sum() == 0 or test_idx.sum() == 0:
                continue
            if len(np.unique(y_train)) < 2:
                continue

            svm_result = fit_linear_svm(
                X_train=X[train_idx],
                y_train=y_train,
                X_test=X[test_idx],
                y_test=y_test,
                C=svm_c,
                random_state=random_state,
            )

            nonzero_trials = metadata.loc[test_idx].copy()
            nonzero_trials["pred_class"] = svm_result["y_pred"].astype(int)
            nonzero_trials["margin"] = svm_result["margins"].astype(float)
            nonzero_trials["choice_cw"] = (nonzero_trials["pred_class"] == 1).astype(int)
            nonzero_trials["subset"] = "test_nonzero"
            nonzero_trials["model_name"] = model_name
            nonzero_trials["layer_name"] = layer_name
            nonzero_trials["split_id"] = split.split_id

            boundary_trials, boundary_summary = evaluate_boundary_m0(
                estimator=svm_result["estimator"],
                scaler=svm_result["scaler"],
                X_boundary=X[boundary_idx],
                metadata_boundary=metadata.loc[boundary_idx],
            )
            boundary_trials["subset"] = "test_boundary_m0"
            boundary_trials["model_name"] = model_name
            boundary_trials["layer_name"] = layer_name
            boundary_trials["split_id"] = split.split_id

            all_trials.extend([nonzero_trials, boundary_trials])

            metric_rows.append(
                {
                    "model_name": model_name,
                    "layer_name": layer_name,
                    "split_id": split.split_id,
                    "held_out_axis": split.held_out_axis,
                    "held_out_value": split.held_out_value,
                    "train_n": int(train_idx.sum()),
                    "test_nonzero_n": int(test_idx.sum()),
                    "balanced_accuracy": svm_result["metrics"]["balanced_accuracy"],
                    "f1": svm_result["metrics"]["f1"],
                    "auroc": svm_result["metrics"]["auroc"],
                    "confusion_matrix": str(svm_result["metrics"]["confusion_matrix"]),
                    "boundary_n": boundary_summary["boundary_n"],
                    "boundary_cw_rate": boundary_summary["boundary_cw_rate"],
                    "boundary_margin_mean": boundary_summary["boundary_margin_mean"],
                    "boundary_margin_std": boundary_summary["boundary_margin_std"],
                }
            )

            if save_artifacts:
                artifact_dir = os.path.join(output_dir, model_name, layer_name, split.split_id)
                split_manifest = _build_split_manifest(metadata, split)
                effective_config = dict(run_config or {})
                if not effective_config:
                    effective_config = {}
                effective_config.update(
                    {
                        "model_name": model_name,
                        "layer_name": layer_name,
                        "split_id": split.split_id,
                        "held_out_axis": split.held_out_axis,
                        "held_out_value": split.held_out_value,
                        "split_mode": split_mode,
                        "holdout_values": list(holdout_values) if holdout_values is not None else None,
                        "pretrained": pretrained,
                        "device": device,
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "svm_c": svm_c,
                        "random_state": random_state,
                        "train_filter": _condition_filter_to_dict(train_filter),
                        "test_filter": _condition_filter_to_dict(test_filter),
                    }
                )
                save_decoder_artifacts(
                    artifact_dir=artifact_dir,
                    estimator=svm_result["estimator"],
                    scaler=svm_result["scaler"],
                    split_manifest=split_manifest,
                    run_config=effective_config,
                    metadata={
                        "model_name": model_name,
                        "layer_name": layer_name,
                        "split_id": split.split_id,
                        "metrics": metric_rows[-1],
                    },
                )

    if not metric_rows:
        raise ValueError("No valid layer/split runs were produced. Check filters and split settings.")

    trial_df = pd.concat(all_trials, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    condition_df = _aggregate_condition_summary(trial_df)

    layer_label = selected_layer if selected_layer != "all" else "all_layers"
    prefix = f"{model_name}_{layer_label}_{split_mode}"
    save_trial_outputs(trial_df, os.path.join(output_dir, f"{prefix}_trial_outputs.csv"))
    save_condition_summaries(
        condition_df, os.path.join(output_dir, f"{prefix}_condition_summary.csv")
    )
    save_condition_summaries(
        metrics_df, os.path.join(output_dir, f"{prefix}_layer_metrics.csv")
    )
    return trial_df, condition_df, metrics_df


def run_default_v1_panel(
    img_dir: str,
    output_dir: str,
    models: Optional[Iterable[str]] = None,
    input_size: int = 224,
    split_mode: str = "leave_one_sd_out",
    holdout_values: Optional[Iterable[int]] = None,
    pretrained: bool = True,
    device: str = "cpu",
    batch_size: int = 64,
    num_workers: int = 0,
    svm_c: float = 1.0,
    random_state: int = 0,
    include_single: bool = False,
    include_zerovar: bool = False,
    selected_layer: str = "all",
    train_filter: Optional[ConditionFilter] = None,
    test_filter: Optional[ConditionFilter] = None,
    save_artifacts: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run v1 defaults over a selected model panel."""
    transform = make_deterministic_transform(input_size=input_size)
    dataset = EPGabors(
        img_dir=img_dir,
        include_vertical=True,
        include_single=include_single,
        include_zerovar=include_zerovar,
        transform=transform,
        return_filename=True,
        return_condition_id=True,
    )

    model_list = list(models) if models is not None else list(DEFAULT_MODEL_NAMES)
    metrics_outputs = {}
    for model_name in model_list:
        _, _, metrics_df = run_layerwise_binary_decoding(
            dataset=dataset,
            model_name=model_name,
            output_dir=output_dir,
            split_mode=split_mode,
            holdout_values=holdout_values,
            pretrained=pretrained,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            svm_c=svm_c,
            random_state=random_state,
            selected_layer=selected_layer,
            train_filter=train_filter,
            test_filter=test_filter,
            save_artifacts=save_artifacts,
            run_config={
                "mode": "panel",
                "selected_layer": selected_layer,
                "split_mode": split_mode,
                "holdout_values": list(holdout_values) if holdout_values is not None else None,
                "input_size": input_size,
                "pretrained": pretrained,
                "device": device,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "svm_c": svm_c,
                "random_state": random_state,
                "include_single": include_single,
                "include_zerovar": include_zerovar,
                "train_filter": _condition_filter_to_dict(train_filter),
                "test_filter": _condition_filter_to_dict(test_filter),
            },
        )
        metrics_outputs[model_name] = metrics_df
    return metrics_outputs


def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _condition_filter_from_args(args: argparse.Namespace, prefix: str) -> Optional[ConditionFilter]:
    data = ConditionFilter(
        means=_parse_int_list(getattr(args, f"{prefix}_means", None)),
        sds=_parse_int_list(getattr(args, f"{prefix}_sds", None)),
        sss=_parse_int_list(getattr(args, f"{prefix}_sss", None)),
        instances=_parse_int_list(getattr(args, f"{prefix}_instances", None)),
        exclude_means=_parse_int_list(getattr(args, f"exclude_{prefix}_means", None)),
        exclude_sds=_parse_int_list(getattr(args, f"exclude_{prefix}_sds", None)),
        exclude_sss=_parse_int_list(getattr(args, f"exclude_{prefix}_sss", None)),
        exclude_instances=_parse_int_list(getattr(args, f"exclude_{prefix}_instances", None)),
    )
    if all(value is None for value in asdict(data).values()):
        return None
    return data


def _print_layer_listing(model_name: str) -> None:
    layer_df = list_available_layers(model_name)
    print(layer_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EPGabor CNN layerwise decoding v1.1.")
    parser.add_argument("--img-dir", type=str, default="images")
    parser.add_argument("--output-dir", type=str, default="results_v1")
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--layer-name", type=str, default="all")
    parser.add_argument("--list-layers", action="store_true", default=False)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="leave_one_sd_out",
        choices=["leave_one_sd_out", "leave_one_ss_out", "explicit"],
    )
    parser.add_argument("--holdout-values", type=str, default=None)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained weights (default: True). Use --no-pretrained to disable.",
    )
    parser.add_argument("--include-single", action="store_true", default=False)
    parser.add_argument("--include-zerovar", action="store_true", default=False)
    parser.add_argument(
        "--save-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save decoder bundle, split manifest, and run config (default: True).",
    )

    parser.add_argument("--dataset-means", type=str, default=None)
    parser.add_argument("--dataset-sds", type=str, default=None)
    parser.add_argument("--dataset-sss", type=str, default=None)
    parser.add_argument("--dataset-instances", type=str, default=None)

    parser.add_argument("--train-means", type=str, default=None)
    parser.add_argument("--train-sds", type=str, default=None)
    parser.add_argument("--train-sss", type=str, default=None)
    parser.add_argument("--train-instances", type=str, default=None)
    parser.add_argument("--exclude-train-means", type=str, default=None)
    parser.add_argument("--exclude-train-sds", type=str, default=None)
    parser.add_argument("--exclude-train-sss", type=str, default=None)
    parser.add_argument("--exclude-train-instances", type=str, default=None)

    parser.add_argument("--test-means", type=str, default=None)
    parser.add_argument("--test-sds", type=str, default=None)
    parser.add_argument("--test-sss", type=str, default=None)
    parser.add_argument("--test-instances", type=str, default=None)
    parser.add_argument("--exclude-test-means", type=str, default=None)
    parser.add_argument("--exclude-test-sds", type=str, default=None)
    parser.add_argument("--exclude-test-sss", type=str, default=None)
    parser.add_argument("--exclude-test-instances", type=str, default=None)

    args = parser.parse_args()

    if args.list_layers:
        _print_layer_listing(args.model_name)
        return

    dataset = EPGabors(
        img_dir=args.img_dir,
        include_vertical=True,
        include_single=args.include_single,
        include_zerovar=args.include_zerovar,
        filter_mean=_parse_int_list(args.dataset_means),
        filter_sd=_parse_int_list(args.dataset_sds),
        filter_ss=_parse_int_list(args.dataset_sss),
        filter_instance=_parse_int_list(args.dataset_instances),
        transform=make_deterministic_transform(input_size=args.input_size),
        return_filename=True,
        return_condition_id=True,
    )

    holdout_values = _parse_int_list(args.holdout_values)
    train_filter = _condition_filter_from_args(args, "train")
    test_filter = _condition_filter_from_args(args, "test")

    run_config = {
        "model_name": args.model_name,
        "layer_name": args.layer_name,
        "split_mode": args.split_mode,
        "holdout_values": holdout_values,
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "svm_c": args.svm_c,
        "random_state": args.random_state,
        "pretrained": args.pretrained,
        "include_single": args.include_single,
        "include_zerovar": args.include_zerovar,
        "save_artifacts": args.save_artifacts,
        "dataset_filters": {
            "means": _parse_int_list(args.dataset_means),
            "sds": _parse_int_list(args.dataset_sds),
            "sss": _parse_int_list(args.dataset_sss),
            "instances": _parse_int_list(args.dataset_instances),
        },
        "train_filter": _condition_filter_to_dict(train_filter),
        "test_filter": _condition_filter_to_dict(test_filter),
    }

    run_layerwise_binary_decoding(
        dataset=dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        split_mode=args.split_mode,
        holdout_values=holdout_values,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        svm_c=args.svm_c,
        random_state=args.random_state,
        selected_layer=args.layer_name,
        train_filter=train_filter,
        test_filter=test_filter,
        save_artifacts=args.save_artifacts,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
