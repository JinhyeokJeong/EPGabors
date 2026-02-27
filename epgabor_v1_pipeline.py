"""v1 pipeline for layerwise mean-orientation decoding in pretrained CNNs."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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
    held_out_value: int


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


def _pool_activation(activation: torch.Tensor, pooling: str = "gap") -> torch.Tensor:
    if pooling != "gap":
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    if activation.ndim == 2:
        return activation
    if activation.ndim == 3:
        # e.g., [B, tokens, C]
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


def build_cross_condition_splits(
    metadata: pd.DataFrame,
    split_mode: str = "leave_one_sd_out",
    holdout_values: Optional[Iterable[int]] = None,
) -> List[SplitSpec]:
    """Build non-overlapping train/test masks with m=0 held out from training."""
    if split_mode not in {"leave_one_sd_out", "leave_one_ss_out"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    axis = "sd" if split_mode == "leave_one_sd_out" else "ss"
    if holdout_values is None:
        holdout_values = sorted(int(v) for v in metadata[axis].unique())
    else:
        holdout_values = [int(v) for v in holdout_values]

    mean_values = metadata["mean"].to_numpy()
    axis_values = metadata[axis].to_numpy()
    splits: List[SplitSpec] = []
    for value in holdout_values:
        in_holdout = axis_values == value
        train_mask = (~in_holdout) & (mean_values != 0)
        test_nonzero_mask = in_holdout & (mean_values != 0)
        boundary_mask = in_holdout & (mean_values == 0)
        split_id = f"{axis}{value}"
        splits.append(
            SplitSpec(
                split_id=split_id,
                train_mask=train_mask,
                test_nonzero_mask=test_nonzero_mask,
                boundary_mask=boundary_mask,
                held_out_axis=axis,
                held_out_value=value,
            )
        )
    return splits


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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full v1 decoding for one model and save outputs."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model, layer_map = get_model_and_layer_map(
        model_name=model_name,
        pretrained=pretrained,
        source="timm",
        device=device,
    )
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
    splits = build_cross_condition_splits(
        metadata=metadata,
        split_mode=split_mode,
        holdout_values=holdout_values,
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

    trial_df = pd.concat(all_trials, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    condition_df = _aggregate_condition_summary(trial_df)

    prefix = f"{model_name}_{split_mode}"
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
) -> Dict[str, pd.DataFrame]:
    """Run v1 defaults over the selected model panel."""
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
        )
        metrics_outputs[model_name] = metrics_df
    return metrics_outputs


def _parse_holdout_values(raw: str) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EPGabor CNN layerwise decoding v1.")
    parser.add_argument("--img-dir", type=str, default="images")
    parser.add_argument("--output-dir", type=str, default="results_v1")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODEL_NAMES))
    parser.add_argument(
        "--split-mode",
        type=str,
        default="leave_one_sd_out",
        choices=["leave_one_sd_out", "leave_one_ss_out"],
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
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    holdout_values = _parse_holdout_values(args.holdout_values)

    run_default_v1_panel(
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        models=model_names,
        input_size=args.input_size,
        split_mode=args.split_mode,
        holdout_values=holdout_values,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        svm_c=args.svm_c,
        random_state=args.random_state,
        include_single=args.include_single,
        include_zerovar=args.include_zerovar,
    )


if __name__ == "__main__":
    main()
