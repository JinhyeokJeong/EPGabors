"""Research-first ResNet50 decoding pipeline for EP Gabor images."""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torchvision import transforms

from EPOriGabors import EPGabors


SPARSE_LAYER_PATHS = {
    "stem": "maxpool",
    "layer1_last": "layer1.2",
    "layer2_last": "layer2.3",
    "layer3_last": "layer3.5",
    "layer4_last": "layer4.2",
}

TRIAL_COLUMNS = [
    "fold_id",
    "layer_name",
    "file_name",
    "condition_id",
    "mean",
    "sd",
    "ss",
    "instance",
    "target",
    "pred_class",
    "decision_value",
    "correct",
    "decoder_name",
]

FOLD_METRIC_COLUMNS = [
    "layer_name",
    "fold_id",
    "train_n",
    "test_n",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auroc",
    "decoder_name",
]

LAYER_SUMMARY_COLUMNS = [
    "layer_name",
    "n_folds",
    "total_test_n",
    "accuracy_mean",
    "accuracy_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "f1_mean",
    "f1_std",
    "auroc_mean",
    "auroc_std",
    "decoder_name",
]

BOUNDARY_TRIAL_COLUMNS = [
    "fold_id",
    "layer_name",
    "file_name",
    "condition_id",
    "mean",
    "sd",
    "ss",
    "instance",
    "pred_class",
    "decision_value",
    "choice_cw",
    "decoder_name",
]

BOUNDARY_SUMMARY_COLUMNS = [
    "layer_name",
    "fold_id",
    "boundary_n",
    "cw_rate",
    "decision_value_mean",
    "decision_value_std",
    "decoder_name",
]

TIMING_COLUMNS = ["stage", "layer_name", "fold_id", "seconds"]


@dataclass
class FoldSpec:
    fold_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_deterministic_transform(input_size: int = 224) -> transforms.Compose:
    """Deterministic preprocessing without augmentation."""
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
) -> Tuple[torch.nn.Module, Dict[str, torch.nn.Module]]:
    """Return a ResNet50 model and the sparse endpoint module map."""
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
) -> Tuple[torch.nn.Module, Dict[str, torch.nn.Module]]:
    """Backward-compatible wrapper for the prior API."""
    if source != "timm":
        raise ValueError("Only 'timm' is supported.")
    if model_name != "resnet50":
        raise ValueError("Only 'resnet50' is supported in this pipeline.")
    return get_resnet50_sparse_layer_map(pretrained=pretrained, device=device)


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
    """Extract flattened endpoint activations for all samples in a dataloader."""
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


def build_stratified_kfold_splits(
    labels: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
) -> List[FoldSpec]:
    """Build stratified K-fold splits from a label vector."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if len(labels) == 0:
        raise ValueError("labels must contain at least one sample")

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError("At least two classes are required for stratified K-fold.")
    if counts.min() < n_splits:
        raise ValueError(
            f"Each class must contain at least {n_splits} samples, got counts {counts.tolist()}."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    splits = []
    dummy = np.zeros(len(labels))
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(dummy, labels), start=1):
        splits.append(
            FoldSpec(
                fold_id=f"fold{fold_idx}",
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )
    return splits


def fit_linear_decoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    decoder_name: str = "linear_svc",
    svm_c: float = 1.0,
    sgd_alpha: float = 0.0001,
    max_iter: int = 1000,
    tol: float = 1e-3,
    random_state: int = 0,
) -> Dict[str, object]:
    """Fit a linear classifier and evaluate it on held-out data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if decoder_name == "linear_svc":
        estimator = LinearSVC(
            C=svm_c,
            class_weight="balanced",
            random_state=random_state,
            max_iter=max_iter,
            dual="auto",
        )
    elif decoder_name == "sgd_hinge":
        estimator = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=sgd_alpha,
            max_iter=max_iter,
            tol=tol,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported decoder_name: {decoder_name}")

    estimator.fit(X_train_scaled, y_train)
    y_pred = estimator.predict(X_test_scaled)
    decision_value = np.asarray(estimator.decision_function(X_test_scaled)).reshape(-1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }
    if len(np.unique(y_test)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_test, decision_value))
    else:
        metrics["auroc"] = np.nan

    return {
        "estimator": estimator,
        "scaler": scaler,
        "y_pred": y_pred,
        "decision_value": decision_value,
        "metrics": metrics,
    }


def fit_linear_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
    random_state: int = 0,
    max_iter: int = 1000,
) -> Dict[str, object]:
    """Backward-compatible SVM wrapper around the unified decoder API."""
    result = fit_linear_decoder(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        decoder_name="linear_svc",
        svm_c=C,
        max_iter=max_iter,
        random_state=random_state,
    )
    return {
        "estimator": result["estimator"],
        "scaler": result["scaler"],
        "y_pred": result["y_pred"],
        "margins": result["decision_value"],
        "metrics": {
            "balanced_accuracy": result["metrics"]["balanced_accuracy"],
            "f1": result["metrics"]["f1"],
            "auroc": result["metrics"]["auroc"],
        },
    }


def evaluate_vertical_boundary(
    estimator,
    scaler: StandardScaler,
    X_vertical: np.ndarray,
    metadata_vertical: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Apply a trained fold model to vertical (mean=0) images."""
    if len(X_vertical) == 0:
        return pd.DataFrame(columns=BOUNDARY_TRIAL_COLUMNS), {
            "boundary_n": 0.0,
            "cw_rate": np.nan,
            "decision_value_mean": np.nan,
            "decision_value_std": np.nan,
        }

    X_vertical_scaled = scaler.transform(X_vertical)
    pred = estimator.predict(X_vertical_scaled)
    decision_value = np.asarray(estimator.decision_function(X_vertical_scaled)).reshape(-1)

    boundary_trials = metadata_vertical.copy()
    boundary_trials["pred_class"] = pred.astype(int)
    boundary_trials["decision_value"] = decision_value.astype(float)
    boundary_trials["choice_cw"] = (boundary_trials["pred_class"] == 1).astype(int)

    summary = {
        "boundary_n": float(len(boundary_trials)),
        "cw_rate": float(boundary_trials["choice_cw"].mean()),
        "decision_value_mean": float(boundary_trials["decision_value"].mean()),
        "decision_value_std": float(boundary_trials["decision_value"].std(ddof=1))
        if len(boundary_trials) > 1
        else 0.0,
    }
    return boundary_trials, summary


def _ensure_metadata_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"mean", "sd", "ss", "instance"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    metadata = metadata.copy()
    if "file_name" not in metadata.columns:
        metadata["file_name"] = np.arange(len(metadata)).astype(str)
    if "condition_id" not in metadata.columns:
        metadata["condition_id"] = metadata.apply(
            lambda row: f"m{int(row['mean'])}_sd{int(row['sd'])}_ss{int(row['ss'])}",
            axis=1,
        )
    return metadata


def _estimate_dense_bytes(array: np.ndarray) -> int:
    return int(np.prod(array.shape) * array.dtype.itemsize)


def _warn_if_large_dense_matrix(array: np.ndarray, decoder_name: str) -> None:
    if decoder_name != "linear_svc":
        return
    dense_bytes = _estimate_dense_bytes(array)
    if dense_bytes >= 1_000_000_000:
        size_gb = dense_bytes / 1_000_000_000
        warnings.warn(
            (
                f"Feature matrix is approximately {size_gb:.2f} GB before scaling. "
                "Consider using decoder_name='sgd_hinge' if memory becomes a bottleneck."
            ),
            RuntimeWarning,
        )


def _aggregate_layer_summary(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics_df.empty:
        return pd.DataFrame(columns=LAYER_SUMMARY_COLUMNS)

    rows = []
    for layer_name, group in fold_metrics_df.groupby("layer_name", sort=False):
        rows.append(
            {
                "layer_name": layer_name,
                "n_folds": int(len(group)),
                "total_test_n": int(group["test_n"].sum()),
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_std": float(group["accuracy"].std(ddof=1)) if len(group) > 1 else 0.0,
                "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
                "balanced_accuracy_std": float(group["balanced_accuracy"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "f1_mean": float(group["f1"].mean()),
                "f1_std": float(group["f1"].std(ddof=1)) if len(group) > 1 else 0.0,
                "auroc_mean": float(group["auroc"].mean()),
                "auroc_std": float(group["auroc"].std(ddof=1)) if len(group) > 1 else 0.0,
                "decoder_name": group["decoder_name"].iloc[0],
            }
        )
    return pd.DataFrame(rows, columns=LAYER_SUMMARY_COLUMNS)


def _save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_resnet50_kfold_decoding(
    dataset: EPGabors,
    output_dir: str,
    selected_layer: str = "all",
    decoder_name: str = "linear_svc",
    n_splits: int = 5,
    pretrained: bool = True,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
    random_state: int = 0,
    svm_c: float = 1.0,
    sgd_alpha: float = 0.0001,
    max_iter: int = 1000,
    tol: float = 1e-3,
    evaluate_vertical: bool = True,
    measure_timing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run sparse-endpoint ResNet50 decoding with K-fold CV."""
    total_start = time.perf_counter() if measure_timing else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model, full_layer_map = get_resnet50_sparse_layer_map(
        pretrained=pretrained,
        device=device,
    )
    selected_layer_map = select_layer_map(full_layer_map, selected_layer=selected_layer)

    timing_rows: List[Dict[str, object]] = []
    trial_rows: List[pd.DataFrame] = []
    fold_metric_rows: List[Dict[str, object]] = []
    boundary_rows: List[pd.DataFrame] = []
    boundary_summary_rows: List[Dict[str, object]] = []
    feature_time_seconds = 0.0

    metadata: Optional[pd.DataFrame] = None
    metadata_nonzero: Optional[pd.DataFrame] = None
    metadata_vertical: Optional[pd.DataFrame] = None
    nonzero_mask: Optional[np.ndarray] = None
    vertical_mask: Optional[np.ndarray] = None
    y_nonzero: Optional[np.ndarray] = None
    folds: Optional[List[FoldSpec]] = None

    for layer_name, module in selected_layer_map.items():
        layer_start = time.perf_counter() if measure_timing else None

        feature_start = time.perf_counter() if measure_timing else None
        layer_features, layer_metadata = extract_layer_features(
            model=model,
            layer_map={layer_name: module},
            dataloader=dataloader,
            device=device,
            feature_mode="flatten",
        )
        if measure_timing and feature_start is not None:
            feature_time_seconds += float(time.perf_counter() - feature_start)

        if metadata is None:
            metadata = _ensure_metadata_columns(layer_metadata)
            metadata["target"] = (metadata["mean"] > 0).astype(int)
            nonzero_mask = metadata["mean"].to_numpy() != 0
            vertical_mask = (
                metadata["mean"].to_numpy() == 0
                if evaluate_vertical
                else np.zeros(len(metadata), dtype=bool)
            )
            metadata_nonzero = metadata.loc[nonzero_mask].reset_index(drop=True)
            metadata_vertical = metadata.loc[vertical_mask].reset_index(drop=True)
            if metadata_nonzero.empty:
                raise ValueError("No nonzero-mean samples are available after filtering.")

            y_nonzero = metadata_nonzero["target"].to_numpy()
            folds = build_stratified_kfold_splits(
                labels=y_nonzero,
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
        else:
            layer_metadata = _ensure_metadata_columns(layer_metadata)
            if not layer_metadata["file_name"].reset_index(drop=True).equals(
                metadata["file_name"].reset_index(drop=True)
            ):
                raise ValueError("Feature extraction order changed across layers.")

        X = layer_features[layer_name]
        X_nonzero = X[nonzero_mask]
        X_vertical = X[vertical_mask]
        _warn_if_large_dense_matrix(X_nonzero, decoder_name=decoder_name)

        for fold in folds:
            fold_start = time.perf_counter() if measure_timing else None

            fold_result = fit_linear_decoder(
                X_train=X_nonzero[fold.train_idx],
                y_train=y_nonzero[fold.train_idx],
                X_test=X_nonzero[fold.test_idx],
                y_test=y_nonzero[fold.test_idx],
                decoder_name=decoder_name,
                svm_c=svm_c,
                sgd_alpha=sgd_alpha,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state,
            )

            test_trials = metadata_nonzero.iloc[fold.test_idx].copy()
            test_trials["fold_id"] = fold.fold_id
            test_trials["layer_name"] = layer_name
            test_trials["pred_class"] = fold_result["y_pred"].astype(int)
            test_trials["decision_value"] = fold_result["decision_value"].astype(float)
            test_trials["correct"] = (
                test_trials["pred_class"].to_numpy() == test_trials["target"].to_numpy()
            ).astype(int)
            test_trials["decoder_name"] = decoder_name
            trial_rows.append(test_trials[TRIAL_COLUMNS])

            fold_metric_rows.append(
                {
                    "layer_name": layer_name,
                    "fold_id": fold.fold_id,
                    "train_n": int(len(fold.train_idx)),
                    "test_n": int(len(fold.test_idx)),
                    "accuracy": fold_result["metrics"]["accuracy"],
                    "balanced_accuracy": fold_result["metrics"]["balanced_accuracy"],
                    "f1": fold_result["metrics"]["f1"],
                    "auroc": fold_result["metrics"]["auroc"],
                    "decoder_name": decoder_name,
                }
            )

            if evaluate_vertical and len(metadata_vertical) > 0:
                boundary_trials, boundary_summary = evaluate_vertical_boundary(
                    estimator=fold_result["estimator"],
                    scaler=fold_result["scaler"],
                    X_vertical=X_vertical,
                    metadata_vertical=metadata_vertical,
                )
                boundary_trials["fold_id"] = fold.fold_id
                boundary_trials["layer_name"] = layer_name
                boundary_trials["decoder_name"] = decoder_name
                boundary_rows.append(boundary_trials[BOUNDARY_TRIAL_COLUMNS])
                boundary_summary_rows.append(
                    {
                        "layer_name": layer_name,
                        "fold_id": fold.fold_id,
                        "boundary_n": boundary_summary["boundary_n"],
                        "cw_rate": boundary_summary["cw_rate"],
                        "decision_value_mean": boundary_summary["decision_value_mean"],
                        "decision_value_std": boundary_summary["decision_value_std"],
                        "decoder_name": decoder_name,
                    }
                )

            if measure_timing and fold_start is not None:
                timing_rows.append(
                    {
                        "stage": "fold_decode",
                        "layer_name": layer_name,
                        "fold_id": fold.fold_id,
                        "seconds": float(time.perf_counter() - fold_start),
                    }
                )

        if measure_timing and layer_start is not None:
            timing_rows.append(
                {
                    "stage": "layer_decode",
                    "layer_name": layer_name,
                    "fold_id": None,
                    "seconds": float(time.perf_counter() - layer_start),
                }
            )

    if measure_timing:
        timing_rows.insert(
            0,
            {
                "stage": "feature_extraction",
                "layer_name": None,
                "fold_id": None,
                "seconds": feature_time_seconds,
            },
        )

    trial_df = (
        pd.concat(trial_rows, ignore_index=True)
        if trial_rows
        else pd.DataFrame(columns=TRIAL_COLUMNS)
    )
    fold_metrics_df = pd.DataFrame(fold_metric_rows, columns=FOLD_METRIC_COLUMNS)
    layer_summary_df = _aggregate_layer_summary(fold_metrics_df)
    boundary_vertical_df = (
        pd.concat(boundary_rows, ignore_index=True)
        if boundary_rows
        else pd.DataFrame(columns=BOUNDARY_TRIAL_COLUMNS)
    )
    boundary_summary_df = pd.DataFrame(boundary_summary_rows, columns=BOUNDARY_SUMMARY_COLUMNS)

    if measure_timing and total_start is not None:
        timing_rows.append(
            {
                "stage": "total",
                "layer_name": None,
                "fold_id": None,
                "seconds": float(time.perf_counter() - total_start),
            }
        )
    timing_df = pd.DataFrame(timing_rows, columns=TIMING_COLUMNS)

    layer_label = selected_layer if selected_layer != "all" else "all_layers"
    prefix = f"resnet50_{layer_label}_kfold"
    _save_dataframe(trial_df, os.path.join(output_dir, f"{prefix}_trial_outputs.csv"))
    _save_dataframe(fold_metrics_df, os.path.join(output_dir, f"{prefix}_fold_metrics.csv"))
    _save_dataframe(layer_summary_df, os.path.join(output_dir, f"{prefix}_layer_summary.csv"))
    if evaluate_vertical and not boundary_vertical_df.empty:
        _save_dataframe(
            boundary_vertical_df,
            os.path.join(output_dir, f"{prefix}_vertical_boundary_outputs.csv"),
        )
        _save_dataframe(
            boundary_summary_df,
            os.path.join(output_dir, f"{prefix}_vertical_boundary_summary.csv"),
        )
    if measure_timing:
        _save_dataframe(timing_df, os.path.join(output_dir, f"{prefix}_timing.csv"))

    return (
        trial_df,
        fold_metrics_df,
        layer_summary_df,
        boundary_vertical_df,
        boundary_summary_df,
        timing_df,
    )


def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _print_layer_listing() -> None:
    layer_df = list_available_layers("resnet50")
    print(layer_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sparse-endpoint ResNet50 K-fold decoding on EP Gabor images."
    )
    parser.add_argument("--img-dir", type=str, default="images")
    parser.add_argument("--output-dir", type=str, default="results_kfold")
    parser.add_argument("--layer-name", type=str, default="all")
    parser.add_argument("--list-layers", action="store_true", default=False)
    parser.add_argument("--decoder-name", type=str, default="linear_svc", choices=["linear_svc", "sgd_hinge"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--sgd-alpha", type=float, default=0.0001)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained ResNet50 weights (default: True).",
    )
    parser.add_argument(
        "--evaluate-vertical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate mean=0 images after each fold (default: True).",
    )
    parser.add_argument(
        "--measure-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure feature extraction, fold, layer, and total runtime (default: True).",
    )
    parser.add_argument("--include-single", action="store_true", default=False)
    parser.add_argument("--include-zerovar", action="store_true", default=False)
    parser.add_argument(
        "--include-vertical",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether mean=0 images are included in the dataset. By default this follows --evaluate-vertical.",
    )

    parser.add_argument("--dataset-means", type=str, default=None)
    parser.add_argument("--dataset-sds", type=str, default=None)
    parser.add_argument("--dataset-sss", type=str, default=None)
    parser.add_argument("--dataset-instances", type=str, default=None)

    args = parser.parse_args()

    if args.list_layers:
        _print_layer_listing()
        return

    include_vertical = args.evaluate_vertical if args.include_vertical is None else args.include_vertical
    dataset = EPGabors(
        img_dir=args.img_dir,
        include_vertical=include_vertical,
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

    run_resnet50_kfold_decoding(
        dataset=dataset,
        output_dir=args.output_dir,
        selected_layer=args.layer_name,
        decoder_name=args.decoder_name,
        n_splits=args.n_splits,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.random_state,
        svm_c=args.svm_c,
        sgd_alpha=args.sgd_alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        evaluate_vertical=args.evaluate_vertical,
        measure_timing=args.measure_timing,
    )


if __name__ == "__main__":
    main()
