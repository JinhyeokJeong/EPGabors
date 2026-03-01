"""Research-first ResNet50 regression pipeline for EP Gabor images."""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
    "target_mean",
    "pred_mean",
    "residual",
    "abs_error",
    "squared_error",
    "regressor_name",
]

FOLD_METRIC_COLUMNS = [
    "layer_name",
    "fold_id",
    "train_n",
    "test_n",
    "r2",
    "rmse",
    "mae",
    "pearson_r",
    "regressor_name",
    "target_scaling",
]

LAYER_SUMMARY_COLUMNS = [
    "layer_name",
    "n_folds",
    "total_test_n",
    "r2_mean",
    "r2_std",
    "rmse_mean",
    "rmse_std",
    "mae_mean",
    "mae_std",
    "pearson_r_mean",
    "pearson_r_std",
    "regressor_name",
    "target_scaling",
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


def build_regression_kfold_splits(
    strata: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
) -> List[FoldSpec]:
    """Build stratified K-fold splits using discrete mean levels as strata."""
    strata = np.asarray(strata)
    if strata.ndim != 1:
        raise ValueError("strata must be a 1D array")
    if len(strata) == 0:
        raise ValueError("strata must contain at least one sample")

    unique_strata, counts = np.unique(strata, return_counts=True)
    if len(unique_strata) < 2:
        raise ValueError("At least two distinct mean levels are required for stratified K-fold.")
    if counts.min() < n_splits:
        detail = ", ".join(f"{int(level)}:{int(count)}" for level, count in zip(unique_strata, counts))
        raise ValueError(
            "Each retained mean level must contain at least "
            f"{n_splits} samples for stratified K-fold. Current counts: {detail}. "
            "Reduce n_splits or widen the dataset subset."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    dummy = np.zeros(len(strata))
    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(dummy, strata), start=1):
        splits.append(
            FoldSpec(
                fold_id=f"fold{fold_idx}",
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )
    return splits


def _compute_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def fit_linear_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    regressor_name: str = "ridge",
    ridge_alpha: float = 1.0,
    sgd_alpha: float = 0.0001,
    max_iter: int = 1000,
    tol: float = 1e-3,
    random_state: int = 0,
    target_scaling: str = "none",
) -> Dict[str, object]:
    """Fit a linear regressor and evaluate it on held-out data."""
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_train_raw = np.asarray(y_train, dtype=float).reshape(-1)
    y_test_raw = np.asarray(y_test, dtype=float).reshape(-1)

    if target_scaling == "none":
        y_scaler = None
        y_train_fit = y_train_raw
    elif target_scaling == "standardize":
        y_scaler = StandardScaler()
        y_train_fit = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    else:
        raise ValueError(f"Unsupported target_scaling: {target_scaling}")

    if regressor_name == "ridge":
        estimator = Ridge(alpha=ridge_alpha)
    elif regressor_name == "sgd_squared_error":
        estimator = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=sgd_alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported regressor_name: {regressor_name}")

    estimator.fit(X_train_scaled, y_train_fit)
    pred_fit_units = np.asarray(estimator.predict(X_test_scaled), dtype=float).reshape(-1)

    if y_scaler is not None:
        pred_mean = y_scaler.inverse_transform(pred_fit_units.reshape(-1, 1)).reshape(-1)
    else:
        pred_mean = pred_fit_units

    residual = pred_mean - y_test_raw
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    pearson_r = _compute_pearson_r(y_test_raw, pred_mean)
    r2 = float(r2_score(y_test_raw, pred_mean)) if len(y_test_raw) >= 2 else np.nan

    return {
        "estimator": estimator,
        "feature_scaler": X_scaler,
        "target_scaler": y_scaler,
        "pred_mean": pred_mean,
        "metrics": {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "pearson_r": pearson_r,
        },
    }


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


def _warn_if_large_dense_matrix(array: np.ndarray, regressor_name: str) -> None:
    if regressor_name != "ridge":
        return
    dense_bytes = _estimate_dense_bytes(array)
    if dense_bytes >= 1_000_000_000:
        size_gb = dense_bytes / 1_000_000_000
        warnings.warn(
            (
                f"Feature matrix is approximately {size_gb:.2f} GB before scaling. "
                "Consider using regressor_name='sgd_squared_error' if memory becomes a bottleneck."
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
                "r2_mean": float(group["r2"].mean()),
                "r2_std": float(group["r2"].std(ddof=1)) if len(group) > 1 else 0.0,
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_std": float(group["rmse"].std(ddof=1)) if len(group) > 1 else 0.0,
                "mae_mean": float(group["mae"].mean()),
                "mae_std": float(group["mae"].std(ddof=1)) if len(group) > 1 else 0.0,
                "pearson_r_mean": float(group["pearson_r"].mean()),
                "pearson_r_std": float(group["pearson_r"].std(ddof=1)) if len(group) > 1 else 0.0,
                "regressor_name": group["regressor_name"].iloc[0],
                "target_scaling": group["target_scaling"].iloc[0],
            }
        )
    return pd.DataFrame(rows, columns=LAYER_SUMMARY_COLUMNS)


def _save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_resnet50_kfold_regression(
    dataset: EPGabors,
    output_dir: str,
    selected_layer: str = "all",
    regressor_name: str = "ridge",
    n_splits: int = 5,
    pretrained: bool = True,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
    random_state: int = 0,
    ridge_alpha: float = 1.0,
    sgd_alpha: float = 0.0001,
    max_iter: int = 1000,
    tol: float = 1e-3,
    target_scaling: str = "none",
    measure_timing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run sparse-endpoint ResNet50 regression with stratified K-fold CV."""
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
    feature_time_seconds = 0.0

    metadata: Optional[pd.DataFrame] = None
    target_mean: Optional[np.ndarray] = None
    strata: Optional[np.ndarray] = None
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
            target_mean = metadata["mean"].astype(float).to_numpy()
            strata = metadata["mean"].to_numpy()
            folds = build_regression_kfold_splits(
                strata=strata,
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
        _warn_if_large_dense_matrix(X, regressor_name=regressor_name)

        for fold in folds:
            fold_start = time.perf_counter() if measure_timing else None

            fit_result = fit_linear_regressor(
                X_train=X[fold.train_idx],
                y_train=target_mean[fold.train_idx],
                X_test=X[fold.test_idx],
                y_test=target_mean[fold.test_idx],
                regressor_name=regressor_name,
                ridge_alpha=ridge_alpha,
                sgd_alpha=sgd_alpha,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state,
                target_scaling=target_scaling,
            )

            test_trials = metadata.iloc[fold.test_idx].copy()
            pred_mean = fit_result["pred_mean"]
            target_test = target_mean[fold.test_idx]
            residual = pred_mean - target_test
            test_trials["fold_id"] = fold.fold_id
            test_trials["layer_name"] = layer_name
            test_trials["target_mean"] = target_test.astype(float)
            test_trials["pred_mean"] = pred_mean.astype(float)
            test_trials["residual"] = residual.astype(float)
            test_trials["abs_error"] = np.abs(residual).astype(float)
            test_trials["squared_error"] = np.square(residual).astype(float)
            test_trials["regressor_name"] = regressor_name
            trial_rows.append(test_trials[TRIAL_COLUMNS])

            fold_metric_rows.append(
                {
                    "layer_name": layer_name,
                    "fold_id": fold.fold_id,
                    "train_n": int(len(fold.train_idx)),
                    "test_n": int(len(fold.test_idx)),
                    "r2": fit_result["metrics"]["r2"],
                    "rmse": fit_result["metrics"]["rmse"],
                    "mae": fit_result["metrics"]["mae"],
                    "pearson_r": fit_result["metrics"]["pearson_r"],
                    "regressor_name": regressor_name,
                    "target_scaling": target_scaling,
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

    trial_df = pd.concat(trial_rows, ignore_index=True) if trial_rows else pd.DataFrame(columns=TRIAL_COLUMNS)
    fold_metrics_df = pd.DataFrame(fold_metric_rows, columns=FOLD_METRIC_COLUMNS)
    layer_summary_df = _aggregate_layer_summary(fold_metrics_df)

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
    prefix = f"resnet50_{layer_label}_kfold_regression"
    _save_dataframe(trial_df, os.path.join(output_dir, f"{prefix}_trial_outputs.csv"))
    _save_dataframe(fold_metrics_df, os.path.join(output_dir, f"{prefix}_fold_metrics.csv"))
    _save_dataframe(layer_summary_df, os.path.join(output_dir, f"{prefix}_layer_summary.csv"))
    if measure_timing:
        _save_dataframe(timing_df, os.path.join(output_dir, f"{prefix}_timing.csv"))

    return trial_df, fold_metrics_df, layer_summary_df, timing_df


def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _print_layer_listing() -> None:
    layer_df = list_available_layers("resnet50")
    print(layer_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sparse-endpoint ResNet50 K-fold regression on EP Gabor images."
    )
    parser.add_argument("--img-dir", type=str, default="images")
    parser.add_argument("--output-dir", type=str, default="results_kfold_regression")
    parser.add_argument("--layer-name", type=str, default="all")
    parser.add_argument("--list-layers", action="store_true", default=False)
    parser.add_argument(
        "--regressor-name",
        type=str,
        default="ridge",
        choices=["ridge", "sgd_squared_error"],
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--sgd-alpha", type=float, default=0.0001)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument(
        "--target-scaling",
        type=str,
        default="none",
        choices=["none", "standardize"],
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained ResNet50 weights (default: True).",
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
        default=True,
        help="Include mean=0 images in the regression dataset (default: True).",
    )

    parser.add_argument("--dataset-means", type=str, default=None)
    parser.add_argument("--dataset-sds", type=str, default=None)
    parser.add_argument("--dataset-sss", type=str, default=None)
    parser.add_argument("--dataset-instances", type=str, default=None)

    args = parser.parse_args()

    if args.list_layers:
        _print_layer_listing()
        return

    dataset = EPGabors(
        img_dir=args.img_dir,
        include_vertical=args.include_vertical,
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

    run_resnet50_kfold_regression(
        dataset=dataset,
        output_dir=args.output_dir,
        selected_layer=args.layer_name,
        regressor_name=args.regressor_name,
        n_splits=args.n_splits,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.random_state,
        ridge_alpha=args.ridge_alpha,
        sgd_alpha=args.sgd_alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        target_scaling=args.target_scaling,
        measure_timing=args.measure_timing,
    )


if __name__ == "__main__":
    main()
