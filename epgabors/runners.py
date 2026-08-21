"""Shared ResNet50 analysis runners and command-line entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from epgabors.data import EPGabors, summarize_conditions, validate_condition_grid
from epgabors.features import extract_layer_features, make_deterministic_transform
from epgabors.models import (
    SPARSE_LAYER_PATHS,
    get_model_and_layer_map,
    get_resnet50_sparse_layer_map,
    list_available_layers,
    select_layer_map,
)
from epgabors.readouts import (
    FoldSpec,
    build_regression_kfold_splits,
    build_stratified_kfold_splits,
    evaluate_vertical_boundary,
    fit_linear_decoder,
    fit_linear_regressor,
    fit_linear_svm,
)


CLASSIFICATION_TRIAL_COLUMNS = [
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

CLASSIFICATION_FOLD_METRIC_COLUMNS = [
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

CLASSIFICATION_LAYER_SUMMARY_COLUMNS = [
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

CLASSIFICATION_COMBINED_COLUMNS = [
    "fold_id",
    "layer_name",
    "split",
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
    "choice_cw",
    "decoder_name",
]

REGRESSION_TRIAL_COLUMNS = [
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

REGRESSION_FOLD_METRIC_COLUMNS = [
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

REGRESSION_LAYER_SUMMARY_COLUMNS = [
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


def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _print_layer_listing() -> None:
    layer_df = list_available_layers("resnet50")
    print(layer_df.to_string(index=False))


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


def _warn_if_large_dense_matrix(array: np.ndarray, estimator_name: str, dense_estimator_name: str) -> None:
    if estimator_name != dense_estimator_name:
        return
    dense_bytes = _estimate_dense_bytes(array)
    if dense_bytes >= 1_000_000_000:
        size_gb = dense_bytes / 1_000_000_000
        warnings.warn(
            (
                f"Feature matrix is approximately {size_gb:.2f} GB before scaling. "
                "Use an SGD readout if memory becomes a bottleneck."
            ),
            RuntimeWarning,
        )


def _save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)


def _save_json(data: Dict[str, object], output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _dataset_run_info(dataset: EPGabors) -> Dict[str, object]:
    validation = validate_condition_grid(dataset.df)
    return {
        "img_dir": dataset.img_dir,
        "include_vertical": bool(dataset.include_vertical),
        "include_single": bool(dataset.include_single),
        "include_zerovar": bool(dataset.include_zerovar),
        **validation,
    }


def _save_dataset_summary(dataset: EPGabors, output_dir: str, prefix: str) -> None:
    _save_dataframe(summarize_conditions(dataset.df), os.path.join(output_dir, f"{prefix}_dataset_summary.csv"))


def _aggregate_classification_layer_summary(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics_df.empty:
        return pd.DataFrame(columns=CLASSIFICATION_LAYER_SUMMARY_COLUMNS)

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
    return pd.DataFrame(rows, columns=CLASSIFICATION_LAYER_SUMMARY_COLUMNS)


def _aggregate_regression_layer_summary(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics_df.empty:
        return pd.DataFrame(columns=REGRESSION_LAYER_SUMMARY_COLUMNS)

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
    return pd.DataFrame(rows, columns=REGRESSION_LAYER_SUMMARY_COLUMNS)


def _build_combined_classification_predictions(
    trial_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
) -> pd.DataFrame:
    combined_parts = []
    if not trial_df.empty:
        nonzero = trial_df.copy()
        nonzero["split"] = "nonzero_cv"
        nonzero["choice_cw"] = nonzero["pred_class"].astype(int)
        combined_parts.append(nonzero)

    if not boundary_df.empty:
        boundary = boundary_df.copy()
        boundary["split"] = "vertical_boundary"
        boundary["target"] = np.nan
        boundary["correct"] = np.nan
        combined_parts.append(boundary)

    if not combined_parts:
        return pd.DataFrame(columns=CLASSIFICATION_COMBINED_COLUMNS)
    return pd.concat(combined_parts, ignore_index=True)[CLASSIFICATION_COMBINED_COLUMNS]


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
    """Run sparse-endpoint ResNet50 binary sign decoding with K-fold CV."""
    total_start = time.perf_counter() if measure_timing else None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model, full_layer_map = get_resnet50_sparse_layer_map(pretrained=pretrained, device=device)
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
        _warn_if_large_dense_matrix(X_nonzero, estimator_name=decoder_name, dense_estimator_name="linear_svc")

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
            trial_rows.append(test_trials[CLASSIFICATION_TRIAL_COLUMNS])

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
        else pd.DataFrame(columns=CLASSIFICATION_TRIAL_COLUMNS)
    )
    fold_metrics_df = pd.DataFrame(fold_metric_rows, columns=CLASSIFICATION_FOLD_METRIC_COLUMNS)
    layer_summary_df = _aggregate_classification_layer_summary(fold_metrics_df)
    boundary_vertical_df = (
        pd.concat(boundary_rows, ignore_index=True)
        if boundary_rows
        else pd.DataFrame(columns=BOUNDARY_TRIAL_COLUMNS)
    )
    boundary_summary_df = pd.DataFrame(boundary_summary_rows, columns=BOUNDARY_SUMMARY_COLUMNS)
    combined_df = _build_combined_classification_predictions(trial_df, boundary_vertical_df)

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
    _save_dataframe(combined_df, os.path.join(output_dir, f"{prefix}_combined_predictions.csv"))
    if evaluate_vertical and not boundary_vertical_df.empty:
        _save_dataframe(boundary_vertical_df, os.path.join(output_dir, f"{prefix}_vertical_boundary_outputs.csv"))
        _save_dataframe(boundary_summary_df, os.path.join(output_dir, f"{prefix}_vertical_boundary_summary.csv"))
    if measure_timing:
        _save_dataframe(timing_df, os.path.join(output_dir, f"{prefix}_timing.csv"))
    _save_dataset_summary(dataset, output_dir, prefix)
    _save_json(
        {
            "analysis": "classification",
            "model_name": "resnet50",
            "selected_layer": selected_layer,
            "decoder_name": decoder_name,
            "n_splits": int(n_splits),
            "pretrained": bool(pretrained),
            "device": device,
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "random_state": int(random_state),
            "svm_c": float(svm_c),
            "sgd_alpha": float(sgd_alpha),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "evaluate_vertical": bool(evaluate_vertical),
            "measure_timing": bool(measure_timing),
            "dataset": _dataset_run_info(dataset),
        },
        os.path.join(output_dir, f"{prefix}_run_config.json"),
    )

    return (
        trial_df,
        fold_metrics_df,
        layer_summary_df,
        boundary_vertical_df,
        boundary_summary_df,
        timing_df,
    )


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
    """Run sparse-endpoint ResNet50 mean-orientation regression with K-fold CV."""
    total_start = time.perf_counter() if measure_timing else None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model, full_layer_map = get_resnet50_sparse_layer_map(pretrained=pretrained, device=device)
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
        _warn_if_large_dense_matrix(X, estimator_name=regressor_name, dense_estimator_name="ridge")

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
            trial_rows.append(test_trials[REGRESSION_TRIAL_COLUMNS])

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

    trial_df = (
        pd.concat(trial_rows, ignore_index=True)
        if trial_rows
        else pd.DataFrame(columns=REGRESSION_TRIAL_COLUMNS)
    )
    fold_metrics_df = pd.DataFrame(fold_metric_rows, columns=REGRESSION_FOLD_METRIC_COLUMNS)
    layer_summary_df = _aggregate_regression_layer_summary(fold_metrics_df)

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
    _save_dataset_summary(dataset, output_dir, prefix)
    _save_json(
        {
            "analysis": "regression",
            "model_name": "resnet50",
            "selected_layer": selected_layer,
            "regressor_name": regressor_name,
            "n_splits": int(n_splits),
            "pretrained": bool(pretrained),
            "device": device,
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "random_state": int(random_state),
            "ridge_alpha": float(ridge_alpha),
            "sgd_alpha": float(sgd_alpha),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "target_scaling": target_scaling,
            "measure_timing": bool(measure_timing),
            "dataset": _dataset_run_info(dataset),
        },
        os.path.join(output_dir, f"{prefix}_run_config.json"),
    )

    return trial_df, fold_metrics_df, layer_summary_df, timing_df


def classification_main() -> None:
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


def regression_main() -> None:
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
    parser.add_argument("--target-scaling", type=str, default="none", choices=["none", "standardize"])
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
