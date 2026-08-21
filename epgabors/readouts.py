"""Linear readout fitting and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass
class FoldSpec:
    fold_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def build_stratified_kfold_splits(
    labels: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
) -> List[FoldSpec]:
    """Build stratified K-fold splits from a class-label vector."""
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

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    dummy = np.zeros(len(labels))
    return [
        FoldSpec(fold_id=f"fold{fold_idx}", train_idx=train_idx, test_idx=test_idx)
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(dummy, labels), start=1)
    ]


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

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    dummy = np.zeros(len(strata))
    return [
        FoldSpec(fold_id=f"fold{fold_idx}", train_idx=train_idx, test_idx=test_idx)
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(dummy, strata), start=1)
    ]


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
    decision_value = np.asarray(estimator.decision_function(X_test_scaled))

    unique_test = np.unique(y_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    if len(unique_test) == 2:
        decision_value_binary = decision_value.reshape(-1)
        metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
        metrics["auroc"] = float(roc_auc_score(y_test, decision_value_binary))
        decision_value = decision_value_binary
    else:
        metrics["f1"] = np.nan
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
    """Apply a trained fold classifier to vertical (mean=0) images."""
    if len(X_vertical) == 0:
        return pd.DataFrame(), {
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
    pred_mean = (
        y_scaler.inverse_transform(pred_fit_units.reshape(-1, 1)).reshape(-1)
        if y_scaler is not None
        else pred_fit_units
    )

    residual = pred_mean - y_test_raw
    return {
        "estimator": estimator,
        "feature_scaler": X_scaler,
        "target_scaler": y_scaler,
        "pred_mean": pred_mean,
        "metrics": {
            "r2": float(r2_score(y_test_raw, pred_mean)) if len(y_test_raw) >= 2 else np.nan,
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "mae": float(np.mean(np.abs(residual))),
            "pearson_r": _compute_pearson_r(y_test_raw, pred_mean),
        },
    }
