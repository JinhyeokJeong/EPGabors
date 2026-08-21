"""Backward-compatible ResNet50 regression CLI wrapper."""

from epgabors.features import extract_layer_features, make_deterministic_transform
from epgabors.models import (
    SPARSE_LAYER_PATHS,
    get_resnet50_sparse_layer_map,
    list_available_layers,
    select_layer_map,
)
from epgabors.readouts import FoldSpec, build_regression_kfold_splits, fit_linear_regressor
from epgabors.runners import (
    REGRESSION_FOLD_METRIC_COLUMNS as FOLD_METRIC_COLUMNS,
    REGRESSION_LAYER_SUMMARY_COLUMNS as LAYER_SUMMARY_COLUMNS,
    REGRESSION_TRIAL_COLUMNS as TRIAL_COLUMNS,
    TIMING_COLUMNS,
    regression_main as main,
    run_resnet50_kfold_regression,
)

__all__ = [
    "FOLD_METRIC_COLUMNS",
    "FoldSpec",
    "LAYER_SUMMARY_COLUMNS",
    "SPARSE_LAYER_PATHS",
    "TIMING_COLUMNS",
    "TRIAL_COLUMNS",
    "build_regression_kfold_splits",
    "extract_layer_features",
    "fit_linear_regressor",
    "get_resnet50_sparse_layer_map",
    "list_available_layers",
    "main",
    "make_deterministic_transform",
    "run_resnet50_kfold_regression",
    "select_layer_map",
]


if __name__ == "__main__":
    main()
