"""Backward-compatible ResNet50 classification CLI wrapper."""

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
    build_stratified_kfold_splits,
    evaluate_vertical_boundary,
    fit_linear_decoder,
    fit_linear_svm,
)
from epgabors.runners import (
    BOUNDARY_SUMMARY_COLUMNS,
    BOUNDARY_TRIAL_COLUMNS,
    CLASSIFICATION_COMBINED_COLUMNS,
    CLASSIFICATION_FOLD_METRIC_COLUMNS as FOLD_METRIC_COLUMNS,
    CLASSIFICATION_LAYER_SUMMARY_COLUMNS as LAYER_SUMMARY_COLUMNS,
    CLASSIFICATION_TRIAL_COLUMNS as TRIAL_COLUMNS,
    TIMING_COLUMNS,
    classification_main as main,
    run_resnet50_kfold_decoding,
)

__all__ = [
    "BOUNDARY_SUMMARY_COLUMNS",
    "BOUNDARY_TRIAL_COLUMNS",
    "CLASSIFICATION_COMBINED_COLUMNS",
    "FOLD_METRIC_COLUMNS",
    "FoldSpec",
    "LAYER_SUMMARY_COLUMNS",
    "SPARSE_LAYER_PATHS",
    "TIMING_COLUMNS",
    "TRIAL_COLUMNS",
    "build_stratified_kfold_splits",
    "evaluate_vertical_boundary",
    "extract_layer_features",
    "fit_linear_decoder",
    "fit_linear_svm",
    "get_model_and_layer_map",
    "get_resnet50_sparse_layer_map",
    "list_available_layers",
    "main",
    "make_deterministic_transform",
    "run_resnet50_kfold_decoding",
    "select_layer_map",
]


if __name__ == "__main__":
    main()
