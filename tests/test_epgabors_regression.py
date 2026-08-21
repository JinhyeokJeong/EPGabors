import json
import os
import tempfile

import numpy as np
from torch.utils.data import DataLoader

from EPOriGabors import EPGabors
from epgabor_regression_pipeline import (
    FOLD_METRIC_COLUMNS,
    TIMING_COLUMNS,
    TRIAL_COLUMNS,
    build_regression_kfold_splits,
    extract_layer_features,
    fit_linear_regressor,
    get_resnet50_sparse_layer_map,
    list_available_layers,
    make_deterministic_transform,
    run_resnet50_kfold_regression,
)


IMG_DIR = "images"


def test_list_available_layers_returns_sparse_resnet50_map():
    layer_df = list_available_layers("resnet50")
    assert layer_df["layer_name"].tolist() == [
        "stem",
        "layer1_last",
        "layer2_last",
        "layer3_last",
        "layer4_last",
    ]
    assert layer_df["module_path"].tolist() == [
        "maxpool",
        "layer1.2",
        "layer2.3",
        "layer3.5",
        "layer4.2",
    ]


def test_feature_extraction_flattens_sparse_endpoints():
    transform = make_deterministic_transform(input_size=32)
    dataset = EPGabors(
        img_dir=IMG_DIR,
        include_vertical=True,
        include_single=False,
        include_zerovar=False,
        filter_ss=[4],
        filter_sd=[4],
        filter_mean=[-4, 0, 4],
        filter_instance=[1, 2],
        transform=transform,
        return_filename=True,
        return_condition_id=True,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    model, layer_map = get_resnet50_sparse_layer_map(pretrained=False, device="cpu")
    layer_features, metadata = extract_layer_features(
        model=model,
        layer_map=layer_map,
        dataloader=loader,
        device="cpu",
        feature_mode="flatten",
    )

    n = len(dataset)
    assert len(metadata) == n
    assert set(layer_features.keys()) == {
        "stem",
        "layer1_last",
        "layer2_last",
        "layer3_last",
        "layer4_last",
    }
    for x in layer_features.values():
        assert x.shape[0] == n
        assert x.ndim == 2
        assert x.shape[1] > 0


def test_build_regression_kfold_splits_has_no_overlap_and_preserves_levels():
    strata = np.repeat(np.array([-12, -8, -4, 0, 4, 8, 12]), 5)
    splits = build_regression_kfold_splits(strata, n_splits=5, shuffle=True, random_state=0)

    assert len(splits) == 5
    test_indices = []
    expected_levels = set(strata.tolist())
    for split in splits:
        assert not np.intersect1d(split.train_idx, split.test_idx).size
        test_indices.extend(split.test_idx.tolist())
        assert set(strata[split.test_idx].tolist()) == expected_levels

    assert sorted(test_indices) == list(range(len(strata)))


def test_linear_regressor_supports_both_modes_and_target_scaling():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(80, 12))
    weights = np.linspace(-2.0, 2.0, 12)
    y_train = X_train @ weights + rng.normal(scale=0.1, size=80)
    X_test = rng.normal(size=(40, 12))
    y_test = X_test @ weights + rng.normal(scale=0.1, size=40)

    for regressor_name in ("ridge", "sgd_squared_error"):
        result = fit_linear_regressor(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            regressor_name=regressor_name,
            random_state=0,
            max_iter=5000,
            target_scaling="standardize",
        )
        assert result["metrics"]["r2"] > 0.8
        assert result["metrics"]["rmse"] < 1.5
        assert result["metrics"]["mae"] < 1.2
        assert result["metrics"]["pearson_r"] > 0.9
        assert result["pred_mean"].shape[0] == len(y_test)


def test_run_resnet50_kfold_regression_outputs_expected_frames():
    transform = make_deterministic_transform(input_size=32)
    dataset = EPGabors(
        img_dir=IMG_DIR,
        include_vertical=True,
        include_single=False,
        include_zerovar=True,
        filter_ss=[4],
        filter_sd=[0, 4],
        filter_mean=[-4, 0, 4],
        filter_instance=[1, 2, 3, 4, 5],
        transform=transform,
        return_filename=True,
        return_condition_id=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trial_df, fold_metrics_df, layer_summary_df, timing_df = run_resnet50_kfold_regression(
            dataset=dataset,
            output_dir=tmpdir,
            selected_layer="layer4_last",
            regressor_name="ridge",
            n_splits=5,
            pretrained=False,
            device="cpu",
            batch_size=2,
            num_workers=0,
            random_state=0,
            target_scaling="none",
            measure_timing=True,
        )

        assert not trial_df.empty
        assert not fold_metrics_df.empty
        assert not layer_summary_df.empty
        assert not timing_df.empty

        assert list(trial_df.columns) == TRIAL_COLUMNS
        assert list(fold_metrics_df.columns) == FOLD_METRIC_COLUMNS
        assert list(timing_df.columns) == TIMING_COLUMNS

        assert fold_metrics_df["fold_id"].nunique() == 5
        assert fold_metrics_df["layer_name"].unique().tolist() == ["layer4_last"]
        assert trial_df["layer_name"].unique().tolist() == ["layer4_last"]
        assert set(timing_df["stage"].tolist()) >= {"feature_extraction", "fold_decode", "layer_decode", "total"}
        assert 0 in set(trial_df["sd"].unique().tolist())

        prefix = os.path.join(tmpdir, "resnet50_layer4_last_kfold_regression")
        assert os.path.exists(f"{prefix}_trial_outputs.csv")
        assert os.path.exists(f"{prefix}_fold_metrics.csv")
        assert os.path.exists(f"{prefix}_layer_summary.csv")
        assert os.path.exists(f"{prefix}_dataset_summary.csv")
        assert os.path.exists(f"{prefix}_run_config.json")
        assert os.path.exists(f"{prefix}_timing.csv")

        with open(f"{prefix}_run_config.json", "r", encoding="utf-8") as handle:
            run_config = json.load(handle)
        assert run_config["analysis"] == "regression"
        assert run_config["dataset"]["include_zerovar"] is True
