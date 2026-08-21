import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from EPOriGabors import EPGabors, summarize_conditions, validate_condition_grid
from epgabor_v1_pipeline import (
    BOUNDARY_SUMMARY_COLUMNS,
    BOUNDARY_TRIAL_COLUMNS,
    CLASSIFICATION_COMBINED_COLUMNS,
    TIMING_COLUMNS,
    build_stratified_kfold_splits,
    evaluate_vertical_boundary,
    extract_layer_features,
    fit_linear_decoder,
    get_resnet50_sparse_layer_map,
    list_available_layers,
    make_deterministic_transform,
    run_resnet50_kfold_decoding,
)


IMG_DIR = "images"


def test_dataset_integrity_counts():
    dataset = EPGabors(
        img_dir=IMG_DIR,
        include_vertical=True,
        include_single=True,
        include_zerovar=True,
    )
    df = dataset.df

    assert len(df) == 14000
    assert sorted(df["mean"].unique().tolist()) == [-12, -8, -4, 0, 4, 8, 12]
    assert sorted(df["sd"].unique().tolist()) == [0, 4, 8, 16]
    assert sorted(df["ss"].unique().tolist()) == [1, 4, 8, 16, 32]
    assert sorted(df["instance"].unique().tolist()) == list(range(1, 101))


def test_dataset_summary_and_condition_validation():
    dataset = EPGabors(
        img_dir=IMG_DIR,
        include_vertical=True,
        include_single=False,
        include_zerovar=True,
        filter_mean=[-4, 0, 4],
        filter_sd=[0, 4],
        filter_ss=[4],
        filter_instance=[1, 2, 3],
    )
    summary = summarize_conditions(dataset.df)
    validation = validate_condition_grid(dataset.df)

    assert len(summary) == 6
    assert set(summary["sd"].tolist()) == {0, 4}
    assert summary["n"].eq(3).all()
    assert validation["n_images"] == 18
    assert validation["n_conditions"] == 6
    assert validation["is_balanced"] is True


def test_default_filtering_supports_vertical_retention():
    dataset = EPGabors(
        img_dir=IMG_DIR,
        include_vertical=True,
        include_single=False,
        include_zerovar=False,
    )
    assert 0 in set(dataset.df["mean"].unique().tolist())
    assert 1 not in set(dataset.df["ss"].unique().tolist())
    assert 0 not in set(dataset.df["sd"].unique().tolist())


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
        include_vertical=False,
        include_single=False,
        include_zerovar=False,
        filter_ss=[4],
        filter_sd=[4],
        filter_mean=[-4, 4],
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


def test_build_stratified_kfold_splits_has_no_overlap():
    labels = np.array([0] * 10 + [1] * 10)
    splits = build_stratified_kfold_splits(labels, n_splits=5, shuffle=True, random_state=0)

    assert len(splits) == 5
    test_indices = []
    for split in splits:
        assert not np.intersect1d(split.train_idx, split.test_idx).size
        test_indices.extend(split.test_idx.tolist())

    assert sorted(test_indices) == list(range(len(labels)))


def test_linear_decoder_supports_both_modes():
    rng = np.random.default_rng(0)
    x0_train = rng.normal(loc=-1.0, scale=0.7, size=(20, 16))
    x1_train = rng.normal(loc=1.0, scale=0.7, size=(20, 16))
    x0_test = rng.normal(loc=-1.0, scale=0.7, size=(10, 16))
    x1_test = rng.normal(loc=1.0, scale=0.7, size=(10, 16))
    X_train = np.concatenate([x0_train, x1_train], axis=0)
    y_train = np.concatenate([np.zeros(20, dtype=int), np.ones(20, dtype=int)], axis=0)
    X_test = np.concatenate([x0_test, x1_test], axis=0)
    y_test = np.concatenate([np.zeros(10, dtype=int), np.ones(10, dtype=int)], axis=0)

    for decoder_name in ("linear_svc", "sgd_hinge"):
        result = fit_linear_decoder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            decoder_name=decoder_name,
            random_state=0,
            max_iter=2000,
        )
        assert result["metrics"]["balanced_accuracy"] > 0.9
        assert result["metrics"]["f1"] > 0.9
        assert result["decision_value"].shape[0] == len(y_test)


def test_evaluate_vertical_boundary_returns_metadata():
    rng = np.random.default_rng(1)
    X_train = np.concatenate(
        [
            rng.normal(-1.0, 0.5, size=(20, 8)),
            rng.normal(1.0, 0.5, size=(20, 8)),
        ],
        axis=0,
    )
    y_train = np.concatenate([np.zeros(20, dtype=int), np.ones(20, dtype=int)], axis=0)
    X_test = np.concatenate(
        [
            rng.normal(-1.0, 0.5, size=(10, 8)),
            rng.normal(1.0, 0.5, size=(10, 8)),
        ],
        axis=0,
    )
    y_test = np.concatenate([np.zeros(10, dtype=int), np.ones(10, dtype=int)], axis=0)
    fit = fit_linear_decoder(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        decoder_name="linear_svc",
        random_state=0,
    )

    metadata_vertical = pd.DataFrame(
        {
            "file_name": ["a.tiff", "b.tiff", "c.tiff"],
            "condition_id": ["m0_sd4_ss4"] * 3,
            "mean": [0, 0, 0],
            "sd": [4, 4, 4],
            "ss": [4, 4, 4],
            "instance": [1, 2, 3],
        }
    )
    X_vertical = rng.normal(0.0, 0.5, size=(3, 8))

    boundary_df, summary = evaluate_vertical_boundary(
        estimator=fit["estimator"],
        scaler=fit["scaler"],
        X_vertical=X_vertical,
        metadata_vertical=metadata_vertical,
    )

    assert boundary_df.columns.tolist() == [
        "file_name",
        "condition_id",
        "mean",
        "sd",
        "ss",
        "instance",
        "pred_class",
        "decision_value",
        "choice_cw",
    ]
    assert summary["boundary_n"] == 3.0
    assert "cw_rate" in summary


def test_run_resnet50_kfold_decoding_outputs_expected_frames():
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
        (
            trial_df,
            fold_metrics_df,
            layer_summary_df,
            boundary_vertical_df,
            boundary_summary_df,
            timing_df,
        ) = run_resnet50_kfold_decoding(
            dataset=dataset,
            output_dir=tmpdir,
            selected_layer="layer4_last",
            decoder_name="linear_svc",
            n_splits=5,
            pretrained=False,
            device="cpu",
            batch_size=2,
            num_workers=0,
            random_state=0,
            evaluate_vertical=True,
            measure_timing=True,
        )

        assert not trial_df.empty
        assert not fold_metrics_df.empty
        assert not layer_summary_df.empty
        assert not boundary_vertical_df.empty
        assert not boundary_summary_df.empty
        assert not timing_df.empty

        assert fold_metrics_df["fold_id"].nunique() == 5
        assert fold_metrics_df["layer_name"].unique().tolist() == ["layer4_last"]
        assert trial_df["layer_name"].unique().tolist() == ["layer4_last"]
        assert boundary_vertical_df["layer_name"].unique().tolist() == ["layer4_last"]

        assert list(boundary_vertical_df.columns) == BOUNDARY_TRIAL_COLUMNS
        assert list(boundary_summary_df.columns) == BOUNDARY_SUMMARY_COLUMNS
        assert list(timing_df.columns) == TIMING_COLUMNS
        assert set(timing_df["stage"].tolist()) >= {"feature_extraction", "fold_decode", "layer_decode", "total"}
        assert 0 not in set(trial_df["mean"].unique().tolist())
        assert 0 in set(dataset.df["sd"].unique().tolist())

        prefix = os.path.join(tmpdir, "resnet50_layer4_last_kfold")
        assert os.path.exists(f"{prefix}_trial_outputs.csv")
        assert os.path.exists(f"{prefix}_fold_metrics.csv")
        assert os.path.exists(f"{prefix}_layer_summary.csv")
        assert os.path.exists(f"{prefix}_vertical_boundary_outputs.csv")
        assert os.path.exists(f"{prefix}_vertical_boundary_summary.csv")
        assert os.path.exists(f"{prefix}_combined_predictions.csv")
        assert os.path.exists(f"{prefix}_dataset_summary.csv")
        assert os.path.exists(f"{prefix}_run_config.json")
        assert os.path.exists(f"{prefix}_timing.csv")

        combined_df = pd.read_csv(f"{prefix}_combined_predictions.csv")
        assert combined_df.columns.tolist() == CLASSIFICATION_COMBINED_COLUMNS
        assert set(combined_df["split"].unique().tolist()) == {"nonzero_cv", "vertical_boundary"}
        assert 0 in set(combined_df["mean"].unique().tolist())

        with open(f"{prefix}_run_config.json", "r", encoding="utf-8") as handle:
            run_config = json.load(handle)
        assert run_config["analysis"] == "classification"
        assert run_config["dataset"]["include_zerovar"] is True


def test_legacy_wrappers_and_package_imports_match():
    import epgabor_regression_pipeline
    import epgabor_v1_pipeline
    import epgabors

    assert epgabors.EPGabors is EPGabors
    assert epgabor_v1_pipeline.run_resnet50_kfold_decoding.__module__ == "epgabors.runners"
    assert epgabor_regression_pipeline.run_resnet50_kfold_regression.__module__ == "epgabors.runners"


def test_resnet50_runner_script_has_valid_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", "scripts/run_resnet50_mean_orientation.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
