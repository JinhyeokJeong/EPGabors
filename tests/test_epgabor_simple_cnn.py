import os
import tempfile

import torch
from torch.utils.data import DataLoader

from epgabor_simple_cnn import (
    CLASSIFICATION_TEST_COLUMNS,
    CLASSIFICATION_BOUNDARY_COLUMNS,
    REGRESSION_TEST_COLUMNS,
    SimpleGaborCNN,
    TaskDataset,
    build_base_dataset,
    build_dataset_from_checkpoint,
    build_fixed_splits,
    build_split_datasets_from_checkpoint,
    extract_activations,
    list_named_modules,
    load_checkpoint_bundle,
    load_trained_model,
    make_basic_tensor_transform,
    predict_with_metadata,
    run_training,
    summarize_predictions_by_group,
)


IMG_DIR = "images"


def _small_dataset():
    return build_base_dataset(
        img_dir=IMG_DIR,
        include_vertical=True,
        include_single=False,
        include_zerovar=False,
        filter_mean=[-4, 0, 4],
        filter_sd=[4],
        filter_ss=[4],
        filter_instance=[1, 2, 3, 4, 5],
    )


def test_make_basic_tensor_transform_preserves_spatial_size():
    dataset = build_base_dataset(
        img_dir=IMG_DIR,
        include_vertical=True,
        include_single=False,
        include_zerovar=False,
        filter_mean=[-4],
        filter_sd=[4],
        filter_ss=[4],
        filter_instance=[1],
    )
    raw_image, _ = dataset[0]
    transform = make_basic_tensor_transform()
    transformed = transform(raw_image.permute(1, 2, 0).numpy())

    assert raw_image.shape == transformed.shape
    assert tuple(transformed.shape) == (3, 377, 377)


def test_build_fixed_splits_classification_excludes_vertical_from_main_sets():
    dataset = _small_dataset()
    splits = build_fixed_splits(
        metadata_df=dataset.df,
        task="classification",
        train_frac=0.6,
        val_frac=0.2,
        seed=0,
        evaluate_vertical=True,
    )

    all_main = set(splits["train_idx"]) | set(splits["val_idx"]) | set(splits["test_idx"])
    boundary = set(splits["boundary_idx"])

    assert not (set(splits["train_idx"]) & set(splits["val_idx"]))
    assert not (set(splits["train_idx"]) & set(splits["test_idx"]))
    assert not (set(splits["val_idx"]) & set(splits["test_idx"]))
    assert len(boundary) == 5
    assert dataset.df.loc[list(all_main), "mean"].ne(0).all()
    assert dataset.df.loc[list(boundary), "mean"].eq(0).all()


def test_build_fixed_splits_regression_includes_vertical():
    dataset = _small_dataset()
    splits = build_fixed_splits(
        metadata_df=dataset.df,
        task="regression",
        train_frac=0.6,
        val_frac=0.2,
        seed=0,
        evaluate_vertical=True,
    )

    all_main = set(splits["train_idx"]) | set(splits["val_idx"]) | set(splits["test_idx"])
    assert len(all_main) == len(dataset)
    assert not splits["boundary_idx"]
    assert 0 in set(dataset.df.loc[list(all_main), "mean"].tolist())


def test_task_dataset_returns_metadata_fields():
    dataset = _small_dataset()
    splits = build_fixed_splits(
        metadata_df=dataset.df,
        task="classification",
        train_frac=0.6,
        val_frac=0.2,
        seed=0,
        evaluate_vertical=True,
    )
    task_dataset = TaskDataset(dataset, splits["train_idx"], task="classification")
    image, target, metadata = task_dataset[0]

    assert tuple(image.shape) == (3, 377, 377)
    assert target.ndim == 0
    assert set(metadata.keys()) == {"file_name", "condition_id", "mean", "sd", "ss", "instance"}


def test_simple_gabor_cnn_outputs_one_scalar_per_sample():
    model = SimpleGaborCNN(conv_channels=(4, 8), fc_hidden_dim=8)
    batch = torch.randn(2, 3, 377, 377)
    output = model(batch)

    assert tuple(output.shape) == (2,)


def test_classification_smoke_run_writes_outputs_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_training(
            img_dir=IMG_DIR,
            output_dir=tmpdir,
            task="classification",
            device="cpu",
            seed=0,
            batch_size=2,
            num_workers=0,
            epochs=1,
            patience=0,
            lr=1e-3,
            train_frac=0.6,
            val_frac=0.2,
            conv_channels=(4, 8),
            kernel_size=3,
            fc_hidden_dim=8,
            dropout=0.0,
            include_single=False,
            include_zerovar=False,
            include_vertical=True,
            evaluate_vertical=True,
            filter_mean=[-4, 0, 4],
            filter_sd=[4],
            filter_ss=[4],
            filter_instance=[1, 2, 3, 4, 5],
        )

        checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")
        predictions = result["test_predictions_df"]

        assert os.path.exists(os.path.join(tmpdir, "simple_cnn_classification_best.pt"))
        assert os.path.exists(os.path.join(tmpdir, "simple_cnn_classification_history.csv"))
        assert os.path.exists(os.path.join(tmpdir, "simple_cnn_classification_test_predictions.csv"))
        assert os.path.exists(os.path.join(tmpdir, "simple_cnn_classification_test_metrics.csv"))
        assert os.path.exists(os.path.join(tmpdir, "simple_cnn_classification_boundary_predictions.csv"))
        assert os.path.exists(os.path.join(tmpdir, "simple_cnn_classification_boundary_summary.csv"))
        assert predictions.columns.tolist() == CLASSIFICATION_TEST_COLUMNS
        assert {
            "model_state_dict",
            "model_config",
            "task",
            "dataset_config",
            "training_config",
            "split_config",
            "best_epoch",
            "best_val_metrics",
        } <= set(checkpoint.keys())


def test_regression_smoke_run_writes_outputs_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_training(
            img_dir=IMG_DIR,
            output_dir=tmpdir,
            task="regression",
            device="cpu",
            seed=0,
            batch_size=2,
            num_workers=0,
            epochs=1,
            patience=0,
            lr=1e-3,
            train_frac=0.6,
            val_frac=0.2,
            conv_channels=(4, 8),
            kernel_size=3,
            fc_hidden_dim=8,
            dropout=0.0,
            include_single=False,
            include_zerovar=False,
            include_vertical=True,
            evaluate_vertical=True,
            filter_mean=[-4, 0, 4],
            filter_sd=[4],
            filter_ss=[4],
            filter_instance=[1, 2, 3, 4, 5],
        )

        checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")
        predictions = result["test_predictions_df"]

        assert os.path.exists(os.path.join(tmpdir, "simple_cnn_regression_best.pt"))
        assert os.path.exists(os.path.join(tmpdir, "simple_cnn_regression_history.csv"))
        assert os.path.exists(os.path.join(tmpdir, "simple_cnn_regression_test_predictions.csv"))
        assert os.path.exists(os.path.join(tmpdir, "simple_cnn_regression_test_metrics.csv"))
        assert predictions.columns.tolist() == REGRESSION_TEST_COLUMNS
        assert checkpoint["task"] == "regression"


def test_checkpoint_loading_and_model_reconstruction():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_training(
            img_dir=IMG_DIR,
            output_dir=tmpdir,
            task="classification",
            device="cpu",
            seed=0,
            batch_size=2,
            num_workers=0,
            epochs=1,
            patience=0,
            lr=1e-3,
            train_frac=0.6,
            val_frac=0.2,
            conv_channels=(4, 8),
            fc_hidden_dim=8,
            include_vertical=True,
            evaluate_vertical=True,
            filter_mean=[-4, 0, 4],
            filter_sd=[4],
            filter_ss=[4],
            filter_instance=[1, 2, 3, 4, 5],
        )

        checkpoint = load_checkpoint_bundle(result["checkpoint_path"], device="cpu")
        model, loaded_checkpoint = load_trained_model(result["checkpoint_path"], device="cpu")
        forward_output = model(torch.randn(2, 3, 377, 377))

        assert checkpoint["task"] == "classification"
        assert loaded_checkpoint["best_epoch"] >= 1
        assert tuple(forward_output.shape) == (2,)


def test_build_dataset_from_older_checkpoint_raises_clear_error():
    legacy_checkpoint = {
        "model_state_dict": {},
        "model_config": {"in_channels": 3, "conv_channels": (4, 8), "kernel_size": 3, "padding": 1, "fc_hidden_dim": 8, "dropout": 0.0},
        "task": "classification",
        "training_config": {},
        "split_config": {},
        "best_epoch": 1,
        "best_val_metrics": {},
    }

    try:
        build_dataset_from_checkpoint(legacy_checkpoint)
    except ValueError as exc:
        assert "dataset_config" in str(exc)
    else:
        raise AssertionError("Expected build_dataset_from_checkpoint to reject older checkpoints.")


def test_rebuild_saved_splits_and_boundary_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_training(
            img_dir=IMG_DIR,
            output_dir=tmpdir,
            task="classification",
            device="cpu",
            seed=0,
            batch_size=2,
            num_workers=0,
            epochs=1,
            patience=0,
            lr=1e-3,
            train_frac=0.6,
            val_frac=0.2,
            conv_channels=(4, 8),
            fc_hidden_dim=8,
            include_vertical=True,
            evaluate_vertical=True,
            filter_mean=[-4, 0, 4],
            filter_sd=[4],
            filter_ss=[4],
            filter_instance=[1, 2, 3, 4, 5],
        )
        checkpoint = load_checkpoint_bundle(result["checkpoint_path"], device="cpu")

        test_dataset = build_split_datasets_from_checkpoint(checkpoint, img_dir=IMG_DIR, split_name="test")
        boundary_dataset = build_split_datasets_from_checkpoint(checkpoint, img_dir=IMG_DIR, split_name="boundary")

        assert isinstance(test_dataset, TaskDataset)
        assert len(test_dataset) == len(result["test_predictions_df"])
        assert boundary_dataset.include_target is False
        assert len(boundary_dataset) == len(result["boundary_predictions_df"])


def test_regression_checkpoint_rejects_boundary_split():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_training(
            img_dir=IMG_DIR,
            output_dir=tmpdir,
            task="regression",
            device="cpu",
            seed=0,
            batch_size=2,
            num_workers=0,
            epochs=1,
            patience=0,
            lr=1e-3,
            train_frac=0.6,
            val_frac=0.2,
            conv_channels=(4, 8),
            fc_hidden_dim=8,
            include_vertical=True,
            filter_mean=[-4, 0, 4],
            filter_sd=[4],
            filter_ss=[4],
            filter_instance=[1, 2, 3, 4, 5],
        )
        checkpoint = load_checkpoint_bundle(result["checkpoint_path"], device="cpu")

        try:
            build_split_datasets_from_checkpoint(checkpoint, img_dir=IMG_DIR, split_name="boundary")
        except ValueError as exc:
            assert "classification" in str(exc)
        else:
            raise AssertionError("Expected regression checkpoint to reject boundary split.")


def test_prediction_helpers_and_group_summaries():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_training(
            img_dir=IMG_DIR,
            output_dir=tmpdir,
            task="classification",
            device="cpu",
            seed=0,
            batch_size=2,
            num_workers=0,
            epochs=1,
            patience=0,
            lr=1e-3,
            train_frac=0.6,
            val_frac=0.2,
            conv_channels=(4, 8),
            fc_hidden_dim=8,
            include_vertical=True,
            evaluate_vertical=True,
            filter_mean=[-4, 0, 4],
            filter_sd=[4],
            filter_ss=[4],
            filter_instance=[1, 2, 3, 4, 5],
        )
        model, checkpoint = load_trained_model(result["checkpoint_path"], device="cpu")
        test_dataset = build_split_datasets_from_checkpoint(checkpoint, img_dir=IMG_DIR, split_name="test")
        boundary_dataset = build_split_datasets_from_checkpoint(checkpoint, img_dir=IMG_DIR, split_name="boundary")

        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
        boundary_loader = DataLoader(boundary_dataset, batch_size=2, shuffle=False, num_workers=0)

        test_predictions = predict_with_metadata(model, test_loader, task="classification", device="cpu")
        boundary_predictions = predict_with_metadata(model, boundary_loader, task="classification", device="cpu")
        grouped = summarize_predictions_by_group(test_predictions, task="classification", group_by=["mean"])

        assert test_predictions.columns.tolist() == CLASSIFICATION_TEST_COLUMNS
        assert boundary_predictions.columns.tolist() == CLASSIFICATION_BOUNDARY_COLUMNS
        assert set(grouped.columns) >= {"mean", "n", "accuracy", "balanced_accuracy", "mean_prob_cw", "mean_logit"}


def test_regression_prediction_summary_helper():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_training(
            img_dir=IMG_DIR,
            output_dir=tmpdir,
            task="regression",
            device="cpu",
            seed=0,
            batch_size=2,
            num_workers=0,
            epochs=1,
            patience=0,
            lr=1e-3,
            train_frac=0.6,
            val_frac=0.2,
            conv_channels=(4, 8),
            fc_hidden_dim=8,
            include_vertical=True,
            filter_mean=[-4, 0, 4],
            filter_sd=[4],
            filter_ss=[4],
            filter_instance=[1, 2, 3, 4, 5],
        )
        model, checkpoint = load_trained_model(result["checkpoint_path"], device="cpu")
        test_dataset = build_split_datasets_from_checkpoint(checkpoint, img_dir=IMG_DIR, split_name="test")
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

        test_predictions = predict_with_metadata(model, test_loader, task="regression", device="cpu")
        grouped = summarize_predictions_by_group(test_predictions, task="regression", group_by=["mean"])

        assert test_predictions.columns.tolist() == REGRESSION_TEST_COLUMNS
        assert set(grouped.columns) >= {"mean", "n", "mae", "rmse", "mean_pred", "mean_target", "mean_residual"}


def test_list_named_modules_and_extract_activations():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_training(
            img_dir=IMG_DIR,
            output_dir=tmpdir,
            task="classification",
            device="cpu",
            seed=0,
            batch_size=2,
            num_workers=0,
            epochs=1,
            patience=0,
            lr=1e-3,
            train_frac=0.6,
            val_frac=0.2,
            conv_channels=(4, 8),
            fc_hidden_dim=8,
            include_vertical=True,
            evaluate_vertical=True,
            filter_mean=[-4, 0, 4],
            filter_sd=[4],
            filter_ss=[4],
            filter_instance=[1, 2, 3, 4, 5],
        )
        model, checkpoint = load_trained_model(result["checkpoint_path"], device="cpu")
        test_dataset = build_split_datasets_from_checkpoint(checkpoint, img_dir=IMG_DIR, split_name="test")
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

        modules_df = list_named_modules(model)
        activations, metadata_df = extract_activations(
            model=model,
            dataloader=test_loader,
            layer_name="features.0",
            device="cpu",
            flatten=True,
        )

        assert set(modules_df["module_name"].tolist()) >= {"features", "features.0", "classifier"}
        assert activations.shape[0] == len(test_dataset)
        assert len(metadata_df) == len(test_dataset)
        assert activations.ndim == 2


def test_notebook_exists_at_repo_root():
    notebook_path = os.path.join(os.getcwd(), "simple-cnn-walkthrough.ipynb")
    assert os.path.exists(notebook_path)
    with open(notebook_path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    assert "Simple CNN Walkthrough" in raw
    assert "run_training(" in raw
