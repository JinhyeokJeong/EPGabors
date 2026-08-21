# How To Use

This repository now exposes a small `epgabors` package plus backward-compatible command-line wrappers.

## Environment

Validated local interpreter:

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python
```

Install dependencies in another environment with:

```bash
python -m pip install -r requirements.txt
```

## One-Command Baseline

Run the first ResNet50 mean-orientation baseline:

```bash
scripts/run_resnet50_mean_orientation.sh
```

Defaults:

- Python: `/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python`
- device: `cpu`
- layer: `all`
- image directory: `images`
- includes `sd=0`
- excludes `ss=1`

Override for a GPU workstation:

```bash
DEVICE=cuda LAYER_NAME=layer4_last BATCH_SIZE=64 NUM_WORKERS=8 \
  scripts/run_resnet50_mean_orientation.sh
```

Use random ResNet50 weights for a quick no-download smoke run:

```bash
PRETRAINED_FLAG=--no-pretrained LAYER_NAME=layer4_last BATCH_SIZE=4 \
  scripts/run_resnet50_mean_orientation.sh
```

## Classification Runner

This predicts the sign of mean orientation:

- class `0`: `mean < 0`
- class `1`: `mean > 0`

`mean=0` images are never used for classifier training or nonzero CV metrics. When present, they are evaluated separately after each fold model and are also written to a combined prediction table.

Example:

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python epgabor_v1_pipeline.py \
  --img-dir images \
  --output-dir results_kfold \
  --layer-name all \
  --decoder-name linear_svc \
  --n-splits 5 \
  --device cpu \
  --include-zerovar \
  --evaluate-vertical
```

Main outputs:

- `resnet50_all_layers_kfold_trial_outputs.csv`
- `resnet50_all_layers_kfold_fold_metrics.csv`
- `resnet50_all_layers_kfold_layer_summary.csv`
- `resnet50_all_layers_kfold_vertical_boundary_outputs.csv`
- `resnet50_all_layers_kfold_vertical_boundary_summary.csv`
- `resnet50_all_layers_kfold_combined_predictions.csv`
- `resnet50_all_layers_kfold_dataset_summary.csv`
- `resnet50_all_layers_kfold_run_config.json`
- `resnet50_all_layers_kfold_timing.csv`

## Regression Runner

This predicts raw mean orientation in degrees using Ridge regression by default.

Example:

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python epgabor_regression_pipeline.py \
  --img-dir images \
  --output-dir results_kfold_regression \
  --layer-name all \
  --regressor-name ridge \
  --n-splits 5 \
  --target-scaling none \
  --device cpu \
  --include-zerovar \
  --include-vertical
```

Main outputs:

- `resnet50_all_layers_kfold_regression_trial_outputs.csv`
- `resnet50_all_layers_kfold_regression_fold_metrics.csv`
- `resnet50_all_layers_kfold_regression_layer_summary.csv`
- `resnet50_all_layers_kfold_regression_dataset_summary.csv`
- `resnet50_all_layers_kfold_regression_run_config.json`
- `resnet50_all_layers_kfold_regression_timing.csv`

## Useful Commands

List supported ResNet50 endpoints:

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python epgabor_v1_pipeline.py --list-layers
```

Run tests:

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python -m pytest -q
```
