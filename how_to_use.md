# How To Use

This repository currently provides two command-line runners:

- `epgabor_v1_pipeline.py` for binary classification
- `epgabor_regression_pipeline.py` for regression

Both use the same sparse ResNet50 endpoints:

- `stem`
- `layer1_last`
- `layer2_last`
- `layer3_last`
- `layer4_last`

## Recommended environment

Use the project conda environment if available:

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python
```

If you activate that environment first, you can also just use `python`.

## 1. Classification runner

This predicts the **sign** of the mean orientation:

- class `0`: `mean < 0`
- class `1`: `mean > 0`

Default behavior:

- excludes `ss=1`
- excludes `sd=0`
- excludes `mean=0` from training/test folds
- optionally evaluates `mean=0` images after training

### Example

Run all sparse endpoints with the default linear SVM:

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python epgabor_v1_pipeline.py \
  --img-dir images \
  --output-dir results_kfold \
  --layer-name all \
  --decoder-name linear_svc \
  --n-splits 5 \
  --device cpu
```

Run one layer only:

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python epgabor_v1_pipeline.py \
  --img-dir images \
  --output-dir results_kfold \
  --layer-name layer4_last \
  --decoder-name linear_svc \
  --n-splits 5 \
  --device cpu
```

### Main arguments

- `--layer-name`: `all`, `stem`, `layer1_last`, `layer2_last`, `layer3_last`, or `layer4_last`
- `--decoder-name`: `linear_svc` or `sgd_hinge`
- `--n-splits`: number of cross-validation folds (default `5`)
- `--device`: usually `cpu`; use `cuda` if available and desired
- `--pretrained` / `--no-pretrained`: use pretrained ResNet50 weights or not
- `--evaluate-vertical` / `--no-evaluate-vertical`: whether `mean=0` images are scored after each fold
- `--include-single`: include `ss=1` images
- `--include-zerovar`: include `sd=0` images
- `--include-vertical` / `--no-include-vertical`: manually control whether `mean=0` images are present in the dataset
- `--dataset-means`: comma-separated means to keep, for example `--dataset-means -4,0,4`
- `--dataset-sds`: comma-separated SD levels to keep, for example `--dataset-sds 4,8,16`
- `--dataset-sss`: comma-separated set sizes to keep, for example `--dataset-sss 4,8,16`
- `--dataset-instances`: comma-separated instance IDs to keep, for example `--dataset-instances 1,2,3,4,5`

### Output files

For `--layer-name all`, the main outputs are written to `results_kfold/` with names like:

- `resnet50_all_layers_kfold_trial_outputs.csv`
- `resnet50_all_layers_kfold_fold_metrics.csv`
- `resnet50_all_layers_kfold_layer_summary.csv`
- `resnet50_all_layers_kfold_vertical_boundary_outputs.csv` (if enabled)
- `resnet50_all_layers_kfold_vertical_boundary_summary.csv` (if enabled)
- `resnet50_all_layers_kfold_timing.csv`

## 2. Regression runner

This predicts the **raw mean orientation in degrees** using L2-regularized regression.

Default behavior:

- includes `mean=0`
- excludes `ss=1`
- excludes `sd=0`

### Example

Run all sparse endpoints with Ridge regression:

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python epgabor_regression_pipeline.py \
  --img-dir images \
  --output-dir results_kfold_regression \
  --layer-name all \
  --regressor-name ridge \
  --n-splits 5 \
  --target-scaling none \
  --device cpu
```

Run one layer only with SGD regression:

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python epgabor_regression_pipeline.py \
  --img-dir images \
  --output-dir results_kfold_regression \
  --layer-name layer4_last \
  --regressor-name sgd_squared_error \
  --n-splits 5 \
  --target-scaling standardize \
  --device cpu
```

### Main arguments

- `--layer-name`: `all`, `stem`, `layer1_last`, `layer2_last`, `layer3_last`, or `layer4_last`
- `--regressor-name`: `ridge` or `sgd_squared_error`
- `--n-splits`: number of cross-validation folds (default `5`)
- `--target-scaling`: `none` or `standardize`
- `--ridge-alpha`: L2 strength for Ridge (default `1.0`)
- `--sgd-alpha`: L2 strength for SGDRegressor
- `--device`: usually `cpu`; use `cuda` if available and desired
- `--pretrained` / `--no-pretrained`: use pretrained ResNet50 weights or not
- `--include-single`: include `ss=1` images
- `--include-zerovar`: include `sd=0` images
- `--include-vertical` / `--no-include-vertical`: include or exclude `mean=0` images (default is to include them)
- `--dataset-means`, `--dataset-sds`, `--dataset-sss`, `--dataset-instances`: same filtering format as the classification runner

### Output files

For `--layer-name all`, the main outputs are written to `results_kfold_regression/` with names like:

- `resnet50_all_layers_kfold_regression_trial_outputs.csv`
- `resnet50_all_layers_kfold_regression_fold_metrics.csv`
- `resnet50_all_layers_kfold_regression_layer_summary.csv`
- `resnet50_all_layers_kfold_regression_timing.csv`

## 3. Helpful utility command

To list the supported sparse endpoint names:

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python epgabor_v1_pipeline.py --list-layers
```

or

```bash
/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python epgabor_regression_pipeline.py --list-layers
```

## 4. Practical notes

- Start with `--layer-name layer4_last` if you want a quick smoke run before using `all`.
- Keep `--batch-size` small on CPU if memory is tight.
- Use `sgd_hinge` or `sgd_squared_error` when flattened features are too large for the default solver.
- After a run, open the review notebooks:
  - `kfold-results-review.ipynb` for classification
  - `kfold-regression-results-review.ipynb` for regression
