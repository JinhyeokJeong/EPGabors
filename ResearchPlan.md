# Research Plan: Sparse-Endpoint ResNet50 K-Fold Decoding

## Objective

Build a simple, research-oriented decoding pipeline to test whether sparse ResNet50 representations linearly encode binary mean orientation (`mean < 0` vs `mean > 0`) in the EP Gabor stimulus set.

## Scope

- Model: `resnet50`
- Endpoints:
  - `stem`
  - `layer1_last`
  - `layer2_last`
  - `layer3_last`
  - `layer4_last`
- Features: flatten raw endpoint activations (no extra spatial pooling)
- Decoder:
  - default: `LinearSVC`
  - fallback: `SGDClassifier(loss="hinge")`
- Evaluation:
  - `StratifiedKFold`
  - default `n_splits=5`
  - stratify only on the binary class label
- Optional boundary analysis:
  - keep `mean=0` images out of training
  - score them after each fold model is fit
- Outputs:
  - trial-level test predictions with metadata
  - fold-level metrics
  - layer-level aggregated summaries
  - optional vertical-image boundary outputs
  - timing summaries

## Default dataset subset

- `include_single=False` (exclude `ss=1`)
- `include_zerovar=False` (exclude `sd=0`)
- `mean=0` excluded from training/test folds
- `mean=0` can still be retained in the dataset for optional boundary evaluation

## Implemented API

- `get_resnet50_sparse_layer_map(pretrained=True, device="cpu")`
- `extract_layer_features(model, layer_map, dataloader, device="cpu", feature_mode="flatten")`
- `build_stratified_kfold_splits(labels, n_splits=5, shuffle=True, random_state=0)`
- `fit_linear_decoder(...)`
- `evaluate_vertical_boundary(estimator, scaler, X_vertical, metadata_vertical)`
- `run_resnet50_kfold_decoding(...)`

## Validation checklist

- Dataset parsing still matches the expected 14,000-image design.
- Sparse endpoint mapping resolves the five expected ResNet50 modules.
- Extracted feature matrices are 2D after flattening and have one row per image.
- K-fold train/test indices are non-overlapping.
- `mean=0` samples are excluded from training folds.
- Vertical-image outputs, when enabled, remain separate from the main CV trial outputs.
- Timing rows include feature extraction, fold decoding, layer decoding, and total runtime.

## Next likely extensions

- Add regression or multiclass readouts for mean orientation.
- Add condition-wise summary helpers for plots by `sd` and `ss`.
- Extend the same pipeline pattern to other network architectures once the ResNet50 baseline is stable.
