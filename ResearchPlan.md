# Research Plan (v1): Layerwise Mean-Orientation Decoding in Pretrained CNNs

## Objective
Build a reproducible pipeline to test whether pretrained CNN representations linearly encode binary mean orientation (CCW vs CW) in EPOriGabors, with cross-condition generalization across orientation variance (`sd`) and set size (`ss`).

## Scope (v1)
- Readout: binary mean orientation (`mean < 0` vs `mean > 0`)
- `mean = 0`: excluded from training, evaluated as boundary cases
- Models: `resnet50`, `convnext_tiny`, `efficientnet_b0`, `vgg16_bn` (timm)
- Layers: stage-end readouts only
- Decoder: `LinearSVC` (scikit-learn), train-only feature scaling
- Preprocessing: deterministic `resize + tensor + ImageNet normalization`, no crop/rotation/augmentation
- Outputs: both trial-level and condition-level CSVs

## Dataset and Defaults
- Dataset class: `EPGabors` in `EPOriGabors.py`
- Default v1 filtering:
  - `include_single=False` (exclude `ss=1`)
  - `include_zerovar=False` (exclude `sd=0`)
  - `include_vertical=True` (`mean=0` retained for boundary evaluation)

## Pipeline
1. Load filtered EPGabors with metadata (`mean`, `sd`, `ss`, `instance`, filename).
2. Extract stage-end activations from each model.
3. Apply global average pooling to obtain per-image feature vectors.
4. Build cross-condition splits:
   - `leave_one_sd_out` or `leave_one_ss_out`
   - training excludes `mean=0`
5. Fit `LinearSVC` per model/layer/split.
6. Evaluate:
   - Nonzero means: balanced accuracy, F1, AUROC, confusion matrix
   - `mean=0`: CW choice rate and margin distribution
7. Save:
   - Trial outputs (`*_trial_outputs.csv`)
   - Condition summaries (`*_condition_summary.csv`)
   - Layer metrics (`*_layer_metrics.csv`)

## Implemented API
- `get_model_and_layer_map(model_name, pretrained=True, source="timm", device="cpu")`
- `extract_layer_features(model, layer_map, dataloader, device="cpu", pooling="gap")`
- `build_cross_condition_splits(metadata, split_mode="leave_one_sd_out", holdout_values=None)`
- `fit_linear_svm(X_train, y_train, X_test, y_test, C=1.0, random_state=0, max_iter=10000)`
- `evaluate_boundary_m0(estimator, scaler, X_boundary, metadata_boundary)`
- `save_trial_outputs(trial_df, output_path)`
- `save_condition_summaries(condition_df, output_path)`
- End-to-end runner:
  - `run_layerwise_binary_decoding(...)`
  - `run_default_v1_panel(...)`

## Validation Checklist
- Dataset parsing and level counts match expected design (14,000 images).
- `include_zerovar=False` excludes `sd=0`.
- Split masks are non-overlapping.
- No `mean=0` images used in training labels.
- Extracted feature matrices are shape-consistent per layer.

## Next Phase (v2)
- Add psychometric fitting (PSE/JND-like analyses) using trial-level outputs.
- Add all-convolution probing mode.
- Extend to continuous decoding and orientation-variance decoding.
