# Research Plan: ResNet50 Mean-Orientation Readouts

## Objective

Revive the EP Gabor project as a small, maintainable package for testing whether deep-network layers represent orientation ensemble statistics. The first baseline uses sparse ResNet50 endpoints and asks whether mean orientation is linearly decodable.

## Current Baseline

- Model: `resnet50`
- Endpoints: `stem`, `layer1_last`, `layer2_last`, `layer3_last`, `layer4_last`
- Features: flattened activations from deterministic preprocessing
- Classification: linear binary readout of `mean < 0` vs `mean > 0`
- Regression: L2-regularized linear regression of raw mean orientation
- Evaluation: stratified K-fold cross-validation

## Dataset Policy

- First runnable analysis includes homogeneous orientation arrays (`sd=0`) together with heterogeneous arrays.
- `ss=1` is excluded by default in the runner script.
- Classification excludes `mean=0` from training and nonzero CV scoring.
- Classification writes separate `mean=0` vertical-boundary predictions and a combined prediction table.
- Regression includes `mean=0` by default.

## Implemented API

- `epgabors.data.EPGabors`
- `epgabors.data.summarize_conditions`
- `epgabors.data.validate_condition_grid`
- `epgabors.models.get_resnet50_sparse_layer_map`
- `epgabors.models.list_available_layers`
- `epgabors.features.extract_layer_features`
- `epgabors.readouts.fit_linear_decoder`
- `epgabors.readouts.fit_linear_regressor`
- `epgabors.runners.run_resnet50_kfold_decoding`
- `epgabors.runners.run_resnet50_kfold_regression`

Legacy imports from `EPOriGabors.py`, `epgabor_v1_pipeline.py`, and `epgabor_regression_pipeline.py` are preserved.

## Validation Checklist

- Dataset parsing matches the expected 14,000-image design.
- Condition summaries report balanced retained conditions.
- Sparse endpoint mapping resolves the five expected ResNet50 modules.
- Feature matrices are 2D and aligned with metadata.
- Classification folds never train or score on `mean=0`.
- Classification combined outputs include both nonzero CV rows and vertical-boundary rows.
- Regression folds preserve mean-orientation levels through stratification.
- Runner scripts pass shell syntax checks.

## Next Extensions

- Add condition-wise plotting notebooks for outputs by `sd`, `ss`, and `mean`.
- Add additional model families after the ResNet50 baseline is stable.
- Consider CORnet as a later biologically motivated model extension.
