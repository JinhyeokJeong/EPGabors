# Description

## EP Gabor Image Set

This repository contains an image set for studying neural representations of orientation ensemble statistics.

All images are in `images/` and follow:

`GaborArray_m{mOri}_sd{SD}_ss{SS}_{instance}.tiff`

- `mOri`: mean orientation (`-12, -8, -4, 0, 4, 8, 12`)
- `SD`: orientation sample standard deviation (`0, 4, 8, 16`)
- `SS`: set size (`1, 4, 8, 16, 32`)
- `instance`: exemplar index (`1..100`)

Total images: `14,000` (`7 x 4 x 5 x 100`).

## Current Analysis Objective

The current baseline tests whether sparse ResNet50 layer representations linearly encode mean orientation in EP Gabor arrays.

- model: `resnet50` from `timm`
- endpoints: `stem`, `layer1_last`, `layer2_last`, `layer3_last`, `layer4_last`
- features: flattened endpoint activations with deterministic ImageNet preprocessing
- classification readout: binary mean-orientation sign
  - class `0`: `mean < 0`
  - class `1`: `mean > 0`
  - `mean=0` is excluded from classifier training and nonzero CV metrics
  - `mean=0` is evaluated afterward as a vertical-boundary condition
- regression readout: raw mean orientation in degrees using L2-regularized Ridge by default

## Dataset Assessment

The current `EPGabors` dataset object is usable for the revived analysis:

- filename parsing recovers `mean`, `sd`, `ss`, and `instance`
- filtering supports mean, SD, set size, and instance subsets
- optional metadata returns `file_name` and `condition_id`
- dataset summaries and condition-grid validation live in `epgabors.data`

The main caution is default filtering: direct `EPGabors()` usage excludes `ss=1` but includes `sd=0`. Analysis scripts make their subset explicit, and the first runnable ResNet50 script includes `sd=0` intentionally.

## Package Structure

- Dataset and condition utilities: `epgabors/data.py`
- ResNet50 model/layer utilities: `epgabors/models.py`
- Preprocessing and feature extraction: `epgabors/features.py`
- Linear readouts and splits: `epgabors/readouts.py`
- Classification/regression runners and CLIs: `epgabors/runners.py`

Backward-compatible wrappers remain:

- `EPOriGabors.py`
- `epgabor_v1_pipeline.py`
- `epgabor_regression_pipeline.py`

## Environment

The workstation environment used for validation is:

`/Users/jeongj/miniconda3/envs/pip-torch-bayes`

Core dependencies are listed in `pyproject.toml` and `requirements.txt`.
