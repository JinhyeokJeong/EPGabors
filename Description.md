# Description

## EPGabor image set

This repository contains an image set for studying neural representations of orientation ensemble statistics.

All images are in `images/` and follow:

`GaborArray_m{mOri}_sd{SD}_ss{SS}_{instance}.tiff`

- `mOri`: mean orientation (`-12, -8, -4, 0, 4, 8, 12`)
- `SD`: orientation sample standard deviation (`0, 4, 8, 16`)
- `SS`: set size (`1, 4, 8, 16, 32`)
- `instance`: exemplar index (`1..100`)

Total images: `14,000` (`7 x 4 x 5 x 100`).

## Current v1 objective

Build a reproducible layerwise decoding pipeline for pretrained CNNs to test whether representations encode **binary mean orientation**:

- Class 0: `mean < 0` (CCW)
- Class 1: `mean > 0` (CW)
- `mean = 0` is excluded from training and used as boundary evaluation.

The v1 model panel is:

- `resnet50`
- `convnext_tiny`
- `efficientnet_b0`
- `vgg16_bn`

Decoding uses `LinearSVC` on stage-end layer features.

## Analysis constraints for v1

- Deterministic preprocessing only:
  - `resize + ToTensor + normalization`
  - no rotation
  - no cropping
  - no augmentation
- Cross-condition generalization:
  - leave-one-`sd`-out or leave-one-`ss`-out splits
- Default data subset:
  - exclude `ss=1`
  - exclude `sd=0`
  - keep `mean=0` for boundary analyses

## Main code files

- Dataset: `EPOriGabors.py`
- v1 pipeline: `epgabor_v1_pipeline.py`
- Plan/spec document: `ResearchPlan.md`

## Environment

Recommended local Python environment:

- `/Users/jeongj/miniconda3/envs/pip-torch-bayes`
- includes PyTorch, torchvision, timm, scikit-learn
