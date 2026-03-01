# Description

## EP Gabor image set

This repository contains an image set for studying neural representations of orientation ensemble statistics.

All images are in `images/` and follow:

`GaborArray_m{mOri}_sd{SD}_ss{SS}_{instance}.tiff`

- `mOri`: mean orientation (`-12, -8, -4, 0, 4, 8, 12`)
- `SD`: orientation sample standard deviation (`0, 4, 8, 16`)
- `SS`: set size (`1, 4, 8, 16, 32`)
- `instance`: exemplar index (`1..100`)

Total images: `14,000` (`7 x 4 x 5 x 100`).

## Current analysis objective

The repository is currently focused on a single research-first baseline:

- model: `resnet50`
- endpoints: sparse checkpoints (`stem`, `layer1_last`, `layer2_last`, `layer3_last`, `layer4_last`)
- features: flattened activations with no extra spatial pooling
- readout: linear classification of binary mean orientation sign
  - class `0`: `mean < 0`
  - class `1`: `mean > 0`
- evaluation: `StratifiedKFold` cross-validation
- optional post-training analysis: evaluate `mean = 0` (vertical) images with each trained fold model

## Default analysis subset

- exclude `ss=1`
- exclude `sd=0`
- exclude `mean=0` from classifier training and fold scoring
- keep `mean=0` available only for optional boundary evaluation

## Main code files

- Dataset: `EPOriGabors.py`
- K-fold decoding pipeline: `epgabor_v1_pipeline.py`
- Current plan/spec document: `ResearchPlan.md`

## Environment

Recommended local Python environment:

- `/Users/jeongj/miniconda3/envs/pip-torch-bayes`

This environment should include at least:

- PyTorch
- torchvision
- timm
- scikit-learn
- pandas
- tifffile
