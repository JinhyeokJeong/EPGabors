#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/Users/jeongj/miniconda3/envs/pip-torch-bayes/bin/python}"
DEVICE="${DEVICE:-cpu}"
LAYER_NAME="${LAYER_NAME:-all}"
IMG_DIR="${IMG_DIR:-images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_resnet50_mean_orientation}"
N_SPLITS="${N_SPLITS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RANDOM_STATE="${RANDOM_STATE:-0}"
PRETRAINED_FLAG="${PRETRAINED_FLAG:---pretrained}"

CLASSIFICATION_OUTPUT_DIR="${CLASSIFICATION_OUTPUT_DIR:-${OUTPUT_ROOT}/classification}"
REGRESSION_OUTPUT_DIR="${REGRESSION_OUTPUT_DIR:-${OUTPUT_ROOT}/regression}"

CLASSIFICATION_CMD=(
  "${PYTHON}" epgabor_v1_pipeline.py
  --img-dir "${IMG_DIR}" \
  --output-dir "${CLASSIFICATION_OUTPUT_DIR}" \
  --layer-name "${LAYER_NAME}" \
  --decoder-name linear_svc \
  --n-splits "${N_SPLITS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --random-state "${RANDOM_STATE}" \
  --device "${DEVICE}"
)
if [[ -n "${PRETRAINED_FLAG}" ]]; then
  CLASSIFICATION_CMD+=("${PRETRAINED_FLAG}")
fi
CLASSIFICATION_CMD+=(
  --include-zerovar \
  --evaluate-vertical
)
"${CLASSIFICATION_CMD[@]}"

REGRESSION_CMD=(
  "${PYTHON}" epgabor_regression_pipeline.py
  --img-dir "${IMG_DIR}" \
  --output-dir "${REGRESSION_OUTPUT_DIR}" \
  --layer-name "${LAYER_NAME}" \
  --regressor-name ridge \
  --n-splits "${N_SPLITS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --random-state "${RANDOM_STATE}" \
  --device "${DEVICE}"
)
if [[ -n "${PRETRAINED_FLAG}" ]]; then
  REGRESSION_CMD+=("${PRETRAINED_FLAG}")
fi
REGRESSION_CMD+=(
  --include-zerovar \
  --include-vertical
)
"${REGRESSION_CMD[@]}"
