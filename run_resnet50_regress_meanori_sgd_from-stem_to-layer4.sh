#!/usr/bin/env bash

set -euo pipefail

ulimit -n 65535 || true
echo "Open file limit: $(ulimit -n)"

SIF="${SIF:-$HOME/pytorch-openclip.sif}"
OUTPUT_DIR="results_resnet50_mean_orientation/regression_sgd_squared_error"

for LAYER in stem layer1_last layer2_last layer3_last layer4_last; do
  echo "Running SGD regression ${LAYER}..."

  singularity exec --nv "${SIF}" python epgabor_regression_pipeline.py \
    --img-dir images \
    --output-dir "${OUTPUT_DIR}" \
    --layer-name "${LAYER}" \
    --regressor-name sgd_squared_error \
    --n-splits 5 \
    --target-scaling standardize \
    --device cuda \
    --batch-size 32 \
    --num-workers 2 \
    --include-zerovar \
    --include-vertical
done

echo "Done."
