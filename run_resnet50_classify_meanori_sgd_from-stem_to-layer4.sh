#!/usr/bin/env bash

set -euo pipefail

ulimit -n 65535 || true
echo "Open file limit: $(ulimit -n)"

SIF="${SIF:-$HOME/pytorch-openclip.sif}"
OUTPUT_DIR="results_resnet50_mean_orientation/classification_sgd_hinge"

for LAYER in stem layer1_last layer2_last layer3_last layer4_last; do
  echo "Running SGD hinge classification ${LAYER}..."

  singularity exec --nv "${SIF}" python epgabor_v1_pipeline.py \
    --img-dir images \
    --output-dir "${OUTPUT_DIR}" \
    --layer-name "${LAYER}" \
    --decoder-name sgd_hinge \
    --n-splits 5 \
    --device cuda \
    --batch-size 32 \
    --num-workers 2 \
    --include-zerovar \
    --evaluate-vertical
done

echo "Done."
