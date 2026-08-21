#!/usr/bin/env bash

set -euo pipefail

ulimit -n 65535 || true
echo "Open file limit: $(ulimit -n)"

SIF="${SIF:-$HOME/pytorch-openclip.sif}"
RANDOM_STATE=0

for MODEL_SEED in 0 1 2 3 4; do
  OUTPUT_DIR="results_resnet50_mean_orientation/regression_random_seed${MODEL_SEED}"

  for LAYER in stem layer1_last layer2_last layer3_last layer4_last; do
    echo "Running random regression seed ${MODEL_SEED}, ${LAYER}..."

    singularity exec --nv "${SIF}" python epgabor_regression_pipeline.py \
      --img-dir images \
      --output-dir "${OUTPUT_DIR}" \
      --layer-name "${LAYER}" \
      --regressor-name ridge \
      --n-splits 5 \
      --target-scaling none \
      --device cuda \
      --batch-size 32 \
      --num-workers 2 \
      --no-pretrained \
      --model-seed "${MODEL_SEED}" \
      --random-state "${RANDOM_STATE}" \
      --include-zerovar \
      --include-vertical
  done
done

echo "Done."
