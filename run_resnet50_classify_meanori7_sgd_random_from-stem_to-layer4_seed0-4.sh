#!/usr/bin/env bash

set -euo pipefail

ulimit -n 65535 || true
echo "Open file limit: $(ulimit -n)"

SIF="${SIF:-$HOME/pytorch-openclip.sif}"
RANDOM_STATE=0

for MODEL_SEED in 0 1 2 3 4; do
  OUTPUT_DIR="results_resnet50_mean_orientation/classification_mean7_sgd_hinge_random_seed${MODEL_SEED}"

  for LAYER in stem layer1_last layer2_last layer3_last layer4_last; do
    echo "Running random 7-class SGD hinge classification seed ${MODEL_SEED}, ${LAYER}..."

    singularity exec --nv "${SIF}" python epgabor_v1_pipeline.py \
      --img-dir images \
      --output-dir "${OUTPUT_DIR}" \
      --classification-mode mean7 \
      --layer-name "${LAYER}" \
      --decoder-name sgd_hinge \
      --n-splits 5 \
      --device cuda \
      --batch-size 32 \
      --num-workers 2 \
      --no-pretrained \
      --model-seed "${MODEL_SEED}" \
      --random-state "${RANDOM_STATE}" \
      --include-zerovar
  done
done

echo "Done."
