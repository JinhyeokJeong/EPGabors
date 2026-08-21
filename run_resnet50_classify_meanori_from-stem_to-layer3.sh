#!/usr/bin/env bash

set -euo pipefail

ulimit -n 65535 || true
echo "Open file limit: $(ulimit -n)"

for LAYER in stem layer1_last layer2_last layer3_last; do
  echo "Running ${LAYER}..."

  singularity exec --nv ~/pytorch-openclip.sif python epgabor_v1_pipeline.py \
    --img-dir images \
    --output-dir results_resnet50_mean_orientation/classification \
    --layer-name "${LAYER}" \
    --decoder-name linear_svc \
    --n-splits 5 \
    --device cuda \
    --batch-size 32 \
    --num-workers 2 \
    --include-zerovar \
    --evaluate-vertical
done

echo "Done."
