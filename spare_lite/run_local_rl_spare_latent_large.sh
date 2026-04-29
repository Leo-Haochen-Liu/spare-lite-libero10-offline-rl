#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/haochenliu/Documents/research/SimpleVLA-RL"
JSONL="${ROOT}/spare_lite/local_data/large_train_localpaths.jsonl"
POLICY_MODEL="${POLICY_MODEL:-openvla/openvla-7b}"
SPATIAL_MODEL="${SPATIAL_MODEL:-facebook/dinov2-base}"
SPATIAL_BACKEND="${SPATIAL_BACKEND:-auto}"

PYTHONPATH="${ROOT}" \
python3 -m spare_lite.rl_spare_latent_smoke \
  --jsonl-path "${JSONL}" \
  --policy-model "${POLICY_MODEL}" \
  --spatial-model "${SPATIAL_MODEL}" \
  --spatial-backend "${SPATIAL_BACKEND}" \
  --batch-size 8 \
  --max-steps 200 \
  --lambda-align 0.2 \
  --candidate-mode orthogonal_noise \
  --align-mode centered_relu \
  --align-threshold 0.72 \
  --device cpu
