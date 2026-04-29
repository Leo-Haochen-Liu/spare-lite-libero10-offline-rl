#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/root/autodl-tmp/SpaRe-lite"
REMOTE_DATA_ROOT="/root/autodl-tmp/data/libero_large"
REMOTE_POLICY="/root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl"
REMOTE_SPATIAL="${REMOTE_SPATIAL:-/root/autodl-tmp/checkpoints/facebook-dinov2-base}"
REMOTE_RESULTS_ROOT="${REMOTE_ROOT}/results"
REMOTE_OUT_DIR="${REMOTE_OUT_DIR:-${REMOTE_RESULTS_ROOT}/r1_large_b16_s400_seed7}"
SPATIAL_BACKEND="${SPATIAL_BACKEND:-auto}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_STEPS="${MAX_STEPS:-400}"
SEED="${SEED:-7}"

mkdir -p "${REMOTE_OUT_DIR}"
cd "${REMOTE_ROOT}"

OUT_JSON="${REMOTE_OUT_DIR}/r1_seed_${SEED}.json"
LOG_TXT="${REMOTE_OUT_DIR}/r1_seed_${SEED}.log"

PYTHONPATH="${REMOTE_ROOT}" \
PYTHONUNBUFFERED=1 \
/root/miniconda3/bin/python -m spare_lite.rl_spare_latent_smoke \
  --jsonl-path "${REMOTE_DATA_ROOT}/large_train.jsonl" \
  --policy-model "${REMOTE_POLICY}" \
  --spatial-model "${REMOTE_SPATIAL}" \
  --spatial-backend "${SPATIAL_BACKEND}" \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --lambda-align 0.0 \
  --candidate-mode orthogonal_noise \
  --align-mode centered_relu \
  --align-threshold 0.72 \
  --seed "${SEED}" \
  --summary-json "${OUT_JSON}" | tee "${LOG_TXT}"

echo "[spare-lite-r1-large] finished. outputs in ${REMOTE_OUT_DIR}"
