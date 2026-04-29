#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/root/autodl-tmp/SpaRe-lite"
REMOTE_DATA_ROOT="/root/autodl-tmp/data/libero_large"
REMOTE_POLICY="/root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl"
REMOTE_SPATIAL="${REMOTE_SPATIAL:-/root/autodl-tmp/checkpoints/facebook-dinov2-base}"
REMOTE_OUT_DIR="/root/autodl-tmp/checkpoints/spare-lite-latent-sweep-large"
SPATIAL_BACKEND="${SPATIAL_BACKEND:-auto}"

mkdir -p "${REMOTE_OUT_DIR}"
cd "${REMOTE_ROOT}"

for SEED in 7 11; do
  OUT_JSON="${REMOTE_OUT_DIR}/seed_${SEED}.json"
  LOG_TXT="${REMOTE_OUT_DIR}/seed_${SEED}.log"
  echo "[spare-lite-sweep-large] running seed=${SEED}"
  PYTHONPATH="${REMOTE_ROOT}" \
  PYTHONUNBUFFERED=1 \
  /root/miniconda3/bin/python -m spare_lite.rl_spare_latent_smoke \
    --jsonl-path "${REMOTE_DATA_ROOT}/large_train.jsonl" \
    --policy-model "${REMOTE_POLICY}" \
    --spatial-model "${REMOTE_SPATIAL}" \
    --spatial-backend "${SPATIAL_BACKEND}" \
    --batch-size 8 \
    --max-steps 200 \
    --lambda-align 0.2 \
    --candidate-mode orthogonal_noise \
    --align-mode centered_relu \
    --align-threshold 0.72 \
    --seed "${SEED}" \
    --summary-json "${OUT_JSON}" | tee "${LOG_TXT}"
done

PYTHONPATH="${REMOTE_ROOT}" \
/root/miniconda3/bin/python -m spare_lite.summarize_rl_spare_latent_sweep \
  --input-dir "${REMOTE_OUT_DIR}" \
  --output-json "${REMOTE_OUT_DIR}/summary.json" \
  --output-md "${REMOTE_OUT_DIR}/summary.md"

PYTHONPATH="${REMOTE_ROOT}" \
/root/miniconda3/bin/python -m spare_lite.decide_next_step_from_summary \
  --summary-json "${REMOTE_OUT_DIR}/summary.json" \
  --output-json "${REMOTE_OUT_DIR}/decision.json"

PYTHONPATH="${REMOTE_ROOT}" \
/root/miniconda3/bin/python -m spare_lite.generate_result_brief \
  --summary-json "${REMOTE_OUT_DIR}/summary.json" \
  --decision-json "${REMOTE_OUT_DIR}/decision.json" \
  --output-md "${REMOTE_OUT_DIR}/result_brief.md"

echo "[spare-lite-sweep-large] finished. outputs in ${REMOTE_OUT_DIR}"
