#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/root/autodl-tmp/SimpleVLA-RL"
REMOTE_OUT_DIR="/root/autodl-tmp/checkpoints/spare-lite-latent-sweep"

cd "${REMOTE_ROOT}"

bash "${REMOTE_ROOT}/spare_lite/run_remote_rl_spare_latent_sweep.sh"

PYTHONPATH="${REMOTE_ROOT}" \
/root/miniconda3/bin/python -m spare_lite.check_phase_artifacts \
  --output-dir "${REMOTE_OUT_DIR}" \
  --output-json "${REMOTE_OUT_DIR}/artifact_status.json" \
  --output-md "${REMOTE_OUT_DIR}/artifact_status.md"

echo "[spare-lite-phase-a] bundle finished. outputs in ${REMOTE_OUT_DIR}"
