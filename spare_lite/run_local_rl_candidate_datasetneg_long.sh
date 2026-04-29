#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/haochenliu/Documents/research/SimpleVLA-RL"
JSONL="${ROOT}/spare_lite/local_data/rl_candidate_train_datasetneg_large.jsonl"

PYTHONPATH="${ROOT}" \
python3 -m spare_lite.rl_real_smoke \
  --jsonl-path "${JSONL}" \
  --batch-size 32 \
  --max-steps 100 \
  --lambda-align 0.2 \
  --device cpu
