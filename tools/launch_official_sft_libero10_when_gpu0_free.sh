#!/usr/bin/env bash
set -euo pipefail
while pgrep -f "sparelite-eval-libero-spatial-r1-v2" >/dev/null; do
  sleep 60
done
cd /root/autodl-tmp/SimpleVLA-RL
nohup env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SIMPLEVLA_LIBERO_WORKER_START_DELAY=2 NUM_STEPS_WAIT=10 \
  EXPERIMENT_NAME=sparelite-eval-official-sft-libero10 \
  SFT_MODEL_PATH=/root/autodl-tmp/checkpoints/Openvla-oft-SFT-libero10-traj1 \
  DATASET_NAME=libero_10 NUM_TRIALS_PER_TASK=50 VAL_BATCH_SIZE=1 \
  bash examples/run_openvla_oft_libero_eval_min.sh \
  > /root/autodl-tmp/SpaRe-lite/results/sparelite-eval-official-sft-libero10.eval.log 2>&1 < /dev/null &
