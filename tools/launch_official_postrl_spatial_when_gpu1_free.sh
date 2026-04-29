#!/usr/bin/env bash
set -euo pipefail
while pgrep -f "sparelite-eval-libero-spatial-r1r2-v2" >/dev/null; do
  sleep 60
done
cd /root/autodl-tmp/SimpleVLA-RL
nohup env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SIMPLEVLA_LIBERO_WORKER_START_DELAY=5 NUM_STEPS_WAIT=40 \
  EXPERIMENT_NAME=sparelite-eval-official-postrl-libero-spatial \
  SFT_MODEL_PATH=/root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl \
  DATASET_NAME=libero_spatial NUM_TRIALS_PER_TASK=50 VAL_BATCH_SIZE=1 \
  bash examples/run_openvla_oft_libero_eval_min.sh \
  > /root/autodl-tmp/SpaRe-lite/results/sparelite-eval-official-postrl-libero-spatial.eval.log 2>&1 < /dev/null &
