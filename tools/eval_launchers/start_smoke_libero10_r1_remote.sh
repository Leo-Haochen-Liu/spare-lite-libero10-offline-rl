#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/SimpleVLA-RL
pkill -f 'sparelite-smoke-libero10-r1' || true
mkdir -p /root/autodl-tmp/SpaRe-lite/results/eval_runtime
: > /root/autodl-tmp/SpaRe-lite/results/eval_runtime/smoke_libero10_r1.log
nohup env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SIMPLEVLA_LIBERO_WORKER_START_DELAY=2 PROJECT_NAME=SpaRe-lite-Eval EXPERIMENT_NAME=sparelite-smoke-libero10-r1 SFT_MODEL_PATH=/root/autodl-tmp/checkpoints/sparelite_eval_models/libero10_3200_r1 DATASET_NAME=libero_10 NUM_GPUS=1 NUM_NODES=1 NUM_TRIALS_PER_TASK=1 VAL_BATCH_SIZE=1 bash examples/run_openvla_oft_libero_eval_min.sh >/root/autodl-tmp/SpaRe-lite/results/eval_runtime/smoke_libero10_r1.log 2>&1 &
echo $! > /root/autodl-tmp/SpaRe-lite/results/eval_runtime/smoke_libero10_r1.pid
sleep 1
cat /root/autodl-tmp/SpaRe-lite/results/eval_runtime/smoke_libero10_r1.pid
