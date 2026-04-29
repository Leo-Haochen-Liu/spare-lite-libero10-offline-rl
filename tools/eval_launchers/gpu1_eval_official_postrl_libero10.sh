#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SIMPLEVLA_LIBERO_WORKER_START_DELAY=2
export PROJECT_NAME='SpaRe-lite-Eval'
export EXPERIMENT_NAME='sparelite-eval-official-postrl-libero10'
export SFT_MODEL_PATH='/root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl'
export DATASET_NAME='libero_10'
export NUM_GPUS=1
export NUM_NODES=1
export NUM_TRIALS_PER_TASK=50
export VAL_BATCH_SIZE=1
cd /root/autodl-tmp/SimpleVLA-RL
bash examples/run_openvla_oft_libero_eval_min.sh > /root/autodl-tmp/SpaRe-lite/results/sparelite-eval-official-postrl-libero10.eval.log 2>&1
