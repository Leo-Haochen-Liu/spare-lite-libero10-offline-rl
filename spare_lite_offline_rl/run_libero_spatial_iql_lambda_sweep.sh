#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/SpaRe-lite}"
LIBERO_DIR="${LIBERO_DIR:-/root/autodl-tmp/data/libero_official/libero_spatial}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$ROOT_DIR/spare_lite/local_data/libero_spatial_transition_3200.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-$ROOT_DIR/spare_lite/local_data/libero_spatial_transition_images}"
POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl}"
SPATIAL_MODEL="${SPATIAL_MODEL:-/root/autodl-tmp/checkpoints/spatialvla-4b-224-pt}"
SPATIAL_BACKEND="${SPATIAL_BACKEND:-spatialvla}"
MAX_TRANSITIONS="${MAX_TRANSITIONS:-3200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-500}"
POSITIVE_BOOST="${POSITIVE_BOOST:-50}"
RETURN_GAMMA="${RETURN_GAMMA:-0.95}"
DISCOUNT="${DISCOUNT:-0.99}"
REWARD_NORM="${REWARD_NORM:-none}"
R1_SCALE="${R1_SCALE:-1.0}"
R2_SCALE="${R2_SCALE:-1.0}"
R2_BIAS="${R2_BIAS:-0.0}"
ALIGN_MODE="${ALIGN_MODE:-raw}"
ALIGN_THRESHOLD="${ALIGN_THRESHOLD:-0.0}"
LAMBDAS="${LAMBDAS:-0.05 0.1 0.2 0.5}"
SEEDS="${SEEDS:-7 11 19}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/results/libero_spatial_iql_sweep}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

mkdir -p "$(dirname "$OUTPUT_JSONL")" "$IMAGE_DIR" "$RESULTS_ROOT"

if [[ ! -f "$OUTPUT_JSONL" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/spare_lite/export_libero_suite_to_transition_jsonl.py" \
    --dataset-dir "$LIBERO_DIR" \
    --output-jsonl "$OUTPUT_JSONL" \
    --image-dir "$IMAGE_DIR" \
    --max-transitions-total "$MAX_TRANSITIONS"
fi

for seed in $SEEDS; do
  for lambda_align in $LAMBDAS; do
    run_name="seed${seed}_lam${lambda_align}_b${BATCH_SIZE}_s${MAX_STEPS}"
    out_dir="$RESULTS_ROOT/$run_name"
    mkdir -p "$out_dir"
    "$PYTHON_BIN" -m spare_lite_offline_rl.train_iql_style_transition \
      --jsonl-path "$OUTPUT_JSONL" \
      --policy-model "$POLICY_MODEL" \
      --spatial-model "$SPATIAL_MODEL" \
      --spatial-backend "$SPATIAL_BACKEND" \
      --batch-size "$BATCH_SIZE" \
      --grad-accumulation-steps "$GRAD_ACCUM" \
      --max-steps "$MAX_STEPS" \
      --lambda-align "$lambda_align" \
      --align-mode "$ALIGN_MODE" \
      --align-threshold "$ALIGN_THRESHOLD" \
      --reward-norm "$REWARD_NORM" \
      --r1-scale "$R1_SCALE" \
      --r2-scale "$R2_SCALE" \
      --r2-bias "$R2_BIAS" \
      --discount "$DISCOUNT" \
      --return-gamma "$RETURN_GAMMA" \
      --positive-sample-boost "$POSITIVE_BOOST" \
      --seed "$seed" \
      --summary-json "$out_dir/summary.json" \
      --checkpoint-dir "$out_dir" \
      > "$out_dir/run.log" 2>&1
  done
done
