# SpaRe-lite Next Task Checklist

Last updated: 2026-04-25

## Immediate task list

1. Export a new RL candidate JSONL using dataset-derived negatives from LIBERO demos.
2. Run a proxy validation on the new dataset-negative JSONL and record the result.
3. Start the real SpatialVLA ego3D encoder integration track with a concrete implementation checklist.

## Current execution status

- Task 1: completed
- Task 2: completed
- Task 3: in progress

## Notes

- The goal of task 1 is to reduce distortion from synthetic noise negatives.
- The goal of task 2 is not to prove final method success, but to check whether a more realistic candidate set changes the validation signal.
- The goal of task 3 is to stop treating the current DINOv2 reference branch as if it were the final SpatialVLA-based implementation.

## Latest concrete result

- A new dataset-negative JSONL was exported to:
  - `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/local_data/rl_candidate_train_datasetneg.jsonl`
- A local proxy smoke was run with:
  - `PYTHONPATH=/Users/haochenliu/Documents/research/SimpleVLA-RL python3 -m spare_lite.rl_real_smoke --jsonl-path /Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/local_data/rl_candidate_train_datasetneg.jsonl --batch-size 8 --max-steps 5 --lambda-align 0.2 --device cpu`
- In this small smoke:
  - baseline average reward was approximately `0.3000`
  - `R1 + 0.2 * R2` average reward was approximately `0.5129`
  - baseline average `R1` was approximately `0.3000`
  - `R1 + 0.2 * R2` average `R1` was approximately `0.3250`
- This does not prove the final method, but it is a cleaner sign than the earlier synthetic-negative-only setup.

## Larger-budget local run

- A larger dataset-negative export was created at:
  - `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/local_data/rl_candidate_train_datasetneg_large.jsonl`
- It contains:
  - `6277` records
- A longer local run entry now exists at:
  - `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_local_rl_candidate_datasetneg_long.sh`
- The mainline local latent run should now use:
  - `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_local_rl_spare_latent_large.sh`
- The mainline remote latent sweep should now use:
  - `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent_sweep_large.sh`
- The older fixed 320-sample subset is now only a fallback subset rather than the default mainline.
- The remote sync helper now supports both the fallback 320-sample subset and the larger 3200-sample mainline subset.
