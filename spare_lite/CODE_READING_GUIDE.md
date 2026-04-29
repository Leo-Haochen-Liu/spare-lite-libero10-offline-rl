# SpaRe-lite Code Reading Guide

Last updated: 2026-04-25

## Best Reading Order

### Core logic

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/rl_spare_latent_smoke.py`
  - Main entry for the current phase-A lightweight validation.
  - `make_candidates(...)` is here.
  - The current candidate set is constructed at runtime here, not stored as a separate file.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/rl_bandit.py`
  - Defines the lightweight bandit policy over candidates.
  - `expert_hit` and `greedy_expert_hit` are defined here.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/rl_reward.py`
  - Implements `R = R1 + lambda * R2`.
  - The current `centered_relu` reward design is here.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/modeling.py`
  - Defines how `policy_latent` and `spatial_latent` are produced.
  - This is the key file for understanding the current `z_pol` and `z_ref`.

### Data and inputs

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/adapters.py`
  - Reads JSONL data, builds batches, and prepares inputs for the policy and spatial encoder.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/local_data/medium_train.jsonl`
  - Current fixed medium local LIBERO subset.
  - Contains 80 `x` examples.

- `/Users/haochenliu/Downloads/libero_demo_cache/small_train.jsonl`
  - Current fixed small LIBERO subset.
  - Contains 16 `x` examples.

### Remote running and aggregation

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent_sweep.sh`
  - Runs the remote multi-seed latent-validation sweep.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_remote_phase_a_bundle.sh`
  - Runs the whole phase-A bundle.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/summarize_rl_spare_latent_sweep.py`
  - Aggregates multi-seed outputs into `summary.json` and `summary.md`.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/decide_next_step_from_summary.py`
  - Turns the summary verdict into the next phase instruction.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/generate_result_brief.py`
  - Generates a concise result brief.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/check_phase_artifacts.py`
  - Verifies that all expected phase-A artifacts exist.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/transfer_remote_assets.sh`
  - Syncs code and data to the remote instance.

### Progress and orientation docs

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/SPARE_LITE_PROGRESS.md`
  - Rolling project progress log.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/PHASE_PLAN_TO_2026-04-30.md`
  - Current phase plan to the April 30 checkpoint.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/SESSION_PROGRESS_2026-04-25.md`
  - What changed in the current session.

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/PROJECT_FRAMEWORK_AND_FILE_MAP.md`
  - Overall project structure and file roles.

## Where The Current Candidate Proxy Experiment Is

The current proxy experiment is centered on:

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/rl_spare_latent_smoke.py`

Important details:

- The candidate set is not loaded from a separate candidate file.
- It is generated online by `make_candidates(...)`.
- For each input `x`, the current setup constructs:
  - 1 expert candidate
  - 3 negative candidates
- So each `x` currently has 4 candidates total.

## What The Current Fixed LIBERO Subsets Are

### Local subsets

- Small subset:
  - `/Users/haochenliu/Downloads/libero_demo_cache/small_train.jsonl`
  - 16 examples

- Medium subset:
  - `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/local_data/medium_train.jsonl`
  - 80 examples

### Remote subsets

- Small subset on remote:
  - `/root/autodl-tmp/data/libero_small/small_train.jsonl`

- Medium subset on remote:
  - `/root/autodl-tmp/data/libero_medium/medium_train.jsonl`

## Important Clarification About The Current Proxy

The current phase-A validation is still a proxy experiment.

- `z_ref` is currently derived from the spatial branch used in `modeling.py`, and the remote validation has been using a DINOv2 checkpoint rather than the final SpatialVLA ego3D encoder.
- `z_pol` exists structurally in `modeling.py` through the learnable policy latent head.
- But in the current lightweight bandit validation, the reward-side candidate chosen for `R2` is still based on the constructed candidate embeddings, not yet the final fully wired policy-side spatial latent used in the intended full implementation.
