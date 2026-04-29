# SpaRe-lite LIBERO-10 Offline RL Reproduction

This repository contains the lightweight reproduction pipeline, dataset manifest,
training/evaluation logs, and result figures for a constrained offline-RL study on
SimpleVLA-RL + LIBERO-10.

## Claim

Under a constrained setting with no online RL interaction and only a small set of
trainable policy components, R1/R2 still provide a measurable positive learning
signal. The goal is not to match the full author-provided online RL checkpoint,
but to show that the offline R1/R2 reward design can improve over a supervised
baseline under limited compute and time.

## Main Results

LIBERO-10 success rates:

| Model / setting | Success | Rate |
|---|---:|---:|
| Official SFT baseline | 80 / 496 | 16.13% |
| Official post-RL baseline | 430 / 496 | 86.69% |
| Offline R1 from official SFT + full409 data | 86 / 500 | 17.20% |
| Offline R1+R2 from official SFT + full409 data | 92 / 500 | 18.40% |
| Earlier R1 checkpoint evaluated on LIBERO-10 | 355 / 448 | 79.24% |
| Earlier R1+R2 checkpoint evaluated on LIBERO-10 | 364 / 448 | 81.25% |

The most important controlled comparison is:

`Official SFT < R1 from SFT < R1+R2 from SFT`

The earlier high-performing pair was trained on a different data source and is
included as an empirical reference rather than the strict LIBERO-10 full409
comparison.

## Repository Layout

- `spare_lite/`: data adapters, reward helpers, and dataset extraction utilities.
- `spare_lite_offline_rl/`: IQL-style offline RL training code.
- `scripts/`: SFT failure rollout collection script.
- `tools/`: checkpoint export/download/launcher utilities.
- `data/`: lightweight final LIBERO-10 JSONL dataset and manifest.
- `results/`: final summaries and eval logs for the SFT-baseline full409 run.
- `artifacts/`: per-task plots and parsed rollout-count JSON.
- `patches/`: minimal SimpleVLA-RL runtime patch used for LIBERO multiprocessing.

## Dataset

The final dataset used for the strict SFT-baseline comparison is:

`data/libero10_expert_plus_sft_failures_409.jsonl`

It contains:

- 138,090 expert transitions from full LIBERO-10 demonstrations.
- 26,176 real failed-policy transitions from 409 official-SFT failed rollout episodes.
- 164,266 total transitions.
- `quality=expert/failure`, `reward`, and `rollout_success`/failure metadata so the
  trainer can avoid treating failure actions as behavior-cloning positives.

Important: the JSONL references image paths from the remote extraction layout.
For a fresh machine, regenerate or remap images using the extraction scripts in
`spare_lite/` and the collection script in `scripts/`. The JSONL itself is tracked
with Git LFS because it is larger than GitHub's normal 100 MB file limit.

## Official Checkpoints

Download the author-provided checkpoints separately:

- `Haozhan72/Openvla-oft-SFT-libero10-traj1`
- `Haozhan72/openvla-oft-libero10-traj1-rl`

The local paths used in our run were:

- `/root/autodl-tmp/official_eval_sandboxes/sft_libero10_full`
- `/root/autodl-tmp/official_eval_sandboxes/postrl_libero10_full`

The post-RL sandbox used hard links to the official checkpoint plus the missing
custom-code files (`train_utils.py`, `constants.py`) needed by
`trust_remote_code=True`.

## Training Configuration

The strict SFT-baseline experiment used:

- Base policy: official SFT checkpoint.
- Dataset: `libero10_expert_plus_sft_failures_409.jsonl`.
- Batch size: 16.
- Gradient accumulation: 2.
- Steps: 500.
- Learning rate: `5e-6`.
- IQL expectile: `0.7`.
- CQL alpha: `0.1`.
- R1-only: `lambda_align=0.0`.
- R1+R2: `lambda_align=0.2`, `r2_bias=0.10`.
- Trainable policy modules:
  - `language_model.model.layers.30.`
  - `language_model.model.layers.31.`
  - `language_model.norm`
  - `language_model.lm_head`
  - `multi_modal_projector`

Example:

```bash
export ROBOT_PLATFORM=LIBERO
export PYTHONPATH=/path/to/SpaRe-lite:/path/to/LIBERO:$PYTHONPATH

python -m spare_lite_offline_rl.train_iql_style_transition \
  --jsonl-path data/libero10_expert_plus_sft_failures_409.jsonl \
  --policy-model /path/to/Openvla-oft-SFT-libero10-traj1 \
  --spatial-model /path/to/spatialvla-4b-224-pt \
  --spatial-backend spatialvla \
  --batch-size 16 \
  --max-steps 500 \
  --grad-accumulation-steps 2 \
  --lambda-align 0.2 \
  --r2-bias 0.10 \
  --policy-trainable-pattern language_model.model.layers.30. \
  --policy-trainable-pattern language_model.model.layers.31. \
  --policy-trainable-pattern language_model.norm \
  --policy-trainable-pattern language_model.lm_head \
  --policy-trainable-pattern multi_modal_projector \
  --checkpoint-dir outputs/r1r2
```

## Evaluation

Evaluation uses the official SimpleVLA-RL LIBERO rollout pipeline. Before running
LIBERO eval on our AutoDL instance, we applied the small runtime patch in:

`patches/simplevla_rl_libero_spawn_runtime.patch`

It changes LIBERO env worker multiprocessing from default fork behavior to a
spawn context to avoid CUDA initialization failures inside Ray workers. It does
not change model logic, action decoding, reward computation, or success criteria.

Generated figures:

- `artifacts/libero10_sft_baseline_full409_offline_rl_per_task.png`
- `artifacts/libero10_postrl_baseline_offline_rl_per_task.png`

Parsed counts:

- `artifacts/libero10_final_comparison_counts.json`

## Large Artifacts

Full exported VLA checkpoints are about 29 GB each and are intentionally not
committed to normal Git. Use Hugging Face Hub, GitHub Releases, or another object
store for those files.

Relevant produced checkpoint directories in our remote run:

- `libero10_full409_r1` (~29 GB)
- `libero10_full409_r1r2` (~29 GB)

The partial training checkpoints are about 2.1 GB each.

## Notes

This repo is meant to preserve the reproducible pipeline and the final
experiment evidence. It is intentionally lightweight enough for GitHub; large
checkpoint binaries and extracted image folders should be distributed separately.
