# EECS 545 Final Report Plan

## Requirement interpretation

Course guidance says the final report is a maximum 8-page report, excluding references and appendix, written in technical-report style. It should clearly state the problem, relevant background, approach, and evaluation. The project criteria emphasize significance, technical quality, novelty, and presentation. For this project, the safest framing is not a polished conference-style narrative, but a reproducible technical report that explains what was implemented, how it was evaluated, what improved, what failed, and how another team could reproduce the pipeline.

## Recommended title

SpaRe-lite: Resource-Constrained Offline Reinforcement Learning for Vision-Language-Action Policies on LIBERO-10

## Core claim

Under a constrained setting with no online RL interaction and only a small subset of trainable policy parameters, SpaRe-lite R1/R2 rewards provide a measurable positive learning signal over the official SFT baseline. The goal is not to beat the full author-provided online RL checkpoint, but to show that offline reward shaping can produce reproducible gains in a limited-compute VLA setting.

## Main technical contributions to emphasize

1. Reproducible SimpleVLA-RL + LIBERO-10 evaluation setup using the official SFT and post-RL checkpoints.
2. Construction of a LIBERO-10 offline RL dataset combining full expert transitions with 409 real failed SFT rollout episodes.
3. IQL-style offline RL training with limited trainable OpenVLA-OFT policy components.
4. Controlled comparison: official SFT vs R1-from-SFT vs R1+R2-from-SFT.
5. Engineering analysis of failure modes: checkpoint custom-code issues, norm-stat keys, Ray/LIBERO multiprocessing initialization, and resource limits.

## Proposed 8-page structure

### 1. Introduction

- Problem: online RL for VLA policies is expensive and fragile; can offline reward learning provide useful improvement under compute constraints?
- Setting: LIBERO-10 manipulation benchmark, OpenVLA-OFT / SimpleVLA-RL pipeline.
- Claim: R1/R2 improves over SFT in a controlled offline setting, but does not match full online RL.

### 2. Background and related work

- VLA policies and OpenVLA-OFT.
- LIBERO benchmark and task-level success rate.
- SimpleVLA-RL official SFT and post-RL checkpoints.
- Offline RL / IQL intuition.
- SpaRe / spatial reward idea at a high level.

### 3. Method

- Dataset construction:
  - full LIBERO-10 expert transitions;
  - 409 real failed SFT rollout episodes;
  - metadata fields: `quality`, `reward`, `rollout_success`.
- Rewards:
  - R1: task/success-style reward signal;
  - R2: spatial/alignment auxiliary reward;
  - why failure actions are negative/low-quality contrast, not behavior-cloning positives.
- Offline RL objective:
  - IQL-style actor/value/Q losses;
  - CQL regularization;
  - limited trainable policy modules.
- Pipeline diagram: `figures/spare_lite_pipeline.jpg`.

### 4. Experimental setup

- Hardware and constraints: single-GPU eval; two-GPU parallel training/eval where available; memory constraints.
- Official checkpoints:
  - `Haozhan72/Openvla-oft-SFT-libero10-traj1`;
  - `Haozhan72/openvla-oft-libero10-traj1-rl`.
- Dataset size: 138,090 expert transitions + 26,176 failure transitions = 164,266 transitions.
- Training hyperparameters:
  - 500 steps, batch size 16, gradient accumulation 2;
  - lr `5e-6`, expectile `0.7`, CQL alpha `0.1`;
  - R1 lambda `0.0`, R1+R2 lambda `0.2`, R2 bias `0.10`.
- Trainable modules:
  - layers 30/31, final norm, LM head, multimodal projector.

### 5. Results

Primary controlled comparison:

| Model | Success | Rate |
|---|---:|---:|
| Official SFT | 80/496 | 16.13% |
| R1 from SFT + full409 | 86/500 | 17.20% |
| R1+R2 from SFT + full409 | 92/500 | 18.40% |
| Official post-RL | 430/496 | 86.69% |

Figures:

- `figures/libero10_sft_baseline_full409_offline_rl_per_task.png`
- `figures/libero10_postrl_baseline_offline_rl_per_task.png`

Interpretation:

- R1 improves over SFT, and R1+R2 improves over R1.
- The gain is modest but directionally consistent under the constrained setup.
- Full online RL remains much stronger, as expected.

### 6. Failure analysis and engineering lessons

This section is important for technical-report style.

- Official checkpoint reproduction required exact environment alignment.
- Some model repos lacked expected custom-code files, so sandbox copies were used without altering model weights.
- LIBERO spatial checkpoint did not generalize directly across benchmark suites; VLA checkpoints can be highly suite-specific.
- Ray/LIBERO multiprocessing needed spawn-based worker initialization to avoid CUDA fork errors.
- Eval can fail due to environment worker init timeout, not model quality; this was handled as a runtime issue.

### 7. Reproducibility

- GitHub repo structure:
  - `data/`: final JSONL and manifest;
  - `spare_lite/`: extraction and reward helpers;
  - `spare_lite_offline_rl/`: training code;
  - `tools/`: export/plot/download utilities;
  - `figures/`, `analysis/`, `results/`, `patches/`.
- Required external downloads:
  - official SFT checkpoint;
  - official post-RL checkpoint;
  - SimpleVLA-RL;
  - LIBERO;
  - SpatialVLA backend checkpoint.
- Mention large checkpoints are distributed separately because each full exported VLA model is about 29-31 GB.

### 8. Conclusion

- Offline R1/R2 can produce measurable improvements over SFT in a limited-compute VLA setting.
- R1+R2 > R1 supports the value of the auxiliary spatial reward signal.
- The work is a reproducible technical study rather than an SOTA claim.

## Appendix candidates

- Exact commands for training and eval.
- Full hyperparameter table.
- Per-task success counts.
- Dataset schema.
- Contribution table.
- Runtime patch explanation.

## What to avoid

- Do not frame this as beating SimpleVLA-RL or online RL.
- Do not overclaim the earlier high-scoring spatial-offline checkpoints as the main controlled result.
- Do not hide failures; the course criteria explicitly value thoughtful negative results and clear analysis.
- Do not make the report too poster-like. The report should read as implementation + method + evaluation + reproducibility.
