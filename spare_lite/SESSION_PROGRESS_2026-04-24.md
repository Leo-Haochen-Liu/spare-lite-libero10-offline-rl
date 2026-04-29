# SpaRe-lite Session Progress

Date: 2026-04-24

## Objective

Push the `spare_lite` branch from "prepared pipeline" to "actually running on
the remote instance," then record the real blocker after execution.

## What Was Confirmed

### Remote access

- DNS for `connect.westb.seetacloud.com` now works.
- SSH login to the remote AutoDL instance works.
- GPU is visible on the remote host.

### Remote assets

Confirmed present:

- `/root/autodl-tmp/SimpleVLA-RL`
- `/root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl`
- `/root/autodl-tmp/checkpoints/facebook-dinov2-base`
- `/root/autodl-tmp/data/libero_small/small_train.jsonl`
- `/root/autodl-tmp/data/libero_medium/medium_train.jsonl`

### Supervised path

Tiny smoke passed:

```text
[spare-lite] step=1 loss=11.7427 action_loss=11.6484 align_loss=0.9430
```

Medium smoke passed:

```text
[spare-lite] epoch=1 completed total_steps=10
```

### Latent-validation path

The `rl_spare_latent_smoke` run also completed on the remote instance.

This means:

- the real checkpoints load
- the VLA branch and DINO branch both run
- `R2` is computed on real latents
- the `R1` vs `R1 + lambda R2` code path is executable

An additional check with `batch_size=4` was more informative than `batch_size=1`:

- baseline branch reached `reward=0.2500`, `expert_hit=0.2500`
- `lambda=0.2` branch reached `reward≈0.40` with the same `expert_hit=0.2500`

So the short validation signal exists; the earlier weak result was partly a
small-batch issue.

## Current Bottleneck

The blocker has shifted from infrastructure to experiment quality.

Right now the issue is:

- the validation path runs
- but the candidate policy head is still too weak for short runs to be very
  informative
- so the next round should focus on better metrics and a more stable short test,
  not on fixing loading or SSH

## Code Changes Made

- added:
  - `run_remote_rl_spare_latent.sh`
- updated:
  - `rl_bandit.py`
  - `rl_spare_latent_smoke.py`
  - `SPARE_LITE_PROGRESS.md`
  - `PROJECT_FRAMEWORK_AND_FILE_MAP.md`
- added later:
  - `run_remote_rl_spare_latent_sweep.sh`
  - `summarize_rl_spare_latent_sweep.py`
  - `decide_next_step_from_summary.py`

## Why These Changes Matter

- The new run script removes friction for repeated remote validation.
- The new logging exposes `expert_prob` and `expert_hit`, which are more useful
  than raw sampled reward alone.
- The updated docs now reflect the real state of the project, not the older SSH
  blocker.

## Recommended Immediate Next Action

Run:

```bash
bash /root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent.sh
```

This helper now uses `batch_size=4`, which is currently the best quick
diagnostic setup we have.

For lower-noise comparison, the next follow-up command should be:

```bash
bash /root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent_sweep.sh
```

That sweep now also writes one aggregated markdown/json summary, so the result
can be reviewed without manually comparing each seed file.
It also writes a direct next-step decision file for phase advancement.
