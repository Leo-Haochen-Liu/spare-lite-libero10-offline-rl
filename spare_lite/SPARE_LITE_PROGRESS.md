# SpaRe-lite Progress State

Last updated: 2026-04-25 after standalone `SpaRe-lite` migration, real SpatialVLA smoke success, and `R1-only large` relaunch

## New Standalone Status

The active workspace is now the standalone project:

```text
/Users/haochenliu/Documents/research/SpaRe-lite/spare_lite
```

with the matching remote root:

```text
/root/autodl-tmp/SpaRe-lite/spare_lite
```

This is now the main project location. We are no longer treating
`SimpleVLA-RL/spare_lite` as the active working tree.

## Current Main Runs

### `R1-only` offline RL on `large` (`3200` samples)

Remote launch path:

```text
/root/autodl-tmp/SpaRe-lite/spare_lite/run_remote_rl_r1_large.sh
```

Current result directory:

```text
/root/autodl-tmp/SpaRe-lite/results/r1_large_b16_s400_seed7
```

Current default run parameters:

- policy checkpoint: `/root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl`
- spatial checkpoint: `/root/autodl-tmp/checkpoints/facebook-dinov2-base`
- spatial backend: `auto`
- batch size: `16`
- max steps: `400`
- seed: `7`
- reward: pure `R1` (`lambda_align=0.0`)

Important correction:

- the first large run stopped at `200` steps because the previous training loop
  only consumed one pass through the dataloader
- the loop has now been fixed to recycle the dataloader until `max_steps`
  is actually reached
- the `R1-only` path also no longer wastes time running a duplicate zero-lambda
  second branch

### Real SpatialVLA reference smoke

The real SpatialVLA checkpoint is now present on the remote instance at:

```text
/root/autodl-tmp/checkpoints/spatialvla-4b-224-pt
```

The real reference smoke now passes with:

```text
[spatialvla-smoke] success
policy_latent shape: (1, 256)
spatial_latent shape: (1, 256)
align_loss: 1.074663
```

This means the remote instance can now produce `z_ref` from the real
SpatialVLA branch, not from DINOv2.

## Current Status

The active workflow is now the single-GPU `SpaRe-lite` real-data path on the
remote AutoDL instance, and it is no longer blocked at SSH or asset transfer.

Current state:

1. Tiny supervised real-data smoke has passed.
2. Medium supervised real-data smoke has passed for 10 steps.
3. Lightweight `R1` vs `R1 + lambda R2` latent validation has started and runs
   end-to-end on the remote instance.
4. A larger micro-validation with `batch_size=4` already gives a more useful
   signal than `batch_size=1`.
5. The main next task is no longer "make it run"; it is "make the validation
   more informative and stable."
6. A real remote `phase A bundle` run completed on 2026-04-25 and produced the
   full artifact set.

## Latest Remote Bundle Result

The remote phase-A bundle was executed with:

```text
bash /root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_phase_a_bundle.sh
```

Artifact directory:

```text
/root/autodl-tmp/checkpoints/spare-lite-latent-sweep
```

Generated files include:

```text
summary.json
summary.md
decision.json
result_brief.md
artifact_status.json
artifact_status.md
```

Bundle-level result:

- `phase_a_ready = false`
- `state = phase_a_incomplete`
- next phase remains `phase_a_validation_stabilization`

Summary metrics across 3 seeds:

```text
baseline avg reward = 0.1833
spare avg reward    = 0.3123
reward delta        = +0.1289
baseline expert hit = 0.1833
spare expert hit    = 0.1667
expert hit delta    = -0.0167
baseline greedy hit = 0.1833
spare greedy hit    = 0.1500
greedy hit delta    = -0.0333
```

Interpretation:

- the SpaRe branch improves average reward in this lightweight setup
- but it regresses on both sampled and greedy expert-hit
- so the current verdict is that phase A is still not ready to advance
- the `orthogonal_noise` candidate change improved overall reward level but did
  not fix the expert-hit regression

## What Passed Today

### Tiny supervised remote smoke

Command path:

```text
/root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_spare_smoke.sh
```

Observed result:

```text
[spare-lite] step=1 loss=11.7427 action_loss=11.6484 align_loss=0.9430
[spare-lite] epoch=1 completed total_steps=1
```

Conclusion:

- the policy model loads
- the frozen spatial branch loads
- the latent-alignment loss path is active
- one full real-data optimization step succeeds

### Medium supervised remote smoke

Command path:

```text
/root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_spare_medium.sh
```

Observed result:

```text
[spare-lite] step=1 loss=11.9184 action_loss=11.8157 align_loss=1.0269
...
[spare-lite] step=10 loss=14.0163 action_loss=14.0129 align_loss=0.0342
[spare-lite] epoch=1 completed total_steps=10
```

Conclusion:

- the supervised `SpaRe-lite` path works beyond a single step
- the model, dataloader, and optimizer all survive a short real-data run
- the current issue is not startup or remote environment failure

### RL-style latent smoke

Command path:

```text
python -m spare_lite.rl_spare_latent_smoke \
  --jsonl-path /root/autodl-tmp/data/libero_medium/medium_train.jsonl \
  --policy-model /root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl \
  --spatial-model /root/autodl-tmp/checkpoints/facebook-dinov2-base \
  --batch-size 1 \
  --max-steps 3 \
  --lambda-align 0.2
```

Observed result:

```text
[rl-spare-latent] lambda=0.000 step=1 loss=-0.0000 reward=0.0000 r1=0.0000 r2=0.6806 expert_reward=1.0000 negative_reward=0.0000
[rl-spare-latent] lambda=0.000 step=2 loss=-0.0000 reward=0.0000 r1=0.0000 r2=0.6665 expert_reward=1.0000 negative_reward=0.0000
[rl-spare-latent] lambda=0.000 step=3 loss=-0.0000 reward=0.0000 r1=0.0000 r2=0.6831 expert_reward=1.0000 negative_reward=0.0000
[rl-spare-latent] lambda=0.200 step=1 loss=-0.0000 reward=0.1357 r1=0.0000 r2=0.6783 expert_reward=1.2000 negative_reward=0.1350
[rl-spare-latent] lambda=0.200 step=2 loss=-0.0000 reward=0.1324 r1=0.0000 r2=0.6622 expert_reward=1.2000 negative_reward=0.1315
[rl-spare-latent] lambda=0.200 step=3 loss=0.0003 reward=0.1339 r1=0.0000 r2=0.6695 expert_reward=1.2000 negative_reward=0.1317
```

Conclusion:

- the latent-validation code path now runs end-to-end on real checkpoints
- `R2` is nontrivial and contributes measurable value when `lambda > 0`
- the current bottleneck is not loading or inference
- the current bottleneck is that the policy head is still weak, so the printed
  reward is dominated by whichever candidate is sampled
- `batch_size=1` is too degenerate for fast diagnosis

### More informative latent-validation check

We also ran the same latent smoke with:

```text
batch_size=4
max_steps=5
```

Observed result:

```text
[rl-spare-latent] lambda=0.000 step=2 loss=-0.0000 reward=0.2500 r1=0.2500 r2=0.7630 expert_prob=0.2500 expert_hit=0.2500 expert_reward=1.0000 negative_reward=0.0000
[rl-spare-latent] lambda=0.200 step=1 loss=-0.0000 reward=0.4005 r1=0.2500 r2=0.7526 expert_prob=0.2500 expert_hit=0.2500 expert_reward=1.2000 negative_reward=0.1366
[rl-spare-latent] lambda=0.200 step=2 loss=0.0763 reward=0.4059 r1=0.2500 r2=0.7793 expert_prob=0.1703 expert_hit=0.2500 expert_reward=1.2000 negative_reward=0.1400
[rl-spare-latent] lambda=0.200 step=3 loss=-0.0023 reward=0.3974 r1=0.2500 r2=0.7371 expert_prob=0.2500 expert_hit=0.2500 expert_reward=1.2000 negative_reward=0.1367
```

Conclusion:

- the validation is not dead
- a slightly larger batch already exposes a usable signal
- the current best quick-check setup is `batch_size=4`, not `batch_size=1`

## Remote Assets Confirmed

The following are confirmed present on the remote instance:

- repo:
  - `/root/autodl-tmp/SimpleVLA-RL`
- policy checkpoint:
  - `/root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl`
- spatial checkpoint:
  - `/root/autodl-tmp/checkpoints/facebook-dinov2-base`
- tiny subset:
  - `/root/autodl-tmp/data/libero_small/small_train.jsonl`
- medium subset:
  - `/root/autodl-tmp/data/libero_medium/medium_train.jsonl`

## Current Practical Blocker

The blocker is no longer infrastructure.

The blocker is now validation quality:

- the bundle completed successfully
- the result is measurable and repeatable enough to summarize
- but the current verdict is still negative because `expert_hit` regressed even
  though reward improved
- so the next work should refine the validation setup instead of claiming
  phase-A completion

Current refinement direction:

- keep the same phase-A bundle structure
- `greedy` evaluation metrics have now been added and tested
- the regression is not just sampling noise, because greedy expert-hit also drops
- the next refinement should therefore target candidate construction or reward design,
  not just evaluation noise
- the current concrete change is to switch negative candidate construction from
  plain noisy perturbations to `orthogonal_noise`, so negatives are less aligned
  with the expert spatial latent by construction
- the first `orthogonal_noise` rerun still fails phase A, so the next refinement
  likely needs reward-design changes instead of only candidate-shape changes
- the current reward-design refinement is to use `align_mode=centered_relu`
  with `align_threshold=0.72`, so moderately aligned negatives no longer get
  positive spatial reward by default

## What Changed In Code Today

- `rl_spare_latent_smoke.py`
  - now prints `expert_prob` and `expert_hit`
  - now supports `seed` and `summary_json` output for repeatable checks
- `rl_bandit.py`
  - now exposes those metrics in `BanditOutput`
- `run_remote_rl_spare_latent.sh`
  - new helper script for the remote latent-validation run, now using
    `batch_size=4`
- `run_remote_rl_spare_latent_sweep.sh`
  - new helper script for multi-seed remote latent validation
- `run_remote_phase_a_bundle.sh`
  - one-command phase-A runner that also writes artifact status files
- `summarize_rl_spare_latent_sweep.py`
  - aggregate multi-seed outputs into `summary.json` and `summary.md`
  - now also emits a simple verdict for whether phase-A validation looks ready
- `decide_next_step_from_summary.py`
  - convert the summary verdict into a direct next-phase instruction
- `generate_result_brief.py`
  - turn the summary and decision outputs into a concise result-framing markdown file
- `check_phase_artifacts.py`
  - check whether the expected phase artifacts are complete or still missing

## Next Step

1. Keep `batch_size=4` as the default quick diagnostic setup.
2. Use the generated summary and decision artifacts as the baseline reference.
3. Refine the validation setup specifically to avoid true expert-hit regression.
4. Focus the next refinement on reward design, not just candidate construction.
5. The current reward-design change is `centered_relu` spatial reward with a
   positive margin threshold.
6. Keep the current `orthogonal_noise` run as the pre-reward-redesign reference.
7. The next rerun is currently blocked from this thread because direct remote
   `ssh/scp` now returns `Operation not permitted` after DNS failure on the
   hostname path.
5. Only move to phase B when `phase_a_ready` becomes `true`.

## New local result with dataset-derived negatives

To reduce dependence on synthetic noise candidates, we added support for
dataset-derived negatives in:

- `export_hdf5_to_rl_jsonl.py`

Using:

- `/Users/haochenliu/Downloads/libero_demo_cache/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5`

we exported:

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/local_data/rl_candidate_train_datasetneg.jsonl`

and ran a local proxy smoke with:

- `PYTHONPATH=/Users/haochenliu/Documents/research/SimpleVLA-RL python3 -m spare_lite.rl_real_smoke --jsonl-path /Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/local_data/rl_candidate_train_datasetneg.jsonl --batch-size 8 --max-steps 5 --lambda-align 0.2 --device cpu`

Observed small-smoke averages:

- baseline reward: about `0.3000`
- `R1 + 0.2 * R2` reward: about `0.5129`
- baseline `R1`: about `0.3000`
- `R1 + 0.2 * R2` `R1`: about `0.3250`

Interpretation:

- this is still only a small proxy smoke
- but it is a cleaner signal than the earlier synthetic-negative-only setup
- the next major realism upgrade should now focus on the true SpatialVLA ego3D
  reference branch
- naming note:
  - `medium320` should only refer to the fixed 320-sample subset
  - the current larger mainline should be referred to as `large`, not `medium320`
  - the default mainline latent validation should now use `large`

## Method Consistency

The lightweight validation still matches the intended reward structure:

```text
R = R1 + lambda * R2
R2 = cosine(z_pol, z_ref)
```

The current effort is now focused on making this validation measurable and
defensible, not merely executable.
