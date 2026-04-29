# SpaRe-lite Session Progress

Date: 2026-04-25

## New Standalone Follow-up

Later on 2026-04-25, the project moved into a separate standalone workspace:

```text
/Users/haochenliu/Documents/research/SpaRe-lite/spare_lite
```

with remote code under:

```text
/root/autodl-tmp/SpaRe-lite/spare_lite
```

### What passed

1. The real SpatialVLA checkpoint was fully downloaded to the remote instance.
2. The real SpatialVLA reference smoke was debugged through multiple
   compatibility issues and now passes end to end.
3. A standalone `R1-only` large run was launched from the new project root.

### Key technical fixes

1. Added a local `gemma2` configuration patch path for the SpatialVLA checkpoint.
2. Added an isolated remote dependency overlay using `transformers==4.47.0`
   without replacing the main training environment.
3. Fixed `SiglipImageProcessor` usage in `adapters.py` so the SpatialVLA path
   calls the processor directly instead of using a nonexistent
   `.image_processor(...)` attribute.
4. Fixed the RL training loop so `max_steps` can exceed one dataloader pass.
5. Fixed the pure `R1` path so it does not redundantly run a second zero-lambda branch.

### Current active run

The new standalone `R1-only` run is:

```text
/root/autodl-tmp/SpaRe-lite/results/r1_large_b16_s400_seed7
```

with:

- dataset: `large` (`3200` samples)
- batch size: `16`
- max steps: `400`
- seed: `7`

## Objective

Turn the heartbeat from passive status polling into active remote execution for
phase A.

## What Changed

### Heartbeat logic

The automation prompt was updated so it no longer just repeats status. It now
prioritizes:

- checking whether a remote phase-A process is already running
- starting the remote phase-A bundle if the artifacts are missing or stale
- monitoring logs and artifacts once the run is active

### Transfer path

`transfer_remote_assets.sh` was made more efficient:

- skip re-copying the DINO checkpoint if it already exists remotely
- skip re-copying tiny/medium JSONL and image assets if they already exist
- sync `spare_lite` code as a tar stream without `local_data` and cache files

This matters because the previous heartbeat could get stuck re-copying large
local directories instead of making real forward progress.

### Remote execution

A real remote phase-A bundle run was started:

```text
bash /root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_phase_a_bundle.sh
```

Log path:

```text
/root/autodl-tmp/checkpoints/spare-lite-latent-sweep/phase_a_bundle.log
```

Initial observed log:

```text
[spare-lite-sweep] running seed=7
Loading checkpoint shards...
```

## Final Outcome Of This Session

The remote bundle finished and produced the full artifact chain:

- `summary.json`
- `decision.json`
- `result_brief.md`
- `artifact_status.json`

Key result from the first bundle:

- `phase_a_ready = false`

Why:

- average reward improved from `0.1500` to `0.2770`
- but average expert-hit dropped from `0.1500` to `0.1333`

So this session did succeed in converting phase A from a hypothetical pipeline
into a real executed bundle with concrete artifacts, but it did not yet clear
the phase-A exit condition.

## Follow-up Refinement

A second bundle run was executed after adding greedy metrics.

New interpretation:

- reward still improves
- sampled expert-hit still regresses slightly
- greedy expert-hit also regresses

So the current blocker is now more specific:

- this does not look like sampling noise alone
- the next refinement should target candidate construction or reward design

## Next Immediate Action

1. Keep phase A active.
2. Use the current artifact set as the new baseline reference.
3. Refine the validation setup to avoid true expert-hit regression.
4. Re-run the phase-A bundle after that refinement.

## New Blocker After Reward-Design Refinement

The next reward-design refinement is now ready locally:

- `align_mode=centered_relu`
- `align_threshold=0.72`

However, the rerun did not start from this environment because direct remote
access is currently blocked locally:

- DNS for `connect.westb.seetacloud.com` still fails
- IP fallback `116.172.66.188:20984` is resolvable
- but direct `ssh/scp` from this thread now returns `Operation not permitted`

So the current state is:

- the reward-design refinement is implemented locally
- the remote artifacts are now stale relative to local code
- phase A is still blocked on restoring a working remote execution path
