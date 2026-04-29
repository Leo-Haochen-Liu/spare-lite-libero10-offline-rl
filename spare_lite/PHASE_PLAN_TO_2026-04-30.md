# SpaRe-lite Phase Plan To 2026-04-30

Last updated: 2026-04-25

## Goal

Finish a defensible `SpaRe-lite` validation package by April 30, 2026:

- the code path runs
- the short validation is repeatable enough to discuss
- the next step is mechanically clear from the generated artifacts

## Phase A: Validation Stabilization

Target:

- get a non-degenerate `R1` vs `R1 + lambda R2` comparison
- make the result readable without hand-parsing logs

Done already:

- tiny supervised smoke passed
- medium supervised smoke passed
- latent smoke passed
- `batch_size=4` identified as a better quick-check setup
- multi-seed sweep script added
- summary script added
- verdict logic added
- next-step decision script added

Exit condition:

- `summary.json` contains `verdict.phase_a_ready = true`

Artifacts:

- `run_remote_phase_a_bundle.sh`
- `run_remote_rl_spare_latent_sweep.sh`
- `summary.json`
- `summary.md`
- `decision.json`
- `result_brief.md`
- `check_phase_artifacts.py`

## Parallel Track B: Reproducibility And Result Framing

Target:

- keep the phase-A setup fixed
- gather one cleaner reproducibility batch
- write the concise explanation of what the current result means and what it does not mean

Entry condition:

- phase A does not need to be fully finished before this track starts
- this track can proceed once there is a result package that is worth framing,
  even if the current proxy setup is still imperfect

Exit condition:

- one stable result package exists and its framing is documented

## Parallel Track C: Realistic Validation Upgrade

Target:

- reduce proxy-specific distortions in the current validation path
- move toward real LIBERO-derived negative candidates instead of only synthetic noise
- prepare the path to a true SpatialVLA ego3D reference encoder

Current direction:

- the current proxy path is still useful for fast reward-signal checks
- but it is not sufficient to claim that `R2` helps real expert-like behavior
- so the realistic-upgrade track can proceed in parallel rather than waiting for
  phase A to become perfect

## Phase D: Final Delivery Packaging

Target:

- freeze the runnable commands
- freeze the current result framing
- leave the repo in a handoff-ready state for the April 30 checkpoint

Entry condition:

- the current state of track A, track B, and track C is documented clearly

Exit condition:

- progress docs, framework map, and commands are aligned with the latest state

## Current Status

We are still in **Phase A**.

The remote multi-seed bundle has now finished and the generated verdict is:

- `phase_a_ready = false`

Current interpretation:

- reward improved over baseline in the sweep
- but both sampled expert-hit and greedy expert-hit regressed
- so phase A is still incomplete and the next work should refine validation
  quality rather than move to phase B yet
- the greedy-metric refinement is now complete
- the regression appears real enough that the next refinement should target
  candidate construction or reward design rather than evaluation noise alone
- the current candidate-construction refinement is to use `orthogonal_noise`
  negatives in the lightweight phase-A sweep
- the first `orthogonal_noise` rerun improved reward level but still failed the
  greedy expert-hit gate, so the next refinement should lean toward reward
  design rather than only changing candidate geometry
- the current reward-design refinement is a thresholded spatial reward:
  `align_mode=centered_relu`, `align_threshold=0.72`
- that reward-design refinement has not been rerun yet from this environment,
  because the current thread cannot reach the remote machine via direct
  `ssh/scp` (`Operation not permitted` on IP fallback after DNS failure on the
  hostname)
