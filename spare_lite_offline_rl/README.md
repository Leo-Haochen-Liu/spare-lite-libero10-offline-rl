# SpaRe-lite Offline RL Track

This directory is the separate development track for a more poster-consistent offline RL implementation.

It is intentionally separated from:

- `/Users/haochenliu/Documents/research/SpaRe-lite/spare_lite`

because the current `spare_lite` directory is being used to produce working checkpoints quickly under a simplified transition-based setup.

## Scope

This track is for:

- transition / trajectory-style LIBERO data
- real `SpatialVLA` reference latents
- real `OpenVLA` policy-side latents
- training the actual policy checkpoint plus the learnable latent head
- a more RL-like objective than plain reward-weighted fine-tuning

## Current implementation status

The main code path here is now an **IQL/CQL-style scaffold**, but it is still not a fully validated paper-grade offline RL implementation.

The current design here:

- reuses transition data exported from LIBERO
- reuses the `OpenVLA` and `SpatialVLA` forward paths
- adds:
  - a value head
  - twin Q heads
  - a target value network updated by EMA
- uses:
  - expectile value regression
  - advantage-weighted actor loss
  - a simple CQL-style conservative penalty
- keeps `R2` from `z_pol` vs `z_ref`
- uses:
  - current-step reward
  - next-state bootstrap target
  - `done` masking

That means:

- it is much closer to standard offline RL than the quick transition fine-tuning line
- but it still needs end-to-end validation and likely more refinement before we can call it final

## Files

- `iql_model.py`
  - wraps the current VLA + spatial latent extraction and adds a value head plus twin Q heads

- `iql_losses.py`
  - contains expectile, advantage-weighting, target soft-update, and conservative penalty helpers

- `train_iql_style_transition.py`
  - main training entry for the separate offline RL track

## Near-term plan

1. Validate that the current actor/value/twin-Q scaffold runs end to end on the exported transition dataset.
2. Compare `R1 only` vs `R1 + R2`.
3. Save checkpoints for both branches.
4. Tune memory usage and target construction if the full branch is unstable.
5. Refine toward a final submission version after the quick-result line is secured.
