# Standard Offline RL Track

## Why this separate folder exists

This folder is for the poster-consistent offline RL implementation track.

The current `spare_lite` directory is being used to produce a checkpoint quickly under a simplified transition-based setup.
That setup is useful for fast iteration, but it is still not a standard offline RL implementation such as IQL or CQL.

To avoid mixing:

- quick-result code
- poster-consistent offline RL code

we keep the standard offline RL track here.

## Current assessment

The larger problem right now is the reward/design signal, not just the number of unfrozen policy parameters.

Reasons:

1. The previous transition export accidentally removed terminal success steps, which made all rewards zero.
2. Even after fixing that, the current reward is still sparse and weak.
3. The current `R2` path only becomes useful if the alignment signal is actually activated and correlated with progress.
4. Unfreezing more layers may help later, but if reward is uninformative, increasing trainable parameters will not fix the core problem.

So the order should be:

1. get the reward/signal definition right
2. then widen trainable policy scope if needed

## Required target

The final code that we submit should be as close as possible to the poster:

- use trajectory/transition data from LIBERO
- use real `SpatialVLA` reference latent
- use policy-side latent from `OpenVLA` hidden state through a learnable head
- compare `R1` vs `R1 + R2`
- train real policy parameters and the learnable head

## Standard offline RL options to prototype here

Candidate methods:

- IQL
- CQL

IQL is likely the better first target because it is simpler to adapt than CQL.

## Implementation plan

1. Reuse the transition dataset exporter and keep the same JSONL contract.
2. Build a clean transition dataset loader for:
   - `obs`
   - `action`
   - `reward`
   - `next_obs`
   - `done`
   - `instruction`
3. Keep `z_ref` from the real `SpatialVLA` reference branch.
4. Keep `z_pol` from the `OpenVLA` hidden state and `policy_latent_head`.
5. Add a proper offline RL training objective:
   - actor/value with expectile regression
   - twin Q critics
   - EMA target value network
   - simple CQL-style conservative penalty
   - incorporate `R2` into the reward definition
6. Save:
   - baseline checkpoint
   - aligned checkpoint
   - training logs
   - evaluation summaries

## Immediate decision

Keep the currently running simplified transition training in `spare_lite` to get a checkpoint and quick result.

In parallel, build the poster-consistent offline RL implementation in this folder, and make sure the final submission points to this version rather than the simplified one.

## Progress update

Already added:

- transition loader support for `next_obs`
- query-only state encoding for policy-side state latent
- value head
- twin Q heads
- target value network updated by soft update
- bootstrap target using `reward + discount * next_value * (1 - done)`
- simple conservative Q penalty

Still missing or not yet validated:

- remote end-to-end run for this new track
- memory/performance tuning
- stronger action-conditioned negative sampling for the conservative term
- final comparison plots and simulator evaluation
