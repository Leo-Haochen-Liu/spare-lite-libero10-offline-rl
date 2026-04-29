# SpaRe-lite Project Framework and File Map

Last updated: 2026-04-24

## Goal

`SpaRe-lite` is the fast validation path for the SpaRe idea when the full
online `SimpleVLA-RL` stack is too unstable or too expensive to iterate on.

The purpose is not to replace the full RL project. The purpose is to validate
the same central idea in a lighter path:

- keep a pretrained VLA policy as the acting model
- keep a frozen spatial encoder as the reference branch
- construct a policy-side latent and a reference spatial latent
- add a spatial alignment signal on top of the original objective

## Current workflow

The active plan is:

1. Start from an author-provided OpenVLA-OFT checkpoint.
2. Run a tiny real-data supervised smoke on a small LIBERO subset.
3. Run a medium real-data supervised smoke on a larger subset.
4. If the real-data path is stable, move to a lightweight `R1` vs `R1 + lambda R2`
   validation path.
5. Only after this lighter path is validated, decide whether to spend more
   time on the heavier online RL route.

Current reality on 2026-04-24:

- tiny supervised smoke: passed
- medium supervised smoke: passed
- latent validation smoke: passed
- `batch_size=4` gives a better latent-validation signal than `batch_size=1`
- next task: improve the quality of the validation signal

## Project structure

### Core modeling and training

- `config.py`
  - Central dataclass config for policy model, spatial encoder, and optimizer.
- `reward.py`
  - Cosine-alignment helpers and the latent alignment loss.
- `modeling.py`
  - Main `SpaReLiteModel` wrapper.
  - Combines the VLA policy, frozen spatial encoder, policy latent head, and
    spatial projection.
- `train_spare_lite.py`
  - Generic training loop for the supervised SpaRe-lite path.
- `run_spare_lite_supervised.py`
  - CLI entry point for real supervised runs.

### Data and tokenization

- `adapters.py`
  - JSONL dataset and collator for real LIBERO-style examples.
  - Supports both legacy text-target mode and OpenVLA-style
    `instruction + continuous action` mode.
- `openvla_oft_utils.py`
  - Prompt formatting and OpenVLA-OFT-style action tokenization helpers.
- `export_hdf5_to_spare_jsonl.py`
  - Export LIBERO-style HDF5 demos into supervised JSONL.
- `export_hdf5_to_rl_jsonl.py`
  - Export HDF5 demos into candidate-action JSONL for lightweight RL-style
    validation.

### Lightweight RL-style validation

- `rl_data.py`
  - Dataset format for candidate-action RL-style JSONL data.
- `rl_reward.py`
  - Reward combiner implementing `R = R1 + lambda * R2`.
- `rl_bandit.py`
  - Minimal categorical policy head for candidate-action validation.
- `rl_smoke.py`
  - Fully local toy RL smoke test.
- `rl_real_smoke.py`
  - Real-data candidate-action smoke using simple action vectors.
- `rl_spare_latent_smoke.py`
  - Main lightweight validation script that compares `lambda=0` and
    `lambda>0` using VLA/DINO latents.

### Local toy smoke tests

- `toy_modeling.py`
  - Tiny stand-in policy and spatial encoder for local development.
- `smoke_train.py`
  - Small local smoke test for the supervised training path.

### Remote workflow helpers

- `transfer_remote_assets.sh`
  - Push code, spatial encoder, and LIBERO subsets to the remote instance.
- `run_remote_spare_smoke.sh`
  - Run the tiny real-data supervised smoke remotely.
- `run_remote_spare_medium.sh`
  - Run the medium real-data supervised smoke remotely after the tiny one passes.
- `run_remote_rl_spare_latent.sh`
  - Run the lightweight `R1` vs `R1 + lambda R2` latent validation remotely.
- `run_remote_rl_spare_latent_sweep.sh`
  - Run the latent validation for several seeds and save summary JSON files.
- `run_remote_phase_a_bundle.sh`
  - One-command phase-A runner: sweep, summarize, decide next step, generate result brief, and write artifact status.
- `summarize_rl_spare_latent_sweep.py`
  - Aggregate the per-seed JSON outputs into one JSON summary and one markdown report.
  - Also computes a simple verdict for whether the current validation stage is ready to advance.
- `decide_next_step_from_summary.py`
  - Turn the summary verdict into the next phase instruction for automation or manual use.
- `generate_result_brief.py`
  - Turn `summary.json` and `decision.json` into a concise markdown brief for phase-B style result framing.
- `check_phase_artifacts.py`
  - Check whether the expected phase artifacts exist and report what is still missing.
- `continue_remote_workflow.sh`
  - Convenience wrapper: transfer assets, then launch the tiny remote smoke.

### Progress and reference docs

- `README.md`
  - High-level overview of SpaRe-lite and how it relates to the main project.
- `SPARE_LITE_PROGRESS.md`
  - Current state log for blockers, ready assets, and next concrete step.
- `PROJECT_FRAMEWORK_AND_FILE_MAP.md`
  - This document. Fast orientation for the project structure and file roles.
- `PHASE_PLAN_TO_2026-04-30.md`
  - Short dated phase plan for what is done, what remains, and how a phase is considered complete.

## Local data assets

Current useful local subsets under `spare_lite/local_data` include:

- `medium_train.jsonl`
  - medium real-data supervised subset
- `medium_images/`
  - images used by `medium_train.jsonl`
- `medium320_train_remote.jsonl`
  - larger remote-oriented subset
- `medium320_train_localpaths.jsonl`
  - local-path version of the same subset
- `rl_candidate_train.jsonl`
  - candidate-action JSONL for lightweight RL-style validation
- `large_train_localpaths.jsonl`
  - larger local-path JSONL
- `large_train_remote.jsonl`
  - remote-path JSONL

## How the files connect

### Supervised SpaRe-lite path

1. `adapters.py`
   - loads JSONL examples and builds model-ready batches
2. `openvla_oft_utils.py`
   - turns actions into OpenVLA-style token supervision
3. `modeling.py`
   - computes policy loss and latent alignment loss
4. `reward.py`
   - defines the latent alignment objective
5. `train_spare_lite.py`
   - runs optimization
6. `run_spare_lite_supervised.py`
   - launches the full supervised run

### Lightweight RL-style path

1. `rl_data.py`
   - loads observation + candidate-action data
2. `rl_bandit.py`
   - selects among candidate actions
3. `rl_reward.py`
   - combines `R1` and `R2`
4. `rl_spare_latent_smoke.py`
   - compares baseline vs SpaRe-style reward on real latents

## Immediate practical next steps

1. Use `run_remote_rl_spare_latent.sh` to extend the latent-validation run.
2. Check `expert_prob` and `expert_hit`, not just the sampled reward.
3. If the signal is still too weak, adjust the validation setup before scaling.
4. Keep the full online RL route as a later step, not the current blocker.

## Why this path matters

This path gives us something we can finish quickly:

- smaller and more controllable than full online RL
- still consistent with the core SpaRe method
- easier to debug under current hardware and time constraints
- easier to explain as a controlled validation story

## File-writing rule for this project

Whenever a new file is added under `spare_lite`, it should satisfy one of these
roles explicitly:

- implementation:
  - a runnable model, reward, dataloader, or trainer component
- launcher:
  - a local or remote script that starts a concrete experiment
- exporter:
  - a script that converts raw demos into JSONL/image assets
- progress log:
  - a dated or rolling state file that records what ran and what failed
- orientation doc:
  - a project map or usage explanation for faster continuation

If a new file does not clearly fit one of those roles, it should probably not be
added yet.
