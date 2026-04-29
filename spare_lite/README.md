# SpaRe-lite

`SpaRe-lite` is a lightweight concept-validation path for the SpaRe idea:

- keep a pretrained VLA policy as the acting model
- attach a small policy-side latent head
- use a frozen spatial encoder to produce a reference latent
- train with the original supervised action loss plus a latent-alignment term

This path is intentionally simpler than full online RL. It is designed for the
case where:

- the full `SimpleVLA-RL` stack is unstable on the current hardware
- we still want a concrete implementation that matches the paper's core claim
- we need something we can finish and iterate on within a few days

## Current status

As of 2026-04-24:

- tiny supervised remote smoke has passed
- medium supervised remote smoke has passed
- remote latent-validation smoke has also passed end-to-end

So the active question is no longer whether the path runs at all. The active
question is whether the lightweight validation is strong enough to produce a
convincing `R1` vs `R1 + lambda * R2` comparison.

## Core objective

For each sample we compute:

- policy hidden state: `h`
- policy latent: `z_policy = p_theta(h)`
- frozen spatial latent: `z_spatial = f_phi(image)`

We then optimize:

```text
total_loss = action_loss + lambda_align * (1 - cosine(z_policy, z_spatial))
```

This is not a full RL reward-shaping loop. Instead, it is a reward-augmented
lightweight training path that preserves the same structural idea:

- the action policy still acts
- the spatial branch does not act
- the extra signal comes from latent agreement

## Recommended first host model

Use `openvla/openvla-7b` first.

Reasons:

- it is closer to the current OpenVLA / OpenVLA-OFT story
- the official code already supports lightweight fine-tuning paths
- we avoid rebuilding a new RL stack around a different VLA family

## Files

- `config.py`: dataclass config for SpaRe-lite experiments
- `reward.py`: latent similarity and alignment loss
- `modeling.py`: wrapper that combines a policy model, frozen spatial encoder,
  and policy latent head
- `adapters.py`: JSONL + image dataset adapter for real LIBERO-style examples
- `openvla_oft_utils.py`: OpenVLA-OFT prompt and action-token helpers
- `train_spare_lite.py`: minimal training entry point
- `run_spare_lite_supervised.py`: runnable supervised training CLI
- `toy_modeling.py`: tiny local stand-in modules for smoke tests
- `smoke_train.py`: local smoke training that validates the full loss path
- `export_hdf5_to_spare_jsonl.py`: HDF5-to-JSONL exporter for LIBERO-style demos
- `export_hdf5_to_rl_jsonl.py`: HDF5-to-JSONL exporter for lightweight `R1` vs `R1+lambda R2` candidate tests
- `rl_data.py`: JSONL format for lightweight RL-style validation
- `rl_reward.py`: `R = R1 + lambda * R2` reward combiner
- `rl_bandit.py`: minimal REINFORCE-style candidate policy head
- `rl_smoke.py`: local smoke run for RL-style validation
- `rl_real_smoke.py`: real-demo candidate-action smoke comparing `lambda=0` and `lambda>0`
- `rl_spare_latent_smoke.py`: VLA/DINO latent-space smoke comparing `lambda=0` and `lambda>0`
- `run_remote_spare_smoke.sh`: remote launcher for the tiny supervised smoke
- `run_remote_spare_medium.sh`: remote launcher for the medium supervised smoke
- `run_remote_rl_spare_latent.sh`: remote launcher for the latent-validation smoke
- `run_remote_rl_spare_latent_sweep.sh`: remote multi-seed launcher that saves summary JSON outputs
- `run_remote_phase_a_bundle.sh`: one-command phase-A bundle that also writes artifact status
- `summarize_rl_spare_latent_sweep.py`: post-process per-seed outputs into one concise summary
- `decide_next_step_from_summary.py`: map the summary verdict into the next phase instruction
- `generate_result_brief.py`: build a concise markdown brief from the summary and decision files
- `check_phase_artifacts.py`: report whether the expected phase artifacts are present and what is still missing

## Expected usage

At minimum, each training batch should provide:

- policy inputs for the VLA model
- labels for the original action/imitation loss
- an image tensor for the frozen spatial encoder

The trainer assumes the base policy model returns a standard Hugging Face style
`loss` field when labels are present.

### JSONL adapter format

The real-data adapter expects one JSON object per line:

```json
{
  "image_path": "/abs/path/to/image.png",
  "prompt": "In: What action should the robot take to ...?\\nOut:",
  "target_text": "<target action text or token text>",
  "spatial_image_path": "/abs/path/to/optional/spatial_image.png"
}
```

If `spatial_image_path` is omitted, the same image is reused for the frozen
spatial branch.

For real OpenVLA-OFT-style supervision, we now also support:

```json
{
  "image_path": "/abs/path/to/image.png",
  "instruction": "pick up the cup",
  "action": [0.1, -0.2, 0.0, 0.3, 0.1, -0.4, 1.0],
  "prompt_style": "pure"
}
```

In this mode the adapter will:

- build the official-style query `What action should the robot take to ...?`
- tokenize the continuous action with the OpenVLA-OFT action discretizer
- build `prompt + action_tokens` as one sequence
- mask the prompt prefix so the loss is taken only on action tokens

## Immediate next step

1. Extend the latent-validation run with:

```bash
bash /root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent.sh
```

2. Use the current `batch_size=4` setup as the default quick diagnostic path.
3. Inspect `expert_prob` and `expert_hit`, not only the sampled reward.
4. If the signal remains too weak, stabilize this short validation before
   scaling toward heavier online RL.

For a slightly more reliable check, run the multi-seed sweep helper after the
single-run smoke is healthy:

```bash
bash /root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent_sweep.sh
```

This now also produces:

- `summary.json`
- `summary.md`
- `decision.json`
- `result_brief.md`

in the same output directory, so the sweep result is immediately readable.
The summary now also includes a simple `phase_a_ready` verdict for the current
validation stage.

To quickly inspect whether the artifact set is complete, run:

```bash
python -m spare_lite.check_phase_artifacts \
  --output-dir /root/autodl-tmp/checkpoints/spare-lite-latent-sweep
```

If you want the full phase-A artifact chain in one command, run:

```bash
bash /root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_phase_a_bundle.sh
```

The provided exporter can be used once demo HDF5 files are available:

```bash
python -m spare_lite.export_hdf5_to_spare_jsonl \
  --hdf5-path /path/to/demos.hdf5 \
  --output-jsonl /path/to/train.jsonl \
  --image-dir /path/to/exported_images \
  --instruction "pick up the cup"
```

For remote runs on the current AutoDL host, the policy model can also be a
local checkpoint path such as:

```text
/root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl
```

Example supervised run:

```bash
python -m spare_lite.run_spare_lite_supervised \
  --jsonl-path /root/autodl-tmp/data/libero_small/train.jsonl \
  --policy-model /root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl \
  --spatial-model facebook/dinov2-base \
  --batch-size 1 \
  --max-steps 100 \
  --save-dir /root/autodl-tmp/checkpoints/spare-lite-supervised
```

## RL-style validation path

For the short-horizon project version, we also support a lightweight RL-style
validation path:

- treat each observation as a contextual decision point
- provide a small set of candidate actions per state
- use `R1` as the environment / success reward attached to each candidate
- compute `R2` from latent alignment
- optimize the policy over candidates with a REINFORCE-style objective

This is not a full simulator rollout stack, but it does validate the central
equation:

```text
R = R1 + lambda * R2
```

The expected JSONL format is:

```json
{
  "image_path": "/abs/path/to/image.png",
  "prompt": "task instruction",
  "candidates": ["candidate action 1", "candidate action 2", "candidate action 3"],
  "env_rewards": [1.0, 0.0, 0.0],
  "spatial_image_path": "/abs/path/to/optional/spatial_image.png"
}
```

For the VLA/DINO-aligned version, use `rl_spare_latent_smoke.py`. It loads the
same policy and spatial encoder as the supervised SpaRe-lite path, extracts
`z_policy` and `z_spatial`, then compares:

- baseline: `R = R1`
- SpaRe: `R = R1 + lambda * R2`

Example:

```bash
python -m spare_lite.rl_spare_latent_smoke \
  --jsonl-path /root/autodl-tmp/data/libero_medium/medium_train.jsonl \
  --policy-model /root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl \
  --spatial-model /root/autodl-tmp/checkpoints/facebook-dinov2-base \
  --batch-size 1 \
  --max-steps 3 \
  --lambda-align 0.2
```
