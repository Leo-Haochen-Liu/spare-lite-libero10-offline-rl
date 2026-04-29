# SpaRe-lite Report Summary

Last updated: 2026-04-25

## 1. Current Goal

The current goal is to check whether our offline RL pipeline can run stably on top of an author-released high baseline model, and then compare two reward settings under the same initialization and data setup:

- `R1` only
- `R1 + R2`

Here, the policy checkpoint we have been using is:

- `/root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl`

This checkpoint should be understood as the author-released post-RL model rather than the original SFT baseline.

## 2. What Has Already Been Verified

We first wrote a very basic offline RL-style proxy training path and verified that the RL loop can run end to end on the remote instance with a real OpenVLA-OFT checkpoint and a real LIBERO subset.

The earliest stable quick check used:

- data: `/root/autodl-tmp/data/libero_medium/medium_train.jsonl`
- policy model: `/root/autodl-tmp/checkpoints/openvla-oft-libero10-traj1-rl`
- spatial branch at that time: `/root/autodl-tmp/checkpoints/facebook-dinov2-base`
- batch size: `4`
- max steps: `5`

The important result is that the reward did increase in this offline RL proxy setup. The recorded numbers were:

- baseline: `reward = 0.2500`
- `R1 + 0.2 * R2`: `reward ≈ 0.4005 ~ 0.4059`

This means the offline RL loop was not dead. It ran, produced updates, and the printed reward trend moved upward.

## 3. Why This Does Not Yet Prove The Final New Method

Although the RL loop runs and reward rises, the current setup is still only a proxy validation rather than the final SpaRe implementation.

The main reasons are:

1. The current negative candidates were initially synthetic.

Only the expert candidate was pulled from real LIBERO data. The negative candidates were constructed by adding noise around the expert candidate. Later we also added an `orthogonal_noise` variant, but this still belongs to synthetic candidate construction.

2. The current `R2` path is not yet using the real SpatialVLA reference branch.

Before the SpatialVLA checkpoint is fully available on the remote instance, the reference branch used in practice has been DINOv2 rather than SpatialVLA.

3. The current validation is still a candidate-selection proxy.

This setup is useful for checking whether the RL machinery runs and whether reward shaping changes behavior, but it is not yet the final fully wired version described in the poster.

## 4. What `checkpoint` Means Here

In this project, a `checkpoint` is not just a single number or a single tensor. It usually refers to the saved model state needed to load and run a model, including:

- model parameter shards such as `.safetensors`
- configuration files
- tokenizer or processor files when needed
- model-specific Python files when the checkpoint uses custom code

So when we say that the SpatialVLA checkpoint is not fully ready yet, we mean that the remote instance does not yet have the full runnable SpatialVLA model package in place.

## 5. What Exists On The Remote Instance Right Now

For the SpatialVLA side, the remote instance currently has a partial download under:

- `/root/autodl-tmp/checkpoints/spatialvla-4b-224-pt`

At this moment, that directory is not yet the full completed SpatialVLA checkpoint. It is still being downloaded. So the correct statement is:

- the remote instance already has part of the SpatialVLA model files
- but it does not yet have the fully completed SpatialVLA checkpoint
- therefore the real SpatialVLA reference branch cannot yet be used as the final running branch

## 6. Current Meaning Of `R2`

The current reward still has the form:

- `R = R1 + lambda * R2`

But the current `R2` should not yet be described as the final poster version.

Right now, the important caveats are:

- the policy branch is running on a proxy candidate-selection setup
- the expert candidate is real, but many negatives were initially synthetic
- the reference branch has been using DINOv2 in practice before the SpatialVLA checkpoint is fully ready

So the current `R2` is still a proxy version of the intended spatial reward.

## 7. Clean Summary For External Explanation

If we explain the current state to someone who does not know the project well, the clean version is:

We have already verified that a basic offline RL-style proxy training loop can run on top of the author-released `openvla-oft-libero10-traj1-rl` checkpoint, and in the early LIBERO medium-subset test the reward increased from about `0.25` to about `0.40` after adding `R2`. This shows that the RL loop is executable and that the reward signal can move upward.

However, this does not yet prove the final SpaRe method, because the early negative candidates were mostly synthetic rather than dataset-native, and the current `R2` path has not yet been fully switched to a completed SpatialVLA checkpoint on the remote instance. In other words, the current result shows that the offline RL pipeline runs and that reward shaping changes behavior, but it is still a proxy validation rather than the final full method.

## 8. Files That Support This Summary

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/SPARE_LITE_NOTES.md`
- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/SPARE_LITE_PROGRESS.md`
- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/SESSION_PROGRESS_2026-04-24.md`
- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/rl_spare_latent_smoke.py`
- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/rl_bandit.py`
- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/modeling.py`
