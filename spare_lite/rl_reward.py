from __future__ import annotations

import torch

from .reward import cosine_alignment


def transform_align_reward(
    align_reward: torch.Tensor,
    align_mode: str,
    align_threshold: float,
) -> torch.Tensor:
    if align_mode == "raw":
        return align_reward
    if align_mode == "centered_relu":
        return torch.relu(align_reward - align_threshold)
    raise ValueError(f"Unsupported align_mode: {align_mode}")


def combine_rewards(
    env_reward: torch.Tensor,
    policy_latent: torch.Tensor,
    spatial_latent: torch.Tensor,
    lambda_align: float,
    align_mode: str = "raw",
    align_threshold: float = 0.0,
) -> dict[str, torch.Tensor]:
    """
    Compute:
      R = R1 + lambda * R2
      R2 = cosine(policy_latent, spatial_latent)
    """
    raw_align_reward = cosine_alignment(policy_latent, spatial_latent)
    align_reward = transform_align_reward(
        align_reward=raw_align_reward,
        align_mode=align_mode,
        align_threshold=align_threshold,
    )
    total_reward = env_reward + lambda_align * align_reward
    return {
        "r1_env": env_reward,
        "r2_align": align_reward,
        "r2_align_raw": raw_align_reward,
        "reward": total_reward,
    }
