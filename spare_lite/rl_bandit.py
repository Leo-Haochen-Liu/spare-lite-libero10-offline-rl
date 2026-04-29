from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from .rl_reward import combine_rewards


@dataclass
class BanditOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    reward: torch.Tensor
    r1_env: torch.Tensor
    r2_align: torch.Tensor
    chosen_index: torch.Tensor
    expert_prob: torch.Tensor
    chosen_is_expert: torch.Tensor
    greedy_index: torch.Tensor
    greedy_reward: torch.Tensor
    greedy_r1_env: torch.Tensor
    greedy_r2_align: torch.Tensor
    greedy_is_expert: torch.Tensor


class CandidateBanditHead(nn.Module):
    """
    Minimal policy head for lightweight RL validation.

    Inputs:
    - context embedding: one vector per sample
    - candidate embeddings: K vectors per sample

    Outputs:
    - a categorical policy over candidates
    - REINFORCE loss using reward = R1 + lambda * R2
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.candidate_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        context: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        env_rewards: torch.Tensor,
        spatial_latent: torch.Tensor,
        lambda_align: float,
        align_mode: str = "raw",
        align_threshold: float = 0.0,
        sample: bool = True,
    ) -> BanditOutput:
        """
        Shapes:
        - context: [B, D]
        - candidate_embeddings: [B, K, D]
        - env_rewards: [B, K]
        - spatial_latent: [B, D]
        """
        q_context = self.context_proj(context).unsqueeze(1)  # [B, 1, D]
        q_candidates = self.candidate_proj(candidate_embeddings)  # [B, K, D]
        logits = (q_context * q_candidates).sum(dim=-1)  # [B, K]

        dist = Categorical(logits=logits)
        probs = dist.probs
        chosen_index = dist.sample() if sample else logits.argmax(dim=-1)

        batch_idx = torch.arange(logits.size(0), device=logits.device)
        chosen_candidate = candidate_embeddings[batch_idx, chosen_index]
        chosen_env_reward = env_rewards[batch_idx, chosen_index]
        greedy_index = logits.argmax(dim=-1)
        greedy_candidate = candidate_embeddings[batch_idx, greedy_index]
        greedy_env_reward = env_rewards[batch_idx, greedy_index]

        rewards = combine_rewards(
            env_reward=chosen_env_reward,
            policy_latent=chosen_candidate,
            spatial_latent=spatial_latent,
            lambda_align=lambda_align,
            align_mode=align_mode,
            align_threshold=align_threshold,
        )
        greedy_rewards = combine_rewards(
            env_reward=greedy_env_reward,
            policy_latent=greedy_candidate,
            spatial_latent=spatial_latent,
            lambda_align=lambda_align,
            align_mode=align_mode,
            align_threshold=align_threshold,
        )
        log_prob = dist.log_prob(chosen_index)
        if rewards["reward"].numel() > 1:
            baseline = rewards["reward"].mean().detach()
        else:
            baseline = torch.zeros_like(rewards["reward"])
        advantage = rewards["reward"] - baseline
        policy_loss = -(log_prob * advantage).mean()

        return BanditOutput(
            loss=policy_loss,
            policy_loss=policy_loss.detach(),
            reward=rewards["reward"].detach(),
            r1_env=rewards["r1_env"].detach(),
            r2_align=rewards["r2_align"].detach(),
            chosen_index=chosen_index.detach(),
            expert_prob=probs[:, 0].detach(),
            chosen_is_expert=(chosen_index == 0).to(dtype=torch.float32).detach(),
            greedy_index=greedy_index.detach(),
            greedy_reward=greedy_rewards["reward"].detach(),
            greedy_r1_env=greedy_rewards["r1_env"].detach(),
            greedy_r2_align=greedy_rewards["r2_align"].detach(),
            greedy_is_expert=(greedy_index == 0).to(dtype=torch.float32).detach(),
        )
