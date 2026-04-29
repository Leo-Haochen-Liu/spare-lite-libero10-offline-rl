from __future__ import annotations

import torch


def expectile_loss(
    value_pred: torch.Tensor,
    target: torch.Tensor,
    expectile: float = 0.7,
) -> torch.Tensor:
    diff = target - value_pred
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def clipped_advantage_weights(
    advantage: torch.Tensor,
    temperature: float = 3.0,
    max_weight: float = 20.0,
) -> torch.Tensor:
    weights = torch.exp(advantage * temperature)
    return weights.clamp(max=max_weight)


def soft_update_(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.lerp_(source_param.data, tau)


def cql_conservative_penalty(
    q_values: torch.Tensor,
    negative_q_values: torch.Tensor,
) -> torch.Tensor:
    combined = torch.cat([q_values.unsqueeze(-1), negative_q_values], dim=-1)
    return (torch.logsumexp(combined, dim=-1) - q_values).mean()
