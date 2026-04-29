from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_alignment(policy_latent: torch.Tensor, spatial_latent: torch.Tensor) -> torch.Tensor:
    """Return per-example cosine similarity in [-1, 1]."""
    policy_latent = F.normalize(policy_latent, dim=-1)
    spatial_latent = F.normalize(spatial_latent, dim=-1)
    return (policy_latent * spatial_latent).sum(dim=-1)


def latent_alignment_loss(policy_latent: torch.Tensor, spatial_latent: torch.Tensor) -> torch.Tensor:
    """
    Convert cosine similarity into a minimization objective.

    Lower is better, with 0 meaning perfect alignment.
    """
    return 1.0 - cosine_alignment(policy_latent, spatial_latent).mean()
