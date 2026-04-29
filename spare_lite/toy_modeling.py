from __future__ import annotations

import torch
from torch import nn

from .reward import latent_alignment_loss


class ToyPolicyModel(nn.Module):
    """
    Small stand-in model for local smoke tests.
    """

    def __init__(self, vocab_size: int = 128, hidden_size: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **_: dict,
    ):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        loss = None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        out = type("ToyOutput", (), {})()
        out.loss = loss
        out.hidden_states = [hidden] if output_hidden_states else None
        return out


class ToySpatialEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, hidden_size)

    def forward(self, pixel_values: torch.Tensor):
        feats = self.net(pixel_values).flatten(1)
        hidden = self.proj(feats).unsqueeze(1)
        out = type("ToySpatialOutput", (), {})()
        out.last_hidden_state = hidden
        return out


class ToySpaReLiteModel(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 64, latent_dim: int = 32) -> None:
        super().__init__()
        self.policy = ToyPolicyModel(vocab_size=vocab_size, hidden_size=hidden_size)
        self.spatial = ToySpatialEncoder(hidden_size=hidden_size)
        self.policy_latent_head = nn.Linear(hidden_size, latent_dim)
        self.spatial_projection = nn.Linear(hidden_size, latent_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.policy(
            input_ids=batch["policy_inputs"]["input_ids"],
            attention_mask=batch["policy_inputs"].get("attention_mask"),
            labels=batch["policy_inputs"]["labels"],
            output_hidden_states=True,
        )
        base_loss = outputs.loss
        policy_hidden = outputs.hidden_states[-1][:, -1, :]
        policy_latent = self.policy_latent_head(policy_hidden)

        spatial_outputs = self.spatial(batch["spatial_pixel_values"])
        spatial_hidden = spatial_outputs.last_hidden_state[:, 0, :]
        spatial_latent = self.spatial_projection(spatial_hidden)

        align_loss = latent_alignment_loss(policy_latent, spatial_latent)
        loss = base_loss + 0.1 * align_loss
        return {
            "loss": loss,
            "action_loss": base_loss.detach(),
            "align_loss": align_loss.detach(),
        }
