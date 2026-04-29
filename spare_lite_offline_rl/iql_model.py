from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from spare_lite.config import SpaReLiteConfig
from spare_lite.modeling import SpaReLiteModel, _get_output_value


class SpaReLiteIQLModel(nn.Module):
    """
    Thin wrapper around the current SpaRe-lite model that adds a scalar value head.

    This is the separate offline RL track. It reuses:
    - real OpenVLA forward pass
    - real SpatialVLA reference branch
    - policy-side latent head from SpaReLiteModel
    """

    def __init__(self, config: SpaReLiteConfig) -> None:
        super().__init__()
        self.base = SpaReLiteModel(config)
        self.value_head = nn.Sequential(
            nn.Linear(config.spatial.output_dim, config.spatial.output_dim),
            nn.GELU(),
            nn.Linear(config.spatial.output_dim, 1),
        )
        q_input_dim = config.spatial.output_dim * 2
        self.q1_head = nn.Sequential(
            nn.Linear(q_input_dim, config.spatial.output_dim),
            nn.GELU(),
            nn.Linear(config.spatial.output_dim, 1),
        )
        self.q2_head = nn.Sequential(
            nn.Linear(q_input_dim, config.spatial.output_dim),
            nn.GELU(),
            nn.Linear(config.spatial.output_dim, 1),
        )

    @property
    def trainable_policy_parameter_names(self) -> list[str]:
        return self.base.trainable_policy_parameter_names

    def encode_policy_state(self, policy_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        query_kwargs = dict(policy_inputs)
        query_kwargs.pop("labels", None)
        query_kwargs["output_hidden_states"] = True
        query_kwargs["return_dict"] = True
        policy_outputs = self.base.policy(**query_kwargs)
        policy_pooled = self.base._pool_policy_hidden(policy_outputs)
        if policy_pooled is None:
            logits = policy_outputs if torch.is_tensor(policy_outputs) else _get_output_value(
                policy_outputs,
                "logits",
                tuple_index=0,
            )
            if logits is None:
                raise ValueError("Policy query forward did not expose hidden states or logits.")
            return self.base.policy_logits_head(logits.mean(dim=1))
        return self.base.policy_latent_head(policy_pooled)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.base(batch)
        state_latent = self.encode_policy_state(batch["policy_query_inputs"])
        value = self.value_head(state_latent).squeeze(-1)
        q_inputs = torch.cat([state_latent, outputs["policy_latent"]], dim=-1)
        q1 = self.q1_head(q_inputs).squeeze(-1)
        q2 = self.q2_head(q_inputs).squeeze(-1)
        outputs["value"] = value
        outputs["state_latent"] = state_latent
        outputs["q1"] = q1
        outputs["q2"] = q2
        if "next_policy_query_inputs" in batch:
            with torch.no_grad():
                next_state_latent = self.encode_policy_state(batch["next_policy_query_inputs"])
                outputs["next_state_latent"] = next_state_latent
        return outputs

    def extra_repr(self) -> str:
        return f"base={asdict(self.base.config)}"
