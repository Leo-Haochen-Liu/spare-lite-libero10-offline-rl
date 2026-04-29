from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForVision2Seq, SiglipImageProcessor

from .config import SpaReLiteConfig
from .reward import latent_alignment_loss


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_matching_parameters(module: nn.Module, patterns: tuple[str, ...]) -> None:
    if not patterns:
        return
    for name, param in module.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = True


def _collect_matching_parameter_names(module: nn.Module, patterns: tuple[str, ...]) -> list[str]:
    matched = []
    if not patterns:
        return matched
    for name, _param in module.named_parameters():
        if any(pattern in name for pattern in patterns):
            matched.append(name)
    return matched


def _load_policy_config_with_compat(model_name: str, trust_remote_code: bool) -> Any:
    policy_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    # Some OpenVLA-OFT checkpoints predate optional proprio fields expected by
    # the dynamic modeling file. The policy we use here is image+language only.
    if not hasattr(policy_config, "use_proprio"):
        policy_config.use_proprio = False
    if not hasattr(policy_config, "proprio_dim"):
        policy_config.proprio_dim = 0
    return policy_config


def _get_output_value(outputs: Any, key: str, tuple_index: int | None = None) -> Any:
    if isinstance(outputs, dict):
        return outputs.get(key)
    value = getattr(outputs, key, None)
    if value is not None:
        return value
    if tuple_index is not None and isinstance(outputs, (tuple, list)) and len(outputs) > tuple_index:
        return outputs[tuple_index]
    return None


def _resolve_spatial_backend(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend
    lowered = model_name.lower()
    if "spatialvla" in lowered:
        return "spatialvla"
    return "hf_image"


class SpaReLiteModel(nn.Module):
    """
    Minimal wrapper for concept validation.

    Assumptions:
    - the policy model can be loaded through `AutoModelForVision2Seq`
    - when labels are passed, the policy model returns a `.loss`
    - the policy model can return hidden states when `output_hidden_states=True`
    - the spatial model can be loaded as an image encoder via `AutoModel`
    """

    def __init__(self, config: SpaReLiteConfig) -> None:
        super().__init__()
        self.config = config

        policy_config = _load_policy_config_with_compat(
            config.policy.model_name,
            config.policy.trust_remote_code,
        )
        self.policy = AutoModelForVision2Seq.from_pretrained(
            config.policy.model_name,
            trust_remote_code=config.policy.trust_remote_code,
            config=policy_config,
        )
        self.trainable_policy_parameter_names: list[str] = []
        if not config.policy.use_lora:
            _freeze_module(self.policy)
        elif config.policy.target_modules:
            _freeze_module(self.policy)
            _unfreeze_matching_parameters(self.policy, config.policy.target_modules)
            self.trainable_policy_parameter_names = _collect_matching_parameter_names(
                self.policy,
                config.policy.target_modules,
            )
            if not self.trainable_policy_parameter_names:
                raise ValueError(
                    "No policy parameters matched the requested trainable patterns: "
                    f"{config.policy.target_modules}"
                )
        self.spatial_backend = _resolve_spatial_backend(
            config.spatial.model_name,
            config.spatial.backend,
        )
        if self.spatial_backend == "spatialvla":
            self.spatial_processor = SiglipImageProcessor.from_pretrained(
                config.spatial.model_name,
                trust_remote_code=config.spatial.trust_remote_code,
            )
            self.spatial_encoder = AutoModel.from_pretrained(
                config.spatial.model_name,
                trust_remote_code=config.spatial.trust_remote_code,
            )
        else:
            self.spatial_processor = AutoImageProcessor.from_pretrained(config.spatial.model_name)
            self.spatial_encoder = AutoModel.from_pretrained(
                config.spatial.model_name,
                trust_remote_code=config.spatial.trust_remote_code,
            )

        if not config.spatial.trainable:
            _freeze_module(self.spatial_encoder)

        policy_hidden_size = getattr(self.policy.config, "hidden_size", None)
        if policy_hidden_size is None:
            policy_text_config = getattr(self.policy.config, "text_config", None)
            policy_hidden_size = getattr(policy_text_config, "hidden_size", None)
        if policy_hidden_size is None:
            raise ValueError("Could not infer policy hidden size from the model config.")

        if self.spatial_backend == "spatialvla":
            spatial_hidden_size = getattr(self.spatial_encoder.config.vision_config, "projection_dim", None)
        else:
            spatial_hidden_size = getattr(self.spatial_encoder.config, "hidden_size", None)
        if spatial_hidden_size is None:
            raise ValueError("Could not infer spatial encoder hidden size from the model config.")

        self.policy_latent_head = nn.Sequential(
            nn.Linear(policy_hidden_size, policy_hidden_size),
            nn.GELU(),
            nn.Linear(policy_hidden_size, config.spatial.output_dim),
        )
        policy_vocab_size = getattr(policy_text_config, "vocab_size", None)
        if policy_vocab_size is None:
            policy_vocab_size = getattr(self.policy.config, "vocab_size", None)
        if policy_vocab_size is None:
            raise ValueError("Could not infer policy vocab size from the model config.")
        self.policy_logits_head = nn.Linear(policy_vocab_size, config.spatial.output_dim)
        self.spatial_projection = nn.Linear(spatial_hidden_size, config.spatial.output_dim)

    def _pool_policy_hidden(self, outputs: Any) -> torch.Tensor | None:
        hidden_states = _get_output_value(outputs, "hidden_states")
        if not hidden_states:
            return None
        last_hidden = hidden_states[-1]
        return last_hidden[:, -1, :]

    def _pool_spatial_hidden(self, outputs: Any) -> torch.Tensor:
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise ValueError("Spatial encoder did not return `last_hidden_state`.")
        return last_hidden[:, 0, :]

    def _encode_spatial(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.spatial_backend == "spatialvla":
            intrinsic = batch.get("spatial_intrinsic")
            if intrinsic is None:
                raise ValueError("SpatialVLA backend requires `spatial_intrinsic` in the batch.")
            image_features = self.spatial_encoder.get_image_features(
                pixel_values=batch["spatial_pixel_values"],
                intrinsic=intrinsic,
            )
            return image_features.mean(dim=1)

        spatial_outputs = self.spatial_encoder(pixel_values=batch["spatial_pixel_values"])
        return self._pool_spatial_hidden(spatial_outputs)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        policy_kwargs = dict(batch["policy_inputs"])
        labels = policy_kwargs.get("labels")
        policy_kwargs["output_hidden_states"] = True
        policy_kwargs["return_dict"] = True
        policy_outputs = self.policy(**policy_kwargs)

        base_loss = _get_output_value(policy_outputs, "loss")
        logits = policy_outputs if torch.is_tensor(policy_outputs) else _get_output_value(
            policy_outputs,
            "logits",
            tuple_index=0,
        )
        if base_loss is None:
            if logits is None or labels is None:
                raise ValueError("Policy model did not return `.loss` or usable logits/labels.")
            target_rows = []
            for row in labels:
                target = row[row != -100]
                if target.numel() < logits.size(1):
                    pad = torch.full(
                        (logits.size(1) - target.numel(),),
                        -100,
                        dtype=target.dtype,
                        device=target.device,
                    )
                    target = torch.cat([target, pad], dim=0)
                else:
                    target = target[: logits.size(1)]
                target_rows.append(target)
            action_labels = torch.stack(target_rows, dim=0)
            base_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                action_labels.reshape(-1),
                ignore_index=-100,
            )
            labels = action_labels
        elif labels is None:
            raise ValueError("Policy labels are required for SpaRe-lite training.")

        if logits is None:
            raise ValueError("Policy model did not expose logits for per-sample action loss.")

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.size())
        valid_mask = (shift_labels != -100).float()
        token_counts = valid_mask.sum(dim=1).clamp_min(1.0)
        per_sample_action_loss = (token_losses * valid_mask).sum(dim=1) / token_counts

        policy_pooled = self._pool_policy_hidden(policy_outputs)
        if policy_pooled is not None:
            policy_latent = self.policy_latent_head(policy_pooled)
        elif logits is not None:
            logits_pooled = logits.mean(dim=1)
            policy_latent = self.policy_logits_head(logits_pooled)
        else:
            raise ValueError("Policy model did not expose hidden states or logits for alignment.")

        spatial_pooled = self._encode_spatial(batch)
        spatial_latent = self.spatial_projection(spatial_pooled)

        align_loss = latent_alignment_loss(policy_latent, spatial_latent)
        total_loss = base_loss + self.config.optim.lambda_align * align_loss

        return {
            "loss": total_loss,
            "action_loss": base_loss,
            "align_loss": align_loss,
            "per_sample_action_loss": per_sample_action_loss,
            "policy_latent": policy_latent,
            "spatial_latent": spatial_latent,
        }

    def extra_repr(self) -> str:
        return f"config={asdict(self.config)}"
