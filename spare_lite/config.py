from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpatialEncoderConfig:
    model_name: str = "facebook/dinov2-base"
    backend: str = "auto"
    trust_remote_code: bool = True
    trainable: bool = False
    output_dim: int = 256


@dataclass
class PolicyConfig:
    model_name: str = "openvla/openvla-7b"
    trust_remote_code: bool = True
    use_lora: bool = False
    lora_rank: int = 32
    target_modules: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class OptimizationConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 1
    grad_accumulation_steps: int = 1
    max_steps: Optional[int] = None
    lambda_align: float = 0.1
    log_every: int = 10
    device: str = "cuda"


@dataclass
class SpaReLiteConfig:
    run_name: str = "spare-lite-openvla"
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    spatial: SpatialEncoderConfig = field(default_factory=SpatialEncoderConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
