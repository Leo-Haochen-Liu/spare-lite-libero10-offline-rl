from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import AdamW

from .config import SpaReLiteConfig
from .modeling import SpaReLiteModel


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            out[key] = {
                inner_key: inner_value.to(device) if hasattr(inner_value, "to") else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def train_epoch(
    model: SpaReLiteModel,
    dataloader: Iterable[dict],
    optimizer: AdamW,
    device: torch.device,
    grad_accumulation_steps: int,
    log_every: int,
    max_steps: int | None = None,
) -> int:
    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, raw_batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch)
        loss = outputs["loss"] / grad_accumulation_steps
        loss.backward()

        if batch_idx % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % log_every == 0:
                print(
                    f"[spare-lite] step={step} "
                    f"loss={outputs['loss'].item():.4f} "
                    f"action_loss={outputs['action_loss'].item():.4f} "
                    f"align_loss={outputs['align_loss'].item():.4f}"
                )

            if max_steps is not None and step >= max_steps:
                break

    return step


def train_spare_lite(config: SpaReLiteConfig, dataloader: Iterable[dict]) -> SpaReLiteModel:
    device = torch.device(config.optim.device if torch.cuda.is_available() else "cpu")
    model = SpaReLiteModel(config).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )

    total_steps = 0
    for epoch in range(config.optim.num_epochs):
        steps = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            grad_accumulation_steps=config.optim.grad_accumulation_steps,
            log_every=config.optim.log_every,
            max_steps=None if config.optim.max_steps is None else config.optim.max_steps - total_steps,
        )
        total_steps += steps
        print(f"[spare-lite] epoch={epoch + 1} completed total_steps={total_steps}")
        if config.optim.max_steps is not None and total_steps >= config.optim.max_steps:
            break

    return model


if __name__ == "__main__":
    raise SystemExit(
        "Import `train_spare_lite` from another script after wiring a concrete "
        "dataset/dataloader for OpenVLA-style batches."
    )
