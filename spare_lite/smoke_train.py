from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .toy_modeling import ToySpaReLiteModel


class ToyDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int):
        input_ids = torch.randint(0, 64, (12,), dtype=torch.long)
        labels = input_ids.clone()
        spatial = torch.rand(3, 64, 64)
        return {
            "policy_inputs": {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            },
            "spatial_pixel_values": spatial,
        }


def collate(batch):
    return {
        "policy_inputs": {
            "input_ids": torch.stack([x["policy_inputs"]["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["policy_inputs"]["attention_mask"] for x in batch]),
            "labels": torch.stack([x["policy_inputs"]["labels"] for x in batch]),
        },
        "spatial_pixel_values": torch.stack([x["spatial_pixel_values"] for x in batch]),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ToySpaReLiteModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    dataloader = DataLoader(ToyDataset(), batch_size=2, shuffle=False, collate_fn=collate)

    model.train()
    for step, batch in enumerate(dataloader, start=1):
        batch = {
            "policy_inputs": {k: v.to(device) for k, v in batch["policy_inputs"].items()},
            "spatial_pixel_values": batch["spatial_pixel_values"].to(device),
        }
        outputs = model(batch)
        outputs["loss"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(
            f"[smoke] step={step} "
            f"loss={outputs['loss'].item():.4f} "
            f"action_loss={outputs['action_loss'].item():.4f} "
            f"align_loss={outputs['align_loss'].item():.4f}"
        )


if __name__ == "__main__":
    main()
