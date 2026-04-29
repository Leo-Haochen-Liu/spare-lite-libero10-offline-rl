from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .rl_bandit import CandidateBanditHead


class ToyRLDataset(Dataset):
    def __len__(self) -> int:
        return 32

    def __getitem__(self, idx: int):
        context = torch.randn(64)
        good = context + 0.1 * torch.randn(64)
        bad1 = torch.randn(64)
        bad2 = torch.randn(64)
        candidates = torch.stack([good, bad1, bad2], dim=0)
        env_rewards = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        spatial_latent = good + 0.05 * torch.randn(64)
        return {
            "context": context,
            "candidate_embeddings": candidates,
            "env_rewards": env_rewards,
            "spatial_latent": spatial_latent,
        }


def collate(batch):
    return {
        "context": torch.stack([x["context"] for x in batch]),
        "candidate_embeddings": torch.stack([x["candidate_embeddings"] for x in batch]),
        "env_rewards": torch.stack([x["env_rewards"] for x in batch]),
        "spatial_latent": torch.stack([x["spatial_latent"] for x in batch]),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = CandidateBanditHead(hidden_dim=64).to(device)
    optimizer = AdamW(policy.parameters(), lr=1e-3)
    dataloader = DataLoader(ToyRLDataset(), batch_size=8, shuffle=True, collate_fn=collate)

    policy.train()
    for step, batch in enumerate(dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = policy(
            context=batch["context"],
            candidate_embeddings=batch["candidate_embeddings"],
            env_rewards=batch["env_rewards"],
            spatial_latent=batch["spatial_latent"],
            lambda_align=0.2,
        )
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(
            f"[rl-smoke] step={step} "
            f"loss={out.loss.item():.4f} "
            f"reward={out.reward.mean().item():.4f} "
            f"r1={out.r1_env.mean().item():.4f} "
            f"r2={out.r2_align.mean().item():.4f}"
        )


if __name__ == "__main__":
    main()
