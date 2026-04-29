from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .rl_bandit import CandidateBanditHead


class RealActionCandidateDataset(Dataset):
    def __init__(self, jsonl_path: str | Path) -> None:
        self.rows = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        candidates = torch.tensor(row["candidates"], dtype=torch.float32)
        env_rewards = torch.tensor(row["env_rewards"], dtype=torch.float32)
        context = candidates.mean(dim=0)
        spatial_latent = candidates[0]
        return {
            "context": context,
            "candidate_embeddings": candidates,
            "env_rewards": env_rewards,
            "spatial_latent": spatial_latent,
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "context": torch.stack([item["context"] for item in batch]),
        "candidate_embeddings": torch.stack([item["candidate_embeddings"] for item in batch]),
        "env_rewards": torch.stack([item["env_rewards"] for item in batch]),
        "spatial_latent": torch.stack([item["spatial_latent"] for item in batch]),
    }


def run_once(lambda_align: float, dataloader: DataLoader, max_steps: int, device: torch.device) -> None:
    policy = CandidateBanditHead(hidden_dim=7).to(device)
    optimizer = AdamW(policy.parameters(), lr=1e-3)

    policy.train()
    for step, batch in enumerate(dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = policy(
            context=batch["context"],
            candidate_embeddings=batch["candidate_embeddings"],
            env_rewards=batch["env_rewards"],
            spatial_latent=batch["spatial_latent"],
            lambda_align=lambda_align,
        )
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(
            f"[rl-real] lambda={lambda_align:.3f} step={step} "
            f"loss={out.loss.item():.4f} "
            f"reward={out.reward.mean().item():.4f} "
            f"r1={out.r1_env.mean().item():.4f} "
            f"r2={out.r2_align.mean().item():.4f}"
        )
        if step >= max_steps:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-data lightweight R1 vs R1+lambda R2 smoke.")
    parser.add_argument("--jsonl-path", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--lambda-align", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = RealActionCandidateDataset(args.jsonl_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    run_once(lambda_align=0.0, dataloader=dataloader, max_steps=args.max_steps, device=device)
    run_once(lambda_align=args.lambda_align, dataloader=dataloader, max_steps=args.max_steps, device=device)


if __name__ == "__main__":
    main()
