from __future__ import annotations

import argparse

import torch

from spare_lite.adapters import build_jsonl_dataloader
from spare_lite.config import SpaReLiteConfig
from spare_lite.modeling import SpaReLiteModel
from spare_lite.train_spare_lite import move_batch_to_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the SpatialVLA reference branch inside SpaRe-lite.")
    parser.add_argument("--jsonl-path", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--spatial-model", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config = SpaReLiteConfig()
    config.policy.model_name = args.policy_model
    config.spatial.model_name = args.spatial_model
    config.spatial.backend = "spatialvla"
    config.optim.device = args.device

    dataloader = build_jsonl_dataloader(
        jsonl_path=args.jsonl_path,
        policy_model_name=args.policy_model,
        spatial_model_name=args.spatial_model,
        spatial_backend="spatialvla",
        batch_size=args.batch_size,
        shuffle=False,
        max_length=args.max_length,
    )

    model = SpaReLiteModel(config).to(device).eval()
    raw_batch = next(iter(dataloader))
    batch = move_batch_to_device(raw_batch, device)
    with torch.no_grad():
        outputs = model(batch)

    print("[spatialvla-smoke] success")
    print(f"policy_latent shape: {tuple(outputs['policy_latent'].shape)}")
    print(f"spatial_latent shape: {tuple(outputs['spatial_latent'].shape)}")
    print(f"align_loss: {outputs['align_loss'].item():.6f}")


if __name__ == "__main__":
    main()
