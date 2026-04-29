from __future__ import annotations

import argparse
from pathlib import Path

import torch

from spare_lite.adapters import build_jsonl_dataloader
from spare_lite.config import SpaReLiteConfig
from spare_lite.train_spare_lite import train_spare_lite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a supervised SpaRe-lite training job.")
    parser.add_argument("--jsonl-path", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--spatial-model", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-prompt-style", default="pure")
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SpaReLiteConfig()
    config.policy.model_name = args.policy_model
    config.spatial.model_name = args.spatial_model
    config.optim.learning_rate = args.learning_rate
    config.optim.weight_decay = args.weight_decay
    config.optim.num_epochs = args.num_epochs
    config.optim.grad_accumulation_steps = args.grad_accumulation_steps
    config.optim.max_steps = args.max_steps
    config.optim.lambda_align = args.lambda_align
    config.optim.log_every = args.log_every
    config.optim.device = args.device

    dataloader = build_jsonl_dataloader(
        jsonl_path=args.jsonl_path,
        policy_model_name=args.policy_model,
        spatial_model_name=args.spatial_model,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_length=args.max_length,
        default_prompt_style=args.default_prompt_style,
    )

    model = train_spare_lite(config=config, dataloader=dataloader)

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.policy.save_pretrained(save_dir / "policy")
        if hasattr(model, "policy_latent_head"):
            torch.save(model.policy_latent_head.state_dict(), save_dir / "policy_latent_head.pt")
        if hasattr(model, "spatial_projection"):
            torch.save(model.spatial_projection.state_dict(), save_dir / "spatial_projection.pt")
        print(f"[spare-lite] saved outputs to {save_dir}")


if __name__ == "__main__":
    main()
