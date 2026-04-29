from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW

from spare_lite.adapters import build_jsonl_dataloader
from spare_lite.config import SpaReLiteConfig
from spare_lite.modeling import SpaReLiteModel
from spare_lite.rl_bandit import CandidateBanditHead
from spare_lite.rl_reward import combine_rewards
from spare_lite.train_spare_lite import move_batch_to_device


def build_config(
    policy_model: str,
    spatial_model: str,
    spatial_backend: str,
    device: str,
    lambda_align: float,
) -> SpaReLiteConfig:
    config = SpaReLiteConfig()
    config.policy.model_name = policy_model
    config.spatial.model_name = spatial_model
    config.spatial.backend = spatial_backend
    config.optim.device = device
    config.optim.lambda_align = lambda_align
    return config


def make_candidates(
    spatial_latent: torch.Tensor,
    num_negative_candidates: int,
    noise_scale: float,
    candidate_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, hidden_dim = spatial_latent.shape
    negatives = []
    for _ in range(num_negative_candidates):
        noise = torch.randn_like(spatial_latent)
        if candidate_mode == "orthogonal_noise":
            denom = spatial_latent.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            proj = (noise * spatial_latent).sum(dim=-1, keepdim=True) / denom
            noise = noise - proj * spatial_latent
        elif candidate_mode != "noise":
            raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")
        negatives.append(spatial_latent + noise_scale * noise)
    candidates = torch.stack([spatial_latent, *negatives], dim=1)
    env_rewards = torch.zeros(
        batch_size,
        num_negative_candidates + 1,
        dtype=spatial_latent.dtype,
        device=spatial_latent.device,
    )
    env_rewards[:, 0] = 1.0
    return candidates, env_rewards


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_branch(
    lambda_align: float,
    model: SpaReLiteModel,
    dataloader,
    max_steps: int,
    num_negative_candidates: int,
    noise_scale: float,
    candidate_mode: str,
    align_mode: str,
    align_threshold: float,
    device: torch.device,
) -> dict[str, float]:
    policy_head: CandidateBanditHead | None = None
    optimizer: AdamW | None = None
    metric_sums = {
        "loss": 0.0,
        "reward": 0.0,
        "r1": 0.0,
        "r2": 0.0,
        "expert_prob": 0.0,
        "expert_hit": 0.0,
        "greedy_reward": 0.0,
        "greedy_r1": 0.0,
        "greedy_r2": 0.0,
        "greedy_expert_hit": 0.0,
        "expert_reward": 0.0,
        "negative_reward": 0.0,
    }
    total_steps = 0

    data_iter = iter(dataloader)
    for step in range(1, max_steps + 1):
        try:
            raw_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            raw_batch = next(data_iter)
        batch = move_batch_to_device(raw_batch, device)
        with torch.no_grad():
            outputs = model(batch)
            policy_latent = outputs["policy_latent"]
            spatial_latent = outputs["spatial_latent"]

        candidate_embeddings, env_rewards = make_candidates(
            spatial_latent=spatial_latent,
            num_negative_candidates=num_negative_candidates,
            noise_scale=noise_scale,
            candidate_mode=candidate_mode,
        )
        expert_rewards = combine_rewards(
            env_reward=env_rewards[:, 0],
            policy_latent=candidate_embeddings[:, 0],
            spatial_latent=spatial_latent,
            lambda_align=lambda_align,
            align_mode=align_mode,
            align_threshold=align_threshold,
        )
        negative_rewards = combine_rewards(
            env_reward=env_rewards[:, 1:].reshape(-1),
            policy_latent=candidate_embeddings[:, 1:].reshape(-1, candidate_embeddings.size(-1)),
            spatial_latent=spatial_latent.unsqueeze(1)
            .expand(-1, num_negative_candidates, -1)
            .reshape(-1, spatial_latent.size(-1)),
            lambda_align=lambda_align,
            align_mode=align_mode,
            align_threshold=align_threshold,
        )
        if policy_head is None:
            policy_head = CandidateBanditHead(hidden_dim=policy_latent.size(-1)).to(device)
            optimizer = AdamW(policy_head.parameters(), lr=1e-3)

        assert optimizer is not None
        out = policy_head(
            context=policy_latent,
            candidate_embeddings=candidate_embeddings,
            env_rewards=env_rewards,
            spatial_latent=spatial_latent,
            lambda_align=lambda_align,
            align_mode=align_mode,
            align_threshold=align_threshold,
        )
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        metric_sums["loss"] += out.loss.item()
        metric_sums["reward"] += out.reward.mean().item()
        metric_sums["r1"] += out.r1_env.mean().item()
        metric_sums["r2"] += out.r2_align.mean().item()
        metric_sums["expert_prob"] += out.expert_prob.mean().item()
        metric_sums["expert_hit"] += out.chosen_is_expert.mean().item()
        metric_sums["greedy_reward"] += out.greedy_reward.mean().item()
        metric_sums["greedy_r1"] += out.greedy_r1_env.mean().item()
        metric_sums["greedy_r2"] += out.greedy_r2_align.mean().item()
        metric_sums["greedy_expert_hit"] += out.greedy_is_expert.mean().item()
        metric_sums["expert_reward"] += expert_rewards["reward"].mean().item()
        metric_sums["negative_reward"] += negative_rewards["reward"].mean().item()
        total_steps = step

        print(
            f"[rl-spare-latent] lambda={lambda_align:.3f} step={step} "
            f"loss={out.loss.item():.4f} "
            f"reward={out.reward.mean().item():.4f} "
            f"r1={out.r1_env.mean().item():.4f} "
            f"r2={out.r2_align.mean().item():.4f} "
            f"expert_prob={out.expert_prob.mean().item():.4f} "
            f"expert_hit={out.chosen_is_expert.mean().item():.4f} "
            f"greedy_reward={out.greedy_reward.mean().item():.4f} "
            f"greedy_expert_hit={out.greedy_is_expert.mean().item():.4f} "
            f"expert_reward={expert_rewards['reward'].mean().item():.4f} "
            f"negative_reward={negative_rewards['reward'].mean().item():.4f}"
        )
    if total_steps == 0:
        return {"lambda_align": lambda_align, "steps": 0}

    summary = {
        "lambda_align": lambda_align,
        "steps": total_steps,
    }
    for key, value in metric_sums.items():
        summary[f"avg_{key}"] = value / total_steps
    print(
        f"[rl-spare-latent-summary] lambda={lambda_align:.3f} "
        f"steps={total_steps} "
        f"avg_reward={summary['avg_reward']:.4f} "
        f"avg_r1={summary['avg_r1']:.4f} "
        f"avg_r2={summary['avg_r2']:.4f} "
        f"avg_expert_prob={summary['avg_expert_prob']:.4f} "
        f"avg_expert_hit={summary['avg_expert_hit']:.4f} "
        f"avg_greedy_reward={summary['avg_greedy_reward']:.4f} "
        f"avg_greedy_expert_hit={summary['avg_greedy_expert_hit']:.4f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R1 vs R1+lambda R2 on real VLA/DINO latents.")
    parser.add_argument("--jsonl-path", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--spatial-model", required=True)
    parser.add_argument("--spatial-backend", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--lambda-align", type=float, default=0.2)
    parser.add_argument("--num-negative-candidates", type=int, default=3)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--candidate-mode", default="orthogonal_noise")
    parser.add_argument("--align-mode", default="centered_relu")
    parser.add_argument("--align-threshold", type=float, default=0.72)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = build_config(
        policy_model=args.policy_model,
        spatial_model=args.spatial_model,
        spatial_backend=args.spatial_backend,
        device=args.device,
        lambda_align=args.lambda_align,
    )
    dataloader = build_jsonl_dataloader(
        jsonl_path=args.jsonl_path,
        policy_model_name=args.policy_model,
        spatial_model_name=args.spatial_model,
        spatial_backend=args.spatial_backend,
        batch_size=args.batch_size,
        shuffle=True,
        max_length=args.max_length,
    )
    model = SpaReLiteModel(config).to(device)
    model.eval()

    if args.skip_baseline:
        baseline_summary = {
            "lambda_align": 0.0,
            "steps": 0,
            "skipped": True,
        }
    else:
        baseline_summary = run_branch(
            lambda_align=0.0,
            model=model,
            dataloader=dataloader,
            max_steps=args.max_steps,
            num_negative_candidates=args.num_negative_candidates,
            noise_scale=args.noise_scale,
            candidate_mode=args.candidate_mode,
            align_mode=args.align_mode,
            align_threshold=args.align_threshold,
            device=device,
        )
    if args.lambda_align == 0.0:
        spare_summary = dict(baseline_summary)
    else:
        spare_summary = run_branch(
            lambda_align=args.lambda_align,
            model=model,
            dataloader=dataloader,
            max_steps=args.max_steps,
            num_negative_candidates=args.num_negative_candidates,
            noise_scale=args.noise_scale,
            candidate_mode=args.candidate_mode,
            align_mode=args.align_mode,
            align_threshold=args.align_threshold,
            device=device,
        )

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "num_negative_candidates": args.num_negative_candidates,
            "noise_scale": args.noise_scale,
            "candidate_mode": args.candidate_mode,
            "align_mode": args.align_mode,
            "align_threshold": args.align_threshold,
            "baseline": baseline_summary,
            "spare": spare_summary,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[rl-spare-latent] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
