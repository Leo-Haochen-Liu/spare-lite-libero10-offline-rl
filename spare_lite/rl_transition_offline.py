from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW, SGD

from spare_lite.adapters import build_jsonl_dataloader
from spare_lite.config import SpaReLiteConfig
from spare_lite.modeling import SpaReLiteModel
from spare_lite.rl_reward import combine_rewards
from spare_lite.train_spare_lite import move_batch_to_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_checkpoint_payload(
    model: SpaReLiteModel,
    summary: dict[str, float],
    step_count: int,
    policy_trainable_patterns: tuple[str, ...],
) -> dict[str, object]:
    policy_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.policy.state_dict().items()
        if name in model.trainable_policy_parameter_names
    }
    extra_state = {
        "policy_latent_head": {k: v.detach().cpu() for k, v in model.policy_latent_head.state_dict().items()},
        "policy_logits_head": {k: v.detach().cpu() for k, v in model.policy_logits_head.state_dict().items()},
        "spatial_projection": {k: v.detach().cpu() for k, v in model.spatial_projection.state_dict().items()},
    }
    return {
        "step_count": step_count,
        "summary": summary,
        "policy_trainable_patterns": list(policy_trainable_patterns),
        "trainable_policy_parameter_names": list(model.trainable_policy_parameter_names),
        "policy_trainable_state_dict": policy_state,
        "extra_trainable_state_dict": extra_state,
    }


def save_branch_checkpoint(
    output_path: Path,
    model: SpaReLiteModel,
    summary: dict[str, float],
    step_count: int,
    policy_trainable_patterns: tuple[str, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint_payload(
        model=model,
        summary=summary,
        step_count=step_count,
        policy_trainable_patterns=policy_trainable_patterns,
    )
    torch.save(payload, output_path)
    print(f"saved checkpoint payload to {output_path}")


def build_config(
    policy_model: str,
    spatial_model: str,
    spatial_backend: str,
    device: str,
    learning_rate: float,
    weight_decay: float,
    policy_trainable_patterns: tuple[str, ...],
) -> SpaReLiteConfig:
    config = SpaReLiteConfig()
    config.policy.model_name = policy_model
    config.policy.use_lora = True
    config.policy.target_modules = policy_trainable_patterns
    config.spatial.model_name = spatial_model
    config.spatial.backend = spatial_backend
    config.optim.device = device
    config.optim.learning_rate = learning_rate
    config.optim.weight_decay = weight_decay
    return config


def compute_branch_loss(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    lambda_align: float,
    align_mode: str,
    align_threshold: float,
    reward_source: str,
    min_reward_weight: float,
    align_aux_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if reward_source == "return_to_go" and "return_to_go" in batch:
        r1_env = batch["return_to_go"]
    elif reward_source == "reward" and "reward" in batch:
        r1_env = batch["reward"]
    elif "return_to_go" in batch:
        r1_env = batch["return_to_go"]
    elif "reward" in batch:
        r1_env = batch["reward"]
    else:
        raise ValueError("The transition batch does not contain `reward` or `return_to_go`.")

    rewards = combine_rewards(
        env_reward=r1_env,
        policy_latent=outputs["policy_latent"],
        spatial_latent=outputs["spatial_latent"],
        lambda_align=lambda_align,
        align_mode=align_mode,
        align_threshold=align_threshold,
    )
    reward_weights = rewards["reward"].detach().clamp_min(min_reward_weight)
    weighted_action_loss = (reward_weights * outputs["per_sample_action_loss"]).mean()
    align_aux_loss = -rewards["r2_align"].mean()
    total_loss = weighted_action_loss + lambda_align * align_aux_weight * align_aux_loss
    metrics = {
        "loss": float(total_loss.detach().item()),
        "weighted_action_loss": float(weighted_action_loss.detach().item()),
        "align_aux_loss": float(align_aux_loss.detach().item()),
        "action_loss": float(outputs["action_loss"].detach().item()),
        "align_loss": float(outputs["align_loss"].detach().item()),
        "reward": float(rewards["reward"].mean().detach().item()),
        "r1": float(rewards["r1_env"].mean().detach().item()),
        "r2": float(rewards["r2_align"].mean().detach().item()),
        "r2_raw": float(rewards["r2_align_raw"].mean().detach().item()),
        "reward_weight": float(reward_weights.mean().detach().item()),
    }
    return total_loss, metrics


def run_branch(
    lambda_align: float,
    model: SpaReLiteModel,
    dataloader,
    device: torch.device,
    max_steps: int,
    grad_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    align_mode: str,
    align_threshold: float,
    reward_source: str,
    min_reward_weight: float,
    align_aux_weight: float,
    policy_trainable_patterns: tuple[str, ...],
    checkpoint_path: Path | None = None,
) -> dict[str, float]:
    model.train()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters were enabled for rl_transition_offline.")
    if optimizer_name == "adamw":
        optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = SGD(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer.zero_grad(set_to_none=True)

    metric_sums = {
        "loss": 0.0,
        "weighted_action_loss": 0.0,
        "align_aux_loss": 0.0,
        "action_loss": 0.0,
        "align_loss": 0.0,
        "reward": 0.0,
        "r1": 0.0,
        "r2": 0.0,
        "r2_raw": 0.0,
        "reward_weight": 0.0,
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
        outputs = model(batch)
        loss, metrics = compute_branch_loss(
            batch=batch,
            outputs=outputs,
            lambda_align=lambda_align,
            align_mode=align_mode,
            align_threshold=align_threshold,
            reward_source=reward_source,
            min_reward_weight=min_reward_weight,
            align_aux_weight=align_aux_weight,
        )
        (loss / grad_accumulation_steps).backward()

        if step % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for key, value in metrics.items():
            metric_sums[key] += value
        total_steps = step

        print(
            f"[rl-transition] lambda={lambda_align:.3f} step={step} "
            f"loss={metrics['loss']:.4f} "
            f"weighted_action_loss={metrics['weighted_action_loss']:.4f} "
            f"align_aux_loss={metrics['align_aux_loss']:.4f} "
            f"r1={metrics['r1']:.4f} "
            f"r2={metrics['r2']:.4f} "
            f"reward={metrics['reward']:.4f} "
            f"reward_weight={metrics['reward_weight']:.4f}"
        )

    if total_steps % grad_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    summary = {
        "lambda_align": lambda_align,
        "steps": total_steps,
    }
    for key, value in metric_sums.items():
        summary[f"avg_{key}"] = value / max(total_steps, 1)
    print(
        f"[rl-transition-summary] lambda={lambda_align:.3f} "
        f"steps={total_steps} "
        f"avg_reward={summary['avg_reward']:.4f} "
        f"avg_r1={summary['avg_r1']:.4f} "
        f"avg_r2={summary['avg_r2']:.4f} "
        f"avg_action_loss={summary['avg_action_loss']:.4f} "
        f"avg_reward_weight={summary['avg_reward_weight']:.4f}"
    )
    if checkpoint_path is not None:
        save_branch_checkpoint(
            output_path=checkpoint_path,
            model=model,
            summary=summary,
            step_count=total_steps,
            policy_trainable_patterns=policy_trainable_patterns,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Poster-closer offline transition training for SpaRe-lite.")
    parser.add_argument("--jsonl-path", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--spatial-model", required=True)
    parser.add_argument("--spatial-backend", default="auto")
    parser.add_argument("--spatial-image-source", choices=["obs", "next"], default="obs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--lambda-align", type=float, default=0.2)
    parser.add_argument("--align-aux-weight", type=float, default=1.0)
    parser.add_argument("--align-mode", default="centered_relu")
    parser.add_argument("--align-threshold", type=float, default=0.72)
    parser.add_argument("--reward-source", choices=["reward", "return_to_go"], default="return_to_go")
    parser.add_argument("--return-gamma", type=float, default=1.0)
    parser.add_argument("--min-reward-weight", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument(
        "--policy-trainable-pattern",
        action="append",
        dest="policy_trainable_patterns",
        default=None,
        help="Substring match for policy parameters to unfreeze. Repeatable.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy_trainable_patterns = tuple(
        args.policy_trainable_patterns
        or (
            "language_model.model.layers.31.",
            "language_model.lm_head",
        )
    )
    config = build_config(
        policy_model=args.policy_model,
        spatial_model=args.spatial_model,
        spatial_backend=args.spatial_backend,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        policy_trainable_patterns=policy_trainable_patterns,
    )
    dataloader = build_jsonl_dataloader(
        jsonl_path=args.jsonl_path,
        policy_model_name=args.policy_model,
        spatial_model_name=args.spatial_model,
        spatial_backend=args.spatial_backend,
        batch_size=args.batch_size,
        shuffle=True,
        max_length=args.max_length,
        transition_spatial_source=args.spatial_image_source,
        return_gamma=args.return_gamma,
    )

    summaries = []
    if args.skip_baseline:
        baseline_summary = {"lambda_align": 0.0, "steps": 0, "skipped": True}
    else:
        baseline_model = SpaReLiteModel(config).to(device)
        print("[rl-transition] trainable policy parameters for baseline:")
        for name in baseline_model.trainable_policy_parameter_names:
            print(f"[rl-transition]   {name}")
        baseline_summary = run_branch(
            lambda_align=0.0,
            model=baseline_model,
            dataloader=dataloader,
            device=device,
            max_steps=args.max_steps,
            grad_accumulation_steps=args.grad_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
            align_mode=args.align_mode,
            align_threshold=args.align_threshold,
            reward_source=args.reward_source,
            min_reward_weight=args.min_reward_weight,
            align_aux_weight=args.align_aux_weight,
            policy_trainable_patterns=policy_trainable_patterns,
            checkpoint_path=(
                Path(args.checkpoint_dir) / "baseline_lambda0.pt"
                if args.checkpoint_dir
                else None
            ),
        )
        del baseline_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summaries.append(baseline_summary)

    align_model = SpaReLiteModel(config).to(device)
    print("[rl-transition] trainable policy parameters for aligned branch:")
    for name in align_model.trainable_policy_parameter_names:
        print(f"[rl-transition]   {name}")
    align_summary = run_branch(
        lambda_align=args.lambda_align,
        model=align_model,
        dataloader=dataloader,
        device=device,
        max_steps=args.max_steps,
        grad_accumulation_steps=args.grad_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        align_mode=args.align_mode,
        align_threshold=args.align_threshold,
        reward_source=args.reward_source,
        min_reward_weight=args.min_reward_weight,
        align_aux_weight=args.align_aux_weight,
        policy_trainable_patterns=policy_trainable_patterns,
        checkpoint_path=(
            Path(args.checkpoint_dir) / f"align_lambda{args.lambda_align:g}.pt"
            if args.checkpoint_dir
            else None
        ),
    )
    summaries.append(align_summary)

    if args.summary_json:
        output_path = Path(args.summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"wrote summary to {output_path}")


if __name__ == "__main__":
    main()
