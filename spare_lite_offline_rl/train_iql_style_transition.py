from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from spare_lite.adapters import build_jsonl_dataloader
from spare_lite.config import SpaReLiteConfig
from spare_lite.rl_reward import combine_rewards
from spare_lite.train_spare_lite import move_batch_to_device

from .iql_losses import cql_conservative_penalty, clipped_advantage_weights, expectile_loss, soft_update_
from .iql_model import SpaReLiteIQLModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def save_checkpoint(
    output_path: Path,
    model: SpaReLiteIQLModel,
    summary: dict[str, float],
    policy_trainable_patterns: tuple[str, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "policy_trainable_patterns": list(policy_trainable_patterns),
        "trainable_policy_parameter_names": list(model.trainable_policy_parameter_names),
        "policy_trainable_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.base.policy.state_dict().items()
            if name in model.trainable_policy_parameter_names
        },
        "policy_latent_head": {
            k: v.detach().cpu() for k, v in model.base.policy_latent_head.state_dict().items()
        },
        "policy_logits_head": {
            k: v.detach().cpu() for k, v in model.base.policy_logits_head.state_dict().items()
        },
        "spatial_projection": {
            k: v.detach().cpu() for k, v in model.base.spatial_projection.state_dict().items()
        },
        "value_head": {
            k: v.detach().cpu() for k, v in model.value_head.state_dict().items()
        },
        "q1_head": {
            k: v.detach().cpu() for k, v in model.q1_head.state_dict().items()
        },
        "q2_head": {
            k: v.detach().cpu() for k, v in model.q2_head.state_dict().items()
        },
    }
    torch.save(payload, output_path)
    print(f"[iql-style] saved checkpoint to {output_path}")


def run_branch(
    lambda_align: float,
    model: SpaReLiteIQLModel,
    dataloader,
    device: torch.device,
    max_steps: int,
    grad_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    align_mode: str,
    align_threshold: float,
    reward_norm: str,
    r1_scale: float,
    r2_scale: float,
    r2_bias: float,
    discount: float,
    expectile: float,
    advantage_temperature: float,
    max_adv_weight: float,
    target_tau: float,
    cql_alpha: float,
    checkpoint_path: Path | None,
    policy_trainable_patterns: tuple[str, ...],
) -> dict[str, float]:
    model.train()
    target_value_head = nn.Sequential(
        nn.Linear(model.value_head[0].in_features, model.value_head[0].out_features),
        nn.GELU(),
        nn.Linear(model.value_head[2].in_features, 1),
    ).to(device)
    target_value_head.load_state_dict(model.value_head.state_dict())
    target_value_head.requires_grad_(False)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True)

    metric_sums = {
        "loss": 0.0,
        "actor_loss": 0.0,
        "value_loss": 0.0,
        "q_loss": 0.0,
        "cql_loss": 0.0,
        "reward": 0.0,
        "r1": 0.0,
        "r2": 0.0,
        "r2_raw": 0.0,
        "value": 0.0,
        "q1": 0.0,
        "q2": 0.0,
        "target_q": 0.0,
        "target_return": 0.0,
        "advantage": 0.0,
        "adv_weight": 0.0,
        "actor_weight": 0.0,
    }
    total_steps = 0
    best_reward_ema = None
    reward_ema = None
    data_iter = iter(dataloader)

    for step in range(1, max_steps + 1):
        try:
            raw_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            raw_batch = next(data_iter)
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch)

        if "return_to_go" not in batch:
            raise ValueError("IQL-style training expects `return_to_go` in the transition batch.")

        rewards = combine_rewards(
            env_reward=batch["reward"],
            policy_latent=outputs["policy_latent"],
            spatial_latent=outputs["spatial_latent"],
            lambda_align=lambda_align,
            align_mode=align_mode,
            align_threshold=align_threshold,
        )
        r1_env = rewards["r1_env"]
        r2_align = rewards["r2_align"]
        if reward_norm == "batch_zscore":
            r1_env = (r1_env - r1_env.mean()) / r1_env.std().clamp_min(1e-6)
            r2_align = (r2_align - r2_align.mean()) / r2_align.std().clamp_min(1e-6)
        elif reward_norm == "batch_minmax":
            r1_env = (r1_env - r1_env.min()) / (r1_env.max() - r1_env.min()).clamp_min(1e-6)
            r2_align = (r2_align - r2_align.min()) / (r2_align.max() - r2_align.min()).clamp_min(1e-6)
        elif reward_norm != "none":
            raise ValueError(f"Unsupported reward_norm: {reward_norm}")
        r1_env = r1_scale * r1_env
        r2_align = r2_scale * (r2_align + r2_bias)
        rewards["r1_env"] = r1_env
        rewards["r2_align"] = r2_align
        rewards["reward"] = r1_env + lambda_align * r2_align

        if "next_state_latent" in outputs and "done" in batch:
            with torch.no_grad():
                next_value = target_value_head(outputs["next_state_latent"]).squeeze(-1)
            target_return = (
                rewards["reward"].detach()
                + discount * (1.0 - batch["done"]) * next_value
            )
        else:
            target_return = rewards["reward"].detach()
        value_pred = outputs["value"]
        q_min = torch.minimum(outputs["q1"], outputs["q2"])
        value_loss = expectile_loss(value_pred=value_pred, target=q_min.detach(), expectile=expectile)
        q_target = target_return
        q1_loss = (outputs["q1"] - q_target).pow(2).mean()
        q2_loss = (outputs["q2"] - q_target).pow(2).mean()

        negative_action_latents = outputs["policy_latent"].roll(shifts=1, dims=0)
        negative_q_inputs = torch.cat([outputs["state_latent"], negative_action_latents], dim=-1)
        negative_q1 = model.q1_head(negative_q_inputs).squeeze(-1).unsqueeze(-1)
        negative_q2 = model.q2_head(negative_q_inputs).squeeze(-1).unsqueeze(-1)
        cql_q1 = cql_conservative_penalty(outputs["q1"], negative_q1)
        cql_q2 = cql_conservative_penalty(outputs["q2"], negative_q2)
        cql_loss = 0.5 * (cql_q1 + cql_q2)
        q_loss = 0.5 * (q1_loss + q2_loss) + cql_alpha * cql_loss

        advantage = (q_min.detach() - value_pred.detach())
        adv_weights = clipped_advantage_weights(
            advantage=advantage,
            temperature=advantage_temperature,
            max_weight=max_adv_weight,
        )
        actor_weights = batch.get("actor_weight")
        if actor_weights is None:
            actor_weights = torch.ones_like(adv_weights)
        weighted_actor = actor_weights * adv_weights
        actor_loss = (
            weighted_actor * outputs["per_sample_action_loss"]
        ).sum() / weighted_actor.sum().clamp_min(1.0)
        loss = actor_loss + value_loss + q_loss

        (loss / grad_accumulation_steps).backward()
        if step % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            soft_update_(target_value_head, model.value_head, target_tau)

        metrics = {
            "loss": float(loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "q_loss": float(q_loss.detach().item()),
            "cql_loss": float(cql_loss.detach().item()),
            "reward": float(rewards["reward"].mean().detach().item()),
            "r1": float(rewards["r1_env"].mean().detach().item()),
            "r2": float(rewards["r2_align"].mean().detach().item()),
            "r2_raw": float(rewards["r2_align_raw"].mean().detach().item()),
            "value": float(value_pred.mean().detach().item()),
            "q1": float(outputs["q1"].mean().detach().item()),
            "q2": float(outputs["q2"].mean().detach().item()),
            "target_q": float(q_target.mean().detach().item()),
            "target_return": float(target_return.mean().detach().item()),
            "advantage": float(advantage.mean().detach().item()),
            "adv_weight": float(adv_weights.mean().detach().item()),
            "actor_weight": float(actor_weights.mean().detach().item()),
        }
        reward_ema = metrics["reward"] if reward_ema is None else (0.9 * reward_ema + 0.1 * metrics["reward"])
        for key, value in metrics.items():
            metric_sums[key] += value
        total_steps = step

        print(
            f"[iql-style] lambda={lambda_align:.3f} step={step} "
            f"loss={metrics['loss']:.4f} actor_loss={metrics['actor_loss']:.4f} "
            f"value_loss={metrics['value_loss']:.4f} q_loss={metrics['q_loss']:.4f} "
            f"cql_loss={metrics['cql_loss']:.4f} reward={metrics['reward']:.4f} "
            f"r1={metrics['r1']:.4f} r2={metrics['r2']:.4f} value={metrics['value']:.4f} "
            f"target_return={metrics['target_return']:.4f} adv_weight={metrics['adv_weight']:.4f} "
            f"actor_weight={metrics['actor_weight']:.4f}"
        )
        if checkpoint_path is not None:
            if best_reward_ema is None or reward_ema > best_reward_ema:
                best_reward_ema = reward_ema
                best_path = checkpoint_path.with_name(checkpoint_path.stem + "_best" + checkpoint_path.suffix)
                save_checkpoint(best_path, model, {"step": step, "reward_ema": reward_ema, **metrics}, policy_trainable_patterns)

    if total_steps % grad_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    summary = {"lambda_align": lambda_align, "steps": total_steps}
    for key, value in metric_sums.items():
        summary[f"avg_{key}"] = value / max(total_steps, 1)
    print(
        f"[iql-style-summary] lambda={lambda_align:.3f} steps={total_steps} "
        f"avg_reward={summary['avg_reward']:.4f} avg_r1={summary['avg_r1']:.4f} "
        f"avg_r2={summary['avg_r2']:.4f} avg_actor_loss={summary['avg_actor_loss']:.4f} "
        f"avg_value_loss={summary['avg_value_loss']:.4f} avg_q_loss={summary['avg_q_loss']:.4f} "
        f"avg_cql_loss={summary['avg_cql_loss']:.4f}"
    )
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, summary, policy_trainable_patterns)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="IQL-style offline RL track for SpaRe-lite.")
    parser.add_argument("--jsonl-path", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--spatial-model", required=True)
    parser.add_argument("--spatial-backend", default="auto")
    parser.add_argument("--spatial-image-source", choices=["obs", "next"], default="obs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--lambda-align", type=float, default=0.2)
    parser.add_argument("--align-mode", default="raw")
    parser.add_argument("--align-threshold", type=float, default=0.0)
    parser.add_argument("--reward-norm", choices=["none", "batch_zscore", "batch_minmax"], default="none")
    parser.add_argument("--r1-scale", type=float, default=1.0)
    parser.add_argument("--r2-scale", type=float, default=1.0)
    parser.add_argument("--r2-bias", type=float, default=0.0)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--return-gamma", type=float, default=0.95)
    parser.add_argument("--positive-sample-boost", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--advantage-temperature", type=float, default=3.0)
    parser.add_argument("--max-adv-weight", type=float, default=20.0)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--cql-alpha", type=float, default=0.1)
    parser.add_argument(
        "--policy-trainable-pattern",
        action="append",
        dest="policy_trainable_patterns",
        default=None,
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
        positive_sample_boost=args.positive_sample_boost,
    )

    summaries = []
    if args.skip_baseline:
        baseline_summary = {"lambda_align": 0.0, "steps": 0, "skipped": True}
    else:
        baseline_model = SpaReLiteIQLModel(config).to(device)
        baseline_summary = run_branch(
            lambda_align=0.0,
            model=baseline_model,
            dataloader=dataloader,
            device=device,
            max_steps=args.max_steps,
            grad_accumulation_steps=args.grad_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            align_mode=args.align_mode,
            align_threshold=args.align_threshold,
            reward_norm=args.reward_norm,
            r1_scale=args.r1_scale,
            r2_scale=args.r2_scale,
            r2_bias=args.r2_bias,
            discount=args.discount,
            expectile=args.expectile,
            advantage_temperature=args.advantage_temperature,
            max_adv_weight=args.max_adv_weight,
            target_tau=args.target_tau,
            cql_alpha=args.cql_alpha,
            checkpoint_path=(
                Path(args.checkpoint_dir) / "baseline_iql_style.pt"
                if args.checkpoint_dir
                else None
            ),
            policy_trainable_patterns=policy_trainable_patterns,
        )
        del baseline_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summaries.append(baseline_summary)

    align_model = SpaReLiteIQLModel(config).to(device)
    align_summary = run_branch(
        lambda_align=args.lambda_align,
        model=align_model,
        dataloader=dataloader,
        device=device,
        max_steps=args.max_steps,
        grad_accumulation_steps=args.grad_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        align_mode=args.align_mode,
        align_threshold=args.align_threshold,
        reward_norm=args.reward_norm,
        r1_scale=args.r1_scale,
        r2_scale=args.r2_scale,
        r2_bias=args.r2_bias,
        discount=args.discount,
        expectile=args.expectile,
        advantage_temperature=args.advantage_temperature,
        max_adv_weight=args.max_adv_weight,
        target_tau=args.target_tau,
        cql_alpha=args.cql_alpha,
        checkpoint_path=(
            Path(args.checkpoint_dir) / f"align_iql_style_lambda{args.lambda_align:g}.pt"
            if args.checkpoint_dir
            else None
        ),
        policy_trainable_patterns=policy_trainable_patterns,
    )
    summaries.append(align_summary)

    if args.summary_json:
        output_path = Path(args.summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"[iql-style] wrote summary to {output_path}")


if __name__ == "__main__":
    main()
