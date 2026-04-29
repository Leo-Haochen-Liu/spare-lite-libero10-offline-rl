from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def _read_value(group: h5py.Group, key: str) -> np.ndarray:
    current = group
    for part in key.split("/"):
        if part not in current:
            raise KeyError(f"Missing key `{key}` in group `{group.name}`")
        current = current[part]
    return np.asarray(current)


def _format_action(action: np.ndarray) -> list[float]:
    return np.asarray(action, dtype=np.float32).round(6).tolist()


def _sample_dataset_negatives(
    rng: np.random.Generator,
    action_pool: list[np.ndarray],
    expert: np.ndarray,
    num_negative_candidates: int,
) -> list[np.ndarray]:
    if not action_pool:
        return []

    stacked = np.stack(action_pool, axis=0)
    expert_flat = expert.reshape(-1)
    pool_flat = stacked.reshape(stacked.shape[0], -1)
    distances = np.linalg.norm(pool_flat - expert_flat[None, :], axis=1)

    # Prefer negatives that are clearly different from the expert action.
    candidate_indices = np.where(distances > 1e-6)[0]
    if candidate_indices.size == 0:
        candidate_indices = np.arange(len(action_pool))

    if candidate_indices.size >= num_negative_candidates:
        chosen = rng.choice(candidate_indices, size=num_negative_candidates, replace=False)
    else:
        chosen = rng.choice(candidate_indices, size=num_negative_candidates, replace=True)
    return [np.asarray(action_pool[idx], dtype=np.float32) for idx in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LIBERO HDF5 demos into lightweight RL JSONL.")
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--episode-group", default="data")
    parser.add_argument("--image-key", default="obs/agentview_rgb")
    parser.add_argument("--action-key", default="actions")
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--max-steps-per-episode", type=int, default=16)
    parser.add_argument("--num-negative-candidates", type=int, default=3)
    parser.add_argument("--noise-scale", type=float, default=0.15)
    parser.add_argument(
        "--negative-mode",
        choices=["noise", "dataset"],
        default="noise",
        help="Use synthetic noisy negatives or sample negatives from other actions in the dataset.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hdf5_path = Path(args.hdf5_path)
    output_jsonl = Path(args.output_jsonl)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records = 0
    with h5py.File(hdf5_path, "r") as f, output_jsonl.open("w", encoding="utf-8") as out_f:
        root = f[args.episode_group]
        episode_names = sorted(root.keys())[: args.max_episodes]
        action_pool: list[np.ndarray] = []

        if args.negative_mode == "dataset":
            for episode_name in episode_names:
                episode = root[episode_name]
                actions = _read_value(episode, args.action_key)
                limit = min(len(actions), args.max_steps_per_episode)
                for step_idx in range(limit):
                    action_pool.append(np.asarray(actions[step_idx], dtype=np.float32))

        for episode_name in episode_names:
            episode = root[episode_name]
            images = _read_value(episode, args.image_key)
            actions = _read_value(episode, args.action_key)
            limit = min(len(images), len(actions), args.max_steps_per_episode)

            for step_idx in range(limit):
                image_path = image_dir / f"{episode_name}_{step_idx:06d}.png"
                Image.fromarray(images[step_idx]).save(image_path)

                expert = np.asarray(actions[step_idx], dtype=np.float32)
                candidates = [_format_action(expert)]
                env_rewards = [1.0]

                if args.negative_mode == "dataset":
                    negatives = _sample_dataset_negatives(
                        rng=rng,
                        action_pool=action_pool,
                        expert=expert,
                        num_negative_candidates=args.num_negative_candidates,
                    )
                else:
                    negatives = []
                    for _ in range(args.num_negative_candidates):
                        noisy = np.clip(
                            expert + rng.normal(0.0, args.noise_scale, size=expert.shape).astype(np.float32),
                            -1.0,
                            1.0,
                        )
                        negatives.append(noisy)

                for negative in negatives:
                    candidates.append(_format_action(negative))
                    env_rewards.append(0.0)

                record = {
                    "image_path": str(image_path),
                    "prompt": args.instruction,
                    "candidates": candidates,
                    "env_rewards": env_rewards,
                    "negative_mode": args.negative_mode,
                }
                out_f.write(json.dumps(record) + "\n")
                records += 1

    print(f"exported {records} RL records to {output_jsonl}")


if __name__ == "__main__":
    main()
