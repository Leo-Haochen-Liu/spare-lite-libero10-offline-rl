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


def _decode_instruction(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, np.ndarray) and raw.shape == ():
        item = raw.item()
        return item.decode("utf-8") if isinstance(item, bytes) else str(item)
    return str(raw)


def _save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LIBERO HDF5 demos into standard offline RL transitions.")
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--episode-group", default="data")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--instruction-key", default=None)
    parser.add_argument("--image-key", default="obs/agentview_rgb")
    parser.add_argument("--action-key", default="actions")
    parser.add_argument("--reward-key", default="rewards")
    parser.add_argument("--done-key", default="dones")
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--next-state-key", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--include-terminal-next-obs", action="store_true")
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path)
    output_jsonl = Path(args.output_jsonl)
    image_dir = Path(args.image_dir)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with h5py.File(hdf5_path, "r") as f, output_jsonl.open("w", encoding="utf-8") as out_f:
        root = f[args.episode_group]
        episode_names = sorted(root.keys())
        if args.max_episodes is not None:
            episode_names = episode_names[: args.max_episodes]

        for episode_name in episode_names:
            episode = root[episode_name]

            instruction = args.instruction
            if args.instruction_key and args.instruction_key in episode:
                instruction = _decode_instruction(episode[args.instruction_key][()])
            if instruction is None:
                instruction = hdf5_path.stem.replace("_demo", "").replace("_", " ")

            images = _read_value(episode, args.image_key)
            actions = _read_value(episode, args.action_key)
            rewards = _read_value(episode, args.reward_key)
            dones = _read_value(episode, args.done_key)
            states = _read_value(episode, args.state_key) if args.state_key else None
            next_states = _read_value(episode, args.next_state_key) if args.next_state_key else None

            limit = min(len(images), len(actions), len(rewards), len(dones))
            if not args.include_terminal_next_obs:
                limit -= 1

            for step_idx in range(limit):
                next_idx = min(step_idx + 1, len(images) - 1)
                current_path = image_dir / f"{episode_name}_{step_idx:06d}.png"
                next_path = image_dir / f"{episode_name}_{next_idx:06d}_next.png"
                _save_image(images[step_idx], current_path)
                _save_image(images[next_idx], next_path)

                record = {
                    "episode_id": episode_name,
                    "step_idx": step_idx,
                    "instruction": instruction,
                    "obs_image_path": str(current_path),
                    "next_obs_image_path": str(next_path),
                    "action": np.asarray(actions[step_idx], dtype=np.float32).round(6).tolist(),
                    "reward": float(np.asarray(rewards[step_idx]).item()),
                    "done": int(np.asarray(dones[step_idx]).item()),
                }
                if states is not None:
                    record["state"] = np.asarray(states[step_idx], dtype=np.float32).round(6).tolist()
                if next_states is not None:
                    record["next_state"] = np.asarray(next_states[step_idx], dtype=np.float32).round(6).tolist()
                elif states is not None:
                    record["next_state"] = np.asarray(states[next_idx], dtype=np.float32).round(6).tolist()

                out_f.write(json.dumps(record) + "\n")
                written += 1

                if args.max_transitions is not None and written >= args.max_transitions:
                    print(f"exported {written} transitions to {output_jsonl}")
                    return

    print(f"exported {written} transitions to {output_jsonl}")


if __name__ == "__main__":
    main()
