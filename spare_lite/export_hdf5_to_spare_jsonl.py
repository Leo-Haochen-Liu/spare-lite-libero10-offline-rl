from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def _iter_step_keys(group: h5py.Group) -> list[str]:
    if "steps" in group and isinstance(group["steps"], h5py.Group):
        return sorted(group["steps"].keys(), key=lambda x: int(x) if x.isdigit() else x)
    numeric = [key for key in group.keys() if key.isdigit() and isinstance(group[key], h5py.Group)]
    return sorted(numeric, key=lambda x: int(x) if x.isdigit() else x)


def _read_value(group: h5py.Group, key: str) -> np.ndarray:
    current = group
    for part in key.split("/"):
        if part not in current:
            raise KeyError(f"Missing key `{key}` in group `{group.name}`")
        current = current[part]
    return np.asarray(current)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HDF5 trajectories into SpaRe-lite JSONL.")
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--instruction", default=None, help="Fallback instruction if not stored in HDF5.")
    parser.add_argument("--instruction-key", default=None, help="Dataset key containing per-episode instruction text.")
    parser.add_argument("--episode-group", default="data", help="Top-level group containing episodes.")
    parser.add_argument("--image-key", default="obs/agentview_rgb", help="Per-step image dataset key.")
    parser.add_argument("--action-key", default="actions", help="Per-step action dataset key.")
    parser.add_argument("--prompt-style", default="pure")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path)
    output_jsonl = Path(args.output_jsonl)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    num_records = 0
    with h5py.File(hdf5_path, "r") as f, output_jsonl.open("w", encoding="utf-8") as out_f:
        root = f[args.episode_group]
        episode_names = sorted(root.keys())
        if args.max_episodes is not None:
            episode_names = episode_names[: args.max_episodes]

        for episode_name in episode_names:
            episode = root[episode_name]

            instruction = args.instruction
            if args.instruction_key and args.instruction_key in episode:
                raw_instruction = episode[args.instruction_key][()]
                if isinstance(raw_instruction, bytes):
                    instruction = raw_instruction.decode("utf-8")
                else:
                    instruction = str(raw_instruction)
            if instruction is None:
                raise ValueError(
                    "Instruction is required. Pass --instruction or provide --instruction-key that exists in the HDF5."
                )

            step_keys = _iter_step_keys(episode)
            if step_keys:
                if args.max_steps_per_episode is not None:
                    step_keys = step_keys[: args.max_steps_per_episode]
                for step_name in step_keys:
                    step = episode["steps"][step_name] if "steps" in episode else episode[step_name]
                    image = _read_value(step, args.image_key)
                    action = _read_value(step, args.action_key).astype(np.float32).tolist()

                    image_path = image_dir / f"{episode_name}_{step_name}.png"
                    Image.fromarray(image).save(image_path)
                    record = {
                        "image_path": str(image_path),
                        "instruction": instruction,
                        "action": action,
                        "prompt_style": args.prompt_style,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    num_records += 1
                continue

            images = _read_value(episode, args.image_key)
            actions = _read_value(episode, args.action_key)
            pairs = zip(images, actions)
            if args.max_steps_per_episode is not None:
                pairs = list(pairs)[: args.max_steps_per_episode]
            for step_idx, (image, action) in enumerate(pairs):
                image_path = image_dir / f"{episode_name}_{step_idx:06d}.png"
                Image.fromarray(image).save(image_path)
                record = {
                    "image_path": str(image_path),
                    "instruction": instruction,
                    "action": np.asarray(action, dtype=np.float32).tolist(),
                    "prompt_style": args.prompt_style,
                }
                out_f.write(json.dumps(record) + "\n")
                num_records += 1

    print(f"exported {num_records} records to {output_jsonl}")


if __name__ == "__main__":
    main()
