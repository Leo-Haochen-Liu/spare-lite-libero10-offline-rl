from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForVision2Seq


def copy_support_files(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.name.endswith(".safetensors") or item.name.endswith(".bin"):
            continue
        if item.name.endswith(".safetensors.index.json") or item.name.endswith(".bin.index.json"):
            continue
        target = dst_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--partial-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    base_model_dir = Path(args.base_model_dir)
    partial_ckpt = Path(args.partial_ckpt)
    output_dir = Path(args.output_dir)

    payload = torch.load(partial_ckpt, map_location="cpu")
    policy_state = payload["policy_trainable_state_dict"]

    config = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)
    if not hasattr(config, "use_proprio"):
        config.use_proprio = False
    if not hasattr(config, "proprio_dim"):
        config.proprio_dim = 0

    model = AutoModelForVision2Seq.from_pretrained(
        base_model_dir,
        trust_remote_code=True,
        config=config,
    )

    current_state = model.state_dict()
    missing = []
    for name, tensor in policy_state.items():
        if name not in current_state:
            missing.append(name)
            continue
        current_state[name] = tensor
    if missing:
        raise KeyError(f"Some checkpoint parameters were not found in base model: {missing[:10]}")

    model.load_state_dict(current_state, strict=False)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    copy_support_files(base_model_dir, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
