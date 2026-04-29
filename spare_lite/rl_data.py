from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class RLSample:
    image_path: str
    prompt: str
    candidates: list[str]
    env_rewards: list[float]
    spatial_image_path: str | None = None


class SpaReLiteRLDataset(Dataset):
    """
    JSONL dataset for lightweight RL-style validation.

    Each line should contain:

    {
      "image_path": "/abs/path/to/image.png",
      "prompt": "task instruction",
      "candidates": ["action 1", "action 2", ...],
      "env_rewards": [0.0, 1.0, ...],
      "spatial_image_path": "/abs/path/to/optional/spatial_image.png"
    }
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.samples: list[RLSample] = []

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                candidates = list(raw["candidates"])
                env_rewards = list(raw["env_rewards"])
                if len(candidates) != len(env_rewards):
                    raise ValueError("candidates and env_rewards must have the same length")
                self.samples.append(
                    RLSample(
                        image_path=raw["image_path"],
                        prompt=raw["prompt"],
                        candidates=candidates,
                        env_rewards=env_rewards,
                        spatial_image_path=raw.get("spatial_image_path"),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        spatial_path = sample.spatial_image_path or sample.image_path
        spatial_image = Image.open(spatial_path).convert("RGB")
        return {
            "image": image,
            "spatial_image": spatial_image,
            "prompt": sample.prompt,
            "candidates": sample.candidates,
            "env_rewards": sample.env_rewards,
        }
