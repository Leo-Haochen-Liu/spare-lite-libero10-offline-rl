from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoImageProcessor, AutoProcessor, SiglipImageProcessor

from .openvla_oft_utils import IGNORE_INDEX, build_openvla_prompt, build_openvla_supervision


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_existing_path(raw_path: str) -> str:
    candidate = Path(raw_path)
    if candidate.exists():
        return str(candidate)
    marker = "SpaRe-lite/spare_lite/"
    normalized = raw_path.replace("\\", "/")
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        remapped = PROJECT_ROOT / suffix
        if remapped.exists():
            return str(remapped)
    return raw_path


@dataclass
class JsonlSample:
    image_path: str
    next_image_path: str | None = None
    prompt: str | None = None
    target_text: str | None = None
    instruction: str | None = None
    action: list[float] | None = None
    action_tokens: str | None = None
    prompt_style: str | None = None
    spatial_image_path: str | None = None
    reward: float | None = None
    done: int | None = None
    quality: str | None = None
    rollout_success: bool | None = None
    episode_id: str | None = None
    step_idx: int | None = None
    return_to_go: float | None = None


class SpaReLiteJsonlDataset(Dataset):
    """
    Simple JSONL-backed dataset for SpaRe-lite.

    Each line should contain:

    {
      "image_path": "/abs/path/to/rgb.png",
      "prompt": "In: What action should the robot take to ...?\\nOut:",
      "target_text": "<action tokens or textual target>",
      "instruction": "pick up the cup",
      "action": [0.1, 0.0, -0.3, ...],
      "action_tokens": "<optional precomputed action-token string>",
      "prompt_style": "pure",
      "spatial_image_path": "/abs/path/to/other_or_same_rgb.png"  // optional
    }

    Supported modes:

    - legacy text-target mode: `prompt` + `target_text`
    - OpenVLA-style mode: `instruction` + (`action` or `action_tokens`)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        transition_spatial_source: str = "obs",
        return_gamma: float = 1.0,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.samples: list[JsonlSample] = []
        if transition_spatial_source not in {"obs", "next"}:
            raise ValueError("transition_spatial_source must be `obs` or `next`.")

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                image_path = raw.get("image_path") or raw.get("obs_image_path")
                if image_path is None:
                    raise ValueError("Each JSONL row must provide `image_path` or `obs_image_path`.")
                image_path = _resolve_existing_path(image_path)
                next_image_path = raw.get("next_obs_image_path")
                if next_image_path is not None:
                    next_image_path = _resolve_existing_path(next_image_path)
                if raw.get("spatial_image_path") is not None:
                    spatial_image_path = _resolve_existing_path(raw["spatial_image_path"])
                elif transition_spatial_source == "next":
                    spatial_image_path = _resolve_existing_path(raw.get("next_obs_image_path") or image_path)
                else:
                    spatial_image_path = _resolve_existing_path(raw.get("obs_image_path") or image_path)
                self.samples.append(
                    JsonlSample(
                        image_path=image_path,
                        next_image_path=next_image_path,
                        prompt=raw.get("prompt"),
                        target_text=raw.get("target_text"),
                        instruction=raw.get("instruction"),
                        action=raw.get("action"),
                        action_tokens=raw.get("action_tokens"),
                        prompt_style=raw.get("prompt_style"),
                        spatial_image_path=spatial_image_path,
                        reward=raw.get("reward"),
                        done=raw.get("done"),
                        quality=raw.get("quality"),
                        rollout_success=raw.get("rollout_success"),
                        episode_id=raw.get("episode_id"),
                        step_idx=raw.get("step_idx"),
                    )
                )
        self._attach_returns(return_gamma=return_gamma)

    def _attach_returns(self, return_gamma: float) -> None:
        running_by_episode: dict[str, float] = {}
        for sample in reversed(self.samples):
            if sample.reward is None:
                continue
            episode_id = sample.episode_id or "__default__"
            if sample.done:
                running_by_episode[episode_id] = 0.0
            running = float(sample.reward) + return_gamma * running_by_episode.get(episode_id, 0.0)
            sample.return_to_go = running
            running_by_episode[episode_id] = running

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        spatial_path = sample.spatial_image_path or sample.image_path
        spatial_image = Image.open(spatial_path).convert("RGB")
        next_image_path = sample.next_image_path or sample.image_path
        next_image = Image.open(next_image_path).convert("RGB")
        return {
            "image": image,
            "next_image": next_image,
            "spatial_image": spatial_image,
            "prompt": sample.prompt,
            "target_text": sample.target_text,
            "instruction": sample.instruction,
            "action": sample.action,
            "action_tokens": sample.action_tokens,
            "prompt_style": sample.prompt_style,
            "reward": sample.reward,
            "done": sample.done,
            "quality": sample.quality,
            "rollout_success": sample.rollout_success,
            "return_to_go": sample.return_to_go,
        }


class SpaReLiteCollator:
    def __init__(
        self,
        policy_model_name: str,
        spatial_model_name: str,
        spatial_backend: str = "auto",
        spatial_trust_remote_code: bool = True,
        trust_remote_code: bool = True,
        max_length: int = 512,
        default_prompt_style: str = "pure",
        action_tokenizer_bins: int = 256,
        action_tokenizer_min_action: int = -1,
        action_tokenizer_max_action: int = 1,
    ) -> None:
        self.policy_processor = AutoProcessor.from_pretrained(
            policy_model_name,
            trust_remote_code=trust_remote_code,
        )
        self.spatial_backend = spatial_backend
        self.spatial_intrinsic: torch.Tensor | None = None
        if spatial_backend == "spatialvla":
            self.spatial_processor = SiglipImageProcessor.from_pretrained(
                spatial_model_name,
                trust_remote_code=spatial_trust_remote_code,
            )
            processor_cfg_path = Path(spatial_model_name) / "processor_config.json"
            if processor_cfg_path.exists():
                processor_cfg = json.loads(processor_cfg_path.read_text())
                intrinsic_cfg = processor_cfg.get("intrinsic_config", {})
                selected = intrinsic_cfg.get("default")
                if selected is None and intrinsic_cfg:
                    selected = next(iter(intrinsic_cfg.values()))
                if selected is not None:
                    self.spatial_intrinsic = torch.tensor(selected["intrinsic"], dtype=torch.float32)
        else:
            self.spatial_processor = AutoImageProcessor.from_pretrained(spatial_model_name)
        self.max_length = max_length
        self.default_prompt_style = default_prompt_style
        self.action_tokenizer_bins = action_tokenizer_bins
        self.action_tokenizer_min_action = action_tokenizer_min_action
        self.action_tokenizer_max_action = action_tokenizer_max_action

    def _build_supervision(self, batch: list[dict[str, Any]]) -> tuple[list[str], torch.Tensor]:
        tokenizer = self.policy_processor.tokenizer
        specs = []
        for item in batch:
            specs.append(
                build_openvla_supervision(
                    tokenizer=tokenizer,
                    instruction=item.get("instruction"),
                    action=item.get("action"),
                    action_tokens=item.get("action_tokens"),
                    prompt=item.get("prompt"),
                    target_text=item.get("target_text"),
                    prompt_style=item.get("prompt_style") or self.default_prompt_style,
                    max_length=self.max_length,
                    bins=self.action_tokenizer_bins,
                    min_action=self.action_tokenizer_min_action,
                    max_action=self.action_tokenizer_max_action,
                )
            )

        full_texts = [spec.full_text for spec in specs]
        tokenized = tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = tokenized["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = IGNORE_INDEX

        for row_idx, spec in enumerate(specs):
            labels[row_idx, : len(spec.labels)] = torch.tensor(spec.labels, dtype=torch.long)

        return full_texts, labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        next_images = [item["next_image"] for item in batch]
        spatial_images = [item["spatial_image"] for item in batch]
        full_texts, labels = self._build_supervision(batch)
        query_texts = [
            build_openvla_prompt(
                instruction=item.get("instruction") or "",
                prompt_style=item.get("prompt_style") or self.default_prompt_style,
            )
            if item.get("instruction")
            else (item.get("prompt") or "")
            for item in batch
        ]

        policy_inputs = self.policy_processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        policy_inputs["labels"] = labels
        policy_query_inputs = self.policy_processor(
            text=query_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        next_policy_query_inputs = self.policy_processor(
            text=query_texts,
            images=next_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if self.spatial_backend == "spatialvla":
            spatial_pixel_values = self.spatial_processor(
                spatial_images,
                return_tensors="pt",
            )["pixel_values"]
        else:
            spatial_pixel_values = self.spatial_processor(
                images=spatial_images,
                return_tensors="pt",
            )["pixel_values"]

        out = {
            "policy_inputs": policy_inputs,
            "policy_query_inputs": policy_query_inputs,
            "next_policy_query_inputs": next_policy_query_inputs,
            "spatial_pixel_values": spatial_pixel_values,
        }
        if any(item.get("reward") is not None for item in batch):
            out["reward"] = torch.tensor(
                [float(item.get("reward") or 0.0) for item in batch],
                dtype=torch.float32,
            )
        if any(item.get("done") is not None for item in batch):
            out["done"] = torch.tensor(
                [float(item.get("done") or 0.0) for item in batch],
                dtype=torch.float32,
            )
        if any(item.get("return_to_go") is not None for item in batch):
            out["return_to_go"] = torch.tensor(
                [float(item.get("return_to_go") or 0.0) for item in batch],
                dtype=torch.float32,
            )
        out["actor_weight"] = torch.tensor(
            [
                0.0
                if item.get("quality") == "failure" or item.get("rollout_success") is False
                else 1.0
                for item in batch
            ],
            dtype=torch.float32,
        )
        if self.spatial_intrinsic is not None:
            out["spatial_intrinsic"] = self.spatial_intrinsic.unsqueeze(0).repeat(len(batch), 1, 1)
        return out


def build_jsonl_dataloader(
    jsonl_path: str | Path,
    policy_model_name: str,
    spatial_model_name: str,
    spatial_backend: str = "auto",
    spatial_trust_remote_code: bool = True,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    trust_remote_code: bool = True,
    max_length: int = 512,
    default_prompt_style: str = "pure",
    transition_spatial_source: str = "obs",
    return_gamma: float = 1.0,
    positive_sample_boost: float = 1.0,
) -> DataLoader:
    dataset = SpaReLiteJsonlDataset(
        jsonl_path,
        transition_spatial_source=transition_spatial_source,
        return_gamma=return_gamma,
    )
    collator = SpaReLiteCollator(
        policy_model_name=policy_model_name,
        spatial_model_name=spatial_model_name,
        spatial_backend=spatial_backend,
        spatial_trust_remote_code=spatial_trust_remote_code,
        trust_remote_code=trust_remote_code,
        max_length=max_length,
        default_prompt_style=default_prompt_style,
    )
    sampler = None
    if positive_sample_boost > 1.0:
        weights = []
        for sample in dataset.samples:
            reward = float(sample.reward or 0.0)
            is_positive = reward > 0.0 and sample.quality != "failure" and sample.rollout_success is not False
            weights.append(positive_sample_boost if is_positive else 1.0)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
    )
