from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

IGNORE_INDEX = -100
OPENVLA_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)


class ActionTokenizer:
    """
    Lightweight copy of the OpenVLA-OFT action tokenizer.

    It discretizes continuous actions into the last `bins` tokens of the base
    tokenizer vocabulary, matching the upstream implementation.
    """

    def __init__(
        self,
        tokenizer: Any,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action

        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> str | list[str]:
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.bin_centers.shape[0] - 1,
        )
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins


def build_openvla_query(instruction: str, lowercase: bool = True) -> str:
    normalized = instruction.strip()
    if lowercase:
        normalized = normalized.lower()
    return f"What action should the robot take to {normalized}?"


def build_openvla_prompt(
    instruction: str,
    prompt_style: str = "pure",
    lowercase: bool = True,
) -> str:
    query = build_openvla_query(instruction, lowercase=lowercase)
    if prompt_style == "pure":
        return f"In: {query}\nOut: "
    if prompt_style == "vicuna_v15":
        return f"{OPENVLA_SYSTEM_PROMPT}USER: {query} ASSISTANT: "
    raise ValueError(f"Unsupported prompt_style: {prompt_style}")


def _prefix_length(full_ids: list[int], prefix_ids: list[int]) -> int:
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return len(prefix_ids)

    limit = min(len(full_ids), len(prefix_ids))
    matched = 0
    for idx in range(limit):
        if full_ids[idx] != prefix_ids[idx]:
            break
        matched += 1
    return matched


@dataclass
class OpenVLASupervision:
    prompt_text: str
    target_text: str
    full_text: str
    input_ids: list[int]
    labels: list[int]


def build_openvla_supervision(
    tokenizer: Any,
    instruction: str | None = None,
    action: list[float] | np.ndarray | None = None,
    action_tokens: str | None = None,
    prompt: str | None = None,
    target_text: str | None = None,
    prompt_style: str = "pure",
    lowercase_instruction: bool = True,
    predict_stop_token: bool = True,
    max_length: int = 512,
    bins: int = 256,
    min_action: int = -1,
    max_action: int = 1,
) -> OpenVLASupervision:
    if prompt is None:
        if instruction is None:
            raise ValueError("Expected either `prompt` or `instruction`.")
        prompt = build_openvla_prompt(
            instruction=instruction,
            prompt_style=prompt_style,
            lowercase=lowercase_instruction,
        )

    if target_text is None:
        if action_tokens is not None:
            target_text = action_tokens
        elif action is not None:
            action_tokenizer = ActionTokenizer(
                tokenizer=tokenizer,
                bins=bins,
                min_action=min_action,
                max_action=max_action,
            )
            target_text = str(action_tokenizer(np.asarray(action, dtype=np.float32)))
        else:
            raise ValueError("Expected `target_text`, `action_tokens`, or `action`.")

    full_text = prompt + target_text
    full_ids = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    ).input_ids
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    ).input_ids

    prefix_len = _prefix_length(full_ids, prompt_ids)
    labels = list(full_ids)
    for idx in range(prefix_len):
        labels[idx] = IGNORE_INDEX

    if not predict_stop_token and labels:
        labels[-1] = IGNORE_INDEX

    return OpenVLASupervision(
        prompt_text=prompt,
        target_text=target_text,
        full_text=full_text,
        input_ids=list(full_ids),
        labels=labels,
    )
