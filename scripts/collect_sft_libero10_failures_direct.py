from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from libero.libero import benchmark

from verl.utils.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    normalize_gripper_action,
)
from verl.workers.rollout.rob_rollout import center_crop_image
from verl.utils.vla_utils.openvla_oft.configuration_prismatic import OpenVLAConfig
from verl.utils.vla_utils.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
from verl.utils.vla_utils.openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def register_openvla_oft() -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def load_policy(model_path: str):
    register_openvla_oft()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_proprio = False
    config.proprio_dim = 7
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()
    if hasattr(model, "vision_backbone"):
        model.vision_backbone.set_num_images_in_input(1)
    stats_path = Path(model_path) / "dataset_statistics.json"
    if stats_path.exists():
        model.norm_stats = json.loads(stats_path.read_text())
    return model, processor


def prepare_inputs(processor, obs, task_description: str, center_crop: bool = True):
    image = Image.fromarray(get_libero_image(obs, 224)).convert("RGB")
    if center_crop:
        image = center_crop_image(image)
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    batch = processor(prompt, image)
    input_ids = batch["input_ids"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")
    pixel_values = batch["pixel_values"].to("cuda", dtype=torch.bfloat16)
    if not torch.all(input_ids[:, -1] == 29871):
        suffix = torch.tensor([[29871]], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat((input_ids, suffix), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(suffix, dtype=attention_mask.dtype)), dim=1)
    return input_ids, attention_mask, pixel_values


@torch.inference_mode()
def predict_action_chunk(model, processor, obs, task_description: str, unnorm_key: str, temperature: float):
    input_ids, attention_mask, pixel_values = prepare_inputs(processor, obs, task_description)
    actions, _ = model.predict_action(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        unnorm_key=unnorm_key,
        do_sample=True,
        temperature=temperature,
    )
    actions = np.asarray(actions)
    if actions.ndim == 3:
        return actions[0]
    return actions


def save_failed_episode(out_dir: Path, episode_id: str, task_name: str, task_description: str, rows: list[dict]) -> None:
    image_dir = out_dir / "images" / episode_id
    episode_dir = out_dir / "episodes"
    image_dir.mkdir(parents=True, exist_ok=True)
    episode_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = episode_dir / f"{episode_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            obs_path = image_dir / f"{idx:06d}.png"
            next_obs_path = image_dir / f"{idx:06d}_next.png"
            Image.fromarray(row.pop("obs_image")).save(obs_path)
            Image.fromarray(row.pop("next_obs_image")).save(next_obs_path)
            row.update(
                {
                    "source": "direct_online_rollout",
                    "quality": "failure",
                    "task_name": task_name,
                    "episode_id": episode_id,
                    "instruction": task_description,
                    "obs_image_path": str(obs_path),
                    "next_obs_image_path": str(next_obs_path),
                    "reward": 0.0,
                    "done": 1 if idx == len(rows) - 1 else 0,
                    "rollout_success": False,
                }
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"SAVED_FAILURE episode={episode_id} transitions={len(rows)} path={jsonl_path}", flush=True)


def rollout_one(model, processor, task_suite, task_suite_name: str, task_id: int, trial_id: int, args) -> bool:
    task = task_suite.get_task(task_id)
    init_state = task_suite.get_task_init_states(task_id)[trial_id]
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    rows: list[dict] = []
    success = False
    try:
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

        step_idx = 0
        while step_idx < args.max_steps:
            action_chunk = predict_action_chunk(
                model,
                processor,
                obs,
                task_description,
                unnorm_key=args.unnorm_key,
                temperature=args.temperature,
            )
            for action in action_chunk:
                obs_image = get_libero_image(obs, 224)
                normalized_action = normalize_gripper_action(np.asarray(action), binarize=True)
                inverted_action = invert_gripper_action(normalized_action)
                next_obs, _, done, _ = env.step(inverted_action.tolist())
                next_image = get_libero_image(next_obs, 224)
                if len(rows) < args.max_transitions_per_episode:
                    rows.append(
                        {
                            "step_idx": step_idx,
                            "action": np.asarray(action, dtype=float).tolist(),
                            "obs_image": obs_image,
                            "next_obs_image": next_image,
                        }
                    )
                obs = next_obs
                step_idx += 1
                if done or step_idx >= args.max_steps:
                    success = bool(done)
                    break
            if success or step_idx >= args.max_steps:
                break
    finally:
        env.close()

    episode_id = f"{task_suite_name}_task_{task_id}_trial_{trial_id}_success_{success}_direct"
    print(f"ROLLOUT_DONE episode={episode_id} success={success} recorded={len(rows)}", flush=True)
    if not success and rows:
        save_failed_episode(Path(args.output_dir), episode_id, f"{task_suite_name}_task_{task_id}", task_description, rows)
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--unnorm-key", default="libero_10_no_noops")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--max-failures", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-transitions-per-episode", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.6)
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--end-task", type=int, default=None)
    parser.add_argument("--start-trial", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, processor = load_policy(args.model_path)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    failures = 0
    total = 0
    end_task = task_suite.n_tasks if args.end_task is None else min(args.end_task, task_suite.n_tasks)
    for task_id in range(args.start_task, end_task):
        trial_start = args.start_trial if task_id == args.start_task else 0
        for trial_id in range(trial_start, args.num_trials_per_task):
            success = rollout_one(model, processor, task_suite, args.task_suite_name, task_id, trial_id, args)
            total += 1
            failures += int(not success)
            print(f"PROGRESS total={total} failures={failures} last_task={task_id} last_trial={trial_id}", flush=True)
            if failures >= args.max_failures:
                print(f"DONE reached max_failures={args.max_failures}", flush=True)
                return


if __name__ == "__main__":
    main()
