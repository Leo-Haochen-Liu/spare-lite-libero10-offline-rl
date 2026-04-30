# Runtime Environment Used for the Reported Runs

This file records the environment used for the final LIBERO-10 offline-RL
experiments. The repository code is intentionally lightweight; SimpleVLA-RL,
LIBERO, SpatialVLA, and the official checkpoints are external dependencies.

## Hardware

The final AutoDL runs used:

- GPU: 2 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each.
- CPU: Intel Xeon Platinum 8470Q, 208 logical CPUs.
- RAM: approximately 1.0 TiB.
- Training/evaluation mode: two independent single-GPU jobs were often run in
  parallel, one per GPU.
- LIBERO evaluation batch size: `VAL_BATCH_SIZE=1`.

## External Repositories and Assets

- SimpleVLA-RL: [`PRIME-RL/SimpleVLA-RL`](https://github.com/PRIME-RL/SimpleVLA-RL)
  at commit `7c51662df27b586f9e8a1ab35fcf849f2b8852f9`.
- LIBERO: installed separately and added to `PYTHONPATH` for rollout evaluation.
- SpatialVLA backend checkpoint: local path used in the run was
  `/root/autodl-tmp/checkpoints/spatialvla-4b-224-pt`.
- Official SFT checkpoint:
  [`Haozhan72/Openvla-oft-SFT-libero10-traj1`](https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero10-traj1).
- Official post-RL checkpoint:
  [`Haozhan72/openvla-oft-libero10-traj1-rl`](https://huggingface.co/Haozhan72/openvla-oft-libero10-traj1-rl).

## Python Environment

The final remote environment was a Conda environment named
`simplevla-official` with Python 3.10.20. Core packages observed in the run
environment were:

| Package | Version |
|---|---:|
| `python` | 3.10.20 |
| `torch` | 2.7.1+cu128 |
| `transformers` | 4.40.1 |
| `tokenizers` | 0.19.1 |
| `ray` | 2.55.0 |
| `accelerate` | 1.13.0 |
| `peft` | 0.11.1 |
| `numpy` | 1.26.4 |
| `h5py` | 3.16.0 |
| `diffusers` | 0.37.1 |
| `einops` | 0.8.2 |
| `safetensors` | 0.7.0 |
| `tensorflow` | 2.15.0 |
| `timm` | 0.9.10 |

The exact package set may include additional dependencies installed by
SimpleVLA-RL, LIBERO, MuJoCo, and the OpenVLA/OFT model code. The two most
important compatibility pins for loading the official OpenVLA-OFT checkpoints
were `transformers==4.40.1` and `tokenizers==0.19.1`.

## Environment Variables

The LIBERO rollout evaluation used headless rendering:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export PYTHONPATH=/root/autodl-tmp/SpaRe-lite:/root/autodl-tmp/LIBERO:$PYTHONPATH
```

## Runtime Patch

The only SimpleVLA-RL code patch included in this repository is
`patches/simplevla_rl_libero_spawn_runtime.patch`. It changes the LIBERO worker
multiprocessing start method from fork-style startup to spawn-style startup to
avoid CUDA initialization failures inside Ray/LIBERO worker processes.

This patch does not change model weights, action decoding, reward computation,
rollout success criteria, or benchmark task definitions.

