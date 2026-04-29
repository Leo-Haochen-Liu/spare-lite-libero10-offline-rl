# SpatialVLA Ego3D Integration Checklist

Last updated: 2026-04-25

## Goal

Replace the current DINOv2-based reference branch in `spare_lite` with the real
SpatialVLA ego3D encoder path.

## Why this track exists

- The current proxy path is useful for fast validation.
- But the current `z_ref` is still DINOv2-based rather than SpatialVLA ego3D-based.
- So current `R2` is still not the final intended reward signal.

## Current status

- This track has started conceptually.
- The current repo now contains an initial local implementation path for the
  real SpatialVLA reference branch under `spare_lite`.
- `spare_lite/modeling.py` now supports a `spatialvla` backend in addition to
  the generic image-encoder backend.
- `spare_lite/adapters.py` now supports passing SpatialVLA-style intrinsic data
  together with spatial images.
- `spare_lite/run_spatialvla_reference_smoke.py` now exists as the minimal smoke
  entry for validating that the reference branch can load and produce `z_ref`.

## Concrete implementation tasks

1. Confirm the exact SpatialVLA model checkpoint path we want to use.
2. Run the new smoke script against a real checkpoint and verify that:
   - the processor loads
   - intrinsic is available
   - `z_ref` is produced without shape mismatch
3. Finalize the pooled feature choice for the reference latent.
4. Replace or extend the current spatial branch in:
   - `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/modeling.py`
5. Verify that the reference latent dimension matches the existing reward path,
   or add the necessary projection layer.
6. After the smoke check passes, rerun the proxy validation with the real
   SpatialVLA ego3D reference branch.

## Files most likely to change

- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/modeling.py`
- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/config.py`
- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/adapters.py`
- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/rl_spare_latent_smoke.py`
- `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent_sweep.sh`

## Current blocker

- The current code path is still designed around a generic Hugging Face image
  encoder, not the final SpatialVLA ego3D implementation.
- So the next concrete step is not another reward tweak. The next concrete step
  is to pin down the exact encoder-loading interface and required preprocessing.
