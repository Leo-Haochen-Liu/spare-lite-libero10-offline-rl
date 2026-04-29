# Data Notes

`libero10_expert_plus_sft_failures_409.jsonl` is the final dataset used for the
strict LIBERO-10 SFT-baseline comparison.

It combines:

- Full LIBERO-10 expert transitions.
- 409 real failed rollout episodes from the official SFT policy.

The manifest records exact transition counts and task distribution:

`libero10_expert_plus_sft_failures_409_manifest.json`

The JSONL contains image-path fields from the original remote extraction layout.
For a fresh machine, either recreate that layout or remap the paths after
regenerating images with the extraction utilities in `spare_lite/`.

This file is tracked via Git LFS because it is larger than GitHub's normal single
file limit.

Use:

```bash
git lfs install
git lfs pull
```

after cloning the repository to materialize the JSONL instead of the small LFS
pointer file.
