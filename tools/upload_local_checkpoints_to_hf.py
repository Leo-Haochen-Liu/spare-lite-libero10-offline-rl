#!/usr/bin/env python3
"""Upload exported SpaRe-lite checkpoints to Hugging Face Hub.

Run after logging in with `hf auth login` or exporting `HF_TOKEN`.
The script is intentionally small so the exact artifact-to-repo mapping is
visible in version control.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi


LOCAL_ROOT = Path("/Users/haochenliu/Downloads/spare-lite-checkpoints-hf-upload-sftp")

UPLOADS = [
    (
        "libero10_full409_r1",
        "liuhaoch/Openvla-oft-SFT-libero10-traj1-offline-r1-libero10-full409",
        "SpaRe-lite R1 checkpoint from official SFT + LIBERO-10 full409 data",
    ),
    (
        "libero10_full409_r1r2",
        "liuhaoch/Openvla-oft-SFT-libero10-traj1-offline-r1r2-libero10-full409",
        "SpaRe-lite R1+R2 checkpoint from official SFT + LIBERO-10 full409 data",
    ),
]


def main() -> None:
    api = HfApi()
    print(api.whoami())

    for local_name, repo_id, description in UPLOADS:
        folder = LOCAL_ROOT / local_name
        if not folder.is_dir():
            raise FileNotFoundError(folder)

        print(f"Creating or reusing {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        readme = folder / "README.md"
        if readme.exists() and readme.stat().st_size <= 1:
            readme.write_text(
                f"# {repo_id.split('/')[-1]}\n\n"
                f"{description}.\n\n"
                "Base checkpoint: "
                "[Haozhan72/Openvla-oft-SFT-libero10-traj1]"
                "(https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero10-traj1).\n\n"
                "Reproduction package: "
                "https://github.com/Leo-Haochen-Liu/spare-lite-libero10-offline-rl\n",
                encoding="utf-8",
            )

        print(f"Uploading {folder} -> {repo_id}")
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=folder,
        )


if __name__ == "__main__":
    main()
