from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Haozhan72/Openvla-oft-SFT-libero10-traj1",
    filename="model-00001-of-00004.safetensors",
    repo_type="model",
    local_dir="/root/autodl-tmp/checkpoints/Openvla-oft-SFT-libero10-traj1",
    endpoint="https://hf-mirror.com",
    resume_download=True,
)
