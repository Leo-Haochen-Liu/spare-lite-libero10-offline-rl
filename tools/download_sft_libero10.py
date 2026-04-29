from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Haozhan72/Openvla-oft-SFT-libero10-traj1',
    repo_type='model',
    local_dir='/root/autodl-tmp/checkpoints/Openvla-oft-SFT-libero10-traj1',
    endpoint='https://hf-mirror.com',
    resume_download=True,
)
print('DONE')
