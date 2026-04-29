#!/usr/bin/env bash
set -euo pipefail

export COPYFILE_DISABLE=1

REMOTE_HOST="${REMOTE_HOST:-connect.westb.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-20984}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_PASS="${REMOTE_PASS:-8YniZj26kHs9}"

LOCAL_SPARE_DIR="${LOCAL_SPARE_DIR:-/Users/haochenliu/Documents/research/SpaRe-lite/spare_lite}"
LOCAL_DINOV2="${LOCAL_DINOV2:-/Users/haochenliu/Downloads/hf_models/facebook-dinov2-base}"
LOCAL_TINY_JSONL="${LOCAL_TINY_JSONL:-/Users/haochenliu/Downloads/libero_demo_cache/small_train.jsonl}"
LOCAL_TINY_IMAGES="${LOCAL_TINY_IMAGES:-/Users/haochenliu/Downloads/libero_demo_cache/small_images}"
LOCAL_MEDIUM_JSONL="${LOCAL_MEDIUM_JSONL:-${LOCAL_SPARE_DIR}/local_data/medium_train.jsonl}"
LOCAL_MEDIUM_IMAGES="${LOCAL_MEDIUM_IMAGES:-${LOCAL_SPARE_DIR}/local_data/medium_images}"
LOCAL_MEDIUM320_JSONL="${LOCAL_MEDIUM320_JSONL:-${LOCAL_SPARE_DIR}/local_data/medium320_train_remote.jsonl}"
LOCAL_MEDIUM320_IMAGES="${LOCAL_MEDIUM320_IMAGES:-${LOCAL_SPARE_DIR}/local_data/medium320_images}"
LOCAL_LARGE_JSONL="${LOCAL_LARGE_JSONL:-${LOCAL_SPARE_DIR}/local_data/large_train_remote.jsonl}"
LOCAL_LARGE_IMAGES="${LOCAL_LARGE_IMAGES:-${LOCAL_SPARE_DIR}/local_data/large_images}"

REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/SpaRe-lite}"
REMOTE_DINOV2="${REMOTE_DINOV2:-/root/autodl-tmp/checkpoints/facebook-dinov2-base}"
REMOTE_TINY="${REMOTE_TINY:-/root/autodl-tmp/data/libero_small}"
REMOTE_MEDIUM="${REMOTE_MEDIUM:-/root/autodl-tmp/data/libero_medium}"
REMOTE_MEDIUM320="${REMOTE_MEDIUM320:-/root/autodl-tmp/data/libero_medium320}"
REMOTE_LARGE="${REMOTE_LARGE:-/root/autodl-tmp/data/libero_large}"

SSH_OPTS=(-p "${REMOTE_PORT}" -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
SCP_OPTS=(-P "${REMOTE_PORT}" -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

export SSHPASS="${REMOTE_PASS}"

sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_ROOT}/spare_lite' '${REMOTE_DINOV2}' '${REMOTE_TINY}/images' '${REMOTE_MEDIUM}/images' '${REMOTE_MEDIUM320}/images' '${REMOTE_LARGE}/images'"

tar -C "${LOCAL_SPARE_DIR}" \
  --exclude='./._*' \
  --exclude='./local_data' \
  --exclude='./__pycache__' \
  --exclude='./*.pyc' \
  -cf - . | sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "tar -C '${REMOTE_ROOT}/spare_lite' -xf -"

if ! sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "test -f '${REMOTE_DINOV2}/config.json'"; then
  sshpass -e scp "${SCP_OPTS[@]}" -r "${LOCAL_DINOV2}/." \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DINOV2}/"
else
  echo "skip dinov2 transfer: remote checkpoint already present"
fi

if ! sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "test -f '${REMOTE_TINY}/small_train.jsonl'"; then
  sshpass -e scp "${SCP_OPTS[@]}" "${LOCAL_TINY_JSONL}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TINY}/small_train.jsonl"
else
  echo "skip tiny jsonl transfer: remote file already present"
fi

if ! sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "find '${REMOTE_TINY}/images' -type f | grep -q ."; then
  sshpass -e scp "${SCP_OPTS[@]}" -r "${LOCAL_TINY_IMAGES}/." \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TINY}/images/"
else
  echo "skip tiny image transfer: remote images already present"
fi

if ! sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "test -f '${REMOTE_MEDIUM}/medium_train.jsonl'"; then
  sshpass -e scp "${SCP_OPTS[@]}" "${LOCAL_MEDIUM_JSONL}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MEDIUM}/medium_train.jsonl"
else
  echo "skip medium jsonl transfer: remote file already present"
fi

if ! sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "find '${REMOTE_MEDIUM}/images' -type f | grep -q ."; then
  sshpass -e scp "${SCP_OPTS[@]}" -r "${LOCAL_MEDIUM_IMAGES}/." \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MEDIUM}/images/"
else
  echo "skip medium image transfer: remote images already present"
fi

if ! sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "test -f '${REMOTE_MEDIUM320}/medium320_train.jsonl'"; then
  sshpass -e scp "${SCP_OPTS[@]}" "${LOCAL_MEDIUM320_JSONL}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MEDIUM320}/medium320_train.jsonl"
else
  echo "skip medium320 jsonl transfer: remote file already present"
fi

if ! sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "find '${REMOTE_MEDIUM320}/images' -type f | grep -q ."; then
  sshpass -e scp "${SCP_OPTS[@]}" -r "${LOCAL_MEDIUM320_IMAGES}/." \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MEDIUM320}/images/"
else
  echo "skip medium320 image transfer: remote images already present"
fi

if ! sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "test -f '${REMOTE_LARGE}/large_train.jsonl'"; then
  sshpass -e scp "${SCP_OPTS[@]}" "${LOCAL_LARGE_JSONL}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_LARGE}/large_train.jsonl"
else
  echo "skip large jsonl transfer: remote file already present"
fi

if ! sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "find '${REMOTE_LARGE}/images' -type f | grep -q ."; then
  sshpass -e scp "${SCP_OPTS[@]}" -r "${LOCAL_LARGE_IMAGES}/." \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_LARGE}/images/"
else
  echo "skip large image transfer: remote images already present"
fi

sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "cd '${REMOTE_TINY}' && /root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

for jsonl_name, image_root in [
    ('/root/autodl-tmp/data/libero_small/small_train.jsonl', '/root/autodl-tmp/data/libero_small/images'),
    ('/root/autodl-tmp/data/libero_medium/medium_train.jsonl', '/root/autodl-tmp/data/libero_medium/images'),
    ('/root/autodl-tmp/data/libero_medium320/medium320_train.jsonl', '/root/autodl-tmp/data/libero_medium320/images'),
    ('/root/autodl-tmp/data/libero_large/large_train.jsonl', '/root/autodl-tmp/data/libero_large/images'),
]:
    p = Path(jsonl_name)
    if not p.exists():
        continue
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row['image_path'] = str(Path(image_root) / Path(row['image_path']).name)
        rows.append(row)
    p.write_text('\\n'.join(json.dumps(r) for r in rows) + '\\n')
    print('rewrote', len(rows), jsonl_name)
PY"

echo "remote assets transferred"
