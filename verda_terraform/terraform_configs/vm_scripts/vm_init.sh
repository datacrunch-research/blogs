#!/bin/bash

export HOST_MODEL_PATH="/mnt/local_nvme/models"
export C_MODEL_PATH="/root/models"
export HF_TOKEN=$HF_TOKEN

# Create a local NVMe volume
mkdir /mnt/local_nvme
sudo parted -s /dev/vdb mklabel gpt
sudo parted -s /dev/vdb mkpart primary ext4 0% 100%
sudo partprobe /dev/vdb
sudo mkfs.ext4 -F /dev/vdb1
sudo mount /dev/vdb1 /mnt/local_nvme

mkdir -p $HOST_MODEL_PATH


# Docker setup (store artifacts in local NVMe)
sudo mkdir -p /mnt/local_nvme/docker
sudo echo '{
    "data-root": "/mnt/local_nvme/docker",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
' > /etc/docker/daemon.json

# Restart docker to apply the changes
sudo systemctl restart docker

# sglang setup
export NCCL_DEBUG="DEBUG"
export FLASHINFER_WORKSPACE_BASE="/nfs_home"
export SGLANG_TORCH_PROFILER_DIR="/nfs_home/profile_result" # Or another directory for storing profiler results
       
# check image version in https://hub.docker.com/r/lmsysorg/sglang/tags
docker run --gpus all --shm-size 32g --network=host --name sglang_server -d --ipc=host \
  -v "$HOST_MODEL_PATH:$C_MODEL_PATH" \
  -e HF_TOKEN="$HF_TOKEN" \
  lmsysorg/sglang:dev-cu13 \
  bash -lc "
    huggingface-cli download nvidia/DeepSeek-R1-0528-NVFP4-v2 --cache-dir "$C_MODEL_PATH"
    exec python3 -m sglang.launch_server \
      --model-path "$C_MODEL_PATH/models--nvidia--DeepSeek-R1-0528-NVFP4-v2/snapshots/25a138f28f49022958b9f2d205f9b7de0cdb6e18/" \
      --served-model-name dsr1 \
      --tp 4 \
      --attention-backend trtllm_mla \
      --disable-radix-cache \
      --moe-runner-backend flashinfer_trtllm \
      --quantization modelopt_fp4 \
      --kv-cache-dtype fp8_e4m3
  "


# Check logs: docker logs -f sglang_server