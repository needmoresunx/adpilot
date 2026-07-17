#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip

# CUDA 12.1 wheels are broadly compatible with recent NVIDIA drivers.
# If your cluster requires another CUDA wheel index, set TORCH_INDEX_URL first.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
python -m pip install transformers accelerate safetensors opencv-python imageio ftfy sentencepiece "rembg[cpu]"

# Wan2.2 support may require a recent Diffusers build. Installing from GitHub is
# slower, but it avoids the common "WanImageToVideoPipeline not found" failure.
python -m pip install --upgrade git+https://github.com/huggingface/diffusers.git

python scripts/check_env.py
