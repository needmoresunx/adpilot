#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ROOT="${ADPILOT_MODEL_ROOT:-$HOME/models/adpilot}"
KEYFRAME_REPO="${ADPILOT_KEYFRAME_REPO:-black-forest-labs/FLUX.1-Kontext-dev}"
VIDEO_REPO="${ADPILOT_VIDEO_REPO:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}"
VLM_REPO="${ADPILOT_VLM_REPO:-Qwen/Qwen2.5-VL-3B-Instruct}"
HF_CLI="${ADPILOT_HF_CLI:-hf}"
HF_MAX_WORKERS="${ADPILOT_HF_MAX_WORKERS:-2}"
U2NET_HOME="${U2NET_HOME:-$HOME/.u2net}"
BIREFNET_URL="${ADPILOT_BIREFNET_URL:-https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx}"
export HF_ENDPOINT="${ADPILOT_HF_ENDPOINT:-${HF_ENDPOINT:-https://huggingface.co}}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "$MODEL_ROOT"

model_weight_bytes() {
  local model_dir="$1"
  if [ ! -d "$model_dir" ]; then
    echo 0
    return
  fi
  find "$model_dir" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' \) -printf '%s\n' 2>/dev/null \
    | awk '{total += $1} END {print total + 0}'
}

download_if_incomplete() {
  local repo="$1"
  local destination="$2"
  local minimum_bytes="$3"
  local label="$4"
  local bytes
  bytes="$(model_weight_bytes "$destination")"
  if [ "$bytes" -ge "$minimum_bytes" ]; then
    echo "$label already has model weights; skipping download."
    return
  fi
  echo "Downloading $label: $repo"
  "$HF_CLI" download "$repo" --local-dir "$destination" --max-workers "$HF_MAX_WORKERS"
}

download_birefnet() {
  local destination="$U2NET_HOME/birefnet-general.onnx"
  local legacy="$U2NET_HOME/BiRefNet-general-epoch_244.onnx"
  if [ -s "$destination" ]; then
    echo "Product segmentation model already exists; skipping download."
    return
  fi
  if [ -s "$legacy" ]; then
    ln -s "$(basename "$legacy")" "$destination"
    echo "Linked existing product segmentation model; skipping download."
    return
  fi
  mkdir -p "$U2NET_HOME"
  echo "Downloading product segmentation model: BiRefNet-general"
  curl --fail --location --retry 3 --output "$destination" "$BIREFNET_URL"
}

if ! python -c "import sentencepiece, qwen_vl_utils"; then
  echo "Missing runtime dependencies. Run python -m pip install -r requirements.txt first." >&2
  exit 1
fi

echo "HF endpoint: $HF_ENDPOINT"
echo "Model root: $MODEL_ROOT"
echo

echo "FLUX Kontext is gated. Accept its Hugging Face license and run hf auth login first."
download_if_incomplete "$KEYFRAME_REPO" "$MODEL_ROOT/flux-kontext-dev" "$((8 * 1024 * 1024 * 1024))" "scene-integration model"

echo "Wan2.2 is large; make sure the destination has sufficient disk space."
download_if_incomplete "$VIDEO_REPO" "$MODEL_ROOT/wan2.2-i2v-a14b-diffusers" "$((50 * 1024 * 1024 * 1024))" "video model"

download_if_incomplete "$VLM_REPO" "$MODEL_ROOT/qwen2.5-vl-3b-instruct" "$((2 * 1024 * 1024 * 1024))" "product-analysis VLM"
download_birefnet

echo
echo "Done. Create an agent project with python -m adpilot.agent --help."
