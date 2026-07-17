#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ROOT="${ADPILOT_MODEL_ROOT:-$HOME/models/adpilot}"
KEYFRAME_REPO="${ADPILOT_KEYFRAME_REPO:-black-forest-labs/FLUX.1-Kontext-dev}"
VIDEO_REPO="${ADPILOT_VIDEO_REPO:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}"
VLM_REPO="${ADPILOT_VLM_REPO:-Salesforce/blip-image-captioning-base}"
HF_CLI="${ADPILOT_HF_CLI:-hf}"
HF_MAX_WORKERS="${ADPILOT_HF_MAX_WORKERS:-2}"
export HF_ENDPOINT="${ADPILOT_HF_ENDPOINT:-${HF_ENDPOINT:-https://huggingface.co}}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "$MODEL_ROOT"

if ! python -c "import sentencepiece"; then
  echo "Installing sentencepiece required by FLUX Kontext"
  python -m pip install sentencepiece
fi

echo "HF endpoint: $HF_ENDPOINT"
echo "Model root: $MODEL_ROOT"
echo

echo "Downloading scene-integration model: $KEYFRAME_REPO"
echo "FLUX Kontext is gated. Accept its Hugging Face license and run hf auth login first."
"$HF_CLI" download "$KEYFRAME_REPO" \
  --local-dir "$MODEL_ROOT/flux-kontext-dev" \
  --max-workers "$HF_MAX_WORKERS"

echo "Downloading video model: $VIDEO_REPO"
echo "Wan2.2 is large. Download it on the login node, then run generation in a GPU allocation."
"$HF_CLI" download "$VIDEO_REPO" \
  --local-dir "$MODEL_ROOT/wan2.2-i2v-a14b-diffusers" \
  --max-workers "$HF_MAX_WORKERS"

echo "Downloading product caption model: $VLM_REPO"
"$HF_CLI" download "$VLM_REPO" \
  --local-dir "$MODEL_ROOT/blip-image-captioning-base" \
  --max-workers "$HF_MAX_WORKERS"

echo
echo "Done. Use these in GPU jobs:"
echo "export ADPILOT_KEYFRAME_MODEL=$MODEL_ROOT/flux-kontext-dev"
echo "export ADPILOT_VIDEO_MODEL=$MODEL_ROOT/wan2.2-i2v-a14b-diffusers"
echo "export ADPILOT_VLM_MODEL=$MODEL_ROOT/blip-image-captioning-base"
