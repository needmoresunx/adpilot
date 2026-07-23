#!/usr/bin/env bash
# Shared strict GPU runner. Source this from a product-specific run script.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${PRODUCT_PATH:?Set PRODUCT_PATH in the product demo script.}"
: "${BRAND:?Set BRAND in the product demo script.}"
: "${PRODUCT_CATEGORY:?Set PRODUCT_CATEGORY in the product demo script.}"
: "${PRODUCT_DESCRIPTION:?Set PRODUCT_DESCRIPTION in the product demo script.}"
: "${TARGET_AUDIENCE:?Set TARGET_AUDIENCE in the product demo script.}"
: "${AD_MOOD:?Set AD_MOOD in the product demo script.}"

OUT_DIR="${OUT_DIR:-outputs}"
REFERENCE_MODE="${REFERENCE_MODE:-front_lock}"
# For multi_view, declare extra clean photos in the product script, for example:
# ADDITIONAL_REFERENCE_IMAGES=("examples/product_45.jpg" "examples/product_side.jpg")
if ! declare -p ADDITIONAL_REFERENCE_IMAGES >/dev/null 2>&1; then
  ADDITIONAL_REFERENCE_IMAGES=()
fi
LOGO_BBOX="${LOGO_BBOX:-}"
IDENTITY_ANCHORS="${IDENTITY_ANCHORS:-}"
PACKAGE_STATE="${PACKAGE_STATE:-}"
FINAL_SHOT_CONSTRAINT="${FINAL_SHOT_CONSTRAINT:-}"
FINAL_SHOT_ENDPOINT_LOCK="${FINAL_SHOT_ENDPOINT_LOCK:-0}"
REQUIRE_READABLE_BRANDING="${REQUIRE_READABLE_BRANDING:-0}"
PLATFORM="${PLATFORM:-landscape}"
AUTO_CUTOUT="${AUTO_CUTOUT:-1}"
AUTO_BRIEF="${AUTO_BRIEF:-1}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models/adpilot}"
KEYFRAME_MODEL="${KEYFRAME_MODEL:-$MODEL_ROOT/flux-kontext-dev}"
VIDEO_MODEL="${VIDEO_MODEL:-$MODEL_ROOT/wan2.2-i2v-a14b-diffusers}"
VLM_MODEL="${VLM_MODEL:-$MODEL_ROOT/qwen2.5-vl-3b-instruct}"
CANVAS_WIDTH="${CANVAS_WIDTH:-1360}"
CANVAS_HEIGHT="${CANVAS_HEIGHT:-768}"
KEYFRAME_SEED="${KEYFRAME_SEED:-17}"
KEYFRAME_STEPS="${KEYFRAME_STEPS:-28}"
KEYFRAME_GUIDANCE="${KEYFRAME_GUIDANCE:-2.5}"
KEYFRAME_CANDIDATES="${KEYFRAME_CANDIDATES:-2}"
KEYFRAME_OFFLOAD="${KEYFRAME_OFFLOAD:-model}"
VIDEO_SEED="${VIDEO_SEED:-11}"
VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:-49}"
VIDEO_FPS="${VIDEO_FPS:-16}"
VIDEO_STEPS="${VIDEO_STEPS:-40}"
VIDEO_GUIDANCE="${VIDEO_GUIDANCE:-3.5}"
VIDEO_CANDIDATES="${VIDEO_CANDIDATES:-1}"
FINAL_SHOT_CANDIDATES="${FINAL_SHOT_CANDIDATES:-$VIDEO_CANDIDATES}"
VIDEO_OFFLOAD="${VIDEO_OFFLOAD:-model}"
VIDEO_RESOLUTION="${VIDEO_RESOLUTION:-480p}"
CUSTOM_VIDEO_WIDTH="${CUSTOM_VIDEO_WIDTH:-832}"
CUSTOM_VIDEO_HEIGHT="${CUSTOM_VIDEO_HEIGHT:-480}"
MINIMUM_IDENTITY_SCORE="${MINIMUM_IDENTITY_SCORE:-75}"

case "$VIDEO_RESOLUTION" in
  480p) VIDEO_WIDTH=832; VIDEO_HEIGHT=480 ;;
  720p) VIDEO_WIDTH=1280; VIDEO_HEIGHT=720 ;;
  custom) VIDEO_WIDTH="$CUSTOM_VIDEO_WIDTH"; VIDEO_HEIGHT="$CUSTOM_VIDEO_HEIGHT" ;;
  *)
    echo "Unknown VIDEO_RESOLUTION: $VIDEO_RESOLUTION. Use 480p, 720p, or custom." >&2
    return 1
    ;;
esac

if [ ! -f "$PRODUCT_PATH" ]; then
  echo "Product image not found: $PRODUCT_PATH" >&2
  return 1
fi
case "$REFERENCE_MODE" in
  front_lock)
    if [ "${#ADDITIONAL_REFERENCE_IMAGES[@]}" -ne 0 ]; then
      echo "front_lock accepts one product photo. Use REFERENCE_MODE=multi_view for additional views." >&2
      return 1
    fi
    REQUIRED_MODELS=("$VLM_MODEL" "$KEYFRAME_MODEL" "$VIDEO_MODEL")
    ;;
  multi_view)
    if [ "${#ADDITIONAL_REFERENCE_IMAGES[@]}" -lt 1 ]; then
      echo "multi_view needs at least one additional real product photo." >&2
      return 1
    fi
    REQUIRED_MODELS=("$VLM_MODEL" "$KEYFRAME_MODEL" "$VIDEO_MODEL")
    ;;
  *)
    echo "Unknown REFERENCE_MODE: $REFERENCE_MODE. Use front_lock or multi_view." >&2
    return 1
    ;;
esac
for reference in "${ADDITIONAL_REFERENCE_IMAGES[@]}"; do
  if [ ! -f "$reference" ]; then
    echo "Additional product reference not found: $reference" >&2
    return 1
  fi
done
for model in "${REQUIRED_MODELS[@]}"; do
  if [ ! -d "$model" ]; then
    echo "Required model not found: $model" >&2
    echo "Download models on a node with Hugging Face access: bash scripts/download_models.sh" >&2
    return 1
  fi
done
# Avoid long-run allocator fragmentation on a single A800.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/check_env.py --strict --model-root "$MODEL_ROOT"

args=(
  --product "$PRODUCT_PATH"
  --reference-mode "$REFERENCE_MODE"
  --brand "$BRAND"
  --out "$OUT_DIR"
  --style auto
  --platform "$PLATFORM"
  --product-category "$PRODUCT_CATEGORY"
  --product-description "$PRODUCT_DESCRIPTION"
  --target-audience "$TARGET_AUDIENCE"
  --ad-mood "$AD_MOOD"
  --planner-backend qwen_vl
  --planner-model "$VLM_MODEL"
  --planner-device cuda
  --canvas-width "$CANVAS_WIDTH"
  --canvas-height "$CANVAS_HEIGHT"
  --keyframe-model "$KEYFRAME_MODEL"
  --keyframe-device cuda
  --keyframe-seed "$KEYFRAME_SEED"
  --keyframe-steps "$KEYFRAME_STEPS"
  --keyframe-guidance-scale "$KEYFRAME_GUIDANCE"
  --keyframe-offload "$KEYFRAME_OFFLOAD"
  --keyframe-candidates "$KEYFRAME_CANDIDATES"
  --critic-model "$VLM_MODEL"
  --critic-device cuda
  --minimum-identity-score "$MINIMUM_IDENTITY_SCORE"
  --video-model "$VIDEO_MODEL"
  --video-device cuda
  --video-seed "$VIDEO_SEED"
  --video-num-frames "$VIDEO_NUM_FRAMES"
  --video-fps "$VIDEO_FPS"
  --video-width "$VIDEO_WIDTH"
  --video-height "$VIDEO_HEIGHT"
  --video-steps "$VIDEO_STEPS"
  --video-guidance-scale "$VIDEO_GUIDANCE"
  --video-offload "$VIDEO_OFFLOAD"
  --video-candidates "$VIDEO_CANDIDATES"
  --final-shot-candidates "$FINAL_SHOT_CANDIDATES"
)
if [ "${#ADDITIONAL_REFERENCE_IMAGES[@]}" -gt 0 ]; then
  args+=(--reference-images "${ADDITIONAL_REFERENCE_IMAGES[@]}")
fi

if [ -n "$LOGO_BBOX" ]; then
  args+=(--logo-bbox "$LOGO_BBOX")
fi
if [ -n "$IDENTITY_ANCHORS" ]; then
  args+=(--identity-anchors "$IDENTITY_ANCHORS")
fi
if [ -n "$PACKAGE_STATE" ]; then
  args+=(--package-state "$PACKAGE_STATE")
fi
if [ -n "$FINAL_SHOT_CONSTRAINT" ]; then
  args+=(--final-shot-constraint "$FINAL_SHOT_CONSTRAINT")
fi
if [ "$FINAL_SHOT_ENDPOINT_LOCK" = "1" ]; then
  args+=(--final-shot-endpoint-lock)
fi
if [ "$REQUIRE_READABLE_BRANDING" = "1" ]; then
  args+=(--require-readable-branding)
fi
if [ "$AUTO_CUTOUT" = "1" ]; then
  args+=(--auto-cutout)
fi
if [ "$AUTO_BRIEF" = "1" ]; then
  args+=(--auto-brief --vlm-model "$VLM_MODEL" --vlm-device cuda)
fi

python app.py "${args[@]}"
python scripts/check_latest_run.py \
  --require-reference-mode "$REFERENCE_MODE" \
  --require-video-backend wan_i2v \
  --require-audited \
  --require-planner-backend qwen2_5_vl
