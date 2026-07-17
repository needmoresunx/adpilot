#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

###############################################################################
# Demo config
#
# Edit this block before running the script.
#
# Required for a real demo:
#   PRODUCT_PATH      Path to your product image. Use a clean white/bright
#                     background photo when possible.
#   BRAND             Brand/product name used for prompt generation and reports.
#   PRODUCT_CATEGORY  Product type. Useful values: fragrance, beverage,
#                     cosmetic, fashion, general.
#
# Optional:
#   LOGO_BBOX         x1,y1,x2,y2 in the ORIGINAL product image. Leave empty if
#                     unknown. If AUTO_CUTOUT=1, the bbox is adjusted after crop.
#   PRODUCT_DESCRIPTION / TARGET_AUDIENCE / AD_MOOD
#                     Manual overrides for product-aware prompt generation.
#   KEYFRAME_STEPS     FLUX Kontext image-editing quality. 28 is the default.
#
# OUT_DIR is fixed to outputs/ so all demo runs land in one place.
#
# FLUX Kontext first turns the supplied product photo into a coherent ad still.
# Wan2.2 then generates each video shot from that integrated still. The final
# video never overlays the product cutout after generation.
# Keep PLATFORM=landscape for this demo.
#
# VIDEO_RESOLUTION choices:
#   480p    832x480, faster and recommended for the first successful demo
#   720p    1280x720, better quality but slower and heavier
#   custom  use CUSTOM_VIDEO_WIDTH and CUSTOM_VIDEO_HEIGHT below
#
# VIDEO_NUM_FRAMES:
#   49      faster first demo, about 3 seconds per shot at 16 fps
#   81      model-card style quality setting, about 5 seconds per shot at 16 fps
###############################################################################

OUT_DIR="outputs"
PRODUCT_PATH="examples/perfume.jpg"
BRAND="Miss Dior"
PRODUCT_CATEGORY="fragrance"
PRODUCT_DESCRIPTION="Miss Dior pink floral perfume bottle"
TARGET_AUDIENCE="luxury beauty shoppers"
AD_MOOD="romantic Parisian luxury, soft floral, cinematic"
LOGO_BBOX=""
STYLE="auto"
PLATFORM="landscape"
AUTO_CUTOUT=0
AUTO_BRIEF=1

MODEL_ROOT="$HOME/models/adpilot"
KEYFRAME_MODEL="$MODEL_ROOT/flux-kontext-dev"
VIDEO_MODEL="$MODEL_ROOT/wan2.2-i2v-a14b-diffusers"
VLM_MODEL="$MODEL_ROOT/blip-image-captioning-base"
VLM_DEVICE="cuda"

CANVAS_WIDTH=1024
CANVAS_HEIGHT=576

KEYFRAME_DEVICE="cuda"
KEYFRAME_SEED=17
KEYFRAME_STEPS=28
KEYFRAME_GUIDANCE=2.5

VIDEO_DEVICE="cuda"
VIDEO_RESOLUTION="480p"
VIDEO_SEED=11
VIDEO_NUM_FRAMES=49
VIDEO_FPS=16
CUSTOM_VIDEO_WIDTH=832
CUSTOM_VIDEO_HEIGHT=480
VIDEO_STEPS=40
VIDEO_GUIDANCE=3.5

###############################################################################

case "$VIDEO_RESOLUTION" in
  480p)
    VIDEO_WIDTH=832
    VIDEO_HEIGHT=480
    ;;
  720p)
    VIDEO_WIDTH=1280
    VIDEO_HEIGHT=720
    ;;
  custom)
    VIDEO_WIDTH="$CUSTOM_VIDEO_WIDTH"
    VIDEO_HEIGHT="$CUSTOM_VIDEO_HEIGHT"
    ;;
  *)
    echo "Unknown VIDEO_RESOLUTION: $VIDEO_RESOLUTION" >&2
    echo "Use one of: 480p, 720p, custom" >&2
    exit 1
    ;;
esac

if [ "$PRODUCT_PATH" = "examples/demo_bottle.png" ] && [ ! -f "$PRODUCT_PATH" ]; then
  python scripts/create_demo_product.py
fi
if [ ! -f "$PRODUCT_PATH" ]; then
  echo "Product image not found: $PRODUCT_PATH" >&2
  exit 1
fi
if [ ! -d "$KEYFRAME_MODEL" ]; then
  echo "FLUX Kontext model not found: $KEYFRAME_MODEL" >&2
  echo "Accept its Hugging Face license, run hf auth login, then run scripts/download_models.sh on the login node." >&2
  exit 1
fi
if [ ! -d "$VIDEO_MODEL" ]; then
  echo "Video model not found: $VIDEO_MODEL" >&2
  echo "Run scripts/download_models.sh on the login node first." >&2
  exit 1
fi
if [ "$AUTO_BRIEF" = "1" ] && [ ! -d "$VLM_MODEL" ]; then
  echo "VLM model not found: $VLM_MODEL" >&2
  echo "Continuing without VLM auto-brief. Product category config will be used." >&2
  AUTO_BRIEF=0
fi

python scripts/check_env.py

args=(
  --product "$PRODUCT_PATH"
  --brand "$BRAND"
  --out "$OUT_DIR"
  --style "$STYLE"
  --platform "$PLATFORM"
  --canvas-width "$CANVAS_WIDTH"
  --canvas-height "$CANVAS_HEIGHT"
  --keyframe-backend flux_kontext
  --keyframe-model "$KEYFRAME_MODEL"
  --keyframe-device "$KEYFRAME_DEVICE"
  --keyframe-seed "$KEYFRAME_SEED"
  --keyframe-steps "$KEYFRAME_STEPS"
  --keyframe-guidance-scale "$KEYFRAME_GUIDANCE"
  --video-backend wan_i2v
  --video-model "$VIDEO_MODEL"
  --video-device "$VIDEO_DEVICE"
  --video-seed "$VIDEO_SEED"
  --video-num-frames "$VIDEO_NUM_FRAMES"
  --video-fps "$VIDEO_FPS"
  --video-width "$VIDEO_WIDTH"
  --video-height "$VIDEO_HEIGHT"
  --video-steps "$VIDEO_STEPS"
  --video-guidance-scale "$VIDEO_GUIDANCE"
  --no-backend-fallback
)

if [ -n "$LOGO_BBOX" ]; then
  args+=(--logo-bbox "$LOGO_BBOX")
fi
if [ -n "$PRODUCT_CATEGORY" ]; then
  args+=(--product-category "$PRODUCT_CATEGORY")
fi
if [ -n "$PRODUCT_DESCRIPTION" ]; then
  args+=(--product-description "$PRODUCT_DESCRIPTION")
fi
if [ -n "$TARGET_AUDIENCE" ]; then
  args+=(--target-audience "$TARGET_AUDIENCE")
fi
if [ -n "$AD_MOOD" ]; then
  args+=(--ad-mood "$AD_MOOD")
fi
if [ "$AUTO_CUTOUT" = "1" ]; then
  args+=(--auto-cutout)
fi
if [ "$AUTO_BRIEF" = "1" ] && [ -n "$VLM_MODEL" ]; then
  args+=(--auto-brief)
  args+=(--vlm-model "$VLM_MODEL" --vlm-device "$VLM_DEVICE")
fi

python app.py "${args[@]}"
