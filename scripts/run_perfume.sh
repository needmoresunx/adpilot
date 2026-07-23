#!/usr/bin/env bash

# Miss Dior demo. Edit only this block before running.
PRODUCT_PATH="examples/perfume.jpg"
BRAND="Miss Dior"
PRODUCT_CATEGORY="fragrance"
PRODUCT_DESCRIPTION="Miss Dior pink floral perfume bottle with a silver bow"
IDENTITY_ANCHORS="wide square clear-glass bottle, pink liquid, large silver bow, exposed silver atomizer, no colored cap"
TARGET_AUDIENCE="luxury beauty shoppers"
AD_MOOD="romantic Parisian luxury, soft floral, cinematic"
LOGO_BBOX=""
REFERENCE_MODE="front_lock"
ADDITIONAL_REFERENCE_IMAGES=()

# Video resolution: 480p (832x480, recommended on one A800), 720p (1280x720),
# or custom (set CUSTOM_VIDEO_WIDTH and CUSTOM_VIDEO_HEIGHT below).
VIDEO_RESOLUTION="480p"
CUSTOM_VIDEO_WIDTH=832
CUSTOM_VIDEO_HEIGHT=480
VIDEO_NUM_FRAMES=49
KEYFRAME_CANDIDATES=2
VIDEO_CANDIDATES=1
KEYFRAME_OFFLOAD="model"
VIDEO_OFFLOAD="model"

source "$(dirname "${BASH_SOURCE[0]}")/run_product_ad.sh"
