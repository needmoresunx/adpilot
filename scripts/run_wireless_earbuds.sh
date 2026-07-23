#!/usr/bin/env bash

# Wireless earbuds demo. Edit only this block before running.
PRODUCT_PATH="examples/wireless-earbuds.jpg"
BRAND="Wireless Earbuds"
PRODUCT_CATEGORY="electronics"
PRODUCT_DESCRIPTION="White wireless earbuds in an open white charging case"
IDENTITY_ANCHORS="open glossy white charging case, exactly two white earbuds, black speaker vents, small green status light"
TARGET_AUDIENCE="premium consumer technology shoppers"
AD_MOOD="minimal, precise, futuristic, polished"
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
