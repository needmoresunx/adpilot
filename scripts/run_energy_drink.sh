#!/usr/bin/env bash

# Monster Energy Ultra demo. Edit only this block before running.
PRODUCT_PATH="examples/energy-drink.jpg"
BRAND="Monster Energy Ultra"
PRODUCT_CATEGORY="beverage"
PRODUCT_DESCRIPTION="Tall slim white zero-sugar energy drink can with black Monster claw logo"
IDENTITY_ANCHORS="tall slim white aluminum can, black claw M logo, black Monster Energy wordmark, ZERO SUGAR text around the rim"
TARGET_AUDIENCE="active young adults and energy drink shoppers"
AD_MOOD="cold, high-energy, sharp, urban, athletic"
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
