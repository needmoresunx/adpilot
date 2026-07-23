#!/usr/bin/env bash

# Minimal white cosmetic tube demo. Edit only this block before running.
PRODUCT_PATH="examples/minimalist-white-cosmetic-tube-with-silver-cap.jpg"
BRAND="Minimal White Cosmetic Tube"
PRODUCT_CATEGORY="cosmetic"
PRODUCT_DESCRIPTION="Minimalist white skincare squeeze tube with a reflective silver cap"
IDENTITY_ANCHORS="tall matte white squeeze tube, blank rectangular front label panel, ribbed top seal, wide reflective silver cap"
TARGET_AUDIENCE="minimal skincare and self-care shoppers"
AD_MOOD="clean, calm, premium, soft, clinical-minimal"
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
