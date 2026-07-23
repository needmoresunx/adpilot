#!/usr/bin/env bash

# Chanel lip gloss demo. Edit only this block before running.
PRODUCT_PATH="examples/lipgloss.jpg"
BRAND="Chanel"
PRODUCT_CATEGORY="cosmetic"
PRODUCT_DESCRIPTION="Chanel pale pink lip gloss in a tall clear rectangular tube"
IDENTITY_ANCHORS="tall clear rectangular lip gloss tube, pale pink liquid, polished gold collar, glossy black cap, vertical white CHANEL wordmark"
TARGET_AUDIENCE="premium beauty and makeup shoppers"
AD_MOOD="modern Parisian beauty, glossy, refined, soft-luxury"
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
