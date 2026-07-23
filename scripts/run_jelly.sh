#!/usr/bin/env bash

# Haribo Goldbaeren demo. Edit only this block before running.
PRODUCT_PATH="examples/jelly.jpg"
BRAND="Haribo Goldbaeren"
PRODUCT_CATEGORY="snack"
PRODUCT_DESCRIPTION="Haribo Goldbaeren gummy candy bag in gold packaging"
IDENTITY_ANCHORS="golden Haribo Goldbaeren candy bag, red HARIBO wordmark, yellow bear mascot"
# A single middle shot may show a few gummies falling from the opened top.
PACKAGE_STATE="open_with_contents"
TARGET_AUDIENCE="snack shoppers and families"
AD_MOOD="playful, bright, appetizing, energetic"
# Final packshot only: preserve the sharp, fixed arrangement visible through the clear pouch.
FINAL_SHOT_CONSTRAINT="The individual gummies visible through the clear pouch keep fixed positions and sharp outlines throughout the shot: no morphing, melting, warping, texture smear, or flicker. Only external reflections and lighting change."
# Native Wan first/last-frame conditioning, not a post-generation composite.
FINAL_SHOT_ENDPOINT_LOCK=1
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
# Generate a second Wan candidate only for the endpoint-locked final packshot.
FINAL_SHOT_CANDIDATES=2
KEYFRAME_OFFLOAD="model"
VIDEO_OFFLOAD="model"

source "$(dirname "${BASH_SOURCE[0]}")/run_product_ad.sh"
