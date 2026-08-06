#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Edit these fields for this product-ad project.
PRODUCT_IMAGE="examples/wireless-earbuds.jpg"
BRAND="Wireless Earbuds"
PROMPT="A minimal, precise, futuristic wireless-earbuds commercial for premium technology shoppers, with a clean high-tech set, polished acrylic reflections, and controlled studio light."
PRODUCT_CATEGORY="electronics"
PRODUCT_DESCRIPTION="White wireless earbuds with their matching open white charging case; the earbuds and case are one complete set and must never appear separately"
IDENTITY_ANCHORS="open glossy white charging case, exactly two white earbuds, black speaker vents, small green status light, show the earbuds and charging case together as one complete matched set, never show either alone"
TARGET_AUDIENCE="premium consumer technology shoppers"
AD_MOOD="minimal, precise, futuristic, polished"
PACKAGE_STATE=""
PROJECT_NAME=""

# guided pauses for storyboard approval; auto runs until completion or a repair limit.
MODE="guided"
PLATFORM="landscape"

# Add real product views here when available, for example:
# REFERENCE_IMAGES=("examples/earbuds-side.jpg" "examples/earbuds-back.jpg")
REFERENCE_IMAGES=()

MODEL_ROOT="${ADPILOT_MODEL_ROOT:-$HOME/models/adpilot}"

reference_args=()
for image in "${REFERENCE_IMAGES[@]}"; do
  reference_args+=(--reference "$image")
done

project_name_args=()
if [ -n "$PROJECT_NAME" ]; then
  project_name_args=(--project-name "$PROJECT_NAME")
fi

brief_args=()
add_brief_arg() {
  local flag="$1"
  local value="$2"
  if [ -n "$value" ]; then
    brief_args+=("$flag" "$value")
  fi
}
add_brief_arg --product-category "$PRODUCT_CATEGORY"
add_brief_arg --product-description "$PRODUCT_DESCRIPTION"
add_brief_arg --identity-anchors "$IDENTITY_ANCHORS"
add_brief_arg --target-audience "$TARGET_AUDIENCE"
add_brief_arg --ad-mood "$AD_MOOD"
add_brief_arg --package-state "$PACKAGE_STATE"

case "$MODE" in
  guided)
    python -m adpilot.agent chat \
      --product "$PRODUCT_IMAGE" \
      --brand "$BRAND" \
      --prompt "$PROMPT" \
      --platform "$PLATFORM" \
      --model-root "$MODEL_ROOT" \
      "${project_name_args[@]}" \
      "${brief_args[@]}" \
      "${reference_args[@]}"
    ;;
  auto)
    create_output="$(python -m adpilot.agent create \
      --product "$PRODUCT_IMAGE" \
      --brand "$BRAND" \
      --prompt "$PROMPT" \
      --mode auto \
      --platform "$PLATFORM" \
      --model-root "$MODEL_ROOT" \
      "${project_name_args[@]}" \
      "${brief_args[@]}" \
      "${reference_args[@]}")"
    printf '%s\n' "$create_output"
    project_id="$(printf '%s\n' "$create_output" | sed -n 's/^project: //p' | head -n 1)"
    if [ -z "$project_id" ]; then
      echo "Could not determine the created project id." >&2
      exit 1
    fi
    python -m adpilot.agent run "$project_id"
    ;;
  *)
    echo "MODE must be guided or auto, got: $MODE" >&2
    exit 1
    ;;
esac
