#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Edit these fields for this product-ad project.
PRODUCT_IMAGE="examples/minimalist-white-cosmetic-tube-with-silver-cap.jpg"
BRAND="Minimal White Cosmetic Tube"
PROMPT="A clean, calm, premium skincare commercial for minimalist self-care shoppers, with soft studio light, pale reflective surfaces, and precise clinical-beauty styling."
PRODUCT_CATEGORY="cosmetic"
PRODUCT_DESCRIPTION="Minimalist white skincare squeeze tube with a reflective silver cap"
IDENTITY_ANCHORS="tall matte white squeeze tube, blank rectangular front label panel, ribbed top seal, wide reflective silver cap"
TARGET_AUDIENCE="minimal skincare and self-care shoppers"
AD_MOOD="clean, calm, premium, soft, clinical-minimal"
PACKAGE_STATE=""
PROJECT_NAME=""

# guided pauses for storyboard approval; auto runs until completion or a repair limit.
MODE="guided"
PLATFORM="landscape"

# Add real product views here when available, for example:
# REFERENCE_IMAGES=("examples/cosmetic-tube-side.jpg" "examples/cosmetic-tube-back.jpg")
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
