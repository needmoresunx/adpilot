#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/create_demo_product.py
python app.py \
  --product examples/demo_bottle.png \
  --brand AquaFlow \
  --style "premium fitness" \
  --logo-bbox 90,75,210,125

