# AdPilot

AdPilot is an auditable product-identity QA and repair agent for AI-generated
ads.

It is not positioned as another ad generator. It focuses on the missing control
layer around generative ad tools:

- represent a product as a structured `IdentityCard`
- create a low-cost proxy preview
- critique product faithfulness with measurable checks
- repair only failed shots with a bounded policy
- export a traceable report for human review

The current MVP runs without a heavy model so the agent loop is testable on any
machine. GPU image/video backends can be plugged in through `adpilot/backends`.

## Project Status

This repository is an MVP/proof of concept, not a finished commercial ad
generator. The current focus is the control layer around generative ad creation:
planning, product-identity representation, critique, bounded repair, and
transparent reporting.

The CPU path is intended to be reproducible on a normal machine. The GPU path
uses FLUX.1 Kontext to integrate the supplied product photo into ad keyframes,
then Wan2.2 I2V for prompt-conditioned video generation.

## Quick Start

```bash
conda activate adpilot
cd /path/to/adpilot
python app.py --product examples/perfume.jpg --brand "Miss Dior" --product-category fragrance --style "romantic floral luxury"
```

Or:

```bash
conda activate adpilot
cd /path/to/adpilot
bash scripts/run_demo.sh
```

Outputs are written to `outputs/run_*`.

## GPU Demo

In a GPU shell/job where CUDA is visible:

```bash
conda activate adpilot
cd /path/to/adpilot
# Edit the Demo config block at the top of scripts/run_strict_gpu_video_demo.sh.
bash scripts/run_strict_gpu_video_demo.sh
python scripts/check_latest_run.py --require-video-backend wan_i2v
```

If the GPU compute node cannot access Hugging Face, download models on a login
node first:

```bash
conda activate adpilot
cd /path/to/adpilot
bash scripts/download_models.sh
```

Before downloading FLUX.1 Kontext, accept its Hugging Face license and run
`hf auth login`. The model is stored at `$HOME/models/adpilot/flux-kontext-dev`.

Then inside the GPU allocation:

```bash
conda activate adpilot
cd /path/to/adpilot
# Edit PRODUCT_PATH, BRAND, and PRODUCT_CATEGORY in scripts/run_strict_gpu_video_demo.sh.
bash scripts/run_strict_gpu_video_demo.sh
python scripts/check_latest_run.py --require-video-backend wan_i2v
```

`run_strict_gpu_video_demo.sh` uses FLUX.1 Kontext to generate scene-integrated
product keyframes and Wan2.2 I2V to export `final_video.mp4`. It does not place
a product PNG over Wan's generated frames. It fails loudly instead of silently
falling back to mock/proxy backends.

The demo script is intentionally config-file style. Edit these variables at the
top of `scripts/run_strict_gpu_video_demo.sh`:

```bash
PRODUCT_PATH="examples/perfume.jpg"
BRAND="Miss Dior"
PRODUCT_CATEGORY="fragrance"
PRODUCT_DESCRIPTION=""
TARGET_AUDIENCE=""
AD_MOOD=""
LOGO_BBOX=""
```

## What The MVP Generates

- `identity_card.json`
- product brief inside `identity_card.json`
- `shot_plan.json`
- `generation_backend.json`
- `render_metadata.json`
- `storyboard.png`
- `proxy_preview.mp4`
- `critique_report.json`
- `repair_log.json`
- `report.html`
- `final_video.mp4` when the Wan I2V backend runs successfully

`proxy_preview.mp4` is a proxy animatic, not a video-generation result. This is
intentional: the proxy is the cheap reasoning layer where identity failures are
detected before expensive generation.

For GPU/Wan runs, always check `video_backend.json`. A successful strict Wan run
should contain:

```json
{
  "name": "wan_i2v",
  "used_fallback": false
}
```

Or run:

```bash
python scripts/check_latest_run.py --require-video-backend wan_i2v
```

## Backend Strategy

Current:

- deterministic CPU proxy path for testing the planning and reporting loop
- FLUX.1 Kontext image editing for scene-integrated product keyframes
- `wan_i2v` backend for Wan2.2 prompt-conditioned image-to-video rendering
- product-aware brief inferred from file name, aspect ratio, brand, and optional category overrides
- structured generation metadata and per-run HTML/JSON reports

Next:

- CLIP/DINO similarity critic
- CLIP/OCR metrics that operate on generated video frames, not just keyframes
- higher-fidelity transparent/glass-product matting

## Current Limitations

- The planner is currently a deterministic product-aware three-shot template,
  not a learned creative planner or full VLM product recognizer.
- The first critic is rule-based. It checks scale, aspect-ratio preservation,
  crop-based color consistency, product/logo bounding boxes, and logo-region
  size, but it is not yet a full CLIP/DINO/OCR visual evaluator.
- FLUX.1 Kontext and Wan2.2 can still alter fine product details. The agent
  should treat logo readability and product identity as a post-generation
  evaluation problem, not promise pixel-exact preservation.

## Roadmap

- Save visual crop artifacts and overlay diagnostics for each rendered frame.
- Evaluate final frame crops against the reference product with CLIP/DINO-style
  similarity metrics.
- Add OCR-based logo readability checks.
- Add a small VLM/LLM planner that adapts shots to product category, platform,
  and target tone.
- Add a rights-cleared real-product demo alongside the synthetic bottle example.

## Environment Check

```bash
conda activate adpilot
python scripts/check_env.py
```
