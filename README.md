# AdPilot

AdPilot turns a clean product reference image into a short, audited product-ad video. It uses Qwen2.5-VL for product analysis and visual review, FLUX.1 Kontext for reference-conditioned keyframes, and Wan2.2 I2V for motion.

## Demos

### Wireless Earbuds

| Reference image | Generated storyboard |
| --- | --- |
| ![Wireless earbuds reference](assets/demo/wireless-earbuds/reference.jpg) | ![Wireless earbuds storyboard](assets/demo/wireless-earbuds/storyboard.png) |

![Wireless earbuds generated preview](assets/demo/wireless-earbuds/preview.gif)

[Watch the full video](assets/demo/wireless-earbuds/final_video.mp4)

### Cosmetic Tube

| Reference image | Generated storyboard |
| --- | --- |
| ![Cosmetic tube reference](assets/demo/cosmetic-tube/reference.jpg) | ![Cosmetic tube storyboard](assets/demo/cosmetic-tube/storyboard.png) |

![Cosmetic tube generated preview](assets/demo/cosmetic-tube/preview.gif)

[Watch the full video](assets/demo/cosmetic-tube/final_video.mp4)

## Pipeline

1. Qwen2.5-VL extracts visible product cues and plans three advertising shots.
2. FLUX.1 Kontext generates reference-conditioned keyframe candidates.
3. Qwen2.5-VL selects candidates that preserve visible product identity.
4. Wan2.2 I2V generates video candidates, followed by a temporal consistency audit.

`front_lock` is used for the included one-image demos: the same reference image conditions every keyframe. The pipeline does not use a post-generation product overlay or background plate.

## Setup

```bash
conda env create -f environment.yml
conda activate adpilot
bash scripts/install_gpu_deps.sh
```

On a machine with Hugging Face access, accept the FLUX.1 Kontext licence, run `hf auth login`, then download the required models:

```bash
bash scripts/download_models.sh
```

## Run

Run inside a shell with an active GPU allocation and CUDA module:

```bash
bash scripts/run_wireless_earbuds.sh
# or
bash scripts/run_cosmetic_tube.sh
```

`VIDEO_RESOLUTION` accepts `480p`, `720p`, or `custom`. Each run writes its storyboard, selected video, and audit report to `outputs/run_*`.

## Repository Layout

- `app.py`: pipeline entry point
- `adpilot/`: planning, generation, identity, audit, and reporting modules
- `scripts/run_*.sh`: product-specific GPU demos
- `assets/demo/`: README demo media
- `examples/`: product reference images
- `tests/`: focused unit tests
