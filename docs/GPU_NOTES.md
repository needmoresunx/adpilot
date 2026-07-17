# GPU Notes

Run model downloads from a machine with Hugging Face access. Run generation in
your scheduler-provided GPU shell or job.

## Install and Download

```bash
conda activate adpilot
cd /path/to/adpilot
bash scripts/install_gpu_deps.sh
hf auth login  # Required after accepting the FLUX.1 Kontext license.
bash scripts/download_models.sh
```

The downloader stores models under `$HOME/models/adpilot` by default. Override
this location with `ADPILOT_MODEL_ROOT` when needed.

## Run a Strict GPU Demo

First make CUDA visible in the current shell according to your cluster's
scheduler and module system. Then run:

```bash
conda activate adpilot
cd /path/to/adpilot
python scripts/check_env.py
bash scripts/run_strict_gpu_video_demo.sh
python scripts/check_latest_run.py --require-video-backend wan_i2v
```

Edit the configuration block at the top of
`scripts/run_strict_gpu_video_demo.sh` to select a product image, product
description, audience, visual mood, and output resolution.

Strict mode uses FLUX.1 Kontext for scene-integrated keyframes and Wan2.2 I2V
for image-to-video generation. It intentionally stops on a missing model or
backend failure rather than substituting a mock output.

## Inspect a Run

```bash
RUN_DIR=$(ls -td outputs/run_* | head -1)
cat "$RUN_DIR/keyframe_backend.json"
cat "$RUN_DIR/video_backend.json"
```

Open `storyboard.png`, `report.html`, and `final_video.mp4` in the selected run
directory. A strict video run must report `"name": "wan_i2v"` and
`"used_fallback": false` in `video_backend.json`.
