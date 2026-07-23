from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def model_weight_bytes(path: str) -> int:
    root = os.path.abspath(path)
    total = 0
    for directory, _, files in os.walk(root):
        for filename in files:
            if filename.endswith((".safetensors", ".bin", ".pt", ".pth")):
                try:
                    total += os.path.getsize(os.path.join(directory, filename))
                except OSError:
                    pass
    return total


def _strict_preflight(model_root: str) -> list[str]:
    failures: list[str] = []
    if not has_module("torch"):
        return ["missing torch"]
    import torch

    if not torch.cuda.is_available():
        failures.append("CUDA is not visible to torch")
    required_modules = ("accelerate", "diffusers", "transformers", "qwen_vl_utils", "cv2", "ftfy", "safetensors")
    for module in required_modules:
        if not has_module(module):
            failures.append(f"missing Python module: {module}")
    if not failures:
        try:
            from diffusers import FluxKontextPipeline, WanImageToVideoPipeline  # noqa: F401
        except Exception as exc:
            failures.append(f"diffusers lacks required FLUX/Wan pipelines: {type(exc).__name__}: {exc}")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
        except Exception as exc:
            failures.append(f"transformers lacks Qwen2.5-VL support: {type(exc).__name__}: {exc}")
    required_models = {
        "flux-kontext-dev": 8 * 1024**3,
        "wan2.2-i2v-a14b-diffusers": 50 * 1024**3,
        "qwen2.5-vl-3b-instruct": 2 * 1024**3,
    }
    for name, minimum_weight_bytes in required_models.items():
        path = os.path.join(model_root, name)
        if not os.path.isdir(path):
            failures.append(f"missing model directory: {path}")
            continue
        weight_bytes = model_weight_bytes(path)
        if weight_bytes < minimum_weight_bytes:
            gib = weight_bytes / 1024**3
            required_gib = minimum_weight_bytes / 1024**3
            failures.append(
                f"incomplete model weights: {path} has {gib:.2f} GiB, requires at least {required_gib:.0f} GiB"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the AdPilot runtime environment.")
    parser.add_argument("--strict", action="store_true", help="Fail unless CUDA, required model libraries, and local strict-demo models are ready.")
    parser.add_argument("--model-root", default=os.path.expanduser("~/models/adpilot"))
    args = parser.parse_args()
    print(f"python: {sys.version.split()[0]}")
    print(f"ffmpeg: {shutil.which('ffmpeg') or 'not found'}")
    print(f"numpy: {'ok' if has_module('numpy') else 'missing'}")
    print(f"PIL: {'ok' if has_module('PIL') else 'missing'}")

    if has_module("torch"):
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch cuda: {torch.version.cuda}")
        print(f"cuda available: {torch.cuda.is_available()}")
        print(f"cuda devices: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"cuda device 0: {torch.cuda.get_device_name(0)}")
    else:
        print("torch: missing")

    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value:
            print(f"{key}: configured")

    if has_module("huggingface_hub"):
        try:
            from huggingface_hub import hf_hub_url

            print(f"huggingface_hub: ok ({hf_hub_url('Qwen/Qwen2.5-VL-3B-Instruct', 'config.json')})")
        except Exception as exc:
            print(f"huggingface_hub: error: {type(exc).__name__}: {exc}")
    else:
        print("huggingface_hub: missing")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run([nvidia_smi], check=False, text=True, capture_output=True)
        print(result.stdout[:2000])
    else:
        print("nvidia-smi: not found")

    if args.strict:
        failures = _strict_preflight(args.model_root)
        if failures:
            print("strict preflight: FAIL")
            for failure in failures:
                print(f"- {failure}")
            raise SystemExit(1)
        print("strict preflight: PASS")


if __name__ == "__main__":
    main()
