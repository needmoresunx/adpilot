from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
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
            print(f"{key}: {value}")

    if has_module("huggingface_hub"):
        try:
            from huggingface_hub import hf_hub_url

            print(f"huggingface_hub: ok ({hf_hub_url('stabilityai/sdxl-turbo', 'model_index.json')})")
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


if __name__ == "__main__":
    main()
