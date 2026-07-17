from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image


class BackgroundBackend(Protocol):
    name: str

    def generate(self, shot_index: int, prompt: str, size: tuple[int, int]) -> Image.Image:
        """Return one background image for a shot."""


class MockBackgroundBackend:
    name = "mock"

    def generate(self, shot_index: int, prompt: str, size: tuple[int, int]) -> Image.Image:
        return make_mock_background(shot_index, prompt, size)


class FolderBackgroundBackend:
    name = "folder"

    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        self.images: list[Path] = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            self.images.extend(sorted(image_dir.glob(pattern)))
        if not self.images:
            raise ValueError(f"No background images found in {image_dir}")

    def generate(self, shot_index: int, prompt: str, size: tuple[int, int]) -> Image.Image:
        path = self.images[shot_index % len(self.images)]
        return cover_resize(Image.open(path).convert("RGB"), size)


class DiffusersBackgroundBackend:
    name = "diffusers"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int = 7,
        generated_size: tuple[int, int] = (768, 1344),
        negative_prompt: str = "",
        fallback: BackgroundBackend | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.generated_size = generated_size
        self.negative_prompt = negative_prompt
        self.fallback = fallback
        self._pipe = None
        self._torch = None
        self.used_fallback = False
        self.errors: list[str] = []

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "device": self.device,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "generated_size": self.generated_size,
            "used_fallback": self.used_fallback,
            "errors": self.errors[-3:],
        }

    def _load(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "Diffusers backend requires torch and diffusers. "
                "Run scripts/install_gpu_deps.sh inside the adpilot environment."
            ) from exc

        if self.device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not visible. Refusing slow CPU diffusion; use --image-device cpu to force CPU.")
            device = "cuda"
        else:
            device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is false.")
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        # Keep dtype decisions in from_pretrained(); this call only moves the
        # pipeline to the selected device.
        pipe = pipe.to(device=device)
        self._pipe = pipe
        self._torch = torch
        self.device = device

    def generate(self, shot_index: int, prompt: str, size: tuple[int, int]) -> Image.Image:
        try:
            self._load()
            width, height = self.generated_size
            generator = self._torch.Generator(device=self.device).manual_seed(self.seed + shot_index)
            orientation = "landscape" if width >= height else "vertical"
            enhanced_prompt = (
                f"{prompt}, {orientation} ad background, realistic light, no text"
            )
            image = self._pipe(
                prompt=enhanced_prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                width=width,
                height=height,
                generator=generator,
            ).images[0]
            return cover_resize(image.convert("RGB"), size)
        except Exception as exc:
            self.used_fallback = True
            self.errors.append(f"{type(exc).__name__}: {exc}")
            if self.fallback is None:
                raise
            return self.fallback.generate(shot_index, prompt, size)


def make_background_backend(
    name: str,
    image_dir: str | None = None,
    image_model: str = "stabilityai/sdxl-turbo",
    image_device: str = "auto",
    image_steps: int = 4,
    image_guidance_scale: float = 0.0,
    image_seed: int = 7,
    generated_size: tuple[int, int] = (768, 1344),
    fallback_on_error: bool = True,
) -> BackgroundBackend:
    if name == "mock":
        return MockBackgroundBackend()
    if name == "folder":
        if not image_dir:
            raise ValueError("--background-dir is required for folder backend")
        return FolderBackgroundBackend(Path(image_dir))
    if name == "diffusers":
        fallback = MockBackgroundBackend() if fallback_on_error else None
        return DiffusersBackgroundBackend(
            model_id=image_model,
            device=image_device,
            steps=image_steps,
            guidance_scale=image_guidance_scale,
            seed=image_seed,
            generated_size=generated_size,
            negative_prompt="text, watermark, logo, product, bottle, distorted objects, clutter",
            fallback=fallback,
        )
    raise ValueError(f"Unknown background backend: {name}")


def cover_resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    source_w, source_h = image.size
    scale = max(target_w / source_w, target_h / source_h)
    resized = image.resize((int(source_w * scale), int(source_h * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - target_w) // 2)
    top = max(0, (resized.height - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


def make_mock_background(index: int, prompt: str, size: tuple[int, int]) -> Image.Image:
    palettes = [
        ((233, 237, 230), (163, 181, 169)),
        ((238, 232, 224), (191, 175, 159)),
        ((224, 231, 238), (148, 163, 184)),
    ]
    top, bottom = palettes[index % len(palettes)]
    width, height = size
    image = Image.new("RGB", size, top)
    pixels = image.load()
    for y in range(height):
        t = y / max(height - 1, 1)
        row = tuple(int(top[c] * (1 - t) + bottom[c] * t) for c in range(3))
        for x in range(width):
            pixels[x, y] = row

    return image
