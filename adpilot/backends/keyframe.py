from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image

from adpilot.backends.background import cover_resize


class KeyframeBackend(Protocol):
    name: str

    def render(
        self,
        product_path: Path,
        prompts: list[str],
        output_dir: Path,
        size: tuple[int, int],
    ) -> list[Path]:
        """Create scene-integrated product keyframes from the reference image."""


class FluxKontextKeyframeBackend:
    name = "flux_kontext"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        seed: int = 17,
        num_inference_steps: int = 28,
        guidance_scale: float = 2.5,
        fallback: KeyframeBackend | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
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
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "used_fallback": self.used_fallback,
            "errors": self.errors[-3:],
        }

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import FluxKontextPipeline
        except Exception as exc:  # pragma: no cover - optional model dependencies
            raise RuntimeError(
                "FLUX Kontext requires a current Diffusers build. Run "
                "scripts/install_gpu_deps.sh inside the adpilot environment."
            ) from exc

        if self.device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not visible. Refusing CPU FLUX Kontext inference.")
            device = "cuda"
        else:
            device = self.device
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FLUX Kontext requires a visible CUDA device for this demo.")

        pipe = FluxKontextPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        self._pipe = pipe.to(device)
        self._torch = torch
        self.device = device

    def render(
        self,
        product_path: Path,
        prompts: list[str],
        output_dir: Path,
        size: tuple[int, int],
    ) -> list[Path]:
        try:
            self._load()
            output_dir.mkdir(parents=True, exist_ok=True)
            reference = Image.open(product_path).convert("RGB")
            reference = cover_resize(reference, size)
            paths: list[Path] = []
            for index, prompt in enumerate(prompts):
                generator = self._torch.Generator(device=self.device).manual_seed(self.seed + index)
                image = self._pipe(
                    image=reference,
                    prompt=prompt,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    generator=generator,
                ).images[0].convert("RGB")
                path = output_dir / f"shot_{index + 1:02d}.jpg"
                image.save(path, quality=95)
                paths.append(path)
            return paths
        except Exception as exc:
            self.used_fallback = True
            self.errors.append(f"{type(exc).__name__}: {exc}")
            if self.fallback is None:
                raise
            return self.fallback.render(product_path, prompts, output_dir, size)


class ReferenceKeyframeBackend:
    """Cheap fallback that exposes the reference image without fake compositing."""

    name = "reference"

    def render(
        self,
        product_path: Path,
        prompts: list[str],
        output_dir: Path,
        size: tuple[int, int],
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        image = cover_resize(Image.open(product_path).convert("RGB"), size)
        paths: list[Path] = []
        for index in range(len(prompts)):
            path = output_dir / f"shot_{index + 1:02d}.jpg"
            image.save(path, quality=95)
            paths.append(path)
        return paths


def make_keyframe_backend(
    name: str,
    model_id: str,
    device: str = "auto",
    seed: int = 17,
    num_inference_steps: int = 28,
    guidance_scale: float = 2.5,
    fallback_on_error: bool = True,
) -> KeyframeBackend:
    if name == "flux_kontext":
        return FluxKontextKeyframeBackend(
            model_id=model_id,
            device=device,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fallback=ReferenceKeyframeBackend() if fallback_on_error else None,
        )
    if name == "reference":
        return ReferenceKeyframeBackend()
    raise ValueError(f"Unknown keyframe backend: {name}")
