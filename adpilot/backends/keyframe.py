from __future__ import annotations

import gc
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


def fit_reference_to_canvas(
    image: Image.Image,
    size: tuple[int, int],
    margin_ratio: float = 0.08,
) -> Image.Image:
    """Fit the complete reference image into a model-sized canvas without cropping it."""
    source = image.convert("RGBA")
    # A segmentation crop already isolates the product. Leave it more breathing
    # room than a full photo so FLUX can construct the requested environment.
    if np.any(np.asarray(source)[:, :, 3] < 16):
        margin_ratio = max(margin_ratio, 0.22)
    target_w, target_h = size
    usable_w = max(1, round(target_w * (1.0 - 2.0 * margin_ratio)))
    usable_h = max(1, round(target_h * (1.0 - 2.0 * margin_ratio)))
    scale = min(usable_w / source.width, usable_h / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    )

    # Use the reference border color so white-background product photos do not
    # acquire a visible hard rectangle before the image-editing model removes it.
    rgba = np.asarray(source)
    corners = np.concatenate(
        [
            rgba[:1, :1, :3].reshape(-1, 3),
            rgba[:1, -1:, :3].reshape(-1, 3),
            rgba[-1:, :1, :3].reshape(-1, 3),
            rgba[-1:, -1:, :3].reshape(-1, 3),
        ]
    )
    background = tuple(int(value) for value in np.median(corners, axis=0))
    canvas = Image.new("RGB", size, background)
    left = (target_w - resized.width) // 2
    top = (target_h - resized.height) // 2
    canvas.paste(resized, (left, top), resized)
    return canvas


def place_product_reference(
    image: Image.Image,
    size: tuple[int, int],
    product_position: str,
    product_scale: float,
) -> Image.Image:
    """Create a per-shot FLUX reference with the real product at its planned layout.

    FLUX Kontext is an image-editing model: it follows the geometry in its input
    image more reliably than geometry mentioned only in text.  This image is an
    input condition, never a layer pasted over the generated keyframe or video.
    """
    source = image.convert("RGBA")
    target_w, target_h = size
    scale = min(max(float(product_scale), 0.18), 0.62)
    product_w = max(1, round(target_w * scale))
    product_h = max(1, round(product_w * source.height / max(source.width, 1)))
    max_h = max(1, round(target_h * 0.82))
    if product_h > max_h:
        ratio = max_h / product_h
        product_w = max(1, round(product_w * ratio))
        product_h = max_h
    product = source.resize((product_w, product_h), Image.Resampling.LANCZOS)

    x_ratio = {"left": 0.28, "center": 0.50, "right": 0.72}.get(product_position, 0.50)
    left = round(target_w * x_ratio - product_w / 2)
    left = max(0, min(target_w - product_w, left))
    # Keep the product grounded on the lower part of the canvas, leaving room
    # above it for the surrounding set and avoiding a floating reference.
    bottom = round(target_h * 0.88)
    top = max(0, min(target_h - product_h, bottom - product_h))

    rgba = np.asarray(source)
    corners = np.concatenate(
        [
            rgba[:1, :1, :3].reshape(-1, 3),
            rgba[:1, -1:, :3].reshape(-1, 3),
            rgba[-1:, :1, :3].reshape(-1, 3),
            rgba[-1:, -1:, :3].reshape(-1, 3),
        ]
    )
    # Segmentation retains the original packshot colours in transparent pixels,
    # so border pixels preserve the neutral source background rather than
    # painting the whole FLUX condition in the product's dominant colour.
    background = tuple(int(value) for value in np.median(corners, axis=0))
    canvas = Image.new("RGB", size, background)
    canvas.paste(product, (left, top), product)
    return canvas


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

    def render_candidates(
        self,
        product_path: Path,
        prompts: list[str],
        output_dir: Path,
        size: tuple[int, int],
        candidates_per_shot: int,
        seed_offset: int = 0,
        reference_layouts: list[tuple[str, float]] | None = None,
        reference_paths: list[Path] | None = None,
    ) -> list[list[Path]]:
        """Create traceable candidates, optionally conditioning each shot on a view."""


class FluxKontextKeyframeBackend:
    name = "flux_kontext"
    clip_prompt_token_limit = 75

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        seed: int = 17,
        num_inference_steps: int = 28,
        guidance_scale: float = 2.5,
        offload_mode: str = "none",
    ):
        self.model_id = model_id
        self.device = device
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.offload_mode = offload_mode
        self._pipe = None
        self._torch = None
        self.used_fallback = False
        self.errors: list[str] = []
        self.reference_path: str | None = None
        self.source_size: tuple[int, int] | None = None
        self.reference_size: tuple[int, int] | None = None
        self.candidates_per_shot = 1
        self.clip_prompt_guard_applied = False
        self.reference_paths: list[str] = []

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "device": self.device,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "offload_mode": self.offload_mode,
            "used_fallback": self.used_fallback,
            "errors": self.errors[-3:],
            "reference_fit": "per_shot_product_layout_when_provided",
            "reference_path": self.reference_path,
            "reference_paths": self.reference_paths,
            "source_size": self.source_size,
            "reference_size": self.reference_size,
            "candidates_per_shot": self.candidates_per_shot,
            "clip_prompt_token_limit": self.clip_prompt_token_limit,
            "clip_prompt_guard_applied": self.clip_prompt_guard_applied,
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
                "python -m pip install -r requirements.txt."
            ) from exc

        if self.device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not visible. Refusing CPU FLUX Kontext inference.")
            device = "cuda"
        else:
            device = self.device
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FLUX Kontext requires a visible CUDA device for this demo.")

        if self.offload_mode not in {"none", "model", "sequential"}:
            raise ValueError(f"Unknown FLUX offload mode: {self.offload_mode}")
        pipe = FluxKontextPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        if self.offload_mode == "model":
            pipe.enable_model_cpu_offload(device=device)
        elif self.offload_mode == "sequential":
            pipe.enable_sequential_cpu_offload(device=device)
        else:
            pipe = pipe.to(device)
        self._pipe = pipe
        self._torch = torch
        self.device = device

    def _clip_safe_prompt(self, prompt: str) -> str:
        """Limit the actual CLIP token sequence, not an approximate word count."""
        tokenizer = getattr(self._pipe, "tokenizer", None)
        if tokenizer is None:
            return prompt
        model_limit = getattr(tokenizer, "model_max_length", self.clip_prompt_token_limit + 2)
        try:
            token_limit = min(self.clip_prompt_token_limit, max(1, int(model_limit) - 2))
            encoded = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=token_limit,
            )
            token_ids = encoded["input_ids"]
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            safe_prompt = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            self.clip_prompt_guard_applied = True
            return safe_prompt or prompt
        except Exception as exc:
            self.errors.append(f"CLIP prompt guard skipped: {type(exc).__name__}: {exc}")
            return prompt

    def render(
        self,
        product_path: Path,
        prompts: list[str],
        output_dir: Path,
        size: tuple[int, int],
    ) -> list[Path]:
        return [paths[0] for paths in self.render_candidates(product_path, prompts, output_dir, size, 1)]

    def render_candidates(
        self,
        product_path: Path,
        prompts: list[str],
        output_dir: Path,
        size: tuple[int, int],
        candidates_per_shot: int,
        seed_offset: int = 0,
        reference_layouts: list[tuple[str, float]] | None = None,
        reference_paths: list[Path] | None = None,
    ) -> list[list[Path]]:
        if candidates_per_shot < 1:
            raise ValueError("candidates_per_shot must be at least 1")
        self.candidates_per_shot = candidates_per_shot
        try:
            self._load()
            output_dir.mkdir(parents=True, exist_ok=True)
            source_paths = [Path(product_path), *(Path(path) for path in (reference_paths or []))]
            self.reference_paths = [str(path) for path in source_paths]
            source = Image.open(source_paths[0]).convert("RGBA")
            self.source_size = source.size
            if reference_layouts is not None and len(reference_layouts) != len(prompts):
                raise ValueError("reference_layouts must have one entry per prompt")
            reference = fit_reference_to_canvas(source, size)
            self.reference_size = reference.size
            reference_path = output_dir / "reference_input.png"
            reference.save(reference_path)
            self.reference_path = str(reference_path)
            paths_by_shot: list[list[Path]] = []
            for index, prompt in enumerate(prompts):
                prompt = self._clip_safe_prompt(prompt)
                shot_dir = output_dir / f"shot_{index + 1:02d}"
                shot_dir.mkdir(parents=True, exist_ok=True)
                source_path = source_paths[index % len(source_paths)]
                shot_source = Image.open(source_path).convert("RGBA")
                if reference_layouts is None:
                    shot_reference = fit_reference_to_canvas(shot_source, size)
                else:
                    position, scale = reference_layouts[index]
                    shot_reference = place_product_reference(shot_source, size, position, scale)
                shot_reference.save(shot_dir / "reference_input.png")
                (shot_dir / "reference_source.txt").write_text(str(source_path), encoding="utf-8")
                shot_paths: list[Path] = []
                for candidate_index in range(candidates_per_shot):
                    seed = self.seed + seed_offset + index * 1009 + candidate_index
                    generator = self._torch.Generator(device=self.device).manual_seed(seed)
                    image = self._pipe(
                        image=shot_reference,
                        prompt=prompt,
                        height=size[1],
                        width=size[0],
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps,
                        generator=generator,
                    ).images[0].convert("RGB")
                    path = shot_dir / f"candidate_{candidate_index + 1:02d}.jpg"
                    image.save(path, quality=95)
                    shot_paths.append(path)
                paths_by_shot.append(shot_paths)
            return paths_by_shot
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            raise

    def release(self) -> None:
        if self._pipe is not None:
            if hasattr(self._pipe, "remove_all_hooks"):
                self._pipe.remove_all_hooks()
            del self._pipe
            self._pipe = None
        gc.collect()
        if self.device == "cuda" and self._torch is not None:
            self._torch.cuda.empty_cache()


def make_keyframe_backend(
    name: str,
    model_id: str,
    device: str = "auto",
    seed: int = 17,
    num_inference_steps: int = 28,
    guidance_scale: float = 2.5,
    offload_mode: str = "none",
) -> KeyframeBackend:
    if name == "flux_kontext":
        return FluxKontextKeyframeBackend(
            model_id=model_id,
            device=device,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            offload_mode=offload_mode,
        )
    raise ValueError(f"Unknown keyframe backend: {name}")
