from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image

from adpilot.backends.background import cover_resize
from adpilot.preview.video import export_motion_preview_if_available


class VideoBackend(Protocol):
    name: str

    def render(self, frame_paths: list[Path], output_path: Path) -> Path | None:
        """Render final video from audited keyframes."""


class ProxyVideoBackend:
    name = "proxy"

    def render(self, frame_paths: list[Path], output_path: Path) -> Path | None:
        return export_motion_preview_if_available(frame_paths, output_path)


class WanImageToVideoBackend:
    name = "wan_i2v"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        seed: int = 11,
        num_frames: int = 81,
        fps: int = 16,
        generated_size: tuple[int, int] = (832, 480),
        prompts: list[str] | None = None,
        negative_prompt: str | None = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.5,
        fallback: VideoBackend | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.seed = seed
        self.num_frames = num_frames
        self.fps = fps
        self.generated_size = generated_size
        self.prompts = prompts or []
        self.negative_prompt = negative_prompt or (
            "static image, no motion, added subtitles, title cards, on-screen captions, "
            "watermark, distorted product, extra product, deformed bottle, blurry, "
            "low quality, cluttered background"
        )
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.fallback = fallback
        self._pipe = None
        self._torch = None
        self._np = None
        self.used_fallback = False
        self.errors: list[str] = []
        self.frames_written = 0
        self.frame_dir: str | None = None
        self.output_size: tuple[int, int] | None = None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "device": self.device,
            "seed": self.seed,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "generated_size": self.generated_size,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "prompts": self.prompts,
            "negative_prompt": self.negative_prompt,
            "frames_written": self.frames_written,
            "frame_dir": self.frame_dir,
            "output_size": self.output_size,
            "render_mode": "native_i2v",
            "used_fallback": self.used_fallback,
            "errors": self.errors[-3:],
        }

    def _load(self):
        if self._pipe is not None:
            return
        try:
            import numpy as np
            import torch
            from diffusers import WanImageToVideoPipeline
        except Exception as exc:  # pragma: no cover - optional deps
            raise RuntimeError(
                "Wan I2V backend requires torch, numpy, and a recent diffusers "
                "with WanImageToVideoPipeline. Run scripts/install_gpu_deps.sh "
                "inside the adpilot environment."
            ) from exc

        if self.device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not visible. Refusing slow CPU Wan video diffusion.")
            device = "cuda"
        else:
            device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is false.")

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        try:
            pipe = WanImageToVideoPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        except TypeError:
            pipe = WanImageToVideoPipeline.from_pretrained(self.model_id, dtype=dtype)
        # Keep dtype decisions in from_pretrained(); this call only moves the
        # pipeline to the selected device.
        pipe = pipe.to(device=device)
        self._pipe = pipe
        self._torch = torch
        self._np = np
        self.device = device

    def _model_aligned_size(self, image: Image.Image) -> tuple[int, int]:
        width, height = self.generated_size
        if self._pipe is None or self._np is None:
            return width, height
        try:
            max_area = width * height
            aspect_ratio = image.height / image.width
            mod_value = self._pipe.vae_scale_factor_spatial * self._pipe.transformer.config.patch_size[1]
            height = round(self._np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(self._np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            return max(int(width), mod_value), max(int(height), mod_value)
        except Exception:
            return width, height

    def _prompt_for(self, index: int) -> str:
        if index < len(self.prompts) and self.prompts[index]:
            return self.prompts[index]
        if self.prompts:
            return self.prompts[0]
        return "cinematic product commercial, premium lighting, smooth camera movement"

    def _to_rgb_image(self, frame) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if self._torch is not None and hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        if self._np is None:
            import numpy as np
        else:
            np = self._np
        array = np.asarray(frame)
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.dtype != np.uint8:
            max_value = float(array.max()) if array.size else 1.0
            if max_value <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array, mode="L").convert("RGB")
        if array.ndim == 3 and array.shape[-1] == 4:
            return Image.fromarray(array, mode="RGBA").convert("RGB")
        return Image.fromarray(array).convert("RGB")

    def render(self, frame_paths: list[Path], output_path: Path) -> Path | None:
        try:
            self._load()
            frame_dir = output_path.parent / "wan_i2v_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            self.frame_dir = str(frame_dir)

            generated_frames: list[Image.Image] = []
            for index, frame_path in enumerate(frame_paths):
                source = Image.open(frame_path).convert("RGB")
                width, height = self._model_aligned_size(source)
                image = cover_resize(source, (width, height))
                generator = self._torch.Generator(device=self.device).manual_seed(self.seed + index)
                frames = self._pipe(
                    image=image,
                    prompt=self._prompt_for(index),
                    negative_prompt=self.negative_prompt,
                    height=height,
                    width=width,
                    num_frames=self.num_frames,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    generator=generator,
                ).frames[0]
                for frame_index, frame in enumerate(frames):
                    rgb = self._to_rgb_image(frame)
                    rgb.save(frame_dir / f"shot_{index + 1:02d}_frame_{frame_index:03d}.png")
                    generated_frames.append(rgb)

            if not generated_frames:
                raise RuntimeError("Wan I2V produced zero frames.")
            self.frames_written = len(generated_frames)
            self.output_size = generated_frames[0].size
            self._write_mp4(generated_frames, output_path)
            if not self._is_readable_video(output_path):
                raise RuntimeError(
                    f"Video file was not written correctly: {output_path} "
                    f"({output_path.stat().st_size if output_path.exists() else 0} bytes)."
                )
            return output_path
        except Exception as exc:
            self.used_fallback = True
            self.errors.append(f"{type(exc).__name__}: {exc}")
            if self.fallback is None:
                raise
            return self.fallback.render(frame_paths, output_path)

    def _write_mp4(self, frames: list[Image.Image], output_path: Path) -> None:
        import cv2
        import numpy as np

        width, height = frames[0].size
        for codec in ("mp4v", "avc1"):
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*codec),
                self.fps,
                (width, height),
            )
            if not writer.isOpened():
                continue
            for frame in frames:
                arr = np.asarray(frame.convert("RGB").resize((width, height)))
                writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            writer.release()
            if self._is_readable_video(output_path):
                return
        raise RuntimeError("OpenCV could not encode a readable mp4 with mp4v or avc1.")

    def _is_readable_video(self, path: Path) -> bool:
        if not path.exists() or path.stat().st_size < 4096:
            return False
        try:
            import cv2
        except Exception:
            return True
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        cap.release()
        return frame_count > 0 and ok


def make_video_backend(
    name: str,
    model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    device: str = "auto",
    seed: int = 11,
    num_frames: int = 14,
    fps: int = 7,
    generated_size: tuple[int, int] = (1024, 576),
    prompts: list[str] | None = None,
    negative_prompt: str | None = None,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.5,
    fallback_on_error: bool = True,
) -> VideoBackend:
    if name == "proxy":
        return ProxyVideoBackend()
    if name == "wan_i2v":
        fallback = ProxyVideoBackend() if fallback_on_error else None
        return WanImageToVideoBackend(
            model_id=model_id,
            device=device,
            seed=seed,
            num_frames=num_frames,
            fps=fps,
            generated_size=generated_size,
            prompts=prompts,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fallback=fallback,
        )
    raise ValueError(f"Unknown video backend: {name}")
