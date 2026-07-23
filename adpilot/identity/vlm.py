from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


PRODUCT_ANALYSIS_PROMPT = """You are inspecting a single product-reference image for an advertising system.
Return only a JSON object with these fields:
{
  "category": "one of fragrance, beverage, cosmetic, fashion, electronics, general",
  "description": "concise factual description of the visible product",
  "visible_traits": ["3 to 6 concrete visual traits needed to preserve identity"],
  "materials": ["visible materials"],
  "colors": ["dominant product colors"],
  "readable_text": "visible brand or label text, or empty string when uncertain"
}
Do not infer claims that are not visible. Do not add markdown or commentary."""


def parse_json_response(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Vision model did not return a JSON object: {text[:240]}")
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as original_error:
        # Local VLMs commonly emit an otherwise valid object with a trailing
        # comma. Repair only that unambiguous JSON defect; broader repairs would
        # silently change a planner or critic decision.
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        if repaired != candidate:
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                pass
            else:
                if not isinstance(parsed, dict):
                    raise ValueError("Vision model returned a non-object JSON value.")
                return parsed
        excerpt = candidate[:700].replace("\n", "\\n")
        raise ValueError(
            f"Vision model returned invalid JSON at line {original_error.lineno}, "
            f"column {original_error.colno}: {excerpt}"
        ) from original_error
    if not isinstance(parsed, dict):
        raise ValueError("Vision model returned a non-object JSON value.")
    return parsed


class QwenVisionSession:
    """Short-lived local Qwen2.5-VL session shared by product analysis and QA."""

    def __init__(self, model_id: str, device: str = "auto"):
        self.model_id = model_id
        self.requested_device = device
        self.device: str | None = None
        self._torch = None
        self._processor = None
        self._model = None

    def __enter__(self) -> "QwenVisionSession":
        self.load()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover - optional GPU dependency
            raise RuntimeError(
                "Qwen2.5-VL requires current transformers. Run scripts/install_gpu_deps.sh."
            ) from exc
        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Qwen2.5-VL but is not visible.")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(device)
        self._torch = torch
        self.device = device

    def ask(self, image_paths: Iterable[Path], prompt: str, max_new_tokens: int = 220) -> str:
        self.load()
        paths = [Path(path) for path in image_paths]
        if not paths:
            raise ValueError("Qwen2.5-VL needs at least one image.")
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": path.resolve().as_uri()} for path in paths],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            image_inputs = [Image.open(path).convert("RGB") for path in paths]
            video_inputs = None
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._processor = None
        if self.device == "cuda" and self._torch is not None:
            self._torch.cuda.empty_cache()


def analyze_product_image(product_path: Path, model_id: str, device: str = "auto") -> dict[str, Any]:
    """Use Qwen2.5-VL to turn a reference product image into structured facts."""
    with QwenVisionSession(model_id=model_id, device=device) as session:
        return parse_json_response(session.ask([product_path], PRODUCT_ANALYSIS_PROMPT))
