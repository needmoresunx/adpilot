from __future__ import annotations

from pathlib import Path

from PIL import Image


def caption_product_image(
    product_path: Path,
    model_id: str,
    device: str = "auto",
    max_new_tokens: int = 28,
) -> str:
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except Exception as exc:
        raise RuntimeError("VLM auto-brief requires torch and transformers.") from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)

    image = Image.open(product_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return " ".join(caption.split())
