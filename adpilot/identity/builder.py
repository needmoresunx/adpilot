from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from adpilot.identity.brief import build_product_brief
from adpilot.identity.card import IdentityCard
from adpilot.identity.color import dominant_rgb
from adpilot.identity.vlm import analyze_product_image
from adpilot.utils.json_io import write_json


def build_identity_card(
    product_path: Path,
    brand: str,
    output_dir: Path,
    logo_bbox: tuple[int, int, int, int] | None = None,
    auto_cutout: bool = False,
    cutout_model: str = "birefnet-general",
    product_category: str | None = None,
    product_description: str | None = None,
    target_audience: str | None = None,
    ad_mood: str | None = None,
    identity_anchors: str | None = None,
    package_state: str | None = None,
    auto_brief: bool = False,
    vlm_model: str | None = None,
    vlm_device: str = "auto",
) -> IdentityCard:
    product = Image.open(product_path).convert("RGBA")
    cutout_method = "original"
    if auto_cutout:
        if cached_rembg_model_path(cutout_model) is not None:
            try:
                product = make_model_cutout(product, model_name=cutout_model)
                cutout_method = f"rembg:{cutout_model}"
            except RuntimeError:
                product = make_simple_cutout(product)
                cutout_method = "heuristic_bright_background"
        else:
            # GPU nodes often cannot reach the rembg release host. A bright,
            # isolated packshot can still be cropped deterministically without
            # downloading a segmentation weight during generation.
            product = make_simple_cutout(product)
            cutout_method = "heuristic_bright_background"
        product, crop_offset = trim_transparent_padding(product)
        if logo_bbox:
            ox, oy = crop_offset
            x1, y1, x2, y2 = logo_bbox
            logo_bbox = (x1 - ox, y1 - oy, x2 - ox, y2 - oy)
    cutout_path = output_dir / "product_cutout.png"
    product.save(cutout_path)

    width, height = product.size
    if logo_bbox:
        x1, y1, x2, y2 = logo_bbox
        logo_bbox = (
            max(0, min(width, x1)),
            max(0, min(height, y1)),
            max(0, min(width, x2)),
            max(0, min(height, y2)),
        )
        if logo_bbox[2] <= logo_bbox[0] or logo_bbox[3] <= logo_bbox[1]:
            logo_bbox = None
    aspect_ratio = round(width / max(height, 1), 4)
    vision_analysis = None
    recognition_error = None
    if auto_brief and vlm_model:
        try:
            vision_analysis = analyze_product_image(product_path, vlm_model, device=vlm_device)
        except Exception as exc:
            recognition_error = f"{type(exc).__name__}: {exc}"
    product_brief = build_product_brief(
        product_path=product_path,
        aspect_ratio=aspect_ratio,
        category=product_category,
        description=product_description,
        audience=target_audience,
        mood=ad_mood,
        visual_caption=(vision_analysis or {}).get("description"),
        vision_analysis=vision_analysis,
        identity_anchors=identity_anchors,
        package_state=package_state,
        recognition_error=recognition_error,
    )
    card = IdentityCard(
        brand=brand,
        product_path=str(product_path),
        cutout_path=str(cutout_path),
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        dominant_rgb=dominant_rgb(product),
        product_brief=product_brief.to_dict(),
        logo_bbox=logo_bbox,
        cutout_method=cutout_method,
        recognition_error=recognition_error,
    )
    write_json(output_dir / "identity_card.json", card.to_dict())
    return card


def make_simple_cutout(image: Image.Image, threshold: int = 42, feather: int = 36) -> Image.Image:
    """Create a soft alpha matte for a product on a nearly solid bright background."""
    rgba = np.asarray(image.convert("RGBA")).copy()
    alpha = rgba[:, :, 3]
    if np.any(alpha < 250):
        return image

    h, w = rgba.shape[:2]
    patches = [
        rgba[: max(1, h // 12), : max(1, w // 12), :3],
        rgba[: max(1, h // 12), -max(1, w // 12) :, :3],
        rgba[-max(1, h // 12) :, : max(1, w // 12), :3],
        rgba[-max(1, h // 12) :, -max(1, w // 12) :, :3],
    ]
    corner_pixels = np.concatenate([patch.reshape(-1, 3) for patch in patches], axis=0)
    background = np.median(corner_pixels, axis=0)
    distance = np.linalg.norm(rgba[:, :, :3].astype(np.float32) - background, axis=2)
    bright_background = np.mean(background) > 200
    if not bright_background:
        return image

    # JPEG product photos rarely have a perfectly uniform white background.
    # A soft matte removes the residual rectangular background without making
    # glass edges look jagged when composited over the generated scene.
    matte = np.clip((distance - threshold) / max(feather, 1), 0.0, 1.0)
    rgba[:, :, 3] = (alpha.astype(np.float32) * matte).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def cached_rembg_model_path(model_name: str) -> Path | None:
    filenames = {
        "birefnet-general": "BiRefNet-general-epoch_244.onnx",
    }
    filename = filenames.get(model_name)
    if not filename:
        return None
    cache_dir = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))
    path = cache_dir / filename
    return path if path.is_file() and path.stat().st_size > 0 else None


def make_model_cutout(image: Image.Image, model_name: str) -> Image.Image:
    """Remove a product-photo background with a real segmentation model."""
    try:
        from rembg import new_session, remove
    except ImportError as exc:  # pragma: no cover - depends on optional GPU deps
        raise RuntimeError(
            "--auto-cutout requires rembg. Run scripts/install_gpu_deps.sh, then "
            "scripts/download_models.sh on the login node."
        ) from exc

    try:
        session = new_session(model_name)
        return remove(image, session=session).convert("RGBA")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the product segmentation model '{model_name}'. "
            "Run scripts/download_models.sh on a node with internet access."
        ) from exc


def trim_transparent_padding(image: Image.Image, padding: int = 8) -> tuple[Image.Image, tuple[int, int]]:
    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 16)
    if xs.size == 0 or ys.size == 0:
        return image, (0, 0)
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(image.width, int(xs.max()) + padding + 1)
    y2 = min(image.height, int(ys.max()) + padding + 1)
    return image.crop((x1, y1, x2, y2)), (x1, y1)
