from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from adpilot.identity.card import IdentityCard
from adpilot.identity.color import normalized_rgb_distance
from adpilot.planner.schema import ShotPlan


@dataclass
class CritiqueReport:
    shot_id: str
    passed: bool
    product_scale: float
    color_delta: float
    shape_score: float
    logo_area_ratio: float | None
    product_bbox: tuple[int, int, int, int] | None
    logo_bbox_in_frame: tuple[int, int, int, int] | None
    ocr_text: str | None
    ocr_available: bool
    failure_reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if not bbox:
        return 0
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _crop(image: Image.Image, bbox: tuple[int, int, int, int] | None) -> Image.Image | None:
    if not bbox or _bbox_area(bbox) == 0:
        return None
    return image.crop(bbox)


def _masked_median_rgb(frame_crop: Image.Image, reference_rgba: Image.Image) -> tuple[int, int, int]:
    frame_rgb = np.asarray(frame_crop.convert("RGB"))
    alpha = np.asarray(reference_rgba.convert("RGBA"))[:, :, 3]
    pixels = frame_rgb[alpha > 16]
    if pixels.size == 0:
        pixels = frame_rgb.reshape(-1, 3)
    rgb = np.median(pixels, axis=0)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _read_logo_text(logo_crop: Image.Image | None) -> tuple[str | None, bool]:
    if logo_crop is None:
        return None, False
    try:
        import pytesseract
    except Exception:
        return None, False
    try:
        text = pytesseract.image_to_string(logo_crop.convert("RGB"), config="--psm 7")
    except Exception:
        return None, True
    return " ".join(text.split()) or None, True


def critique_shot(
    identity_card: IdentityCard,
    shot: ShotPlan,
    frame_path: Path,
    render_metadata: dict | None = None,
) -> CritiqueReport:
    frame = Image.open(frame_path).convert("RGBA")
    product_bbox = None
    logo_bbox_in_frame = None
    if render_metadata:
        product_bbox = tuple(render_metadata.get("product_bbox") or ()) or None
        logo_bbox_in_frame = tuple(render_metadata.get("logo_bbox_in_frame") or ()) or None

    if product_bbox:
        x1, y1, x2, y2 = product_bbox
        target_width = max(1, x2 - x1)
        target_height = max(1, y2 - y1)
    else:
        target_width = int(frame.width * shot.product_scale)
        target_height = int(target_width / max(identity_card.aspect_ratio, 0.01))

    shape_ratio = (target_width / max(target_height, 1)) / max(identity_card.aspect_ratio, 0.01)
    shape_score = round(max(0.0, 1.0 - abs(1.0 - shape_ratio)), 4)

    product = Image.open(identity_card.cutout).convert("RGBA")
    product_crop = _crop(frame, product_bbox)
    if product_crop is not None:
        reference = product.resize(product_crop.size, Image.Resampling.LANCZOS)
        observed_rgb = _masked_median_rgb(product_crop, reference)
    else:
        observed_rgb = identity_card.dominant_rgb
    color_delta = normalized_rgb_distance(identity_card.dominant_rgb, observed_rgb)

    logo_area_ratio = None
    if logo_bbox_in_frame:
        logo_area_ratio = round(_bbox_area(logo_bbox_in_frame) / (frame.width * frame.height), 6)

    ocr_text, ocr_available = _read_logo_text(_crop(frame, logo_bbox_in_frame))

    failure_reasons: list[str] = []
    min_scale = 0.24 if frame.width >= frame.height else 0.38
    if shot.product_scale < min_scale:
        failure_reasons.append("product_too_small")
    if color_delta > 0.12:
        failure_reasons.append("color_shift")
    if shape_score < 0.9:
        failure_reasons.append("shape_distortion")
    if logo_area_ratio is not None and logo_area_ratio < 0.002:
        failure_reasons.append("logo_region_too_small")
    if product_bbox is not None and _bbox_area(product_bbox) == 0:
        failure_reasons.append("product_out_of_frame")

    return CritiqueReport(
        shot_id=shot.shot_id,
        passed=not failure_reasons,
        product_scale=shot.product_scale,
        color_delta=color_delta,
        shape_score=shape_score,
        logo_area_ratio=logo_area_ratio,
        product_bbox=product_bbox,
        logo_bbox_in_frame=logo_bbox_in_frame,
        ocr_text=ocr_text,
        ocr_available=ocr_available,
        failure_reasons=failure_reasons,
    )
