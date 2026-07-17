from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageFilter

from adpilot.identity.card import IdentityCard
from adpilot.planner.schema import ShotPlan


POSITIONS = {
    "center": (0.5, 0.58),
    "left": (0.32, 0.6),
    "right": (0.68, 0.6),
}


@dataclass
class RenderedShot:
    path: Path
    background_path: Path
    product_bbox: tuple[int, int, int, int]
    logo_bbox_in_frame: tuple[int, int, int, int] | None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["path"] = str(self.path)
        data["background_path"] = str(self.background_path)
        return data


def _clamp_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def composite_product(
    background: Image.Image,
    identity_card: IdentityCard,
    shot: ShotPlan,
    motion_progress: float = 0.0,
) -> tuple[Image.Image, tuple[int, int, int, int], tuple[int, int, int, int] | None]:
    canvas = background.convert("RGBA")
    product = Image.open(identity_card.cutout).convert("RGBA")
    # The foreground product is deliberately composited after video generation.
    # It stays the exact reference asset instead of being redrawn by the video model.
    zoom = 1.0 + 0.035 * max(0.0, min(1.0, motion_progress))
    target_width = int(canvas.width * shot.product_scale * zoom)
    target_height = int(target_width / max(identity_card.aspect_ratio, 0.01))
    product = product.resize((target_width, target_height), Image.Resampling.LANCZOS)

    cx, cy = POSITIONS.get(shot.product_position, POSITIONS["center"])
    x = int(canvas.width * cx - product.width / 2)
    y = int(canvas.height * cy - product.height / 2)

    shadow = Image.new("RGBA", product.size, (0, 0, 0, 0))
    alpha = product.getchannel("A")
    shadow_alpha = alpha.point(lambda value: int(value * 0.26))
    shadow.putalpha(shadow_alpha.filter(ImageFilter.GaussianBlur(18)))
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_layer.alpha_composite(shadow, (x + 18, y + 24))
    canvas = Image.alpha_composite(canvas, shadow_layer)
    canvas.alpha_composite(product, (x, y))

    product_bbox = _clamp_bbox((x, y, x + product.width, y + product.height), canvas.width, canvas.height)
    logo_bbox_in_frame = None
    if identity_card.logo_bbox:
        x1, y1, x2, y2 = identity_card.logo_bbox
        scale = product.width / max(identity_card.width, 1)
        logo_bbox_in_frame = _clamp_bbox(
            (
                x + int(x1 * scale),
                y + int(y1 * scale),
                x + int(x2 * scale),
                y + int(y2 * scale),
            ),
            canvas.width,
            canvas.height,
        )

    return canvas, product_bbox, logo_bbox_in_frame


def render_shot(
    background: Image.Image,
    identity_card: IdentityCard,
    shot: ShotPlan,
    output_path: Path,
) -> RenderedShot:
    background_path = output_path.with_name(f"{output_path.stem}_background.jpg")
    background.convert("RGB").save(background_path, quality=95)
    canvas, product_bbox, logo_bbox_in_frame = composite_product(background, identity_card, shot)
    # Keep keyframes free of graphics. Wan treats the full keyframe as scene
    # content, so a baked-in caption box becomes an unwanted moving object.
    canvas.convert("RGB").save(output_path, quality=95)
    return RenderedShot(
        path=output_path,
        background_path=background_path,
        product_bbox=product_bbox,
        logo_bbox_in_frame=logo_bbox_in_frame,
    )
