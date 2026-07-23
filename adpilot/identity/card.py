from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class IdentityCard:
    brand: str
    product_path: str
    cutout_path: str
    width: int
    height: int
    aspect_ratio: float
    dominant_rgb: tuple[int, int, int]
    product_brief: dict
    logo_bbox: tuple[int, int, int, int] | None = None
    cutout_method: str = "none"
    recognition_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def cutout(self) -> Path:
        return Path(self.cutout_path)
