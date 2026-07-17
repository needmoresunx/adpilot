from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ProductBrief:
    category: str
    description: str
    audience: str
    mood: str
    scene_keywords: list[str]
    negative_keywords: list[str]
    visual_caption: str | None
    recognition_source: str

    def to_dict(self) -> dict:
        return asdict(self)


CATEGORY_PRESETS = {
    "fragrance": {
        "description": "a premium fragrance or perfume bottle",
        "audience": "style-conscious luxury shoppers",
        "mood": "elegant, sensual, refined, cinematic",
        "scene_keywords": [
            "luxury vanity table",
            "soft evening light",
            "glass reflections",
            "silk fabric",
            "subtle floral atmosphere",
            "high-end beauty commercial",
        ],
        "negative_keywords": ["gym", "sports", "fitness", "plastic bottle", "kitchen", "office clutter"],
    },
    "beverage": {
        "description": "a beverage bottle or drink product",
        "audience": "active everyday consumers",
        "mood": "fresh, crisp, energetic, clean",
        "scene_keywords": [
            "cold condensation",
            "fresh splash",
            "clean studio product ad",
            "bright lifestyle setting",
            "refreshing commercial lighting",
        ],
        "negative_keywords": ["perfume", "cosmetics vanity", "dark luxury room", "medical product"],
    },
    "cosmetic": {
        "description": "a beauty or skincare product",
        "audience": "beauty and self-care consumers",
        "mood": "clean, premium, soft, polished",
        "scene_keywords": [
            "bathroom vanity",
            "soft daylight",
            "cream ceramic surface",
            "botanical skincare ingredients",
            "beauty product commercial",
        ],
        "negative_keywords": ["gym", "industrial tools", "food plate", "sports bottle"],
    },
    "fashion": {
        "description": "a fashion item",
        "audience": "style-conscious shoppers",
        "mood": "editorial, expressive, modern",
        "scene_keywords": [
            "fashion editorial set",
            "styled wardrobe details",
            "studio lighting",
            "premium retail campaign",
        ],
        "negative_keywords": ["food", "kitchen", "medical", "bottle condensation"],
    },
    "general": {
        "description": "a consumer product",
        "audience": "online shoppers",
        "mood": "premium, clear, product-focused",
        "scene_keywords": [
            "clean commercial product set",
            "premium studio lighting",
            "minimal background",
            "lifestyle context",
        ],
        "negative_keywords": ["distorted product", "extra logo", "random text", "clutter"],
    },
}


def infer_category(product_path: Path, aspect_ratio: float, visual_caption: str | None = None) -> str:
    text = product_path.stem.lower().replace("_", " ").replace("-", " ")
    if visual_caption:
        text = f"{text} {visual_caption.lower()}"
    checks = [
        ("fragrance", ("perfume", "fragrance", "cologne", "parfum", "eau de toilette", "bottle of perfume")),
        ("cosmetic", ("cosmetic", "skincare", "serum", "cream", "lipstick", "beauty", "lotion")),
        ("beverage", ("bottle", "drink", "water", "juice", "coffee", "tea", "soda", "beverage")),
        ("fashion", ("shirt", "shoe", "bag", "dress", "jacket", "fashion", "wear")),
    ]
    for category, keywords in checks:
        if any(keyword in text for keyword in keywords):
            return category
    if aspect_ratio < 0.55:
        return "beverage"
    return "general"


def build_product_brief(
    product_path: Path,
    aspect_ratio: float,
    category: str | None = None,
    description: str | None = None,
    audience: str | None = None,
    mood: str | None = None,
    visual_caption: str | None = None,
) -> ProductBrief:
    resolved_category = category or infer_category(product_path, aspect_ratio, visual_caption=visual_caption)
    preset = CATEGORY_PRESETS.get(resolved_category, CATEGORY_PRESETS["general"])
    resolved_description = description or preset["description"]
    if visual_caption and not description:
        resolved_description = f"{preset['description']} ({visual_caption})"
    return ProductBrief(
        category=resolved_category,
        description=resolved_description,
        audience=audience or preset["audience"],
        mood=mood or preset["mood"],
        scene_keywords=list(preset["scene_keywords"]),
        negative_keywords=list(preset["negative_keywords"]),
        visual_caption=visual_caption,
        recognition_source="vlm_caption" if visual_caption else ("user_override" if category else "filename_aspect_ratio"),
    )
