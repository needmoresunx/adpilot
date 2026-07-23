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
    visible_traits: list[str]
    materials: list[str]
    colors: list[str]
    readable_text: str | None
    recognition_source: str
    identity_anchors: str | None = None
    package_state: str | None = None
    recognition_error: str | None = None

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
    "electronics": {
        "description": "a consumer electronic device",
        "audience": "technology-minded consumers",
        "mood": "precise, modern, clean, high-tech",
        "scene_keywords": [
            "minimal technology set",
            "polished acrylic surface",
            "controlled light reflections",
            "precise product commercial",
        ],
        "negative_keywords": ["food", "flowers", "fabric", "kitchen", "messy cables"],
    },
    "snack": {
        "description": "a packaged snack or candy product",
        "audience": "snack shoppers and families",
        "mood": "playful, bright, appetizing, energetic",
        "scene_keywords": [
            "sunny tabletop",
            "clean gold backdrop",
            "bright commercial studio light",
            "cheerful snack campaign",
        ],
        "negative_keywords": ["perfume", "cosmetics vanity", "alcohol", "medical packaging", "dark luxury room"],
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
        ("snack", ("jelly", "gummy", "candy", "haribo", "snack", "sweets", "confection")),
        ("cosmetic", ("cosmetic", "skincare", "serum", "cream", "lipstick", "beauty", "lotion")),
        ("beverage", ("bottle", "drink", "water", "juice", "coffee", "tea", "soda", "beverage")),
        ("electronics", ("earbud", "headphone", "electronic", "speaker", "keyboard", "camera")),
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
    vision_analysis: dict | None = None,
    identity_anchors: str | None = None,
    package_state: str | None = None,
    recognition_error: str | None = None,
) -> ProductBrief:
    analysis = vision_analysis or {}
    analyzed_category = analysis.get("category")
    if analyzed_category not in CATEGORY_PRESETS:
        analyzed_category = None
    analyzed_description = analysis.get("description")
    resolved_category = category or analyzed_category or infer_category(product_path, aspect_ratio, visual_caption=visual_caption)
    preset = CATEGORY_PRESETS.get(resolved_category, CATEGORY_PRESETS["general"])
    resolved_description = description or analyzed_description or preset["description"]
    if visual_caption and not description and not analyzed_description:
        resolved_description = f"{preset['description']} ({visual_caption})"
    visible_traits = [str(value).strip() for value in analysis.get("visible_traits", []) if str(value).strip()]
    materials = [str(value).strip() for value in analysis.get("materials", []) if str(value).strip()]
    colors = [str(value).strip() for value in analysis.get("colors", []) if str(value).strip()]
    readable_text = str(analysis.get("readable_text") or "").strip() or None
    resolved_anchors = str(identity_anchors or "").strip() or None
    resolved_package_state = str(package_state or "").strip().lower() or None
    if vision_analysis:
        recognition_source = "qwen2_5_vl"
    elif visual_caption:
        recognition_source = "vlm_caption"
    elif category or description or audience or mood:
        recognition_source = "user_override"
    else:
        recognition_source = "filename_aspect_ratio"
    return ProductBrief(
        category=resolved_category,
        description=resolved_description,
        audience=audience or preset["audience"],
        mood=mood or preset["mood"],
        scene_keywords=list(preset["scene_keywords"]),
        negative_keywords=list(preset["negative_keywords"]),
        visual_caption=visual_caption,
        visible_traits=visible_traits[:8],
        materials=materials[:6],
        colors=colors[:6],
        readable_text=readable_text,
        recognition_source=recognition_source,
        identity_anchors=resolved_anchors,
        package_state=resolved_package_state,
        recognition_error=recognition_error,
    )
