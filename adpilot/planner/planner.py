from __future__ import annotations

from adpilot.identity.card import IdentityCard
from adpilot.planner.schema import AdPlan, ShotPlan


def _join_keywords(values: list[str], limit: int = 3) -> str:
    return ", ".join(value for value in values[:limit] if value)


def _compact_text(text: str, max_words: int = 10) -> str:
    words = text.replace("(", "").replace(")", "").split()
    return " ".join(words[:max_words])


def make_default_plan(identity_card: IdentityCard, style: str, duration: int, platform: str = "shorts") -> AdPlan:
    per_shot = max(2, duration // 3)
    brand = identity_card.brand
    brief = identity_card.product_brief
    category = brief.get("category", "general")
    mood = _compact_text(brief.get("mood", "premium, clear, product-focused"), max_words=8)
    audience = brief.get("audience", "online shoppers")
    scene_keywords = _join_keywords(brief.get("scene_keywords", []), limit=3)
    style_text = mood if style == "auto" else f"{style}, {mood}"
    platform_context = {
        "shorts": "vertical social ad",
        "instagram": "polished instagram ad",
        "tiktok": "energetic social ad",
        "youtube": "youtube shorts ad",
        "landscape": "16:9 cinematic commercial",
    }.get(platform, "vertical social short ad")
    hero_scale, context_scale, packshot_scale = (
        (0.3, 0.26, 0.34) if platform == "landscape" else (0.42, 0.34, 0.48)
    )
    return AdPlan(
        concept=f"{brand} {category} ad for {audience}",
        style=f"{style_text}, {platform_context}",
        shots=[
            ShotPlan(
                shot_id="shot_01",
                duration=per_shot,
                goal="product-aware hook",
                background_prompt=(
                    f"{style_text}, {platform_context}, {scene_keywords}, "
                    f"luxury {category} campaign environment, empty center foreground, no product or bottle"
                ),
                product_position="center",
                product_scale=hero_scale,
            ),
            ShotPlan(
                shot_id="shot_02",
                duration=per_shot,
                goal="category-specific lifestyle context",
                background_prompt=(
                    f"{style_text}, {platform_context}, {scene_keywords}, "
                    f"luxury {category} campaign environment, clean right foreground, no product or bottle"
                ),
                product_position="right",
                product_scale=context_scale,
            ),
            ShotPlan(
                shot_id="shot_03",
                duration=per_shot,
                goal="final packshot",
                background_prompt=(
                    f"{style_text}, {platform_context}, {scene_keywords}, "
                    f"premium packshot surface, soft shadow, no product or bottle"
                ),
                product_position="center",
                product_scale=packshot_scale,
            ),
        ],
    )
