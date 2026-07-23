from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from adpilot.identity.card import IdentityCard
from adpilot.identity.vlm import QwenVisionSession, parse_json_response
from adpilot.planner.schema import AdPlan, ShotPlan


def _platform_context(platform: str) -> str:
    return {
        "shorts": "vertical 9:16 social ad",
        "instagram": "polished vertical Instagram ad",
        "tiktok": "energetic vertical TikTok ad",
        "youtube": "vertical YouTube Shorts ad",
        "landscape": "16:9 cinematic commercial",
    }.get(platform, "vertical social ad")


def _category_direction(identity_card: IdentityCard) -> str:
    brief = identity_card.product_brief
    category = brief.get("category", "general")
    package_state = str(brief.get("package_state") or "").lower()
    if category != "snack":
        return "Make the three shots visually connected through a shared palette and campaign world, not three unrelated backgrounds."
    if package_state == "open_with_contents":
        return (
            "For this snack ad, make a coherent gold tabletop candy campaign: first show the upright package hero; "
            "second show a few matching gummies naturally falling from a freshly opened top onto the tabletop; "
            "third is a clean upright final packshot with a few gummies resting beside it. Never show candy floating "
            "inside an opaque package, a huge spill, or the package rotating on a turntable."
        )
    return (
        "For this snack ad, make a coherent gold tabletop candy campaign: first show the upright package hero; "
        "second show a few matching gummies rolling or settling beside the sealed package; third is a clean upright "
        "final packshot with a few gummies at its base. Never show candy inside the sealed package or the package rotating."
    )


def _planner_prompt(identity_card: IdentityCard, style: str, duration: int, platform: str, reference_count: int = 1) -> str:
    brief = json.dumps(identity_card.product_brief, ensure_ascii=True)
    reference_context = (
        "The supplied images show the same product from multiple real views. Assign one supplied view to each shot; "
        "do not invent a new product angle."
        if reference_count > 1
        else "The supplied image is the only product reference. Do not request an unseen product angle."
    )
    return f"""You are an art director planning a short product advertisement.
{reference_context} Product facts: {brief}
Brand: {identity_card.brand}. Platform: {_platform_context(platform)}. Total duration: about {duration} seconds.
Requested style: {style}.
Category direction: {_category_direction(identity_card)}
Return only JSON with exactly three visually distinct shots:
{{
  "concept": "specific campaign idea",
  "style": "specific visual style",
  "shots": [
    {{"goal": "hook or detail or packshot", "scene_prompt": "physical set only", "product_position": "center|left|right", "product_scale": 0.42, "motion_prompt": "camera move plus visible environmental motion"}}
  ]
}}
The sequence must progress from hook, to contextual detail, to final packshot. Use product_scale
near 0.42 for the hook, 0.32 for context, and 0.46 for the final packshot.
For every scene_prompt, write 8-14 words describing only the physical set: at least two concrete
props or materials plus a light, color, or atmosphere cue. Never include product, bottle, perfume,
brand, close-up, camera, or shot. Example: "blush silk, peony petals, warm side light, mirrored tray".
For every motion_prompt, name one camera movement and one visible event appropriate to the category.
Keep exactly one product package visible and do not request typography, people, extra packages, or a
different package."""


def _planner_retry_prompt(
    identity_card: IdentityCard,
    style: str,
    duration: int,
    platform: str,
    previous: str,
    failure_reason: str,
) -> str:
    return f"""Your previous plan was invalid or unusable. Return a new complete, compact JSON object only.
Use this exact schema and no markdown:
{{"concept":"campaign idea","style":"visual style","shots":[
{{"goal":"hook","scene_prompt":"physical set only","product_position":"center","product_scale":0.42,"motion_prompt":"camera plus scene motion"}},
{{"goal":"context","scene_prompt":"physical set only","product_position":"left","product_scale":0.32,"motion_prompt":"camera plus scene motion"}},
{{"goal":"packshot","scene_prompt":"physical set only","product_position":"center","product_scale":0.46,"motion_prompt":"camera plus scene motion"}}
]}}
Product facts: {json.dumps(identity_card.product_brief, ensure_ascii=True)}
Brand: {identity_card.brand}. Platform: {_platform_context(platform)}. Style: {style}. Duration: {duration} seconds.
Category direction: {_category_direction(identity_card)}
The previous plan failed this creative validation: {failure_reason}
Each scene_prompt must have 8-14 words, two concrete set materials or props, and a light/color cue.
It must not mention product, bottle, perfume, brand, camera, close-up, or shot. Each motion_prompt
must include both a camera move and a visible environmental motion. Do not copy or explain the previous output.
Previous output for correction context: {previous[:1800]}"""


def _text(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"VLM planner returned an empty {field}.")
    return " ".join(text.split())


def _scale(value: object) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("VLM planner returned a non-numeric product_scale.") from exc
    if not 0.18 <= scale <= 0.58:
        raise ValueError(f"VLM planner product_scale {scale} is outside [0.18, 0.58].")
    return round(scale, 3)


def _scene_prompt(value: object) -> str:
    """Keep the planner resilient; downstream prompt building repairs weak scene text."""
    return _text(value, "scene_prompt")


def build_vlm_plan(response: dict, duration: int) -> AdPlan:
    shots_data = response.get("shots")
    if not isinstance(shots_data, list) or len(shots_data) != 3:
        raise ValueError("VLM planner must return exactly three shots.")
    per_shot = max(2, duration // 3)
    shots: list[ShotPlan] = []
    minimum_scales = (0.34, 0.28, 0.38)
    for index, shot_data in enumerate(shots_data, start=1):
        if not isinstance(shot_data, dict):
            raise ValueError("VLM planner returned a non-object shot.")
        position = _text(shot_data.get("product_position"), "product_position").lower()
        if position not in {"center", "left", "right"}:
            raise ValueError(f"VLM planner returned unsupported product_position '{position}'.")
        scene = _scene_prompt(shot_data.get("scene_prompt"))
        shots.append(
            ShotPlan(
                shot_id=f"shot_{index:02d}",
                duration=per_shot,
                goal=_text(shot_data.get("goal"), "goal"),
                background_prompt=scene,
                product_position=position,
                product_scale=max(_scale(shot_data.get("product_scale")), minimum_scales[index - 1]),
                motion_prompt=_text(shot_data.get("motion_prompt"), "motion_prompt"),
            )
        )
    return AdPlan(
        concept=_text(response.get("concept"), "concept"),
        style=_text(response.get("style"), "style"),
        shots=shots,
    )


def validate_creative_plan(identity_card: IdentityCard, plan: AdPlan) -> None:
    """Reject category-specific physical actions that would make an ad implausible."""
    brief = identity_card.product_brief
    if brief.get("category") != "snack":
        return

    all_motion = " ".join(shot.motion_prompt.lower() for shot in plan.shots)
    if any(word in all_motion for word in ("rotate", "rotating", "rotation", "turntable", "spin", "spinning")):
        raise ValueError("snack plan requests package rotation")

    if str(brief.get("package_state") or "").lower() == "open_with_contents":
        middle_beat = f"{plan.shots[1].goal} {plan.shots[1].motion_prompt}".lower()
        natural_actions = ("fall", "tumble", "drop", "land", "roll", "settle")
        if not any(action in middle_beat for action in natural_actions):
            raise ValueError("open snack plan lacks a natural middle gummy action")


def apply_category_storyboard(identity_card: IdentityCard, plan: AdPlan) -> AdPlan:
    """Separate snack opening and final beats while preserving the model-planned middle action."""
    if identity_card.product_brief.get("category") != "snack":
        return plan

    shots = list(plan.shots)
    shots[0] = replace(
        shots[0],
        goal="opening hero reveal with bright candy-world energy",
        product_position="left",
        product_scale=0.38,
        motion_prompt="gentle lateral glide as a warm light sweep moves across the upright pouch",
    )
    shots[2] = replace(
        shots[2],
        goal="signature final packshot",
        product_position="center",
        product_scale=0.46,
        motion_prompt="locked hero composition as reflections and soft tabletop shadows shift",
    )
    return replace(plan, shots=shots)


def make_vlm_plan(
    identity_card: IdentityCard,
    product_path: Path,
    style: str,
    duration: int,
    platform: str,
    model_id: str,
    device: str = "auto",
    reference_paths: list[Path] | None = None,
) -> AdPlan:
    references = [Path(product_path), *(Path(path) for path in (reference_paths or []))]
    with QwenVisionSession(model_id=model_id, device=device) as session:
        raw_response = session.ask(
            references,
            _planner_prompt(identity_card, style, duration, platform, reference_count=len(references)),
            max_new_tokens=480,
        )
        try:
            response = parse_json_response(raw_response)
            plan = build_vlm_plan(response, duration)
            validate_creative_plan(identity_card, plan)
            return apply_category_storyboard(identity_card, plan)
        except ValueError as first_error:
            retry_response = session.ask(
                references,
                _planner_retry_prompt(identity_card, style, duration, platform, raw_response, str(first_error)),
                max_new_tokens=480,
            )
            try:
                response = parse_json_response(retry_response)
                plan = build_vlm_plan(response, duration)
                validate_creative_plan(identity_card, plan)
                return apply_category_storyboard(identity_card, plan)
            except ValueError as retry_error:
                raise ValueError(
                    "Qwen planner returned an invalid or unusable plan twice. "
                    f"First error: {first_error}. Retry error: {retry_error}"
                ) from retry_error
