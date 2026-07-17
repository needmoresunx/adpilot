from __future__ import annotations

from dataclasses import replace

from adpilot.critic.critique import CritiqueReport
from adpilot.planner.schema import ShotPlan


def repair_shot(shot: ShotPlan, report: CritiqueReport) -> tuple[ShotPlan, list[str]]:
    actions: list[str] = []
    repaired = shot

    if "product_too_small" in report.failure_reasons or "logo_region_too_small" in report.failure_reasons:
        new_scale = min(0.56, repaired.product_scale + 0.1)
        repaired = replace(repaired, product_scale=round(new_scale, 2), product_position="center")
        actions.append(f"scale_product_to_{repaired.product_scale}_and_center")
    if "color_shift" in report.failure_reasons:
        actions.append("force_reference_cutout_composite")
    if "shape_distortion" in report.failure_reasons:
        actions.append("preserve_original_aspect_ratio")

    return repaired, actions

