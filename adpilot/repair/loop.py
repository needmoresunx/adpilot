from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adpilot.backends.background import BackgroundBackend
from adpilot.critic.critique import CritiqueReport, critique_shot
from adpilot.identity.card import IdentityCard
from adpilot.planner.schema import AdPlan
from adpilot.preview.composite import RenderedShot, render_shot
from adpilot.repair.policy import repair_shot
from adpilot.utils.json_io import write_json


@dataclass
class PipelineResult:
    final_frames: list[Path]
    final_background_frames: list[Path]
    final_reports: list[CritiqueReport]
    final_render_metadata: list[dict]
    repair_log: list[dict]
    generation_metadata: dict


def _render_plan(
    identity_card: IdentityCard,
    plan: AdPlan,
    output_dir: Path,
    attempt: int,
    background_backend: BackgroundBackend,
    canvas_size: tuple[int, int],
) -> list[RenderedShot]:
    frames_dir = output_dir / f"frames_attempt_{attempt}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedShot] = []
    for index, shot in enumerate(plan.shots):
        background = background_backend.generate(index, shot.background_prompt, size=canvas_size)
        frame_path = frames_dir / f"{shot.shot_id}.jpg"
        rendered.append(render_shot(background, identity_card, shot, frame_path))
    return rendered


def run_preview_repair_loop(
    identity_card: IdentityCard,
    plan: AdPlan,
    output_dir: Path,
    background_backend: BackgroundBackend,
    max_attempts: int = 2,
    canvas_size: tuple[int, int] = (1080, 1920),
) -> PipelineResult:
    write_json(output_dir / "shot_plan.json", plan.to_dict())
    repair_log: list[dict] = []
    current_plan = plan
    final_frames: list[Path] = []
    final_background_frames: list[Path] = []
    final_reports: list[CritiqueReport] = []
    final_render_metadata: list[dict] = []

    for attempt in range(max_attempts + 1):
        rendered_shots = _render_plan(identity_card, current_plan, output_dir, attempt, background_backend, canvas_size)
        frame_paths = [rendered.path for rendered in rendered_shots]
        reports = [
            critique_shot(identity_card, shot, rendered.path, rendered.to_dict())
            for shot, rendered in zip(current_plan.shots, rendered_shots)
        ]
        if all(report.passed for report in reports) or attempt == max_attempts:
            final_frames = frame_paths
            final_background_frames = [rendered.background_path for rendered in rendered_shots]
            final_reports = reports
            final_render_metadata = [rendered.to_dict() for rendered in rendered_shots]
            break

        repaired_shots = []
        for shot, report in zip(current_plan.shots, reports):
            if report.passed:
                repaired_shots.append(shot)
                continue
            repaired, actions = repair_shot(shot, report)
            repair_log.append(
                {
                    "attempt": attempt,
                    "shot_id": shot.shot_id,
                    "failure_reasons": report.failure_reasons,
                    "actions": actions,
                    "old_scale": shot.product_scale,
                    "new_scale": repaired.product_scale,
                }
            )
            repaired_shots.append(repaired)
        current_plan.shots = repaired_shots

    write_json(output_dir / "critique_report.json", [report.to_dict() for report in final_reports])
    write_json(output_dir / "render_metadata.json", final_render_metadata)
    write_json(output_dir / "repair_log.json", repair_log)
    write_json(output_dir / "final_shot_plan.json", current_plan.to_dict())
    metadata = (
        background_backend.metadata()
        if hasattr(background_backend, "metadata")
        else {"name": background_backend.name}
    )
    metadata["note"] = "proxy_preview.mp4 is rendered from audited keyframes; real image generation depends on backend."
    write_json(output_dir / "generation_backend.json", metadata)
    return PipelineResult(
        final_frames=final_frames,
        final_background_frames=final_background_frames,
        final_reports=final_reports,
        final_render_metadata=final_render_metadata,
        repair_log=repair_log,
        generation_metadata=metadata,
    )
