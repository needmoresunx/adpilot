from __future__ import annotations

import argparse
from pathlib import Path

from adpilot.backends.keyframe import make_keyframe_backend
from adpilot.backends.video import make_video_backend
from adpilot.critic.vlm import critique_keyframe_candidates, critique_video_candidates
from adpilot.identity.builder import build_identity_card
from adpilot.planner.planner import make_default_plan
from adpilot.planner.vlm import make_vlm_plan
from adpilot.preview.storyboard import make_storyboard
from adpilot.report.html import write_html_report
from adpilot.utils.json_io import write_json
from adpilot.utils.paths import make_run_dir


def parse_bbox(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--logo-bbox must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("--logo-bbox must have x2>x1 and y2>y1")
    return x1, y1, x2, y2


def compact_words(text: str | None, max_words: int) -> str:
    return " ".join((text or "").replace("(", "").replace(")", "").split()[:max_words])


def resolve_reference_mode(
    product_path: Path,
    additional_paths: list[str],
    requested_mode: str,
) -> tuple[str, list[Path]]:
    candidates = [product_path, *(Path(path) for path in additional_paths)]
    references: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"Product reference image not found: {path}")
        resolved = path.resolve()
        if resolved not in seen:
            references.append(path)
            seen.add(resolved)
    mode = "front_lock" if requested_mode == "auto" and len(references) == 1 else requested_mode
    if requested_mode == "auto" and len(references) > 1:
        mode = "multi_view"
    if mode == "front_lock" and len(references) != 1:
        raise ValueError("front_lock accepts exactly one product photo.")
    if mode == "multi_view" and len(references) < 2:
        raise ValueError("multi_view needs the primary photo plus at least one additional real product view.")
    return mode, references


def make_keyframe_prompts(identity_card, plan, reference_mode: str = "front_lock", final_shot_constraint: str = "") -> list[str]:
    brief = identity_card.product_brief
    description = compact_words(brief.get("description"), 14) or "the supplied product"
    mood = compact_words(brief.get("mood"), 12) or "premium cinematic"
    category = brief.get("category", "product")
    view_instruction = (
        "Keep each supplied product part's exact geometry, count, and assembly relationship. Articulated parts may move only as rigid pieces around their existing hinges; never stretch, melt, reshape, add, or remove parts."
        if reference_mode == "front_lock"
        else "Use the supplied product view for this shot; preserve each part's exact geometry, count, and assembly relationship."
    )
    positions = {"center": "centered", "left": "on the left third", "right": "on the right third"}
    prompts = []
    for index, shot in enumerate(plan.shots):
        set_description = compact_words(shot.background_prompt, 18)
        final_constraint = final_shot_constraint if index == len(plan.shots) - 1 else ""
        prompts.append(
            compact_words(
                f"{view_instruction} {shot.goal}. {final_constraint} Set: {set_description}. Create a photorealistic high-end "
                f"{category} advertising still of {description}, {mood}. Place one product package "
                f"{positions.get(shot.product_position, 'centered')} with realistic light and reflections.",
                52,
            )
        )
    return prompts


def make_video_prompts(identity_card, plan, final_shot_constraint: str = "") -> list[str]:
    brief = identity_card.product_brief
    description = compact_words(brief.get("description"), 14) or "the reference product"
    mood = compact_words(brief.get("mood"), 12) or "premium cinematic"
    category = brief.get("category", "product")
    prompts = []
    for index, shot in enumerate(plan.shots):
        final_constraint = final_shot_constraint if index == len(plan.shots) - 1 else ""
        prompts.append(
            compact_words(
                f"Keep each supplied product part's exact geometry, count, and assembly relationship. Articulated parts may move only as rigid pieces around their existing hinges; never stretch, melt, reshape, add, or remove parts. "
                f"{shot.goal}. {compact_words(shot.motion_prompt, 16)}. {final_constraint} "
                f"Cinematic {category} commercial of {description}, {mood}, high-end advertising cinematography",
                58,
            )
        )
    return prompts


def final_shot_endpoint_locks(shot_count: int, enabled: bool) -> list[bool]:
    """Lock only the final native I2V shot when the product script requests it."""
    return [enabled and index == shot_count - 1 for index in range(shot_count)]


def video_candidate_counts(shot_count: int, default_count: int, final_count: int) -> list[int]:
    if shot_count < 1 or default_count < 1 or final_count < 1:
        raise ValueError("Video candidate counts must be positive.")
    return [default_count] * (shot_count - 1) + [final_count]


def select_keyframe_candidates(candidate_paths, candidate_reports):
    selected_frames, selected_reports, selection_log = [], [], []
    readability = {"readable": 2, "uncertain": 1, "not_applicable": 1, "unreadable": 0}
    for paths, reports in zip(candidate_paths, candidate_reports):
        index, report = max(
            enumerate(reports),
            key=lambda item: (
                item[1].passed,
                item[1].identity_score or 0,
                readability.get(item[1].label_readability or "uncertain", 0),
            ),
        )
        selected_frames.append(paths[index])
        selected_reports.append(report)
        selection_log.append(
            {
                "shot_id": report.shot_id,
                "selected_candidate": index + 1,
                "selected_path": str(paths[index]),
                "candidates": [
                    {
                        "candidate": candidate_index + 1,
                        "path": str(path),
                        "passed": candidate_report.passed,
                        "identity_score": candidate_report.identity_score,
                        "failure_reasons": candidate_report.failure_reasons,
                    }
                    for candidate_index, (path, candidate_report) in enumerate(zip(paths, reports))
                ],
            }
        )
    return selected_frames, selected_reports, selection_log


def select_video_candidates(candidate_sequences, candidate_reports):
    selected_sequences, selected_reports, selection_log = [], [], []
    for sequences, reports in zip(candidate_sequences, candidate_reports):
        index, report = max(
            enumerate(reports),
            key=lambda item: (
                item[1].passed,
                item[1].identity_score or 0,
                item[1].temporal_consistency == "stable",
            ),
        )
        selected_sequences.append(sequences[index])
        selected_reports.append(report)
        selection_log.append(
            {
                "shot_id": report.shot_id,
                "selected_candidate": index + 1,
                "selected_first_frame": str(sequences[index][0]),
                "failure_reasons": report.failure_reasons,
            }
        )
    return selected_sequences, selected_reports, selection_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and audit a short product advertisement.")
    parser.add_argument("--product", required=True)
    parser.add_argument("--reference-mode", choices=["auto", "front_lock", "multi_view"], default="auto")
    parser.add_argument("--reference-images", nargs="*", default=[])
    parser.add_argument("--brand", required=True)
    parser.add_argument("--style", default="auto")
    parser.add_argument("--platform", choices=["shorts", "instagram", "tiktok", "youtube", "landscape"], default="landscape")
    parser.add_argument("--product-category")
    parser.add_argument("--product-description")
    parser.add_argument("--target-audience")
    parser.add_argument("--ad-mood")
    parser.add_argument("--final-shot-constraint", default="")
    parser.add_argument("--final-shot-endpoint-lock", action="store_true")
    parser.add_argument("--require-readable-branding", action="store_true")
    parser.add_argument("--identity-anchors")
    parser.add_argument("--package-state", choices=["sealed", "open_with_contents"])
    parser.add_argument("--auto-brief", action="store_true")
    parser.add_argument("--vlm-model")
    parser.add_argument("--vlm-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--planner-backend", choices=["template", "qwen_vl"], default="template")
    parser.add_argument("--planner-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--planner-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--duration", type=int, default=9)
    parser.add_argument("--logo-bbox", type=parse_bbox)
    parser.add_argument("--auto-cutout", action="store_true")
    parser.add_argument("--cutout-model", default="birefnet-general")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--keyframe-model", default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--keyframe-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--keyframe-seed", type=int, default=17)
    parser.add_argument("--keyframe-steps", type=int, default=28)
    parser.add_argument("--keyframe-guidance-scale", type=float, default=2.5)
    parser.add_argument("--keyframe-offload", choices=["none", "model", "sequential"], default="none")
    parser.add_argument("--keyframe-candidates", type=int, default=1)
    parser.add_argument("--canvas-width", type=int, default=1360)
    parser.add_argument("--canvas-height", type=int, default=768)
    parser.add_argument("--critic-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--critic-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--minimum-identity-score", type=int, default=75)
    parser.add_argument("--video-model", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--video-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--video-seed", type=int, default=11)
    parser.add_argument("--video-num-frames", type=int, default=49)
    parser.add_argument("--video-fps", type=int, default=16)
    parser.add_argument("--video-width", type=int, default=832)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-steps", type=int, default=40)
    parser.add_argument("--video-guidance-scale", type=float, default=3.5)
    parser.add_argument("--video-offload", choices=["none", "model", "sequential"], default="model")
    parser.add_argument("--video-candidates", type=int, default=1)
    parser.add_argument("--final-shot-candidates", type=int, default=1)
    args = parser.parse_args()

    reference_mode, references = resolve_reference_mode(Path(args.product), args.reference_images, args.reference_mode)
    run_dir = make_run_dir(Path(args.out))
    identity_card = build_identity_card(
        product_path=references[0],
        brand=args.brand,
        output_dir=run_dir,
        logo_bbox=args.logo_bbox,
        auto_cutout=args.auto_cutout,
        cutout_model=args.cutout_model,
        product_category=args.product_category,
        product_description=args.product_description,
        target_audience=args.target_audience,
        ad_mood=args.ad_mood,
        identity_anchors=args.identity_anchors,
        package_state=args.package_state,
        auto_brief=args.auto_brief,
        vlm_model=args.vlm_model,
        vlm_device=args.vlm_device,
    )

    if args.planner_backend == "qwen_vl":
        plan = make_vlm_plan(
            identity_card,
            references[0],
            args.style,
            args.duration,
            args.platform,
            args.planner_model,
            args.planner_device,
            reference_paths=references[1:],
        )
        planner_metadata = {"name": "qwen2_5_vl", "used_fallback": False}
    else:
        plan = make_default_plan(identity_card, args.style, args.duration, args.platform)
        planner_metadata = {"name": "template", "used_fallback": False}

    write_json(
        run_dir / "reference_mode.json",
        {
            "mode": reference_mode,
            "reference_images": [str(path) for path in references],
            "shot_reference_assignment": [
                {"shot_id": shot.shot_id, "reference_image": str(references[index % len(references)])}
                for index, shot in enumerate(plan.shots)
            ],
            "post_generation_foreground_composite": False,
        },
    )

    keyframe_backend = make_keyframe_backend(
        "flux_kontext",
        args.keyframe_model,
        device=args.keyframe_device,
        seed=args.keyframe_seed,
        num_inference_steps=args.keyframe_steps,
        guidance_scale=args.keyframe_guidance_scale,
        offload_mode=args.keyframe_offload,
    )
    candidate_paths = keyframe_backend.render_candidates(
        references[0],
        make_keyframe_prompts(identity_card, plan, reference_mode, args.final_shot_constraint),
        run_dir / "keyframes",
        (args.canvas_width, args.canvas_height),
        args.keyframe_candidates,
        reference_layouts=[(shot.product_position, shot.product_scale) for shot in plan.shots],
        reference_paths=references[1:],
    )
    keyframe_backend.release()
    candidate_reports = critique_keyframe_candidates(
        identity_card,
        plan.shots,
        candidate_paths,
        args.critic_model,
        args.critic_device,
        args.minimum_identity_score,
        reference_paths=references[1:],
        final_shot_constraint=args.final_shot_constraint,
        require_readable_branding=args.require_readable_branding,
    )
    final_frames, reports, keyframe_selection = select_keyframe_candidates(candidate_paths, candidate_reports)
    generation_metadata = keyframe_backend.metadata()
    generation_metadata.update({"reference_mode": reference_mode, "post_generation_foreground_composite": False})

    video_backend = make_video_backend(
        "wan_i2v",
        args.video_model,
        device=args.video_device,
        seed=args.video_seed,
        num_frames=args.video_num_frames,
        fps=args.video_fps,
        generated_size=(args.video_width, args.video_height),
        prompts=make_video_prompts(identity_card, plan, args.final_shot_constraint),
        negative_prompt="static image, blurry, low quality, watermark, deformed product, morphing parts, stretched parts, melted surfaces, extra parts, missing parts",
        num_inference_steps=args.video_steps,
        guidance_scale=args.video_guidance_scale,
        offload_mode=args.video_offload,
        endpoint_locked_shots=final_shot_endpoint_locks(
            len(plan.shots), args.final_shot_endpoint_lock
        ),
    )
    video_candidates = video_backend.render_candidates(
        final_frames,
        run_dir / "video_candidates",
        video_candidate_counts(len(plan.shots), args.video_candidates, args.final_shot_candidates),
    )
    video_backend.release()
    video_candidate_reports = critique_video_candidates(
        identity_card,
        plan.shots,
        video_candidates,
        args.critic_model,
        args.critic_device,
        args.minimum_identity_score,
        reference_paths=references[1:],
        final_shot_constraint=args.final_shot_constraint,
        require_readable_branding=args.require_readable_branding,
    )
    selected_sequences, video_reports, video_selection = select_video_candidates(video_candidates, video_candidate_reports)
    video_path = video_backend.render_selected(selected_sequences, run_dir / "final_video.mp4")
    video_metadata = video_backend.metadata()
    video_metadata["reference_mode"] = reference_mode
    video_metadata["post_generation_foreground_composite"] = False

    storyboard_path = make_storyboard(final_frames, run_dir / "storyboard.png")
    write_json(run_dir / "shot_plan.json", plan.to_dict())
    write_json(run_dir / "planner_metadata.json", planner_metadata)
    write_json(run_dir / "generation_backend.json", generation_metadata)
    write_json(run_dir / "video_backend.json", video_metadata)
    write_json(run_dir / "critique_report.json", [report.to_dict() for report in reports])
    write_json(run_dir / "video_critique_report.json", [report.to_dict() for report in video_reports])
    write_json(run_dir / "candidate_selection.json", keyframe_selection)
    write_json(run_dir / "video_candidate_selection.json", video_selection)
    report_path = write_html_report(
        run_dir,
        identity_card,
        final_frames,
        reports,
        storyboard_path,
        video_path,
        generation_metadata,
        video_metadata,
        video_reports,
        planner_metadata,
    )
    print(f"Run directory: {run_dir}")
    print(f"Storyboard: {storyboard_path}")
    print(f"Video: {video_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
