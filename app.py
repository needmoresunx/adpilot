from __future__ import annotations

import argparse
from pathlib import Path

from adpilot.backends.background import make_background_backend
from adpilot.backends.keyframe import make_keyframe_backend
from adpilot.backends.video import make_video_backend
from adpilot.identity.builder import build_identity_card
from adpilot.planner.planner import make_default_plan
from adpilot.preview.storyboard import make_storyboard
from adpilot.repair.loop import run_preview_repair_loop
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


def compact_words(text: str | None, max_words: int = 28) -> str:
    if not text:
        return ""
    return " ".join(text.replace("(", "").replace(")", "").split()[:max_words])


def make_keyframe_prompts(identity_card, plan) -> list[str]:
    brief = identity_card.product_brief
    description = compact_words(brief.get("description"), 14) or "the supplied product"
    mood = compact_words(brief.get("mood"), 12) or "premium cinematic"
    category = brief.get("category", "product")
    scene = ", ".join(brief.get("scene_keywords", [])[:3])
    prompts = []
    positions = {"center": "centered", "left": "on the left third", "right": "on the right third"}
    for shot in plan.shots:
        prompt = (
            f"Use the supplied product photograph as the sole product reference. Create a single, "
            f"photorealistic high-end {category} advertising still of {description}, {mood}, "
            f"{scene}, {shot.goal}. Replace the plain studio background with a coherent luxury "
            f"scene and place the product {positions.get(shot.product_position, 'centered')} on a "
            "real reflective surface with physically plausible contact shadow, reflections, and "
            "matching light. Preserve the exact bottle shape, bow, pink liquid, and front label. "
            "Remove every other bottle, package, duplicate product, caption, and watermark."
        )
        prompts.append(compact_words(prompt, 100))
    return prompts


def make_video_prompts(identity_card, plan) -> list[str]:
    brief = identity_card.product_brief
    description = compact_words(brief.get("description"), 14) or "the reference product"
    mood = compact_words(brief.get("mood"), 12) or "premium cinematic"
    category = brief.get("category", "product")
    prompts = []
    shot_motion = [
        "slow dolly-in, elegant reveal, glossy reflections",
        "gentle lateral camera move, intimate beauty atmosphere",
        "smooth packshot push-in, soft shadow, clean finish",
    ]
    for index, shot in enumerate(plan.shots):
        motion = shot_motion[min(index, len(shot_motion) - 1)]
        prompts.append(
            compact_words(
                f"cinematic {category} commercial of {description}, {mood}, {shot.goal}, "
                f"{motion}, preserve one exact product from the input image, preserve bottle shape "
                "and front label, realistic optical reflections, high-end advertising cinematography",
                60,
            )
        )
    return prompts


def make_video_negative_prompt() -> str:
    return (
        "static image, no motion, added subtitles, title cards, on-screen captions, "
        "watermark, subtitles, title cards, extra bottle, duplicate product, package, "
        "warped label, unreadable label, blurry, low quality, messy background, deformed object"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AdPilot MVP pipeline.")
    parser.add_argument("--product", required=True, help="Path to product image.")
    parser.add_argument("--brand", required=True, help="Brand or product name.")
    parser.add_argument("--style", default="auto", help="Ad style. Use 'auto' to derive style from product analysis.")
    parser.add_argument("--platform", default="shorts", choices=["shorts", "instagram", "tiktok", "youtube", "landscape"], help="Target ad format/platform.")
    parser.add_argument("--product-category", default=None, help="Optional product category override, e.g. fragrance, beverage, cosmetic.")
    parser.add_argument("--product-description", default=None, help="Optional natural-language product description.")
    parser.add_argument("--target-audience", default=None, help="Optional target audience override.")
    parser.add_argument("--ad-mood", default=None, help="Optional visual mood override.")
    parser.add_argument("--auto-brief", action="store_true", help="Use a VLM captioner to derive a product brief.")
    parser.add_argument("--vlm-model", default=None, help="Local path or HF id for a BLIP image captioning model.")
    parser.add_argument("--vlm-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--duration", type=int, default=9, help="Total duration in seconds.")
    parser.add_argument("--logo-bbox", type=parse_bbox, default=None, help="Optional x1,y1,x2,y2 logo box.")
    parser.add_argument("--auto-cutout", action="store_true", help="Remove the product-photo background with the configured segmentation model.")
    parser.add_argument("--cutout-model", default="birefnet-general", help="Local rembg model used by --auto-cutout.")
    parser.add_argument("--out", default="outputs", help="Output root directory.")
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument("--background-backend", choices=["mock", "folder", "diffusers"], default="mock")
    parser.add_argument("--background-dir", default=None, help="Directory of real/generated backgrounds.")
    parser.add_argument("--image-model", default="stabilityai/sdxl-turbo", help="Diffusers text-to-image model id.")
    parser.add_argument("--image-device", default="auto", choices=["auto", "cuda", "cpu"], help="Device for diffusers backend.")
    parser.add_argument("--image-steps", type=int, default=4, help="Diffusion inference steps.")
    parser.add_argument("--image-guidance-scale", type=float, default=0.0, help="Diffusion guidance scale.")
    parser.add_argument("--image-seed", type=int, default=7, help="Base seed for generated backgrounds.")
    parser.add_argument("--generated-width", type=int, default=768, help="Generated background width before cover crop.")
    parser.add_argument("--generated-height", type=int, default=1344, help="Generated background height before cover crop.")
    parser.add_argument("--canvas-width", type=int, default=1080, help="Final keyframe canvas width.")
    parser.add_argument("--canvas-height", type=int, default=1920, help="Final keyframe canvas height.")
    parser.add_argument("--keyframe-backend", choices=["preview_composite", "flux_kontext"], default="preview_composite")
    parser.add_argument("--keyframe-model", default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--keyframe-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--keyframe-seed", type=int, default=17)
    parser.add_argument("--keyframe-steps", type=int, default=28)
    parser.add_argument("--keyframe-guidance-scale", type=float, default=2.5)
    parser.add_argument("--video-backend", choices=["proxy", "wan_i2v"], default="proxy")
    parser.add_argument("--video-model", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--video-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--video-seed", type=int, default=11)
    parser.add_argument("--video-num-frames", type=int, default=14)
    parser.add_argument("--video-fps", type=int, default=7)
    parser.add_argument("--video-width", type=int, default=1024)
    parser.add_argument("--video-height", type=int, default=576)
    parser.add_argument("--video-steps", type=int, default=40, help="Video diffusion inference steps.")
    parser.add_argument("--video-guidance-scale", type=float, default=3.5, help="Video diffusion guidance scale.")
    parser.add_argument("--no-backend-fallback", action="store_true", help="Raise backend errors instead of falling back to mock.")
    args = parser.parse_args()

    run_dir = make_run_dir(Path(args.out))
    identity_card = build_identity_card(
        product_path=Path(args.product),
        brand=args.brand,
        output_dir=run_dir,
        logo_bbox=args.logo_bbox,
        auto_cutout=args.auto_cutout,
        cutout_model=args.cutout_model,
        product_category=args.product_category,
        product_description=args.product_description,
        target_audience=args.target_audience,
        ad_mood=args.ad_mood,
        auto_brief=args.auto_brief,
        vlm_model=args.vlm_model,
        vlm_device=args.vlm_device,
    )
    plan = make_default_plan(identity_card, style=args.style, duration=args.duration, platform=args.platform)
    if args.keyframe_backend == "flux_kontext":
        keyframe_backend = make_keyframe_backend(
            "flux_kontext",
            model_id=args.keyframe_model,
            device=args.keyframe_device,
            seed=args.keyframe_seed,
            num_inference_steps=args.keyframe_steps,
            guidance_scale=args.keyframe_guidance_scale,
            fallback_on_error=not args.no_backend_fallback,
        )
        final_frames = keyframe_backend.render(
            Path(args.product),
            make_keyframe_prompts(identity_card, plan),
            run_dir / "context_keyframes",
            (args.canvas_width, args.canvas_height),
        )
        final_reports = []
        repair_log = []
        generation_metadata = keyframe_backend.metadata()
        write_json(run_dir / "shot_plan.json", plan.to_dict())
        write_json(run_dir / "final_shot_plan.json", plan.to_dict())
        write_json(run_dir / "critique_report.json", [])
        write_json(run_dir / "repair_log.json", repair_log)
        write_json(run_dir / "generation_backend.json", generation_metadata)
        write_json(run_dir / "keyframe_backend.json", generation_metadata)
    else:
        background_backend = make_background_backend(
            args.background_backend,
            image_dir=args.background_dir,
            image_model=args.image_model,
            image_device=args.image_device,
            image_steps=args.image_steps,
            image_guidance_scale=args.image_guidance_scale,
            image_seed=args.image_seed,
            generated_size=(args.generated_width, args.generated_height),
            fallback_on_error=not args.no_backend_fallback,
        )
        result = run_preview_repair_loop(
            identity_card=identity_card,
            plan=plan,
            output_dir=run_dir,
            background_backend=background_backend,
            max_attempts=args.max_repair_attempts,
            canvas_size=(args.canvas_width, args.canvas_height),
        )
        final_frames = result.final_frames
        final_reports = result.final_reports
        repair_log = result.repair_log
        generation_metadata = result.generation_metadata

    storyboard_path = make_storyboard(final_frames, run_dir / "storyboard.png")
    video_prompts = make_video_prompts(identity_card, plan)
    video_backend = make_video_backend(
        args.video_backend,
        model_id=args.video_model,
        device=args.video_device,
        seed=args.video_seed,
        num_frames=args.video_num_frames,
        fps=args.video_fps,
        generated_size=(args.video_width, args.video_height),
        prompts=video_prompts,
        negative_prompt=make_video_negative_prompt(),
        num_inference_steps=args.video_steps,
        guidance_scale=args.video_guidance_scale,
        fallback_on_error=not args.no_backend_fallback,
    )
    video_name = "proxy_preview.mp4" if args.video_backend == "proxy" else "final_video.mp4"
    video_path = video_backend.render(final_frames, run_dir / video_name)
    video_metadata = (
        video_backend.metadata()
        if hasattr(video_backend, "metadata")
        else {"name": video_backend.name}
    )
    write_json(run_dir / "video_backend.json", video_metadata)
    report_path = write_html_report(
        run_dir=run_dir,
        identity_card=identity_card,
        final_frames=final_frames,
        reports=final_reports,
        repair_log=repair_log,
        storyboard_path=storyboard_path,
        preview_path=video_path,
        generation_metadata=generation_metadata,
        video_metadata=video_metadata,
    )

    print(f"Run directory: {run_dir}")
    print(f"Identity card: {run_dir / 'identity_card.json'}")
    print(f"Shot plan: {run_dir / 'shot_plan.json'}")
    print(f"Critique report: {run_dir / 'critique_report.json'}")
    print(f"Repair log: {run_dir / 'repair_log.json'}")
    print(f"Storyboard: {storyboard_path}")
    print(f"Video: {video_path if video_path else 'skipped; renderer unavailable'}")
    print(f"Video backend: {run_dir / 'video_backend.json'}")
    print(f"HTML report: {report_path}")


if __name__ == "__main__":
    main()
