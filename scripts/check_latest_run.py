from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run(outputs: Path) -> Path:
    runs = sorted(outputs.glob("run_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run_* directories found in {outputs}")
    return runs[0]


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def video_frame_count(path: Path) -> int | None:
    if not path.exists() or file_size(path) == 0:
        return 0
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and validate the latest AdPilot run.")
    parser.add_argument("--outputs", default="outputs", help="Output root containing run_* directories.")
    parser.add_argument("--run-dir", default=None, help="Specific run directory to inspect.")
    parser.add_argument("--require-video-backend", default=None, help="Fail unless the run used this video backend without fallback.")
    parser.add_argument("--require-audited", action="store_true", help="Fail unless keyframe and final-video VLM audit artifacts were completed.")
    parser.add_argument("--require-planner-backend", default=None, help="Fail unless this planner ran without fallback.")
    parser.add_argument("--require-reference-mode", choices=["front_lock", "multi_view"], default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run(Path(args.outputs))
    generation_path = run_dir / "generation_backend.json"
    video_path = run_dir / "video_backend.json"
    critique_path = run_dir / "critique_report.json"
    planner_path = run_dir / "planner_metadata.json"
    video_critique_path = run_dir / "video_critique_report.json"
    reference_mode_path = run_dir / "reference_mode.json"

    generation = read_json(generation_path) if generation_path.exists() else {}
    video = read_json(video_path) if video_path.exists() else {}
    critique = read_json(critique_path) if critique_path.exists() else []
    planner = read_json(planner_path) if planner_path.exists() else {}
    video_critique = read_json(video_critique_path) if video_critique_path.exists() else []
    reference_mode = read_json(reference_mode_path) if reference_mode_path.exists() else {}
    final_video = run_dir / "final_video.mp4"
    report = run_dir / "report.html"

    print(f"run_dir: {run_dir}")
    print(f"generation_backend: {generation.get('name', 'missing')}")
    print(f"generation_used_fallback: {generation.get('used_fallback', False)}")
    print(f"video_backend: {video.get('name', 'missing')}")
    print(f"video_used_fallback: {video.get('used_fallback', False)}")
    print(f"video_frames_written: {video.get('frames_written', 'unknown')}")
    print(f"video_frame_dir: {video.get('frame_dir', 'missing')}")
    print(f"report_html: {report} ({file_size(report)} bytes)")
    print(f"final_video: {final_video} ({file_size(final_video)} bytes)")
    final_video_frames = video_frame_count(final_video)
    if final_video_frames is not None:
        print(f"final_video_frames: {final_video_frames}")
    print(f"critique_items: {len(critique) if isinstance(critique, list) else 'unknown'}")
    print(f"planner_backend: {planner.get('name', 'missing')}")
    print(f"planner_used_fallback: {planner.get('used_fallback', False)}")
    print(f"video_critique_items: {len(video_critique) if isinstance(video_critique, list) else 'unknown'}")
    print(f"reference_mode: {reference_mode.get('mode', 'missing')}")

    failures: list[str] = []
    if not report.exists() or file_size(report) == 0:
        failures.append("missing_or_empty_report")
    if generation.get("used_fallback"):
        failures.append("generation_used_fallback")

    if args.require_video_backend:
        if video.get("name") != args.require_video_backend:
            failures.append(f"video_backend_not_{args.require_video_backend}")
        if video.get("used_fallback"):
            failures.append("video_used_fallback")
        if not final_video.exists() or file_size(final_video) < 4096:
            failures.append("missing_or_empty_final_video")
        if final_video_frames == 0:
            failures.append("final_video_not_readable")

    if args.require_audited:
        if not isinstance(critique, list) or not critique:
            failures.append("missing_keyframe_critique")
        elif any(not item.get("passed", False) for item in critique if isinstance(item, dict)):
            failures.append("unresolved_keyframe_identity_failures")
        if not isinstance(video_critique, list) or not video_critique:
            failures.append("missing_video_critique")
        elif any(not item.get("passed", False) for item in video_critique if isinstance(item, dict)):
            failures.append("unresolved_video_identity_failures")

    if args.require_planner_backend:
        if planner.get("name") != args.require_planner_backend:
            failures.append(f"planner_backend_not_{args.require_planner_backend}")
        if planner.get("used_fallback"):
            failures.append("planner_used_fallback")

    if args.require_reference_mode:
        if reference_mode.get("mode") != args.require_reference_mode:
            failures.append(f"reference_mode_not_{args.require_reference_mode}")

    if failures:
        print(f"status: FAIL ({', '.join(failures)})")
        raise SystemExit(1)
    print("status: PASS")


if __name__ == "__main__":
    main()
