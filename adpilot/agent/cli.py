from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .store import ProjectStore


DEFAULT_PROJECT_ROOT = Path("projects")
DEFAULT_MODEL_ROOT = Path(os.environ.get("ADPILOT_MODEL_ROOT", "~/models/adpilot")).expanduser()


def _model_settings(args: argparse.Namespace) -> dict[str, str]:
    root = Path(args.model_root).expanduser()
    return {
        "vlm": args.vlm_model or str(root / "qwen2.5-vl-3b-instruct"),
        "keyframe": args.keyframe_model or str(root / "flux-kontext-dev"),
        "video": args.video_model or str(root / "wan2.2-i2v-a14b-diffusers"),
    }


def _brief_settings(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "product_category": args.product_category,
        "product_description": args.product_description,
        "target_audience": args.target_audience,
        "ad_mood": args.ad_mood,
        "identity_anchors": args.identity_anchors,
        "package_state": args.package_state,
    }


def _project_store(project: str, root: Path = DEFAULT_PROJECT_ROOT) -> ProjectStore:
    candidate = Path(project)
    if not candidate.is_dir():
        candidate = Path(root) / project
    return ProjectStore(candidate)


def _graph_for(store: ProjectStore):
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - dependency installation is external
        raise RuntimeError(
            "Interactive agent commands require langgraph and langgraph-checkpoint-sqlite. "
            "Install the project requirements first."
        ) from exc
    from .workflow import AdPilotGraph

    connection = sqlite3.connect(store.database_path, check_same_thread=False)
    return AdPilotGraph(store).build(SqliteSaver(connection)), connection


def _config(store: ProjectStore) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": store.config.project_id},
        "recursion_limit": store.config.max_agent_steps * 4,
    }


def _interrupt_payload(result: Any) -> dict[str, Any] | None:
    interrupts = getattr(result, "interrupts", None)
    if interrupts is None and isinstance(result, dict):
        interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    interrupt = interrupts[0]
    return dict(getattr(interrupt, "value", interrupt))


def _print_interrupt(
    payload: dict[str, Any],
    *,
    project_id: str | None = None,
    interactive: bool = False,
) -> None:
    print()
    if payload.get("type") == "storyboard_review":
        print(f"PAUSED: storyboard review for revision {payload['revision']}")
        print(f"Storyboard: {payload['storyboard_path']}")
        for shot in payload.get("shots", []):
            print(
                f"  {shot['shot_id']}: visual_score={shot.get('identity_audit_score')}, "
                f"identity={shot.get('identity_verdict')}, "
                f"selection={shot.get('selection_method')}, "
                f"candidates={shot.get('candidate_count')}, selected={shot.get('selected_path')}"
            )
            for candidate in shot.get("candidates", []):
                status = "pass" if candidate.get("passed") else "fail"
                reasons = ", ".join(candidate.get("failure_reasons", [])) or "none"
                print(
                    f"    [{candidate['candidate']}] {status}, visual_score={candidate.get('identity_audit_score')}, "
                    f"identity={candidate.get('identity_verdict')}, "
                    f"checks={candidate.get('identity_checks', {})}: "
                    f"{candidate.get('path')} (reasons: {reasons})"
                )
                evidence_path = candidate.get("identity_evidence", {}).get("evidence_image")
                if evidence_path:
                    print(f"        evidence: {evidence_path}")
                metric = candidate.get("identity_evidence", {}).get("visual_metric", {})
                if metric.get("available"):
                    print(
                        f"        metric: IoU={metric['silhouette_iou']}, "
                        f"aspect={metric['aspect_similarity']}, "
                        f"color={metric['color_similarity']}, source={metric.get('source')}"
                    )
                elif metric:
                    print(f"        metric: unavailable ({metric.get('reason', 'unknown')})")
            comparisons = shot.get("pairwise_comparisons", [])
            for comparison in comparisons:
                preferred = comparison.get("preferred_candidate")
                preference = f"candidate {preferred}" if preferred is not None else "tie"
                reason = comparison.get("reason") or "No discriminative visual reason returned."
                print(
                    f"    pairwise: {comparison.get('left_candidate')} vs {comparison.get('right_candidate')} "
                    f"-> {preference} ({reason})"
                )
        print("Actions: approve | select <shot_id> <candidate> | feedback <shot_id|all> <message> | add-reference <image> | cancel")
        if interactive:
            print("Enter one action at the adpilot> prompt below.")
        elif project_id:
            print("Resume from this shell with:")
            print(f"  python -m adpilot.agent approve {project_id}")
        return
    print(f"PAUSED: {payload.get('type', 'guidance')}")
    print(payload.get("message", "The agent needs your input."))
    if payload.get("shots"):
        print("Affected shots: " + ", ".join(payload["shots"]))
    print("Actions: feedback <shot_id|all> <message> | add-reference <image> | cancel")
    if interactive:
        print("Enter one action at the adpilot> prompt below.")
    elif project_id:
        print("Resume from this shell with a concrete revision request, for example:")
        print(
            f'  python -m adpilot.agent feedback {project_id} '
            '--shot all --message "Keep the product identity; simplify the scene and motion."'
        )


def _print_result(store: ProjectStore, result: Any) -> int:
    payload = _interrupt_payload(result)
    if payload:
        _print_interrupt(payload, project_id=store.config.project_id)
        return 0
    state = getattr(result, "value", result)
    if not isinstance(state, dict):
        print("Agent finished without a readable state.")
        return 1
    status = state.get("status", "unknown")
    print(f"status: {status}")
    if state.get("final_video"):
        print(f"video: {state['final_video']}")
    if state.get("report_path"):
        print(f"report: {state['report_path']}")
    if state.get("error"):
        print(f"message: {state['error']}")
    return 0 if status in {"completed", "cancelled", "running"} else 1


def _invoke_new(store: ProjectStore) -> int:
    graph, connection = _graph_for(store)
    try:
        from .workflow import initial_state

        result = graph.invoke(initial_state(store.config), _config(store))
        return _print_result(store, result)
    finally:
        connection.close()


def _resume(store: ProjectStore, decision: dict[str, Any]) -> int:
    graph, connection = _graph_for(store)
    try:
        from langgraph.types import Command

        result = graph.invoke(Command(resume=decision), _config(store))
        return _print_result(store, result)
    finally:
        connection.close()


def command_create(args: argparse.Namespace) -> int:
    store = ProjectStore.create(
        Path(args.projects_root),
        Path(args.product),
        args.brand,
        args.prompt,
        args.mode,
        [Path(path) for path in args.reference],
        project_name=args.project_name,
        platform=args.platform,
        duration=args.duration,
        max_agent_steps=args.max_agent_steps,
        max_keyframe_attempts=args.max_keyframe_attempts,
        max_video_attempts=args.max_video_attempts,
        keyframe_candidates=args.keyframe_candidates,
        video_candidates=args.video_candidates,
        models=_model_settings(args),
        **_brief_settings(args),
    )
    print(f"project: {store.config.project_id}")
    print(f"directory: {store.project_dir}")
    print(f"mode: {store.config.mode}")
    print(f"next: python -m adpilot.agent run {store.config.project_id}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    return _invoke_new(_project_store(args.project, Path(args.projects_root)))


def command_approve(args: argparse.Namespace) -> int:
    return _resume(_project_store(args.project, Path(args.projects_root)), {"action": "approve"})


def command_select(args: argparse.Namespace) -> int:
    return _resume(
        _project_store(args.project, Path(args.projects_root)),
        {"action": "select", "shot_id": args.shot, "candidate": args.candidate},
    )


def command_feedback(args: argparse.Namespace) -> int:
    return _resume(
        _project_store(args.project, Path(args.projects_root)),
        {"action": "feedback", "shot_id": args.shot, "message": args.message},
    )


def command_add_reference(args: argparse.Namespace) -> int:
    store = _project_store(args.project, Path(args.projects_root))
    asset_id = store.add_asset(Path(args.image), role="product_reference", origin="uploaded")
    print(f"added reference asset: {asset_id}")
    return _resume(store, {"action": "add_reference", "asset_id": asset_id, "shot_id": args.shot})


def command_cancel(args: argparse.Namespace) -> int:
    return _resume(_project_store(args.project, Path(args.projects_root)), {"action": "cancel"})


def command_status(args: argparse.Namespace) -> int:
    store = _project_store(args.project, Path(args.projects_root))
    config = store.config
    print(f"project: {config.project_id}")
    print(f"mode: {config.mode}")
    print(f"assets: {len(store.asset_records())}")
    events = store.events()
    if events:
        event = events[-1]
        print(f"last_event: {event['event_type']} ({event['created_at']})")
    try:
        graph, connection = _graph_for(store)
    except RuntimeError as exc:
        print(f"checkpoint: unavailable ({exc})")
        return 0
    try:
        snapshot = graph.get_state(_config(store))
        values = getattr(snapshot, "values", {})
        if values:
            print(f"status: {values.get('status', 'unknown')}")
            print(f"revision: {values.get('revision', 'unknown')}")
            print(f"agent_steps: {values.get('agent_steps_used', 0)}/{config.max_agent_steps}")
            if getattr(snapshot, "tasks", None):
                interrupts = [task for task in snapshot.tasks if getattr(task, "interrupts", None)]
                if interrupts:
                    print("waiting_for: user decision")
        else:
            print("status: not started")
    finally:
        connection.close()
    return 0


def _chat_decision(store: ProjectStore) -> dict[str, Any]:
    raw = _read_terminal_line("adpilot> ")
    if raw == "approve":
        return {"action": "approve"}
    if raw == "cancel":
        return {"action": "cancel"}
    if raw.startswith("select "):
        _, shot_id, candidate = raw.split(maxsplit=2)
        return {"action": "select", "shot_id": shot_id, "candidate": int(candidate)}
    if raw.startswith("feedback "):
        _, shot_id, message = raw.split(maxsplit=2)
        return {"action": "feedback", "shot_id": shot_id, "message": message}
    if raw.startswith("add-reference "):
        _, image = raw.split(maxsplit=1)
        asset_id = store.add_asset(Path(image), role="product_reference", origin="uploaded")
        return {"action": "add_reference", "asset_id": asset_id, "shot_id": "all"}
    raise ValueError("Use approve, select <shot_id> <candidate>, feedback <shot_id|all> <message>, add-reference <image>, or cancel.")


def _decode_terminal_bytes(raw: bytes) -> str:
    """Decode terminal feedback when an SSH client does not send UTF-8 bytes."""
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        pass

    candidates: list[tuple[float, str]] = []
    for encoding in ("cp949", "gb18030"):
        try:
            decoded = raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
        letters = [character for character in decoded if character.isalpha()]
        if not letters:
            candidates.append((0.0, decoded))
            continue
        hangul = sum("\uac00" <= character <= "\ud7a3" for character in letters)
        han = sum("\u4e00" <= character <= "\u9fff" for character in letters)
        latin = sum(character.isascii() for character in letters)
        dominant = max(hangul, han, latin) / len(letters)
        mixed_scripts = (hangul + han + latin - max(hangul, han, latin)) / len(letters)
        candidates.append((dominant - 0.5 * mixed_scripts, decoded))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    raise UnicodeDecodeError("utf-8", raw, 0, min(len(raw), 1), "terminal input is not UTF-8, CP949, or GB18030")


def _read_terminal_line(prompt: str) -> str:
    """Read feedback bytes directly so non-UTF-8 SSH terminals remain usable."""
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:
        return input(prompt).strip()
    print(prompt, end="", flush=True)
    raw = buffer.readline()
    if not raw:
        raise EOFError("Terminal input closed.")
    return _decode_terminal_bytes(raw)


def command_chat(args: argparse.Namespace) -> int:
    store = ProjectStore.create(
        Path(args.projects_root),
        Path(args.product),
        args.brand,
        args.prompt,
        "guided",
        [Path(path) for path in args.reference],
        project_name=args.project_name,
        platform=args.platform,
        models=_model_settings(args),
        **_brief_settings(args),
    )
    graph, connection = _graph_for(store)
    try:
        from langgraph.types import Command
        from .workflow import initial_state

        result = graph.invoke(initial_state(store.config), _config(store))
        while payload := _interrupt_payload(result):
            _print_interrupt(payload, interactive=True)
            try:
                decision = _chat_decision(store)
            except ValueError as exc:
                print(exc)
                continue
            result = graph.invoke(Command(resume=decision), _config(store))
        return _print_result(store, result)
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AdPilot interactive product-ad agent")
    parser.add_argument("--projects-root", default=str(DEFAULT_PROJECT_ROOT))
    commands = parser.add_subparsers(dest="command", required=True)

    def add_model_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT), help="Directory containing downloaded AdPilot models.")
        command.add_argument("--vlm-model", help="Override the local Qwen model path.")
        command.add_argument("--keyframe-model", help="Override the local FLUX Kontext model path.")
        command.add_argument("--video-model", help="Override the local Wan I2V model path.")

    def add_brief_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--product-category", help="Optional product category, such as electronics or cosmetic.")
        command.add_argument("--product-description", help="Visible product description to preserve during generation.")
        command.add_argument("--target-audience", help="Intended audience for the ad plan.")
        command.add_argument("--ad-mood", help="Visual mood for the ad plan.")
        command.add_argument("--identity-anchors", help="Comma-separated visible details that must be preserved.")
        command.add_argument("--package-state", choices=["sealed", "open_with_contents"])

    create = commands.add_parser("create", help="Create a project from a product image and creative prompt.")
    create.add_argument("--product", required=True)
    create.add_argument("--brand", required=True)
    create.add_argument("--prompt", required=True, help="Creative direction, such as 'warm editorial beauty ad with glass reflections'.")
    create.add_argument("--mode", choices=["auto", "guided"], default="guided")
    create.add_argument("--reference", action="append", default=[], help="Optional additional real product view. Repeat for more views.")
    create.add_argument("--project-name", help="Optional project label. Defaults to the product image filename.")
    create.add_argument("--platform", choices=["shorts", "instagram", "tiktok", "youtube", "landscape"], default="landscape")
    create.add_argument("--duration", type=int, default=9)
    create.add_argument("--max-agent-steps", type=int, default=16)
    create.add_argument("--max-keyframe-attempts", type=int, default=2)
    create.add_argument("--max-video-attempts", type=int, default=2)
    create.add_argument("--keyframe-candidates", type=int, default=2)
    create.add_argument("--video-candidates", type=int, default=1)
    add_model_options(create)
    add_brief_options(create)
    create.set_defaults(handler=command_create)

    run = commands.add_parser("run", help="Start a newly created project until it pauses or finishes.")
    run.add_argument("project")
    run.set_defaults(handler=command_run)

    approve = commands.add_parser("approve", help="Approve the current storyboard and continue.")
    approve.add_argument("project")
    approve.set_defaults(handler=command_approve)

    select = commands.add_parser("select", help="Select a keyframe candidate before video generation.")
    select.add_argument("project")
    select.add_argument("--shot", required=True)
    select.add_argument("--candidate", required=True, type=int)
    select.set_defaults(handler=command_select)

    feedback = commands.add_parser("feedback", help="Give feedback and regenerate the selected shot or all shots.")
    feedback.add_argument("project")
    feedback.add_argument("--shot", default="all")
    feedback.add_argument("--message", required=True)
    feedback.set_defaults(handler=command_feedback)

    reference = commands.add_parser("add-reference", help="Add a real product reference and revise the project.")
    reference.add_argument("project")
    reference.add_argument("--image", required=True)
    reference.add_argument("--shot", default="all")
    reference.set_defaults(handler=command_add_reference)

    cancel = commands.add_parser("cancel", help="Cancel a project waiting for a decision.")
    cancel.add_argument("project")
    cancel.set_defaults(handler=command_cancel)

    status = commands.add_parser("status", help="Show project assets, checkpoint status, and latest event.")
    status.add_argument("project")
    status.set_defaults(handler=command_status)

    chat = commands.add_parser("chat", help="Create and run a guided project in one terminal session.")
    chat.add_argument("--product", required=True)
    chat.add_argument("--brand", required=True)
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--reference", action="append", default=[])
    chat.add_argument("--project-name", help="Optional project label. Defaults to the product image filename.")
    chat.add_argument("--platform", choices=["shorts", "instagram", "tiktok", "youtube", "landscape"], default="landscape")
    add_model_options(chat)
    add_brief_options(chat)
    chat.set_defaults(handler=command_chat)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
