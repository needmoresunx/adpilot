from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from adpilot.creative import (
    compact_words,
    make_keyframe_prompts,
    make_repair_prompt,
    make_video_prompts,
    select_keyframe_candidates,
    select_video_candidates,
)
from adpilot.backends.keyframe import make_keyframe_backend
from adpilot.backends.video import make_video_backend
from adpilot.critic.critique import CritiqueReport
from adpilot.critic.evidence import build_keyframe_identity_evidence
from adpilot.critic.vlm import critique_keyframe_candidates, critique_video_candidates
from adpilot.identity.builder import build_identity_card
from adpilot.identity.card import IdentityCard
from adpilot.identity.vlm import diffusion_safe_feedback
from adpilot.planner.schema import AdPlan, ShotPlan
from adpilot.planner.vlm import make_vlm_plan
from adpilot.preview.storyboard import make_storyboard
from adpilot.report.html import write_html_report
from adpilot.utils.json_io import write_json

from .store import ProjectConfig, ProjectStore


class AgentState(TypedDict, total=False):
    project_id: str
    mode: str
    revision: int
    status: str
    prompt: str
    product_asset_id: str
    reference_asset_ids: list[str]
    identity_card: dict[str, Any]
    plan: dict[str, Any]
    attempts: dict[str, int]
    agent_steps_used: int
    keyframe_candidates: dict[str, list[str]]
    keyframe_candidate_reports: dict[str, list[dict[str, Any]]]
    selected_keyframes: dict[str, str]
    keyframe_reports: dict[str, dict[str, Any]]
    keyframe_selection: dict[str, dict[str, Any]]
    storyboard_path: str
    generation_metadata: dict[str, Any]
    video_candidates: dict[str, list[list[str]]]
    video_candidate_reports: dict[str, list[dict[str, Any]]]
    selected_videos: dict[str, list[str]]
    video_reports: dict[str, dict[str, Any]]
    video_selection: dict[str, dict[str, Any]]
    video_metadata: dict[str, Any]
    feedback_by_shot: dict[str, list[str]]
    generation_feedback_by_shot: dict[str, list[str]]
    repair_log: list[dict[str, Any]]
    pending_regenerate_shots: list[str]
    repair_stage: str | None
    decision: dict[str, Any]
    decision_outcome: str
    guidance: dict[str, Any]
    final_video: str
    report_path: str
    error: str


def initial_state(config: ProjectConfig) -> AgentState:
    return {
        "project_id": config.project_id,
        "mode": config.mode,
        "revision": 1,
        "status": "running",
        "prompt": config.prompt,
        "product_asset_id": config.product_asset_id,
        "reference_asset_ids": list(config.reference_asset_ids),
        "attempts": {},
        "agent_steps_used": 0,
        "feedback_by_shot": {},
        "generation_feedback_by_shot": {},
        "repair_log": [],
    }


def _identity(value: dict[str, Any]) -> IdentityCard:
    return IdentityCard(**value)


def _plan(value: dict[str, Any]) -> AdPlan:
    return AdPlan(
        concept=str(value["concept"]),
        style=str(value["style"]),
        shots=[ShotPlan(**shot) for shot in value["shots"]],
    )


def _report(value: dict[str, Any]) -> CritiqueReport:
    return CritiqueReport(**value)


def _paths(paths: list[str]) -> list[Path]:
    return [Path(path) for path in paths]


def _shot_ids(plan: AdPlan) -> list[str]:
    return [shot.shot_id for shot in plan.shots]


def _attempt_key(stage: str, shot_id: str) -> str:
    return f"{stage}:{shot_id}"


def _has_budget(state: AgentState, config: ProjectConfig, cost: int) -> bool:
    return state.get("agent_steps_used", 0) + cost <= config.max_agent_steps


def _budget_failure(state: AgentState, config: ProjectConfig, stage: str) -> dict[str, Any]:
    return {
        "status": "needs_user_input",
        "guidance": {
            "type": "step_budget_exhausted",
            "stage": stage,
            "message": (
                f"The revision reached its maximum of {config.max_agent_steps} generation actions. "
                "Change the direction, choose an existing candidate, or start a new revision."
            ),
            "allowed_actions": ["feedback", "select", "cancel"],
        },
    }


def _feedback_prompt(prompt: str, feedback: list[str], maximum_words: int) -> str:
    if not feedback:
        return prompt
    direction = compact_words(" ".join(feedback[-2:]), 16)
    return compact_words(f"User creative direction: {direction}. {prompt}", maximum_words)


class AdPilotGraph:
    """LangGraph orchestration over the existing local Qwen, FLUX, and Wan tools."""

    def __init__(self, store: ProjectStore):
        self.store = store
        self.config = store.config

    def _reference_paths(self, state: AgentState) -> list[Path]:
        return [self.store.asset_path(asset_id) for asset_id in state["reference_asset_ids"]]

    def _revision_dir(self, state: AgentState) -> Path:
        return self.store.revision_dir(state["revision"])

    def prepare(self, state: AgentState) -> dict[str, Any]:
        if state.get("identity_card") and state.get("plan"):
            return {}
        if not _has_budget(state, self.config, 1):
            return _budget_failure(state, self.config, "planning")
        references = self._reference_paths(state)
        revision_dir = self._revision_dir(state)
        identity = build_identity_card(
            product_path=references[0],
            brand=self.config.brand,
            output_dir=revision_dir,
            auto_cutout=True,
            product_category=self.config.product_category,
            product_description=self.config.product_description,
            target_audience=self.config.target_audience,
            ad_mood=self.config.ad_mood,
            identity_anchors=self.config.identity_anchors,
            package_state=self.config.package_state,
            auto_brief=True,
            vlm_model=self.config.models["vlm"],
            vlm_device="auto",
        )
        plan = make_vlm_plan(
            identity,
            references[0],
            state["prompt"],
            self.config.duration,
            self.config.platform,
            self.config.models["vlm"],
            "auto",
            reference_paths=references[1:],
        )
        write_json(revision_dir / "identity_card.json", identity.to_dict())
        write_json(revision_dir / "shot_plan.json", plan.to_dict())
        self.store.record_event("plan_created", {"revision": state["revision"]})
        return {
            "identity_card": identity.to_dict(),
            "plan": plan.to_dict(),
            "agent_steps_used": state.get("agent_steps_used", 0) + 1,
            "status": "running",
        }

    def generate_keyframes(self, state: AgentState) -> dict[str, Any]:
        plan = _plan(state["plan"])
        identity = _identity(state["identity_card"])
        candidates = {key: list(value) for key, value in state.get("keyframe_candidates", {}).items()}
        requested = state.get("pending_regenerate_shots") or _shot_ids(plan)
        repair_mode = state.get("repair_stage") == "keyframe"
        attempts = dict(state.get("attempts", {}))
        limit = self.config.max_keyframe_attempts
        blocked = [shot_id for shot_id in requested if attempts.get(_attempt_key("keyframe", shot_id), 0) >= limit]
        if blocked:
            return {
                "status": "needs_user_input",
                "guidance": {
                    "type": "keyframe_attempt_limit",
                    "shots": blocked,
                    "message": "Keyframe identity repair reached its limit. Add a product view, simplify the direction, or choose an existing candidate.",
                    "allowed_actions": ["feedback", "select", "add_reference", "cancel"],
                },
            }
        if not _has_budget(state, self.config, len(requested)):
            return _budget_failure(state, self.config, "keyframe_generation")

        references = self._reference_paths(state)
        assignments = [references[index % len(references)] for index in range(len(plan.shots))]
        shot_by_id = {shot.shot_id: shot for shot in plan.shots}
        base_prompts = make_keyframe_prompts(identity, plan, "front_lock" if len(references) == 1 else "multi_view")
        prompt_by_id = {shot.shot_id: prompt for shot, prompt in zip(plan.shots, base_prompts)}
        prompts = []
        layouts = []
        selected_references = []
        for shot_id in requested:
            shot = shot_by_id[shot_id]
            prompt = _feedback_prompt(
                prompt_by_id[shot_id],
                state.get("generation_feedback_by_shot", state.get("feedback_by_shot", {})).get(shot_id, []),
                60,
            )
            if repair_mode and state.get("keyframe_reports", {}).get(shot_id):
                prompt = make_repair_prompt(prompt, _report(state["keyframe_reports"][shot_id]), 62)
            prompts.append(prompt)
            layouts.append((shot.product_position, shot.product_scale))
            selected_references.append(assignments[plan.shots.index(shot)])

        backend = make_keyframe_backend(
            "flux_kontext",
            self.config.models["keyframe"],
            device="auto",
            seed=self.config.keyframe_seed + (state["revision"] - 1) * 10_000,
            num_inference_steps=28,
            guidance_scale=2.5,
            offload_mode="model",
        )
        attempt_number = max(attempts.get(_attempt_key("keyframe", shot_id), 0) for shot_id in requested) + 1
        new_groups = backend.render_candidates(
            selected_references[0],
            prompts,
            self._revision_dir(state) / "keyframes" / f"attempt_{attempt_number:02d}",
            (1360, 768),
            1 if repair_mode else self.config.keyframe_candidates,
            seed_offset=(attempt_number - 1) * 100_000,
            reference_layouts=layouts,
            reference_paths=selected_references[1:],
        )
        metadata = backend.metadata()
        backend.release()
        for shot_id, group in zip(requested, new_groups):
            existing = candidates.get(shot_id, []) if repair_mode else []
            candidates[shot_id] = [*existing, *(str(path) for path in group)]
            for candidate_index, path in enumerate(group, start=len(existing) + 1):
                self.store.register_generated_asset(
                    path,
                    "keyframe_candidate",
                    state["revision"],
                    state["reference_asset_ids"],
                    {"shot_id": shot_id, "candidate": candidate_index, "repair": repair_mode},
                )
            attempts[_attempt_key("keyframe", shot_id)] = attempts.get(_attempt_key("keyframe", shot_id), 0) + 1
        self.store.record_event(
            "keyframes_generated",
            {"revision": state["revision"], "shots": requested, "repair": repair_mode},
        )
        return {
            "keyframe_candidates": candidates,
            "attempts": attempts,
            "agent_steps_used": state.get("agent_steps_used", 0) + len(requested),
            "pending_regenerate_shots": [],
            "repair_stage": None,
            "generation_metadata": metadata,
            "status": "running",
        }

    def audit_keyframes(self, state: AgentState) -> dict[str, Any]:
        plan = _plan(state["plan"])
        identity = _identity(state["identity_card"])
        references = self._reference_paths(state)
        candidates = state["keyframe_candidates"]
        groups = [_paths(candidates[shot.shot_id]) for shot in plan.shots]
        reports, rankings = critique_keyframe_candidates(
            identity,
            plan.shots,
            groups,
            self.config.models["vlm"],
            "auto",
            75,
            reference_assignments=[references[index % len(references)] for index in range(len(plan.shots))],
            include_rankings=True,
        )
        evidence_groups = build_keyframe_identity_evidence(
            identity,
            plan.shots,
            groups,
            reports,
            self._revision_dir(state) / "identity_evidence",
            enforce_visual_gate=len(references) == 1,
        )
        for shot_reports, shot_evidence in zip(reports, evidence_groups):
            for report, evidence in zip(shot_reports, shot_evidence):
                report.identity_evidence = evidence
        selected, selected_reports, selection = select_keyframe_candidates(groups, reports, rankings)
        selected_map = {shot.shot_id: str(path) for shot, path in zip(plan.shots, selected)}
        report_map = {shot.shot_id: report.to_dict() for shot, report in zip(plan.shots, selected_reports)}
        candidate_report_map = {
            shot.shot_id: [report.to_dict() for report in shot_reports]
            for shot, shot_reports in zip(plan.shots, reports)
        }
        selection_map = {shot.shot_id: item for shot, item in zip(plan.shots, selection)}
        repair_log = list(state.get("repair_log", []))
        for shot in plan.shots:
            attempt = state.get("attempts", {}).get(_attempt_key("keyframe", shot.shot_id), 0)
            if attempt > 1:
                report = report_map[shot.shot_id]
                repair_log.append(
                    {
                        "stage": "keyframe",
                        "shot_id": shot.shot_id,
                        "attempt": attempt,
                        "repaired": bool(report["passed"]),
                        "failure_reasons": report.get("failure_reasons", []),
                    }
                )
        storyboard = make_storyboard(selected, self._revision_dir(state) / "storyboard.png")
        write_json(self._revision_dir(state) / "keyframe_critique.json", candidate_report_map)
        self.store.register_generated_asset(
            storyboard,
            "storyboard",
            state["revision"],
            state["reference_asset_ids"],
            {"stage": "keyframe_review"},
        )
        self.store.record_event("keyframes_audited", {"revision": state["revision"]})
        return {
            "selected_keyframes": selected_map,
            "keyframe_reports": report_map,
            "keyframe_candidate_reports": candidate_report_map,
            "keyframe_selection": selection_map,
            "storyboard_path": str(storyboard),
            "repair_log": repair_log,
            "status": "running",
        }

    def queue_keyframe_repair(self, state: AgentState) -> dict[str, Any]:
        failed = [shot_id for shot_id, report in state["keyframe_reports"].items() if not report["passed"]]
        return {"pending_regenerate_shots": failed, "repair_stage": "keyframe", "status": "running"}

    def review_storyboard(self, state: AgentState) -> dict[str, Any]:
        from langgraph.types import interrupt

        plan = _plan(state["plan"])
        payload = {
            "type": "storyboard_review",
            "revision": state["revision"],
            "storyboard_path": state["storyboard_path"],
            "shots": [
                {
                    "shot_id": shot.shot_id,
                    "selected_path": state["selected_keyframes"][shot.shot_id],
                    "identity_verdict": state["keyframe_reports"][shot.shot_id].get("identity_verdict"),
                    "identity_audit_score": state["keyframe_reports"][shot.shot_id].get("identity_audit_score"),
                    "selection_method": state.get("keyframe_selection", {}).get(shot.shot_id, {}).get("selection_method"),
                    "pairwise_comparisons": state.get("keyframe_selection", {}).get(shot.shot_id, {}).get("pairwise_comparisons", []),
                    "failure_reasons": state["keyframe_reports"][shot.shot_id].get("failure_reasons", []),
                    "candidate_count": len(state["keyframe_candidates"][shot.shot_id]),
                    "candidates": [
                        {
                            "candidate": index,
                            "path": path,
                            "identity_verdict": report.get("identity_verdict"),
                            "identity_checks": report.get("identity_checks", {}),
                            "identity_audit_score": report.get("identity_audit_score"),
                            "identity_evidence": report.get("identity_evidence", {}),
                            "passed": report.get("passed"),
                            "failure_reasons": report.get("failure_reasons", []),
                        }
                        for index, (path, report) in enumerate(
                            zip(
                                state["keyframe_candidates"][shot.shot_id],
                                state["keyframe_candidate_reports"][shot.shot_id],
                            ),
                            start=1,
                        )
                    ],
                }
                for shot in plan.shots
            ],
            "allowed_actions": ["approve", "select", "feedback", "add_reference", "cancel"],
        }
        decision = interrupt(payload)
        return {"decision": decision}

    def apply_storyboard_decision(self, state: AgentState) -> dict[str, Any]:
        decision = dict(state.get("decision") or {})
        action = str(decision.get("action", "")).lower()
        if action == "approve":
            self.store.record_event("storyboard_approved", {"revision": state["revision"]})
            return {"status": "running", "decision_outcome": "generate_video"}
        if action == "cancel":
            self.store.record_event("project_cancelled", {"revision": state["revision"]})
            return {"status": "cancelled", "decision_outcome": "end"}
        if action == "select":
            shot_id = str(decision.get("shot_id", ""))
            candidate = int(decision.get("candidate", 0))
            if shot_id not in state["keyframe_candidates"] or candidate < 1:
                return {"status": "needs_user_input", "error": "Invalid shot or candidate selection.", "decision_outcome": "guidance"}
            choices = state["keyframe_candidates"][shot_id]
            if candidate > len(choices):
                return {"status": "needs_user_input", "error": "Candidate number is outside the available range.", "decision_outcome": "guidance"}
            selected = dict(state["selected_keyframes"])
            selected[shot_id] = choices[candidate - 1]
            self.store.record_event("candidate_selected", {"shot_id": shot_id, "candidate": candidate})
            return {"selected_keyframes": selected, "status": "running", "decision_outcome": "generate_video"}
        if action in {"feedback", "add_reference"}:
            feedback = dict(state.get("feedback_by_shot", {}))
            generation_feedback = dict(state.get("generation_feedback_by_shot", {}))
            target = str(decision.get("shot_id") or "all")
            targets = list(state["keyframe_candidates"]) if target == "all" or action == "add_reference" else [target]
            if any(shot_id not in state["keyframe_candidates"] for shot_id in targets):
                return {"status": "needs_user_input", "error": "Unknown shot id in feedback.", "decision_outcome": "guidance"}
            message = str(decision.get("message") or "").strip()
            if action == "feedback" and not message:
                return {"status": "needs_user_input", "error": "Feedback needs a message.", "decision_outcome": "guidance"}
            generation_message = (
                diffusion_safe_feedback(message, self.config.models["vlm"])
                if action == "feedback"
                else ""
            )
            if action == "add_reference":
                asset_id = str(decision.get("asset_id") or "")
                try:
                    self.store.asset_path(asset_id)
                except KeyError:
                    return {"status": "needs_user_input", "error": "Unknown reference asset.", "decision_outcome": "guidance"}
                reference_ids = [*state["reference_asset_ids"], asset_id]
            else:
                reference_ids = state["reference_asset_ids"]
            for shot_id in targets:
                feedback[shot_id] = [*feedback.get(shot_id, []), message] if message else feedback.get(shot_id, [])
                generation_feedback[shot_id] = (
                    [*generation_feedback.get(shot_id, []), generation_message]
                    if generation_message
                    else generation_feedback.get(shot_id, [])
                )
            revision = state["revision"] + 1
            self.store.record_event("revision_created", {"revision": revision, "action": action, "shots": targets})
            update = {
                "revision": revision,
                "reference_asset_ids": reference_ids,
                "feedback_by_shot": feedback,
                "generation_feedback_by_shot": generation_feedback,
                "pending_regenerate_shots": targets,
                "video_candidates": {},
                "selected_videos": {},
                "video_reports": {},
                "attempts": {},
                "agent_steps_used": 0,
                "status": "running",
                "decision_outcome": "replan" if action == "add_reference" else "regenerate_keyframes",
            }
            if action == "add_reference":
                update["identity_card"] = {}
                update["plan"] = {}
            return update
        return {"status": "needs_user_input", "error": "Choose approve, select, feedback, add_reference, or cancel.", "decision_outcome": "guidance"}

    def generate_videos(self, state: AgentState) -> dict[str, Any]:
        plan = _plan(state["plan"])
        identity = _identity(state["identity_card"])
        requested = state.get("pending_regenerate_shots") or _shot_ids(plan)
        repair_mode = state.get("repair_stage") == "video"
        attempts = dict(state.get("attempts", {}))
        blocked = [
            shot_id
            for shot_id in requested
            if attempts.get(_attempt_key("video", shot_id), 0) >= self.config.max_video_attempts
        ]
        if blocked:
            return {
                "status": "needs_user_input",
                "guidance": {
                    "type": "video_attempt_limit",
                    "shots": blocked,
                    "message": "Video repair reached its limit. Change the creative direction or add stronger references.",
                    "allowed_actions": ["feedback", "add_reference", "cancel"],
                },
            }
        if not _has_budget(state, self.config, len(requested)):
            return _budget_failure(state, self.config, "video_generation")

        shot_by_id = {shot.shot_id: shot for shot in plan.shots}
        base_prompts = make_video_prompts(identity, plan)
        prompt_by_id = {shot.shot_id: prompt for shot, prompt in zip(plan.shots, base_prompts)}
        prompts = []
        frames = []
        for shot_id in requested:
            prompt = _feedback_prompt(
                prompt_by_id[shot_id],
                state.get("generation_feedback_by_shot", state.get("feedback_by_shot", {})).get(shot_id, []),
                66,
            )
            if repair_mode and state.get("video_reports", {}).get(shot_id):
                prompt = make_repair_prompt(prompt, _report(state["video_reports"][shot_id]), 68)
            prompts.append(prompt)
            frames.append(Path(state["selected_keyframes"][shot_id]))
        attempt_number = max((attempts.get(_attempt_key("video", shot_id), 0) for shot_id in requested), default=0) + 1
        video_seed = self.config.video_seed + (state["revision"] - 1) * 10_000
        if repair_mode:
            video_seed += (attempt_number - 1) * 100_000
        backend = make_video_backend(
            "wan_i2v",
            self.config.models["video"],
            device="auto",
            seed=video_seed,
            num_frames=49,
            fps=16,
            generated_size=(832, 480),
            prompts=prompts,
            negative_prompt="static image, blurry, low quality, watermark, deformed product, morphing parts, stretched parts, melted surfaces, extra parts, missing parts",
            num_inference_steps=40,
            guidance_scale=3.5,
            offload_mode="model",
            endpoint_locked_shots=[False] * len(requested),
        )
        output_dir = self._revision_dir(state) / "video_candidates" / f"attempt_{attempt_number:02d}"
        new_groups = backend.render_candidates(frames, output_dir, 1 if repair_mode else self.config.video_candidates)
        metadata = backend.metadata()
        backend.release()
        candidates = {key: [list(sequence) for sequence in value] for key, value in state.get("video_candidates", {}).items()}
        for shot_id, groups in zip(requested, new_groups):
            existing = candidates.get(shot_id, []) if repair_mode else []
            candidates[shot_id] = [*existing, *[[str(path) for path in sequence] for sequence in groups]]
            attempts[_attempt_key("video", shot_id)] = attempts.get(_attempt_key("video", shot_id), 0) + 1
        self.store.record_event("videos_generated", {"revision": state["revision"], "shots": requested, "repair": repair_mode})
        return {
            "video_candidates": candidates,
            "attempts": attempts,
            "agent_steps_used": state.get("agent_steps_used", 0) + len(requested),
            "pending_regenerate_shots": [],
            "repair_stage": None,
            "video_metadata": metadata,
            "status": "running",
        }

    def audit_videos(self, state: AgentState) -> dict[str, Any]:
        plan = _plan(state["plan"])
        identity = _identity(state["identity_card"])
        references = self._reference_paths(state)
        candidates = state["video_candidates"]
        groups = [[_paths(sequence) for sequence in candidates[shot.shot_id]] for shot in plan.shots]
        reports = critique_video_candidates(
            identity,
            plan.shots,
            groups,
            self.config.models["vlm"],
            "auto",
            75,
            reference_assignments=[references[index % len(references)] for index in range(len(plan.shots))],
        )
        selected, selected_reports, selection = select_video_candidates(groups, reports)
        selected_video_map = {shot.shot_id: [str(path) for path in sequence] for shot, sequence in zip(plan.shots, selected)}
        selected_report_map = {shot.shot_id: report.to_dict() for shot, report in zip(plan.shots, selected_reports)}
        candidate_report_map = {
            shot.shot_id: [report.to_dict() for report in shot_reports]
            for shot, shot_reports in zip(plan.shots, reports)
        }
        repair_log = list(state.get("repair_log", []))
        for shot in plan.shots:
            attempt = state.get("attempts", {}).get(_attempt_key("video", shot.shot_id), 0)
            if attempt > 1:
                report = selected_report_map[shot.shot_id]
                repair_log.append(
                    {
                        "stage": "video",
                        "shot_id": shot.shot_id,
                        "attempt": attempt,
                        "repaired": bool(report["passed"]),
                        "failure_reasons": report.get("failure_reasons", []),
                    }
                )
        write_json(self._revision_dir(state) / "video_critique.json", candidate_report_map)
        return {
            "selected_videos": selected_video_map,
            "video_reports": selected_report_map,
            "video_candidate_reports": candidate_report_map,
            "video_selection": {shot.shot_id: item for shot, item in zip(plan.shots, selection)},
            "repair_log": repair_log,
            "status": "running",
        }

    def queue_video_repair(self, state: AgentState) -> dict[str, Any]:
        failed = [shot_id for shot_id, report in state["video_reports"].items() if not report["passed"]]
        return {"pending_regenerate_shots": failed, "repair_stage": "video", "status": "running"}

    def request_guidance(self, state: AgentState) -> dict[str, Any]:
        from langgraph.types import interrupt

        guidance = state.get("guidance") or {
            "type": "agent_input_needed",
            "message": state.get("error", "The agent needs a creative decision."),
            "allowed_actions": ["feedback", "add_reference", "cancel"],
        }
        decision = interrupt({"type": "guidance", "revision": state["revision"], **guidance})
        return {"decision": decision}

    def apply_guidance(self, state: AgentState) -> dict[str, Any]:
        decision = dict(state.get("decision") or {})
        action = str(decision.get("action", "")).lower()
        if action == "cancel":
            self.store.record_event("project_cancelled", {"revision": state["revision"]})
            return {"status": "cancelled", "decision_outcome": "end"}
        if action == "feedback" and not state.get("keyframe_candidates"):
            message = str(decision.get("message") or "").strip()
            if not message:
                return {"status": "needs_user_input", "error": "Feedback needs a message.", "decision_outcome": "guidance"}
            generation_message = diffusion_safe_feedback(message, self.config.models["vlm"])
            self.store.record_event("planning_feedback", {"revision": state["revision"]})
            return {
                "revision": state["revision"] + 1,
                "prompt": compact_words(f"{state['prompt']}. User creative direction: {generation_message}", 80),
                "identity_card": {},
                "plan": {},
                "attempts": {},
                "agent_steps_used": 0,
                "status": "running",
                "decision_outcome": "replan",
            }
        if action == "add_reference" and not state.get("keyframe_candidates"):
            asset_id = str(decision.get("asset_id") or "")
            try:
                self.store.asset_path(asset_id)
            except KeyError:
                return {"status": "needs_user_input", "error": "Unknown reference asset.", "decision_outcome": "guidance"}
            self.store.record_event("reference_added_before_plan", {"revision": state["revision"] + 1})
            return {
                "revision": state["revision"] + 1,
                "reference_asset_ids": [*state["reference_asset_ids"], asset_id],
                "identity_card": {},
                "plan": {},
                "attempts": {},
                "agent_steps_used": 0,
                "status": "running",
                "decision_outcome": "replan",
            }
        if action in {"feedback", "add_reference"}:
            return self.apply_storyboard_decision({**state, "decision": decision})
        return {"status": "needs_user_input", "error": "Provide feedback, add a reference, or cancel.", "decision_outcome": "guidance"}

    def finish(self, state: AgentState) -> dict[str, Any]:
        plan = _plan(state["plan"])
        identity = _identity(state["identity_card"])
        sequences = [_paths(state["selected_videos"][shot.shot_id]) for shot in plan.shots]
        backend = make_video_backend("wan_i2v", self.config.models["video"], fps=16)
        final_video = backend.render_selected(sequences, self._revision_dir(state) / "final_video.mp4")
        frames = [Path(state["selected_keyframes"][shot.shot_id]) for shot in plan.shots]
        storyboard = make_storyboard(frames, self._revision_dir(state) / "storyboard.png")
        keyframe_reports = [_report(state["keyframe_reports"][shot.shot_id]) for shot in plan.shots]
        video_reports = [_report(state["video_reports"][shot.shot_id]) for shot in plan.shots]
        report_path = write_html_report(
            self._revision_dir(state),
            identity,
            frames,
            keyframe_reports,
            storyboard,
            final_video,
            state.get("generation_metadata", {}),
            state.get("video_metadata", {}),
            video_reports,
            {"name": "qwen2_5_vl", "used_fallback": False},
            state.get("repair_log", []),
        )
        write_json(self._revision_dir(state) / "repair_log.json", state.get("repair_log", []))
        write_json(
            self._revision_dir(state) / "workflow_summary.json",
            {
                "project_id": state["project_id"],
                "revision": state["revision"],
                "mode": state["mode"],
                "agent_steps_used": state.get("agent_steps_used", 0),
                "keyframe_selection": state.get("keyframe_selection", {}),
                "video_selection": state.get("video_selection", {}),
                "final_video": str(final_video),
                "report": str(report_path),
            },
        )
        self.store.register_generated_asset(final_video, "final_video", state["revision"], state["reference_asset_ids"])
        self.store.register_generated_asset(report_path, "report", state["revision"], state["reference_asset_ids"])
        self.store.record_event("project_completed", {"revision": state["revision"]})
        return {"final_video": str(final_video), "report_path": str(report_path), "status": "completed"}

    def _route_after_prepare(self, state: AgentState) -> str:
        return "request_guidance" if state.get("status") == "needs_user_input" else "generate_keyframes"

    def _route_after_keyframe_audit(self, state: AgentState) -> str:
        failed = [shot_id for shot_id, report in state["keyframe_reports"].items() if not report["passed"]]
        if failed:
            attempts = state.get("attempts", {})
            if all(attempts.get(_attempt_key("keyframe", shot_id), 0) >= self.config.max_keyframe_attempts for shot_id in failed):
                return "request_guidance"
            return "queue_keyframe_repair"
        return "review_storyboard" if state["mode"] == "guided" else "generate_videos"

    def _route_after_storyboard_decision(self, state: AgentState) -> str:
        outcome = state.get("decision_outcome")
        if outcome == "generate_video":
            return "generate_videos"
        if outcome == "regenerate_keyframes":
            return "generate_keyframes"
        if outcome == "replan":
            return "prepare"
        if outcome == "end":
            return "__end__"
        return "request_guidance"

    def _route_after_video_audit(self, state: AgentState) -> str:
        failed = [shot_id for shot_id, report in state["video_reports"].items() if not report["passed"]]
        if not failed:
            return "finish"
        attempts = state.get("attempts", {})
        if all(attempts.get(_attempt_key("video", shot_id), 0) >= self.config.max_video_attempts for shot_id in failed):
            return "request_guidance"
        return "queue_video_repair"

    def _route_after_guidance(self, state: AgentState) -> str:
        outcome = state.get("decision_outcome")
        if outcome == "regenerate_keyframes":
            return "generate_keyframes"
        if outcome == "replan":
            return "prepare"
        if outcome == "generate_video":
            return "generate_videos"
        if outcome == "end":
            return "__end__"
        return "request_guidance"

    def build(self, checkpointer):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - dependency installation is external
            raise RuntimeError(
                "Interactive agent support requires langgraph and langgraph-checkpoint-sqlite. "
                "Install the project requirements first."
            ) from exc

        graph = StateGraph(AgentState)
        graph.add_node("prepare", self.prepare)
        graph.add_node("generate_keyframes", self.generate_keyframes)
        graph.add_node("audit_keyframes", self.audit_keyframes)
        graph.add_node("queue_keyframe_repair", self.queue_keyframe_repair)
        graph.add_node("review_storyboard", self.review_storyboard)
        graph.add_node("apply_storyboard_decision", self.apply_storyboard_decision)
        graph.add_node("generate_videos", self.generate_videos)
        graph.add_node("audit_videos", self.audit_videos)
        graph.add_node("queue_video_repair", self.queue_video_repair)
        graph.add_node("request_guidance", self.request_guidance)
        graph.add_node("apply_guidance", self.apply_guidance)
        graph.add_node("finish", self.finish)
        graph.add_edge(START, "prepare")
        graph.add_conditional_edges("prepare", self._route_after_prepare)
        graph.add_edge("generate_keyframes", "audit_keyframes")
        graph.add_conditional_edges("audit_keyframes", self._route_after_keyframe_audit)
        graph.add_edge("queue_keyframe_repair", "generate_keyframes")
        graph.add_edge("review_storyboard", "apply_storyboard_decision")
        graph.add_conditional_edges("apply_storyboard_decision", self._route_after_storyboard_decision)
        graph.add_edge("generate_videos", "audit_videos")
        graph.add_conditional_edges("audit_videos", self._route_after_video_audit)
        graph.add_edge("queue_video_repair", "generate_videos")
        graph.add_edge("request_guidance", "apply_guidance")
        graph.add_conditional_edges("apply_guidance", self._route_after_guidance)
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=checkpointer)
