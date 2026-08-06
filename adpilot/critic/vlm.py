from __future__ import annotations

from pathlib import Path

from adpilot.critic.critique import CritiqueReport
from adpilot.identity.card import IdentityCard
from adpilot.identity.vlm import QwenVisionSession, parse_json_response
from adpilot.planner.schema import ShotPlan


KEYFRAME_CRITIQUE_PROMPT = """You are a strict visual quality reviewer for an AI product-ad pipeline.
Image 1 is the ground-truth product reference. Image 2 is a generated advertising keyframe.
Judge only the physical product identity. Do not penalize a changed background, lighting, surface,
camera angle, position, or product scale. Return only JSON:
{
  "identity_verdict": "pass|fail",
  "silhouette_match": "match|minor_drift|mismatch",
  "component_match": "match|minor_drift|mismatch",
  "color_match": "match|minor_drift|mismatch",
  "product_visible": true,
  "product_count": 1,
  "label_readability": "readable|uncertain|unreadable|not_applicable",
  "visual_drift": ["specific visible differences from the reference"],
  "failure_reasons": ["only product_absent, product_duplicate, silhouette_mismatch, color_mismatch, component_mismatch, branding_unreadable"],
  "repair_instruction": "one concise targeted regeneration instruction",
  "evidence": "one concise visual justification"
}
Set identity_verdict to pass only when silhouette_match, component_match, and color_match are not mismatch.
Use "match" only when you can verify the same visible feature in both images. Use "minor_drift" for
any uncertain or approximate match; do not assume a generated product matches because its category is similar.
Compare component inventory, not just the overall category: any extra detached part, duplicate component,
missing component, or altered assembly relationship is component_mismatch and must fail. For example, an
extra loose earbud beside a charging case is a mismatch even though earbuds normally belong to that product.
Readability is required only when the run explicitly asks for branding. Return compact JSON only: each array has at most one item of
at most six words; failure_reasons contains only the listed code values; repair_instruction and evidence
are each at most eight words. Never copy field descriptions or generic phrases such as "specific visible differences" into the answer."""


VIDEO_CRITIQUE_PROMPT = """You are a strict visual quality reviewer for an AI product-ad pipeline.
Image 1 is the ground-truth product reference. The remaining images are the first, middle, and
last sampled frames from one generated product-video shot. Judge only visible evidence.
Judge only the physical product identity and temporal stability. Do not penalize changed background,
lighting, camera angle, position, or scale. Return only JSON:
{
  "identity_verdict": "pass|fail",
  "silhouette_match": "match|minor_drift|mismatch",
  "component_match": "match|minor_drift|mismatch",
  "color_match": "match|minor_drift|mismatch",
  "product_visible": true,
  "product_count": 1,
  "label_readability": "readable|uncertain|unreadable|not_applicable",
  "temporal_consistency": "stable|minor_drift|severe_drift",
  "constraint_verdict": "pass|fail|not_applicable",
  "visual_drift": ["specific differences across frames or from the reference"],
  "failure_reasons": ["only product_absent, product_duplicate, silhouette_mismatch, color_mismatch, component_mismatch, branding_unreadable, temporal_drift, temporal_artifact, final_constraint_violation"],
  "repair_instruction": "one concise targeted regeneration instruction",
  "evidence": "one concise visual justification"
}
Set identity_verdict to pass only when silhouette_match, component_match, and color_match are not mismatch,
and the product does not disappear, duplicate, or change shape/color/components across frames. Readability
is required only when the run explicitly asks for branding. A temporal artifact includes morphing, melting, warping, texture smear, or
flicker inside the product, even when its outer silhouette remains stable. Return compact JSON only: each array has at most one item of at most six
words; failure_reasons contains only the listed code values; repair_instruction and evidence are each
at most eight words."""


ALLOWED_CRITIC_FAILURES = {
    "product_absent",
    "product_duplicate",
    "silhouette_mismatch",
    "color_mismatch",
    "component_mismatch",
    "branding_unreadable",
    "temporal_drift",
    "temporal_artifact",
    "final_constraint_violation",
}

PAIRWISE_KEYFRAME_RANKING_PROMPT = """Image 1 is the ground-truth product reference. Images 2 and 3 are two generated
advertising candidates for the same shot. Choose the candidate that better preserves the reference product's
silhouette, visible components, and colors. Ignore background, lighting, camera angle, scale, and composition.
Treat an extra detached part, duplicate component, missing component, or altered assembly relationship as worse identity preservation.
Do not prefer Image 2 by default. Return only JSON:
{
  "preferred_candidate": "A|B|tie",
  "reason": "one concise identity comparison"
}
Return tie when the visible product identity is equally preserved or cannot be distinguished."""


def reference_identity_constraint(
    identity_card: IdentityCard,
    shot: ShotPlan | None = None,
    final_shot_constraint: str = "",
) -> str:
    """Give the VLM concrete reference cues instead of an abstract identity request."""
    brief = identity_card.product_brief
    category = brief.get("category", "product")
    ratio = float(identity_card.aspect_ratio)
    open_snack = category == "snack" and str(brief.get("package_state") or "").strip().lower() == "open_with_contents"
    if category == "fragrance":
        silhouette = "a wide square bottle" if ratio >= 0.82 else "a tall rectangular bottle"
    elif category == "snack":
        silhouette = "an upright pouch naturally open at its top seam" if open_snack else "a sealed unopened upright pouch"
    else:
        silhouette = "the reference product silhouette"
    anchors = str(brief.get("identity_anchors") or "").strip()
    traits = [str(value).strip() for value in brief.get("visible_traits", [])[:4] if str(value).strip()]
    trait_text = anchors or "; ".join(traits) or "its visible components and colors"
    if open_snack:
        snack_constraint = (
            " A naturally open top and a few visible gummy bears are expected. "
            "Mark fail for a torn pouch, a huge spill, floating candy, or changed branding."
        )
    elif category == "snack":
        snack_constraint = " Mark fail if the package is torn/open or candy visibly exits it."
    else:
        snack_constraint = ""
    final_snack_constraint = ""
    if shot is not None and shot.shot_id == "shot_03" and final_shot_constraint:
        final_snack_constraint = (
            f" Final-packshot constraint: {final_shot_constraint} Mark fail if this constraint is visibly violated."
        )
    return (
        f"Reference-specific constraint: it is {silhouette}. Required visible traits: {trait_text}. "
        "Mark fail for a changed aspect ratio/silhouette or a missing required component, even if it is still a plausible product."
        f"{snack_constraint}{final_snack_constraint}"
    )


def _as_count(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _critic_failure_codes(value: object) -> list[str]:
    return [reason for reason in _as_list(value) if reason.lower() in ALLOWED_CRITIC_FAILURES]


def _ask_critic_json(session: QwenVisionSession, image_paths: list[Path], prompt: str) -> tuple[str, dict]:
    """Ask for a compact critic response and retry once if generation was cut off."""
    raw_response = session.ask(image_paths, prompt, max_new_tokens=160)
    try:
        return raw_response, parse_json_response(raw_response)
    except ValueError as first_error:
        retry_prompt = (
            f"{prompt}\n\nYour previous response was incomplete. Return one compact JSON object only, "
            "with no markdown and no explanation outside the required fields."
        )
        retry_response = session.ask(image_paths, retry_prompt, max_new_tokens=320)
        try:
            return retry_response, parse_json_response(retry_response)
        except ValueError as retry_error:
            raise ValueError(
                "Qwen critic returned invalid JSON twice. "
                f"First error: {first_error}. Retry error: {retry_error}."
            ) from retry_error


def _pairwise_keyframe_ranking(
    session: QwenVisionSession,
    identity_card: IdentityCard,
    shot: ShotPlan,
    reference_path: Path,
    candidates: list[Path],
    reports: list[CritiqueReport],
) -> dict:
    """Rank candidates comparatively; VLM-provided absolute scores are not calibrated."""
    viable = [index for index, report in enumerate(reports) if report.passed]
    pool = viable or list(range(len(candidates)))
    if len(pool) == 1:
        return {
            "selected_candidate": pool[0] + 1,
            "selection_method": "audit_filter",
            "comparisons": [],
        }

    winner = pool[0]
    decisive = False
    comparisons: list[dict] = []
    for round_index, challenger in enumerate(pool[1:], start=1):
        # Vary the displayed order by shot and tournament round so a deterministic
        # model cannot systematically favour candidate A.
        display_winner_first = (sum(ord(char) for char in shot.shot_id) + round_index) % 2 == 0
        left, right = (winner, challenger) if display_winner_first else (challenger, winner)
        prompt = (
            f"{PAIRWISE_KEYFRAME_RANKING_PROMPT}\n\n"
            f"{reference_identity_constraint(identity_card, shot)}"
        )
        raw_response, result = _ask_critic_json(
            session,
            [reference_path, candidates[left], candidates[right]],
            prompt,
        )
        preference = str(result.get("preferred_candidate") or "tie").strip().lower()
        preferred_index = None
        if preference in {"a", "candidate_a", "left"}:
            preferred_index = left
        elif preference in {"b", "candidate_b", "right"}:
            preferred_index = right
        if preferred_index is not None:
            winner = preferred_index
            decisive = True
        comparisons.append(
            {
                "left_candidate": left + 1,
                "right_candidate": right + 1,
                "preferred_candidate": preferred_index + 1 if preferred_index is not None else None,
                "reason": str(result.get("reason") or "").strip(),
                "raw_response": raw_response,
            }
        )
    return {
        "selected_candidate": winner + 1 if decisive else None,
        "selection_method": "pairwise_identity" if decisive else "pairwise_indeterminate",
        "comparisons": comparisons,
    }


def critique_generated_keyframes(
    identity_card: IdentityCard,
    shots: list[ShotPlan],
    frame_paths: list[Path],
    model_id: str,
    device: str = "auto",
    minimum_identity_score: int = 75,
) -> list[CritiqueReport]:
    if len(shots) != len(frame_paths):
        raise ValueError("Each generated keyframe must have one corresponding shot plan.")
    with QwenVisionSession(model_id=model_id, device=device) as session:
        return [
            _critique_one_keyframe(session, identity_card, shot, frame_path, minimum_identity_score)
            for shot, frame_path in zip(shots, frame_paths)
        ]


def critique_keyframe_candidates(
    identity_card: IdentityCard,
    shots: list[ShotPlan],
    candidate_paths: list[list[Path]],
    model_id: str,
    device: str = "auto",
    minimum_identity_score: int = 75,
    reference_paths: list[Path] | None = None,
    reference_assignments: list[Path] | None = None,
    final_shot_constraint: str = "",
    require_readable_branding: bool = False,
    include_rankings: bool = False,
) -> list[list[CritiqueReport]] | tuple[list[list[CritiqueReport]], list[dict]]:
    if len(shots) != len(candidate_paths):
        raise ValueError("Each shot plan must have one candidate group.")
    if reference_assignments is not None and len(reference_assignments) != len(shots):
        raise ValueError("reference_assignments must have one entry per shot plan.")
    with QwenVisionSession(model_id=model_id, device=device) as session:
        references = [Path(identity_card.product_path), *(Path(path) for path in (reference_paths or []))]
        reports = [
            [
                _critique_one_keyframe(
                    session,
                    identity_card,
                    shot,
                    candidate,
                    minimum_identity_score,
                    reference_path=(
                        Path(reference_assignments[index])
                        if reference_assignments is not None
                        else references[index % len(references)]
                    ),
                    final_shot_constraint=final_shot_constraint,
                    require_readable_branding=require_readable_branding,
                )
                for candidate in candidates
            ]
            for index, (shot, candidates) in enumerate(zip(shots, candidate_paths))
        ]
        if not include_rankings:
            return reports
        rankings = [
            _pairwise_keyframe_ranking(
                session,
                identity_card,
                shot,
                (
                    Path(reference_assignments[index])
                    if reference_assignments is not None
                    else references[index % len(references)]
                ),
                candidates,
                shot_reports,
            )
            for index, (shot, candidates, shot_reports) in enumerate(zip(shots, candidate_paths, reports))
        ]
        return reports, rankings


def sample_wan_frames(frame_dir: Path, shot_count: int) -> list[list[Path]]:
    """Pick first/middle/last evidence frames from each Wan shot without decoding the mp4."""
    groups: list[list[Path]] = []
    for shot_index in range(1, shot_count + 1):
        frames = sorted(frame_dir.glob(f"shot_{shot_index:02d}_frame_*.png"))
        if not frames:
            raise FileNotFoundError(f"No Wan frames found for shot {shot_index} in {frame_dir}")
        indices = sorted({0, len(frames) // 2, len(frames) - 1})
        groups.append([frames[index] for index in indices])
    return groups


def sample_frame_sequence(frames: list[Path], sample_count: int = 3) -> list[Path]:
    """Select evenly spaced evidence frames from one generated sequence."""
    if not frames:
        raise ValueError("Cannot sample an empty generated video sequence.")
    if sample_count < 1:
        raise ValueError("sample_count must be at least one.")
    if sample_count == 1:
        return [frames[0]]
    if len(frames) <= sample_count:
        return frames
    indices = sorted({round(index * (len(frames) - 1) / (sample_count - 1)) for index in range(sample_count)})
    return [frames[index] for index in indices]


def critique_generated_video_frames(
    identity_card: IdentityCard,
    shots: list[ShotPlan],
    frame_groups: list[list[Path]],
    model_id: str,
    device: str = "auto",
    minimum_identity_score: int = 75,
) -> list[CritiqueReport]:
    if len(shots) != len(frame_groups):
        raise ValueError("Each shot plan must have one generated video-frame group.")
    with QwenVisionSession(model_id=model_id, device=device) as session:
        reports: list[CritiqueReport] = []
        for shot, frames in zip(shots, frame_groups):
            reports.append(_critique_one_video(session, identity_card, shot, frames, minimum_identity_score))
        return reports


def critique_video_candidates(
    identity_card: IdentityCard,
    shots: list[ShotPlan],
    candidate_sequences: list[list[list[Path]]],
    model_id: str,
    device: str = "auto",
    minimum_identity_score: int = 75,
    reference_paths: list[Path] | None = None,
    reference_assignments: list[Path] | None = None,
    final_shot_constraint: str = "",
    require_readable_branding: bool = False,
) -> list[list[CritiqueReport]]:
    """Audit every Wan candidate and retain sampled evidence frames in its report."""
    if len(shots) != len(candidate_sequences):
        raise ValueError("Each shot plan must have one generated video candidate group.")
    if reference_assignments is not None and len(reference_assignments) != len(shots):
        raise ValueError("reference_assignments must have one entry per shot plan.")
    with QwenVisionSession(model_id=model_id, device=device) as session:
        references = [Path(identity_card.product_path), *(Path(path) for path in (reference_paths or []))]
        return [
            [
                _critique_one_video(
                    session,
                    identity_card,
                    shot,
                    sample_frame_sequence(sequence, sample_count=5 if final_shot_constraint and shot.shot_id == "shot_03" else 3),
                    minimum_identity_score,
                    reference_path=(
                        Path(reference_assignments[index])
                        if reference_assignments is not None
                        else references[index % len(references)]
                    ),
                    final_shot_constraint=final_shot_constraint,
                    require_readable_branding=require_readable_branding,
                )
                for sequence in sequences
            ]
            for index, (shot, sequences) in enumerate(zip(shots, candidate_sequences))
        ]


def _critique_one_video(
    session: QwenVisionSession,
    identity_card: IdentityCard,
    shot: ShotPlan,
    frames: list[Path],
    minimum_identity_score: int,
    reference_path: Path | None = None,
    final_shot_constraint: str = "",
    require_readable_branding: bool = False,
) -> CritiqueReport:
    branding_policy = "Readable brand text is required for this run." if require_readable_branding else "Brand text readability is not required for this run; judge product form, components, and color instead."
    prompt = f"{VIDEO_CRITIQUE_PROMPT}\n\n{branding_policy}\n\n{reference_identity_constraint(identity_card, shot, final_shot_constraint)}"
    raw_response, result = _ask_critic_json(session, [reference_path or Path(identity_card.product_path), *frames], prompt)
    report = _report_from_result(
        result,
        shot,
        minimum_identity_score,
        raw_response,
        require_readable_branding=require_readable_branding,
    )
    temporal = str(result.get("temporal_consistency") or "severe_drift").lower()
    if temporal == "severe_drift":
        report.failure_reasons = list(dict.fromkeys([*report.failure_reasons, "severe_temporal_drift"]))
        report.passed = False
    if final_shot_constraint and shot.shot_id == "shot_03":
        constraint_verdict = str(result.get("constraint_verdict") or "").strip().lower()
        if constraint_verdict != "pass":
            report.failure_reasons = list(
                dict.fromkeys([*report.failure_reasons, "final_constraint_violation"])
            )
            report.passed = False
    report.temporal_consistency = temporal
    report.evidence_frames = [str(frame) for frame in frames]
    return report


def _critique_one_keyframe(
    session: QwenVisionSession,
    identity_card: IdentityCard,
    shot: ShotPlan,
    frame_path: Path,
    minimum_identity_score: int,
    reference_path: Path | None = None,
    final_shot_constraint: str = "",
    require_readable_branding: bool = False,
) -> CritiqueReport:
    branding_policy = "Readable brand text is required for this run." if require_readable_branding else "Brand text readability is not required for this run; judge product form, components, and color instead."
    prompt = f"{KEYFRAME_CRITIQUE_PROMPT}\n\n{branding_policy}\n\n{reference_identity_constraint(identity_card, shot, final_shot_constraint)}"
    raw_response, result = _ask_critic_json(session, [reference_path or Path(identity_card.product_path), frame_path], prompt)
    report = _report_from_result(
        result,
        shot,
        minimum_identity_score,
        raw_response,
        require_readable_branding=require_readable_branding,
    )
    report.evidence_frames = [str(frame_path)]
    return report


def _report_from_result(
    result: dict,
    shot: ShotPlan,
    minimum_identity_score: int,
    raw_response: str | None = None,
    require_readable_branding: bool = False,
) -> CritiqueReport:
    product_visible = bool(result.get("product_visible"))
    product_count = _as_count(result.get("product_count"))
    readability = str(result.get("label_readability") or "uncertain").lower()
    verdict = str(result.get("identity_verdict") or "").strip().lower()
    if verdict not in {"pass", "fail"}:
        verdict = None
    identity_checks = {
        name: str(result.get(name) or "unknown").strip().lower()
        for name in ("silhouette_match", "component_match", "color_match")
    }
    reasons = _critic_failure_codes(result.get("failure_reasons"))
    check_failures = {
        "silhouette_match": "silhouette_mismatch",
        "component_match": "component_mismatch",
        "color_match": "color_mismatch",
    }
    for check_name, failure_code in check_failures.items():
        if identity_checks[check_name] == "mismatch":
            reasons.append(failure_code)
    if any(value == "unknown" for value in identity_checks.values()):
        reasons.append("incomplete_identity_evidence")
    if verdict == "fail":
        reasons.append("critic_identity_verdict_fail")
    if not product_visible:
        reasons.append("product_not_visible")
    if product_count is not None and product_count != 1:
        reasons.append("unexpected_product_count")
    if require_readable_branding and readability == "unreadable":
        reasons.append("label_unreadable")
    reasons = list(dict.fromkeys(reasons))
    passed = not reasons
    return CritiqueReport(
        shot_id=shot.shot_id,
        passed=passed,
        product_scale=shot.product_scale,
        color_delta=0.0,
        shape_score=None,
        logo_area_ratio=None,
        product_bbox=None,
        logo_bbox_in_frame=None,
        ocr_text=None,
        ocr_available=False,
        failure_reasons=reasons,
        critic_name="qwen2_5_vl",
        identity_score=None,
        identity_verdict=verdict,
        identity_checks=identity_checks,
        # The score is attached later from pixel-level visual comparison, not Qwen's labels.
        identity_audit_score=None,
        product_visible=product_visible,
        product_count=product_count,
        label_readability=readability,
        visual_drift=_as_list(result.get("visual_drift")),
        repair_instruction=str(result.get("repair_instruction") or "").strip() or None,
        evidence=str(result.get("evidence") or "").strip() or None,
        raw_response=raw_response,
    )
