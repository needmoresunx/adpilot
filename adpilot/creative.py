from __future__ import annotations

from pathlib import Path
from typing import Any


def compact_words(text: str | None, max_words: int) -> str:
    return " ".join((text or "").replace("(", "").replace(")", "").split()[:max_words])


def make_keyframe_prompts(identity_card: Any, plan: Any, reference_mode: str = "front_lock") -> list[str]:
    brief = identity_card.product_brief
    description = compact_words(brief.get("description"), 14) or "the supplied product"
    mood = compact_words(brief.get("mood"), 12) or "premium cinematic"
    category = brief.get("category", "product")
    view_instruction = (
        "Keep each supplied product part's exact geometry, count, and assembly relationship. "
        "Articulated parts may move only as rigid pieces around their existing hinges; never stretch, melt, reshape, add, or remove parts."
        if reference_mode == "front_lock"
        else "Use the supplied product view for this shot; preserve each part's exact geometry, count, and assembly relationship."
    )
    positions = {"center": "centered", "left": "on the left third", "right": "on the right third"}
    prompts = []
    for shot in plan.shots:
        set_description = compact_words(shot.background_prompt, 18)
        prompts.append(
            compact_words(
                f"{view_instruction} {shot.goal}. Set: {set_description}. Create a photorealistic high-end "
                f"{category} advertising still of {description}, {mood}. Place one product package "
                f"{positions.get(shot.product_position, 'centered')} with realistic light and reflections.",
                52,
            )
        )
    return prompts


def make_video_prompts(identity_card: Any, plan: Any) -> list[str]:
    brief = identity_card.product_brief
    description = compact_words(brief.get("description"), 14) or "the reference product"
    mood = compact_words(brief.get("mood"), 12) or "premium cinematic"
    category = brief.get("category", "product")
    prompts = []
    for shot in plan.shots:
        prompts.append(
            compact_words(
                "Keep each supplied product part's exact geometry, count, and assembly relationship. "
                "Articulated parts may move only as rigid pieces around their existing hinges; never stretch, melt, reshape, add, or remove parts. "
                f"{shot.goal}. {compact_words(shot.motion_prompt, 16)}. "
                f"Cinematic {category} commercial of {description}, {mood}, high-end advertising cinematography",
                58,
            )
        )
    return prompts


def make_repair_prompt(prompt: str, report: Any, max_words: int) -> str:
    instruction = compact_words(report.repair_instruction, 12)
    if not instruction:
        instruction = " ".join(str(reason).replace("_", " ") for reason in report.failure_reasons[:2])
    if not instruction:
        instruction = "preserve the reference product identity"
    return compact_words(f"Repair target: {instruction}. {prompt}", max_words)


def select_keyframe_candidates(
    candidate_paths: list[list[Path]],
    candidate_reports: list[list[Any]],
    rankings: list[dict[str, Any]] | None = None,
):
    selected_frames, selected_reports, selection_log = [], [], []
    readability = {"readable": 2, "uncertain": 1, "not_applicable": 1, "unreadable": 0}

    def trusted_visual_score(report: Any) -> int | None:
        metric = getattr(report, "identity_evidence", {}).get("visual_metric", {})
        if metric.get("available") and metric.get("source") == "rembg_birefnet":
            return report.identity_audit_score
        return None

    for group_index, (paths, reports) in enumerate(zip(candidate_paths, candidate_reports)):
        ranking = rankings[group_index] if rankings is not None else {}
        preferred = ranking.get("selected_candidate")
        passing_indices = [index for index, report in enumerate(reports) if report.passed]
        visual_scores = [trusted_visual_score(report) for report in reports]
        audit_scores = [score if score is not None else 0 for score in visual_scores]
        best_score = max((audit_scores[index] for index in passing_indices), default=0)
        score_indices = [index for index in passing_indices if visual_scores[index] is not None]
        if len(score_indices) != len(passing_indices):
            # Do not compare a learned product mask to an untrusted fallback mask.
            score_indices = []
        score_tied_indices = [index for index in score_indices if audit_scores[index] >= best_score - 5]
        pairwise_can_decide = (
            isinstance(preferred, int)
            and 1 <= preferred <= len(reports)
            and (
                ((preferred - 1) in score_tied_indices)
                or (not score_indices and (preferred - 1) in passing_indices)
            )
        )
        if pairwise_can_decide:
            preferred_index = preferred - 1
            index = preferred_index
            selection_method = str(ranking.get("selection_method") or "pairwise_identity")
        elif score_indices:
            index = max(
                score_indices,
                key=lambda candidate_index: (
                    audit_scores[candidate_index],
                    readability.get(reports[candidate_index].label_readability or "uncertain", 0),
                ),
            )
            selection_method = "audit_score" if len(score_tied_indices) == 1 else "audit_score_tie_break"
        elif passing_indices:
            index, _ = max(
                ((candidate_index, reports[candidate_index]) for candidate_index in passing_indices),
                key=lambda item: readability.get(item[1].label_readability or "uncertain", 0),
            )
            selection_method = str(ranking.get("selection_method") or "unscored_tie_break")
        else:
            index, _ = max(
                enumerate(reports),
                key=lambda item: readability.get(item[1].label_readability or "uncertain", 0),
            )
            selection_method = str(ranking.get("selection_method") or "audit_tie_break")
        report = reports[index]
        selected_frames.append(paths[index])
        selected_reports.append(report)
        selection_log.append(
            {
                "shot_id": report.shot_id,
                "selected_candidate": index + 1,
                "selected_path": str(paths[index]),
                "selection_method": selection_method,
                "pairwise_comparisons": ranking.get("comparisons", []),
                "candidates": [
                    {
                        "candidate": candidate_index + 1,
                        "path": str(path),
                        "passed": candidate_report.passed,
                        "identity_verdict": candidate_report.identity_verdict,
                        "identity_checks": candidate_report.identity_checks,
                        "identity_audit_score": candidate_report.identity_audit_score,
                        "identity_evidence": candidate_report.identity_evidence,
                        "failure_reasons": candidate_report.failure_reasons,
                    }
                    for candidate_index, (path, candidate_report) in enumerate(zip(paths, reports))
                ],
            }
        )
    return selected_frames, selected_reports, selection_log


def select_video_candidates(candidate_sequences: list[list[list[Path]]], candidate_reports: list[list[Any]]):
    selected_sequences, selected_reports, selection_log = [], [], []
    for sequences, reports in zip(candidate_sequences, candidate_reports):
        def audit_key(item: tuple[int, Any]) -> tuple[bool, bool, bool]:
            report = item[1]
            return (
                report.passed,
                report.temporal_consistency == "stable",
                report.temporal_consistency == "minor_drift",
            )

        ranked = [audit_key(item) for item in enumerate(reports)]
        index, report = max(enumerate(reports), key=audit_key)
        selection_method = "audit_temporal" if ranked.count(max(ranked)) == 1 else "audit_temporal_tie_break"
        selected_sequences.append(sequences[index])
        selected_reports.append(report)
        selection_log.append(
            {
                "shot_id": report.shot_id,
                "selected_candidate": index + 1,
                "selected_first_frame": str(sequences[index][0]),
                "selection_method": selection_method,
                "failure_reasons": report.failure_reasons,
            }
        )
    return selected_sequences, selected_reports, selection_log
