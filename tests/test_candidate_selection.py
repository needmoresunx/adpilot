import unittest
from pathlib import Path

from adpilot.critic.critique import CritiqueReport
from adpilot.creative import select_keyframe_candidates


def report(score: int, passed: bool, readability: str = "uncertain", metric_source: str = "rembg_birefnet") -> CritiqueReport:
    return CritiqueReport(
        shot_id="shot_01",
        passed=passed,
        product_scale=0.4,
        color_delta=0.0,
        shape_score=score / 100,
        logo_area_ratio=None,
        product_bbox=None,
        logo_bbox_in_frame=None,
        ocr_text=None,
        ocr_available=False,
        failure_reasons=[] if passed else ["low_identity_score"],
        identity_score=score,
        identity_audit_score=score,
        identity_evidence={"visual_metric": {"available": True, "source": metric_source}},
        label_readability=readability,
    )


class CandidateSelectionTests(unittest.TestCase):
    def test_passing_candidate_beats_higher_scoring_failed_candidate(self):
        paths = [[Path("candidate_01.jpg"), Path("candidate_02.jpg")]]
        reports = [[report(91, False), report(82, True)]]

        selected_frames, selected_reports, selection = select_keyframe_candidates(paths, reports)

        self.assertEqual(selected_frames, [Path("candidate_02.jpg")])
        self.assertTrue(selected_reports[0].passed)
        self.assertEqual(selection[0]["selected_candidate"], 2)

    def test_pairwise_preference_beats_an_untrusted_absolute_score(self):
        paths = [[Path("candidate_01.jpg"), Path("candidate_02.jpg")]]
        reports = [[report(85, True), report(85, True)]]

        selected_frames, _, selection = select_keyframe_candidates(
            paths,
            reports,
            [{"selected_candidate": 2, "selection_method": "pairwise_identity", "comparisons": []}],
        )

        self.assertEqual(selected_frames, [Path("candidate_02.jpg")])
        self.assertEqual(selection[0]["selection_method"], "pairwise_identity")

    def test_higher_rule_based_audit_score_wins_before_pairwise_tie_break(self):
        paths = [[Path("candidate_01.jpg"), Path("candidate_02.jpg")]]
        reports = [[report(100, True), report(75, True)]]

        selected_frames, _, selection = select_keyframe_candidates(
            paths,
            reports,
            [{"selected_candidate": 2, "selection_method": "pairwise_identity", "comparisons": []}],
        )

        self.assertEqual(selected_frames, [Path("candidate_01.jpg")])
        self.assertEqual(selection[0]["selection_method"], "audit_score")

    def test_heuristic_score_does_not_override_pairwise_identity_choice(self):
        paths = [[Path("candidate_01.jpg"), Path("candidate_02.jpg")]]
        reports = [[report(91, True, metric_source="corner_background_heuristic"), report(30, True, metric_source="corner_background_heuristic")]]

        selected_frames, _, selection = select_keyframe_candidates(
            paths,
            reports,
            [{"selected_candidate": 2, "selection_method": "pairwise_identity", "comparisons": []}],
        )

        self.assertEqual(selected_frames, [Path("candidate_02.jpg")])
        self.assertEqual(selection[0]["selection_method"], "pairwise_identity")


if __name__ == "__main__":
    unittest.main()
