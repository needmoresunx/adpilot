import unittest
from pathlib import Path

from adpilot.critic.critique import CritiqueReport
from app import select_keyframe_candidates


def report(score: int, passed: bool, readability: str = "uncertain") -> CritiqueReport:
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


if __name__ == "__main__":
    unittest.main()
