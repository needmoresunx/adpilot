import unittest
from pathlib import Path

from adpilot.critic.critique import CritiqueReport
from adpilot.critic.vlm import sample_frame_sequence
from adpilot.creative import select_video_candidates


def report(score: int, passed: bool, temporal: str) -> CritiqueReport:
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
        label_readability="readable",
        temporal_consistency=temporal,
    )


class VideoCandidateSelectionTests(unittest.TestCase):
    def test_passing_stable_sequence_beats_higher_scoring_failure(self):
        candidates = [[
            [Path("shot_01/candidate_01/frame_000.png")],
            [Path("shot_01/candidate_02/frame_000.png")],
        ]]
        reports = [[report(92, False, "severe_drift"), report(84, True, "stable")]]

        selected, selected_reports, log = select_video_candidates(candidates, reports)

        self.assertEqual(selected[0][0], Path("shot_01/candidate_02/frame_000.png"))
        self.assertTrue(selected_reports[0].passed)
        self.assertEqual(log[0]["selected_candidate"], 2)

    def test_samples_first_middle_and_last_without_duplicates(self):
        frames = [Path(f"frame_{index:03d}.png") for index in range(5)]
        self.assertEqual(
            sample_frame_sequence(frames),
            [Path("frame_000.png"), Path("frame_002.png"), Path("frame_004.png")],
        )
        self.assertEqual(sample_frame_sequence([Path("frame_000.png")]), [Path("frame_000.png")])


if __name__ == "__main__":
    unittest.main()
