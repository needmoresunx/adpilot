import unittest

from adpilot.critic.critique import CritiqueReport
from adpilot.creative import make_repair_prompt


def report(score: int, passed: bool, instruction: str | None = None) -> CritiqueReport:
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
        failure_reasons=[] if passed else ["silhouette_mismatch"],
        identity_score=score,
        repair_instruction=instruction,
    )


class RepairLoopTests(unittest.TestCase):
    def test_repair_prompt_keeps_the_critic_instruction(self):
        prompt = make_repair_prompt(
            "Keep the exact supplied product shape with premium lighting.",
            report(40, False, "restore the original cap silhouette"),
            20,
        )

        self.assertIn("Repair target: restore the original cap silhouette.", prompt)
        self.assertLessEqual(len(prompt.split()), 20)

if __name__ == "__main__":
    unittest.main()
