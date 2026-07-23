import unittest
from types import SimpleNamespace

from adpilot.critic.vlm import _ask_critic_json, _critique_one_video, _report_from_result, reference_identity_constraint
from adpilot.planner.schema import ShotPlan


class VlmCriticTests(unittest.TestCase):
    def setUp(self):
        self.shot = ShotPlan("shot_01", 3, "hook", "scene", "center", 0.4)

    def test_pass_verdict_overrides_malformed_low_numeric_score(self):
        report = _report_from_result(
            {
                "identity_verdict": "pass",
                "identity_score": 1,
                "product_visible": True,
                "product_count": 1,
                "label_readability": "readable",
                "failure_reasons": ["the product looks slightly different"],
            },
            self.shot,
            minimum_identity_score=75,
            raw_response="{...}",
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.identity_verdict, "pass")
        self.assertEqual(report.failure_reasons, [])
        self.assertEqual(report.raw_response, "{...}")

    def test_explicit_failure_remains_a_failure(self):
        report = _report_from_result(
            {
                "identity_verdict": "fail",
                "identity_score": 95,
                "product_visible": True,
                "product_count": 1,
                "label_readability": "readable",
                "failure_reasons": ["component_mismatch"],
            },
            self.shot,
            minimum_identity_score=75,
        )

        self.assertFalse(report.passed)
        self.assertIn("component_mismatch", report.failure_reasons)
        self.assertIn("critic_identity_verdict_fail", report.failure_reasons)

    def test_unreadable_branding_is_optional_by_default(self):
        result = {
            "identity_verdict": "pass",
            "identity_score": 85,
            "product_visible": True,
            "product_count": 1,
            "label_readability": "unreadable",
        }

        self.assertTrue(_report_from_result(result, self.shot, 75).passed)
        self.assertFalse(
            _report_from_result(result, self.shot, 75, require_readable_branding=True).passed
        )

    def test_reference_constraint_names_distinctive_product_shape_and_component(self):
        identity = SimpleNamespace(
            aspect_ratio=0.97,
            product_brief={"category": "fragrance", "visible_traits": ["pink cap", "silver bow"]},
        )

        constraint = reference_identity_constraint(identity)

        self.assertIn("wide square bottle", constraint)
        self.assertIn("silver bow", constraint)
        self.assertIn("Mark fail", constraint)

    def test_reference_constraint_prefers_user_identity_anchors(self):
        identity = SimpleNamespace(
            aspect_ratio=0.97,
            product_brief={
                "category": "fragrance",
                "identity_anchors": "exposed silver atomizer, no colored cap",
                "visible_traits": ["pink cap"],
            },
        )

        constraint = reference_identity_constraint(identity)

        self.assertIn("exposed silver atomizer", constraint)
        self.assertNotIn("pink cap", constraint)

    def test_snack_constraint_rejects_an_open_or_spilling_package(self):
        identity = SimpleNamespace(
            aspect_ratio=0.74,
            product_brief={"category": "snack", "identity_anchors": "sealed candy bag"},
        )

        constraint = reference_identity_constraint(identity)

        self.assertIn("sealed unopened upright pouch", constraint)
        self.assertIn("candy visibly exits", constraint)

    def test_open_snack_constraint_allows_a_few_contents_but_rejects_bad_physics(self):
        identity = SimpleNamespace(
            aspect_ratio=0.74,
            product_brief={"category": "snack", "package_state": "open_with_contents", "identity_anchors": "gold candy pouch"},
        )

        constraint = reference_identity_constraint(identity)

        self.assertIn("naturally open at its top seam", constraint)
        self.assertIn("few visible gummy bears", constraint)
        self.assertIn("floating candy", constraint)

    def test_final_snack_constraint_allows_visible_but_static_gummies(self):
        identity = SimpleNamespace(
            aspect_ratio=0.74,
            product_brief={"category": "snack", "package_state": "open_with_contents", "identity_anchors": "gold candy pouch"},
        )
        shot = ShotPlan("shot_03", 3, "packshot", "scene", "center", 0.46)

        constraint = reference_identity_constraint(
            identity,
            shot,
            "Gummies inside the transparent pouch remain completely motionless: no jiggling, sliding, bouncing, or tumbling.",
        )

        self.assertIn("transparent pouch remain completely motionless", constraint)
        self.assertIn("jiggling, sliding, bouncing", constraint)

    def test_critic_retries_once_after_a_truncated_json_response(self):
        class SessionStub:
            def __init__(self):
                self.calls = []

            def ask(self, _paths, prompt, max_new_tokens):
                self.calls.append((prompt, max_new_tokens))
                if len(self.calls) == 1:
                    return '{"identity_verdict": "pass"'
                return '{"identity_verdict": "pass", "identity_score": 85}'

        session = SessionStub()
        raw, result = _ask_critic_json(session, [], "Return JSON")

        self.assertEqual(result["identity_verdict"], "pass")
        self.assertEqual(raw, '{"identity_verdict": "pass", "identity_score": 85}')
        self.assertEqual([call[1] for call in session.calls], [160, 320])

    def test_final_video_requires_an_explicit_constraint_pass(self):
        class SessionStub:
            def ask(self, _paths, _prompt, max_new_tokens):
                return '{"identity_verdict":"pass","identity_score":90,"product_visible":true,"product_count":1,"label_readability":"readable","temporal_consistency":"stable","constraint_verdict":"fail"}'

        identity = SimpleNamespace(
            product_path="examples/jelly.jpg",
            aspect_ratio=0.74,
            product_brief={"category": "snack", "identity_anchors": "gold candy pouch"},
        )
        shot = ShotPlan("shot_03", 3, "packshot", "scene", "center", 0.46)
        report = _critique_one_video(
            SessionStub(),
            identity,
            shot,
            [],
            minimum_identity_score=75,
            final_shot_constraint="no texture smear",
        )

        self.assertFalse(report.passed)
        self.assertIn("final_constraint_violation", report.failure_reasons)


if __name__ == "__main__":
    unittest.main()
