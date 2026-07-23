import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adpilot.planner.vlm import apply_category_storyboard, build_vlm_plan, make_vlm_plan, validate_creative_plan


class VlmPlannerTests(unittest.TestCase):
    def test_builds_valid_three_shot_plan(self):
        plan = build_vlm_plan(
            {
                "concept": "A floral evening ritual",
                "style": "refined soft-focus beauty campaign",
                "shots": [
                    {"goal": "hook", "scene_prompt": "blush silk peony petals warm side light mirrored tray", "product_position": "center", "product_scale": 0.4, "motion_prompt": "slow reveal with drifting petals"},
                    {"goal": "detail", "scene_prompt": "pink marble vanity glass reflections soft evening window light", "product_position": "right", "product_scale": 0.3, "motion_prompt": "gentle pan with shifting reflections"},
                    {"goal": "packshot", "scene_prompt": "pale flowers mirrored tray soft haze warm studio glow", "product_position": "center", "product_scale": 0.42, "motion_prompt": "final push in with moving highlights"},
                ],
            },
            duration=9,
        )

        self.assertEqual(plan.concept, "A floral evening ritual")
        self.assertEqual([shot.shot_id for shot in plan.shots], ["shot_01", "shot_02", "shot_03"])
        self.assertEqual(plan.shots[1].motion_prompt, "gentle pan with shifting reflections")

    def test_rejects_invalid_position(self):
        response = {
            "concept": "x",
            "style": "y",
            "shots": [{"goal": "x", "scene_prompt": "x", "product_position": "top", "product_scale": 0.3, "motion_prompt": "x"}] * 3,
        }
        with self.assertRaises(ValueError):
            build_vlm_plan(response, duration=9)

    def test_keeps_scene_text_for_the_keyframe_prompt(self):
        response = {
            "concept": "x",
            "style": "y",
            "shots": [
                {"goal": "hook", "scene_prompt": "close-up perfume bottle on a soft romantic background", "product_position": "center", "product_scale": 0.4, "motion_prompt": "slow dolly with drifting petals"},
                {"goal": "detail", "scene_prompt": "blush silk with peony petals and warm side light", "product_position": "left", "product_scale": 0.3, "motion_prompt": "gentle pan with shifting reflections"},
                {"goal": "packshot", "scene_prompt": "mirrored tray with pale flowers and evening window glow", "product_position": "center", "product_scale": 0.42, "motion_prompt": "push in with moving highlights"},
            ],
        }
        plan = build_vlm_plan(response, duration=9)
        self.assertIn("perfume bottle", plan.shots[0].background_prompt)

    def test_rejects_snack_package_rotation(self):
        identity = SimpleNamespace(product_brief={"category": "snack", "package_state": "open_with_contents"})
        plan = build_vlm_plan(
            {
                "concept": "x",
                "style": "y",
                "shots": [
                    {"goal": "hook", "scene_prompt": "gold tabletop bright studio light candy props", "product_position": "center", "product_scale": 0.42, "motion_prompt": "slow dolly with glints"},
                    {"goal": "detail gummies tumble naturally", "scene_prompt": "gold tabletop bright studio light candy props", "product_position": "left", "product_scale": 0.32, "motion_prompt": "low pan as gummies land"},
                    {"goal": "packshot", "scene_prompt": "gold tabletop bright studio light candy props", "product_position": "center", "product_scale": 0.46, "motion_prompt": "rotating package"},
                ],
            },
            duration=9,
        )

        with self.assertRaises(ValueError):
            validate_creative_plan(identity, plan)

    def test_snack_storyboard_separates_opening_and_final_without_changing_middle(self):
        identity = SimpleNamespace(product_brief={"category": "snack"})
        plan = build_vlm_plan(
            {
                "concept": "x",
                "style": "y",
                "shots": [
                    {"goal": "hook", "scene_prompt": "gold tabletop bright studio light candy props", "product_position": "center", "product_scale": 0.42, "motion_prompt": "zoom in"},
                    {"goal": "middle gummy action", "scene_prompt": "gold tabletop bright studio light candy props", "product_position": "center", "product_scale": 0.32, "motion_prompt": "gummies tumble and land"},
                    {"goal": "packshot", "scene_prompt": "gold tabletop bright studio light candy props", "product_position": "center", "product_scale": 0.46, "motion_prompt": "zoom out"},
                ],
            },
            duration=9,
        )

        storyboard = apply_category_storyboard(identity, plan)

        self.assertIn("opening hero reveal", storyboard.shots[0].goal)
        self.assertEqual(storyboard.shots[0].product_position, "left")
        self.assertEqual(storyboard.shots[1], plan.shots[1])
        self.assertIn("signature final packshot", storyboard.shots[2].goal)
        self.assertIn("reflections", storyboard.shots[2].motion_prompt)
        self.assertNotIn("zoom", storyboard.shots[2].motion_prompt)

    def test_retries_once_when_the_first_vlm_response_is_invalid_json(self):
        class Session:
            def __init__(self):
                self.responses = [
                    '{"concept": "broken" "style": "broken"}',
                    (
                        '{"concept":"floral ritual","style":"soft luxury",'
                        '"shots":[{"goal":"hook","scene_prompt":"blush silk peony petals warm side light mirrored tray","product_position":"center",'
                        '"product_scale":0.42,"motion_prompt":"dolly in with drifting petals"},'
                        '{"goal":"detail","scene_prompt":"pink marble vanity glass reflections soft evening window light","product_position":"left",'
                        '"product_scale":0.32,"motion_prompt":"pan with shifting reflections"},'
                        '{"goal":"packshot","scene_prompt":"pale flowers mirrored tray soft haze warm studio glow","product_position":"center",'
                        '"product_scale":0.46,"motion_prompt":"push in with moving highlights"}]}'
                    ),
                ]
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def ask(self, *_args, **_kwargs):
                response = self.responses[self.calls]
                self.calls += 1
                return response

        session = Session()
        identity = SimpleNamespace(brand="Miss Dior", product_brief={"category": "fragrance"})
        with patch("adpilot.planner.vlm.QwenVisionSession", return_value=session):
            plan = make_vlm_plan(identity, Path("product.jpg"), "luxury", 9, "landscape", "unused")

        self.assertEqual(session.calls, 2)
        self.assertEqual(plan.concept, "floral ritual")


if __name__ == "__main__":
    unittest.main()
