import unittest
from types import SimpleNamespace

from adpilot.creative import make_keyframe_prompts, make_video_prompts
from adpilot.planner.schema import AdPlan, ShotPlan


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.identity = SimpleNamespace(
            product_brief={
                "category": "cosmetic",
                "description": "pale pink lip gloss in a clear tube",
                "mood": "modern refined beauty campaign",
                "scene_keywords": ["cream vanity", "soft daylight", "botanical decor"],
            }
        )
        self.plan = AdPlan(
            concept="test",
            style="beauty",
            shots=[
                ShotPlan("shot_01", 3, "hero", "vanity", "center", 0.4, "slow dolly-in with changing highlights"),
                ShotPlan("shot_02", 3, "detail", "glass counter and soft daylight", "right", 0.3, "gentle lateral move with shifting reflections"),
                ShotPlan("shot_03", 3, "packshot", "marble tray and warm studio glow", "center", 0.4, "smooth push-in with moving highlights"),
            ],
        )

    def test_keyframe_prompt_is_short_and_uses_the_source_photo_as_reference(self):
        prompt = make_keyframe_prompts(self.identity, self.plan)[0]

        self.assertLessEqual(len(prompt.split()), 60)
        self.assertIn("supplied product part", prompt)
        self.assertIn("pale pink lip gloss", prompt)
        self.assertIn("exact geometry", prompt)
        self.assertIn("Set: vanity", prompt)

    def test_keyframes_follow_the_planner_shot_goals(self):
        prompts = make_keyframe_prompts(self.identity, self.plan)

        self.assertIn("hero", prompts[0])
        self.assertIn("detail", prompts[1])
        self.assertIn("packshot", prompts[2])

    def test_video_prompts_use_basic_ad_camera_moves(self):
        prompts = make_video_prompts(self.identity, self.plan)

        self.assertIn("slow dolly-in", prompts[0])
        self.assertIn("gentle lateral move", prompts[1])
        self.assertIn("smooth push-in", prompts[2])
        self.assertIn("cinematic cosmetic commercial", prompts[0].lower())
        self.assertIn("exact geometry", prompts[0])
        self.assertIn("existing hinges", prompts[0])

if __name__ == "__main__":
    unittest.main()
