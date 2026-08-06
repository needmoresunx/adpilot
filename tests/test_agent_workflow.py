import unittest
from pathlib import Path
from unittest.mock import patch

from adpilot.agent.store import ProjectConfig
from adpilot.agent.cli import _decode_terminal_bytes
from adpilot.agent.workflow import _feedback_prompt, initial_state
from adpilot.identity.vlm import diffusion_safe_feedback


class AgentWorkflowTests(unittest.TestCase):
    def test_initial_state_is_serializable_and_respects_config_mode(self):
        config = ProjectConfig(
            project_id="project_test",
            brand="Example",
            prompt="warm beauty campaign",
            mode="guided",
            product_asset_id="asset_product",
            reference_asset_ids=["asset_product"],
        )

        state = initial_state(config)

        self.assertEqual(state["mode"], "guided")
        self.assertEqual(state["revision"], 1)
        self.assertEqual(state["agent_steps_used"], 0)
        self.assertEqual(state["generation_feedback_by_shot"], {})

    def test_feedback_is_scoped_and_prompt_limited(self):
        prompt = _feedback_prompt(
            "High-end cosmetic advertising still with a single product and warm reflections.",
            ["make the background softer pink", "keep glass reflections"],
            20,
        )

        self.assertIn("User creative direction", prompt)
        self.assertIn("softer pink", prompt)
        self.assertLessEqual(len(prompt.split()), 20)

    def test_non_ascii_feedback_is_translated_before_generation(self):
        with patch("adpilot.identity.vlm.QwenVisionSession") as session:
            session.return_value.__enter__.return_value.ask_text.return_value = (
                "Show one pair of earbuds and preserve the original structure."
            )
            result = diffusion_safe_feedback("画面内只保留一副耳机，并且注意保留结构特征", "local-qwen")

        self.assertEqual(result, "Show one pair of earbuds and preserve the original structure.")

    def test_ascii_feedback_skips_translation(self):
        self.assertEqual(diffusion_safe_feedback("keep the lid geometry", "unused"), "keep the lid geometry")

    def test_terminal_feedback_decodes_chinese_and_korean_legacy_encodings(self):
        chinese = "画面只保留一副耳机"
        korean = "원래 형태를 유지해"

        self.assertEqual(_decode_terminal_bytes(chinese.encode("gb18030")), chinese)
        self.assertEqual(_decode_terminal_bytes(korean.encode("cp949")), korean)


if __name__ == "__main__":
    unittest.main()
