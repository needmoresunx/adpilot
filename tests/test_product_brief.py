import unittest
from pathlib import Path

from adpilot.identity.brief import build_product_brief


class ProductBriefTests(unittest.TestCase):
    def test_vision_analysis_overrides_filename_heuristics(self):
        brief = build_product_brief(
            product_path=Path("examples/product.jpg"),
            aspect_ratio=1.0,
            vision_analysis={
                "category": "fragrance",
                "description": "a pink glass perfume bottle",
                "visible_traits": ["silver bow", "square glass bottle"],
                "materials": ["glass", "fabric"],
                "colors": ["pink", "silver"],
                "readable_text": "Miss Dior",
            },
        )

        self.assertEqual(brief.category, "fragrance")
        self.assertEqual(brief.description, "a pink glass perfume bottle")
        self.assertEqual(brief.visible_traits, ["silver bow", "square glass bottle"])
        self.assertEqual(brief.recognition_source, "qwen2_5_vl")

    def test_user_description_is_not_replaced_by_vision_caption(self):
        brief = build_product_brief(
            product_path=Path("examples/perfume.jpg"),
            aspect_ratio=0.66,
            description="a pink perfume bottle with a silver bow",
            visual_caption="a bottle on a white background",
            vision_analysis={"category": "fragrance", "description": "a glass perfume bottle"},
        )

        self.assertEqual(brief.description, "a pink perfume bottle with a silver bow")

    def test_jelly_filename_infers_snack_category(self):
        brief = build_product_brief(product_path=Path("examples/jelly.jpg"), aspect_ratio=0.8)

        self.assertEqual(brief.category, "snack")
        self.assertIn("snack", " ".join(brief.scene_keywords + [brief.description]).lower())

    def test_wireless_earbuds_filename_infers_electronics_category(self):
        brief = build_product_brief(product_path=Path("examples/wireless-earbuds.jpg"), aspect_ratio=1.1)

        self.assertEqual(brief.category, "electronics")

    def test_user_identity_anchors_are_retained_separately_from_vlm_traits(self):
        brief = build_product_brief(
            product_path=Path("examples/perfume.jpg"),
            aspect_ratio=0.97,
            identity_anchors="exposed silver atomizer, no colored cap",
            vision_analysis={"visible_traits": ["pink cap"]},
        )

        self.assertEqual(brief.identity_anchors, "exposed silver atomizer, no colored cap")
        self.assertEqual(brief.visible_traits, ["pink cap"])

    def test_user_package_state_is_retained_separately_from_identity_traits(self):
        brief = build_product_brief(
            product_path=Path("examples/jelly.jpg"),
            aspect_ratio=0.74,
            package_state="open_with_contents",
        )

        self.assertEqual(brief.package_state, "open_with_contents")


if __name__ == "__main__":
    unittest.main()
