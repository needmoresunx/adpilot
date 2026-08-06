import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from adpilot.critic.critique import CritiqueReport
from adpilot.critic.evidence import _corner_background_cutout, _visual_identity_metrics, build_keyframe_identity_evidence
from adpilot.identity.card import IdentityCard
from adpilot.planner.schema import ShotPlan


class IdentityEvidenceTests(unittest.TestCase):
    def test_visual_metric_scores_matching_cutouts_higher_than_changed_shape(self):
        reference = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
        ImageDraw.Draw(reference).rounded_rectangle((35, 15, 85, 105), radius=8, fill=(220, 70, 80, 255))
        matching = reference.copy()
        changed = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
        ImageDraw.Draw(changed).rounded_rectangle((15, 45, 105, 75), radius=8, fill=(220, 70, 80, 255))

        matching_metric = _visual_identity_metrics(reference, matching)
        changed_metric = _visual_identity_metrics(reference, changed)

        self.assertEqual(matching_metric["visual_score"], 100)
        self.assertLess(changed_metric["visual_score"], matching_metric["visual_score"])

    def test_corner_background_fallback_returns_a_product_mask(self):
        image = Image.new("RGB", (120, 120), (240, 240, 240))
        ImageDraw.Draw(image).rounded_rectangle((35, 15, 85, 105), radius=8, fill=(220, 70, 80))

        cutout = _corner_background_cutout(image)

        self.assertIsNotNone(cutout)
        self.assertGreater(sum(pixel[3] > 16 for pixel in cutout.getdata()), 100)

    def test_low_birefnet_score_fails_the_single_view_visual_gate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
            ImageDraw.Draw(reference).rounded_rectangle((35, 15, 85, 105), radius=8, fill=(220, 70, 80, 255))
            reference_path = root / "reference.png"
            reference.save(reference_path)
            candidate_path = root / "candidate.png"
            Image.new("RGB", (640, 360), (240, 240, 240)).save(candidate_path)
            changed = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
            ImageDraw.Draw(changed).rounded_rectangle((15, 45, 105, 75), radius=8, fill=(220, 70, 80, 255))
            identity = IdentityCard(
                brand="Example",
                product_path=str(reference_path),
                cutout_path=str(reference_path),
                width=120,
                height=120,
                aspect_ratio=1.0,
                dominant_rgb=(220, 70, 80),
                product_brief={},
            )
            report = CritiqueReport(
                shot_id="shot_01",
                passed=True,
                product_scale=0.4,
                color_delta=0.0,
                shape_score=None,
                logo_area_ratio=None,
                product_bbox=None,
                logo_bbox_in_frame=None,
                ocr_text=None,
                ocr_available=False,
                failure_reasons=[],
            )

            with patch("adpilot.critic.evidence._segmentation_session", return_value=object()), patch(
                "adpilot.critic.evidence._candidate_cutout", return_value=changed
            ):
                build_keyframe_identity_evidence(
                    identity,
                    [ShotPlan("shot_01", 3, "goal", "scene", "center", 0.4)],
                    [[candidate_path]],
                    [[report]],
                    root / "evidence",
                    enforce_visual_gate=True,
                )

            self.assertFalse(report.passed)
            self.assertIn("visual_identity_below_threshold", report.failure_reasons)

    def test_writes_reference_frame_and_product_region_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_path = root / "reference.png"
            candidate_path = root / "candidate.png"
            Image.new("RGBA", (100, 80), (240, 240, 240, 255)).save(reference_path)
            Image.new("RGB", (640, 360), (120, 150, 180)).save(candidate_path)
            identity = IdentityCard(
                brand="Example",
                product_path=str(reference_path),
                cutout_path=str(reference_path),
                width=100,
                height=80,
                aspect_ratio=1.25,
                dominant_rgb=(240, 240, 240),
                product_brief={},
            )
            report = CritiqueReport(
                shot_id="shot_01",
                passed=True,
                product_scale=0.4,
                color_delta=0.0,
                shape_score=None,
                logo_area_ratio=None,
                product_bbox=None,
                logo_bbox_in_frame=None,
                ocr_text=None,
                ocr_available=False,
                failure_reasons=[],
                identity_verdict="pass",
                identity_checks={
                    "silhouette_match": "match",
                    "component_match": "match",
                    "color_match": "minor_drift",
                },
            )

            evidence = build_keyframe_identity_evidence(
                identity,
                [ShotPlan("shot_01", 3, "goal", "scene", "center", 0.4)],
                [[candidate_path]],
                [[report]],
                root / "evidence",
            )

            item = evidence[0][0]
            self.assertTrue(Path(item["product_region"]).is_file())
            self.assertTrue(Path(item["evidence_image"]).is_file())
            self.assertTrue((root / "evidence" / "identity_evidence.json").is_file())
            self.assertEqual(item["identity_checks"]["color_match"], "minor_drift")


if __name__ == "__main__":
    unittest.main()
