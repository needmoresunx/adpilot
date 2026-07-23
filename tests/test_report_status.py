import tempfile
import unittest
from pathlib import Path

from PIL import Image

from adpilot.critic.critique import CritiqueReport
from adpilot.identity.card import IdentityCard
from adpilot.report.html import write_html_report


class ReportStatusTests(unittest.TestCase):
    def test_report_marks_failed_identity_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            image_path = run_dir / "frame.jpg"
            Image.new("RGB", (10, 10), "white").save(image_path)
            storyboard = run_dir / "storyboard.png"
            Image.new("RGB", (10, 10), "white").save(storyboard)
            card = IdentityCard("brand", str(image_path), str(image_path), 10, 10, 1.0, (255, 255, 255), {})
            report = CritiqueReport(
                shot_id="shot_01",
                passed=False,
                product_scale=0.4,
                color_delta=0.0,
                shape_score=0.5,
                logo_area_ratio=None,
                product_bbox=None,
                logo_bbox_in_frame=None,
                ocr_text=None,
                ocr_available=False,
                failure_reasons=["low_identity_score"],
            )

            output = write_html_report(run_dir, card, [image_path], [report], storyboard, None)

            self.assertIn("IDENTITY CHECK FAILED", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
