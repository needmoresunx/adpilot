import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app import resolve_reference_mode


class ReferenceModeTests(unittest.TestCase):
    def test_auto_selects_front_lock_for_one_photo_and_multi_view_for_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "front.png"
            side = root / "side.png"
            Image.new("RGB", (20, 20), "white").save(primary)
            Image.new("RGB", (20, 20), "white").save(side)

            mode, references = resolve_reference_mode(primary, [], "auto")
            self.assertEqual(mode, "front_lock")
            self.assertEqual(references, [primary])

            mode, references = resolve_reference_mode(primary, [str(side)], "auto")
            self.assertEqual(mode, "multi_view")
            self.assertEqual(references, [primary, side])

    def test_front_lock_rejects_extra_views_and_multi_view_rejects_one_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "front.png"
            side = root / "side.png"
            Image.new("RGB", (20, 20), "white").save(primary)
            Image.new("RGB", (20, 20), "white").save(side)

            with self.assertRaises(ValueError):
                resolve_reference_mode(primary, [str(side)], "front_lock")
            with self.assertRaises(ValueError):
                resolve_reference_mode(primary, [], "multi_view")

if __name__ == "__main__":
    unittest.main()
