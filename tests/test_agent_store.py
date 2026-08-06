import tempfile
import unittest
from pathlib import Path

from PIL import Image

from adpilot.agent.store import ProjectStore


class ProjectStoreTests(unittest.TestCase):
    def test_project_keeps_assets_relative_and_records_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "product image.png"
            Image.new("RGB", (16, 20), "white").save(source)

            store = ProjectStore.create(
                root / "projects",
                source,
                brand="Example Brand",
                prompt="warm editorial product campaign",
                mode="guided",
            )

            config = store.config
            self.assertEqual(config.mode, "guided")
            self.assertRegex(config.project_id, r"^\d{8}_product-image$")
            self.assertIsNone(config.product_description)
            self.assertEqual(config.keyframe_seed, 17)
            copied = store.asset_path(config.product_asset_id)
            self.assertTrue(copied.is_file())
            self.assertTrue(copied.is_relative_to(store.project_dir))
            self.assertEqual(store.events()[-1]["event_type"], "project_created")

    def test_generated_asset_inside_project_is_registered_without_copying(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "product.png"
            Image.new("RGB", (16, 16), "white").save(source)
            store = ProjectStore.create(root / "projects", source, "Example", "clean product ad", "auto")
            generated = store.revision_dir(1) / "storyboard.png"
            Image.new("RGB", (20, 20), "black").save(generated)

            asset_id = store.register_generated_asset(generated, "storyboard", 1)

            self.assertEqual(store.asset_path(asset_id), generated)
            self.assertEqual(len(list((store.project_dir / "assets").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
