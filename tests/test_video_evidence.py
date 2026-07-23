import tempfile
import unittest
from pathlib import Path

from adpilot.critic.vlm import sample_wan_frames


class VideoEvidenceTests(unittest.TestCase):
    def test_samples_first_middle_and_last_frame_for_each_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shot in (1, 2):
                for frame in range(5):
                    (root / f"shot_{shot:02d}_frame_{frame:03d}.png").touch()

            groups = sample_wan_frames(root, shot_count=2)

            self.assertEqual([path.name for path in groups[0]], [
                "shot_01_frame_000.png",
                "shot_01_frame_002.png",
                "shot_01_frame_004.png",
            ])
            self.assertEqual([path.name for path in groups[1]], [
                "shot_02_frame_000.png",
                "shot_02_frame_002.png",
                "shot_02_frame_004.png",
            ])


if __name__ == "__main__":
    unittest.main()
