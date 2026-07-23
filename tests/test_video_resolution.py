import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from adpilot.backends.video import WanImageToVideoBackend


class VideoResolutionTests(unittest.TestCase):
    def test_configured_landscape_resolution_is_preserved_when_model_aligned(self):
        backend = WanImageToVideoBackend(model_id="unused", generated_size=(832, 480))
        backend._np = np
        backend._pipe = SimpleNamespace(
            vae_scale_factor_spatial=8,
            transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
        )

        self.assertEqual(backend._model_aligned_size(Image.new("RGB", (1024, 576))), (832, 480))

    def test_unaligned_custom_resolution_rounds_to_wan_grid(self):
        backend = WanImageToVideoBackend(model_id="unused", generated_size=(831, 479))
        backend._np = np
        backend._pipe = SimpleNamespace(
            vae_scale_factor_spatial=8,
            transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
        )

        self.assertEqual(backend._model_aligned_size(Image.new("RGB", (1024, 576))), (816, 464))


if __name__ == "__main__":
    unittest.main()
