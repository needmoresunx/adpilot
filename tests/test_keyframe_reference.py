import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from adpilot.backends.keyframe import FluxKontextKeyframeBackend, fit_reference_to_canvas, place_product_reference


class ReferenceCanvasTests(unittest.TestCase):
    def test_portrait_reference_is_contained_without_vertical_crop(self):
        source = Image.new("RGB", (20, 100), "red")
        for y in range(50, 100):
            for x in range(20):
                source.putpixel((x, y), (0, 0, 255))

        result = fit_reference_to_canvas(source, (160, 90))
        colors = set(result.getdata())

        self.assertIn((255, 0, 0), colors)
        self.assertIn((0, 0, 255), colors)
        self.assertEqual(result.size, (160, 90))

    def test_per_shot_reference_preserves_product_aspect_ratio_and_position(self):
        source = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
        for y in range(2, 18):
            for x in range(5, 35):
                source.putpixel((x, y), (255, 0, 0, 255))

        result = place_product_reference(source, (200, 100), "right", 0.30)
        pixels = result.load()
        red_coordinates = [
            (x, y)
            for y in range(result.height)
            for x in range(result.width)
            if pixels[x, y] == (255, 0, 0)
        ]
        xs = [x for x, _ in red_coordinates]
        ys = [y for _, y in red_coordinates]

        self.assertGreater(min(xs), 110)
        visible_width = max(xs) - min(xs) + 1
        visible_height = max(ys) - min(ys) + 1
        self.assertAlmostEqual(visible_width / visible_height, 30 / 16, delta=0.2)

    def test_flux_backend_passes_requested_landscape_size_to_pipeline(self):
        calls = []

        class TorchStub:
            class Generator:
                def __init__(self, device):
                    self.device = device

                def manual_seed(self, seed):
                    return self

        class PipeStub:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "white")])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = root / "product.png"
            Image.new("RGB", (100, 150), "white").save(product)
            backend = FluxKontextKeyframeBackend(model_id="unused", device="cuda")
            backend._pipe = PipeStub()
            backend._torch = TorchStub()
            paths = backend.render_candidates(product, ["ad still"], root / "frames", (1024, 576), 1)

            self.assertEqual(calls[0]["width"], 1024)
            self.assertEqual(calls[0]["height"], 576)
            with Image.open(paths[0][0]) as generated:
                self.assertEqual(generated.size, (1024, 576))

    def test_flux_backend_uses_a_separate_layout_reference_for_each_shot(self):
        calls = []

        class TorchStub:
            class Generator:
                def __init__(self, device):
                    self.device = device

                def manual_seed(self, seed):
                    return self

        class PipeStub:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "white")])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = root / "product.png"
            image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
            image.paste(Image.new("RGBA", (80, 80), "red"), (10, 10))
            image.save(product)
            backend = FluxKontextKeyframeBackend(model_id="unused", device="cuda")
            backend._pipe = PipeStub()
            backend._torch = TorchStub()
            backend.render_candidates(
                product,
                ["left", "right"],
                root / "frames",
                (200, 100),
                1,
                reference_layouts=[("left", 0.25), ("right", 0.25)],
            )

            left_reference = calls[0]["image"]
            right_reference = calls[1]["image"]
            self.assertNotEqual(left_reference.tobytes(), right_reference.tobytes())
            self.assertTrue((root / "frames" / "shot_01" / "reference_input.png").is_file())
            self.assertTrue((root / "frames" / "shot_02" / "reference_input.png").is_file())

    def test_flux_backend_uses_actual_clip_token_limit(self):
        calls = []

        class TorchStub:
            class Generator:
                def __init__(self, device):
                    self.device = device
                    pass

                def manual_seed(self, _seed):
                    return self

        class TokenizerStub:
            model_max_length = 77

            def __call__(self, prompt, **kwargs):
                self.max_length = kwargs["max_length"]
                return {"input_ids": list(range(min(len(prompt.split()), self.max_length)))}

            def decode(self, token_ids, **_kwargs):
                return "safe " * len(token_ids)

        class PipeStub:
            tokenizer = TokenizerStub()

            def __call__(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "white")])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = root / "product.png"
            Image.new("RGB", (100, 150), "white").save(product)
            backend = FluxKontextKeyframeBackend(model_id="unused", device="cuda")
            backend._pipe = PipeStub()
            backend._torch = TorchStub()
            backend.render_candidates(product, ["word " * 200], root / "frames", (1024, 576), 1)

            self.assertEqual(PipeStub.tokenizer.max_length, 75)
            self.assertEqual(len(calls[0]["prompt"].split()), 75)
            self.assertTrue(backend.clip_prompt_guard_applied)


if __name__ == "__main__":
    unittest.main()
