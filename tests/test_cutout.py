import unittest

from PIL import Image

from adpilot.identity.builder import make_simple_cutout, trim_transparent_padding


class CutoutTests(unittest.TestCase):
    def test_bright_background_cutout_produces_a_cropped_alpha_subject(self):
        image = Image.new("RGB", (100, 120), "white")
        for y in range(30, 100):
            for x in range(25, 75):
                image.putpixel((x, y), (220, 80, 100))

        cutout = make_simple_cutout(image)
        cropped, offset = trim_transparent_padding(cutout)

        self.assertLess(cropped.width, image.width)
        self.assertLess(cropped.height, image.height)
        self.assertGreater(offset[0], 0)
        self.assertGreater(offset[1], 0)


if __name__ == "__main__":
    unittest.main()
