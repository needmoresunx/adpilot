from __future__ import annotations

import math

import numpy as np
from PIL import Image


def visible_pixels(image: Image.Image) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    pixels = rgba[alpha > 16][:, :3]
    if pixels.size == 0:
        return rgba[:, :, :3].reshape(-1, 3)
    return pixels


def dominant_rgb(image: Image.Image) -> tuple[int, int, int]:
    rgb = np.median(visible_pixels(image), axis=0)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def normalized_rgb_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return round(distance / math.sqrt(3 * 255**2), 4)

