from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def make_storyboard(frame_paths: list[Path], output_path: Path) -> Path:
    thumbs = [Image.open(path).convert("RGB").resize((270, 480)) for path in frame_paths]
    width = 270 * max(len(thumbs), 1)
    height = 540
    storyboard = Image.new("RGB", (width, height), (245, 245, 242))
    draw = ImageDraw.Draw(storyboard)
    for index, thumb in enumerate(thumbs):
        x = index * 270
        storyboard.paste(thumb, (x, 0))
        draw.text((x + 16, 500), f"Shot {index + 1}", fill=(40, 40, 40))
    storyboard.save(output_path)
    return output_path

