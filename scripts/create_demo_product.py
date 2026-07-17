from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    out = Path("examples/demo_bottle.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (300, 620), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((80, 70, 220, 570), radius=48, fill=(20, 22, 22, 255))
    draw.rounded_rectangle((105, 20, 195, 105), radius=22, fill=(32, 35, 34, 255))
    draw.rectangle((90, 80, 210, 125), fill=(245, 245, 238, 255))
    draw.text((118, 94), "AQUA", fill=(20, 22, 22, 255))
    draw.ellipse((115, 505, 185, 545), fill=(36, 39, 38, 255))
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()

