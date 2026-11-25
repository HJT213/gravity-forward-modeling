"""Draw the Gravity Lab mark and save standard Windows icon sizes."""

from PIL import Image, ImageDraw


def write_icon(path):
    scale = 4
    image = Image.new("RGBA", (256 * scale, 256 * scale))
    draw = ImageDraw.Draw(image)

    def bounds(*coordinates):
        return tuple(coordinate * scale for coordinate in coordinates)

    draw.rounded_rectangle(bounds(4, 4, 251, 251), radius=48 * scale, fill="#1664DC")
    draw.arc(bounds(51, 51, 205, 205), start=0, end=318, fill="white", width=29 * scale)
    draw.rectangle(bounds(128, 120, 205, 147), fill="white")

    icon = image.resize((256, 256), Image.Resampling.LANCZOS)
    icon.save(path, sizes=[(n, n) for n in (16, 20, 24, 32, 40, 48, 64, 128, 256)], bitmap_format="bmp")
