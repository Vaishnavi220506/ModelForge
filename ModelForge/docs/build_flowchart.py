from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review_1_proposal_design" / "model_forge_flowchart.png"


WIDTH, HEIGHT = 2000, 960
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (246, 246, 246)


def font(size):
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def centered_text(draw, box, text, text_font):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = left + (right - left - text_width) / 2
    y = top + (bottom - top - text_height) / 2 - bounds[1]
    draw.text((x, y), text, fill=BLACK, font=text_font)


def box(draw, left, top, right, bottom, text, text_font):
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=18,
        fill=LIGHT_GRAY,
        outline=BLACK,
        width=3,
    )
    centered_text(draw, (left, top, right, bottom), text, text_font)


def arrow(draw, start, end, width=5):
    draw.line((start[0], start[1], end[0], end[1]), fill=BLACK, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) >= abs(dy):
        direction = 1 if dx >= 0 else -1
        tip = (end[0], end[1])
        points = [
            tip,
            (end[0] - direction * 20, end[1] - 12),
            (end[0] - direction * 20, end[1] + 12),
        ]
    else:
        direction = 1 if dy >= 0 else -1
        tip = (end[0], end[1])
        points = [
            tip,
            (end[0] - 12, end[1] - direction * 20),
            (end[0] + 12, end[1] - direction * 20),
        ]
    draw.polygon(points, fill=BLACK)


def build():
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    node_font = font(28)

    box(draw, 70, 25, 510, 105, "ONNX model", node_font)
    box(draw, 70, 175, 510, 255, "ONNX loader", node_font)
    box(draw, 660, 175, 1140, 255, "semantic validator", node_font)
    box(draw, 1290, 175, 1930, 255, "ModelForge IR", node_font)

    arrow(draw, (290, 105), (290, 163))
    arrow(draw, (510, 215), (648, 215))
    arrow(draw, (1140, 215), (1278, 215))

    box(draw, 1120, 325, 1880, 405, "optimization passes", node_font)
    box(draw, 1120, 465, 1880, 545, "C++ code generator", node_font)
    box(draw, 1020, 605, 1980, 685, "generated source and header", node_font)
    arrow(draw, (1610, 255), (1610, 313))
    arrow(draw, (1500, 405), (1500, 453))
    arrow(draw, (1500, 545), (1500, 593))

    box(draw, 1030, 760, 1300, 840, "native build", node_font)
    box(draw, 1375, 760, 1625, 840, "inference", node_font)
    box(draw, 1700, 760, 1970, 840, "comparison", node_font)
    arrow(draw, (1500, 685), (1500, 748))
    arrow(draw, (1300, 800), (1363, 800))
    arrow(draw, (1625, 800), (1688, 800))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, format="PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    build()
