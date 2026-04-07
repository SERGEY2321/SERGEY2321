"""
Render a math expression string as a clean PNG image (bytes).
Used by the /generate API endpoint.
"""

from __future__ import annotations
import io
import random
from PIL import Image, ImageDraw, ImageFont


# ── Random expression bank ────────────────────────────────────────────────────

_EXPRESSIONS = {
    "Арифметика": [
        "15 + 28", "100 - 47", "12 * 8", "144 / 12",
        "256 / 16", "37 + 85", "200 - 63", "25 * 4",
        "(18 + 7) * 3", "(100 - 25) / 5",
    ],
    "Алгебра": [
        "2*x + 3 = 7", "5*x - 10 = 0", "3*x + 1 = 10",
        "x**2 - 5*x + 6 = 0", "x**2 - 4 = 0",
        "2*x**2 + 3*x - 2 = 0", "x**2 + 6*x + 9 = 0",
        "(x + 3) * (x - 2) = 0",
    ],
    "Вычисления": [
        "integrate(x**2, x)", "integrate(2*x + 1, x)",
        "integrate(x**3, x)", "diff(x**3 + 2*x, x)",
        "diff(x**4 - 3*x**2, x)", "diff(5*x**2 + 3*x - 7, x)",
    ],
    "Упрощение": [
        "(x + 1)**2", "(x - 3)**2",
        "(x + 2) * (x - 2)", "x**2 + 2*x + 1",
        "(2*x + 1)**2",
    ],
}


def random_expression() -> dict:
    """Return a random expression with its category."""
    category = random.choice(list(_EXPRESSIONS.keys()))
    expr = random.choice(_EXPRESSIONS[category])
    return {"expression": expr, "category": category}


def all_categories() -> dict:
    """Return all categories with example expressions."""
    return {cat: exprs[0] for cat, exprs in _EXPRESSIONS.items()}


# ── Image rendering ───────────────────────────────────────────────────────────

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_expression_image(
    expression: str,
    width: int = 520,
    height: int = 100,
    font_size: int = 36,
    bg_color: tuple = (15, 17, 23),       # dark background
    text_color: tuple = (255, 255, 255),   # white text
    accent_color: tuple = (91, 106, 245),  # accent border
    padding: int = 20,
) -> bytes:
    """
    Render math expression as a styled PNG image.

    Returns PNG bytes.
    """
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Accent border on left
    draw.rectangle([(0, 0), (5, height)], fill=accent_color)

    # Light grid lines for "notebook" feel
    for y in range(0, height, 20):
        draw.line([(6, y), (width, y)], fill=(255, 255, 255, 15), width=1)

    font = _load_font(font_size)
    bbox = draw.textbbox((0, 0), expression, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Auto-shrink font if text too wide
    while text_w > width - padding * 2 - 10 and font_size > 14:
        font_size -= 2
        font = _load_font(font_size)
        bbox = draw.textbbox((0, 0), expression, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    x = max(padding + 10, (width - text_w) // 2)
    y = max(padding // 2, (height - text_h) // 2)

    # Subtle shadow
    draw.text((x + 2, y + 2), expression, fill=(0, 0, 0), font=font)
    # Main text
    draw.text((x, y), expression, fill=text_color, font=font)

    # Corner label
    label_font = _load_font(11)
    draw.text((width - 80, 6), "math solver", fill=accent_color, font=label_font)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
