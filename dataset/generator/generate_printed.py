"""
Synthetic dataset generator for PRINTED math expressions.

Renders math expressions as images using matplotlib + PIL,
then splits the result 80% train / 20% val.

Usage:
    python -m dataset.generator.generate_printed --count 5000 --out dataset
"""

from __future__ import annotations

import argparse
import json
import os
import random
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ── Expression templates ──────────────────────────────────────────────────────

def _rand_int(lo: int = 1, hi: int = 20) -> int:
    return random.randint(lo, hi)

def _rand_var() -> str:
    return random.choice(["x", "y", "z", "n", "a", "b"])

def _rand_expr() -> str:
    """Generate a random math expression string."""
    kind = random.choice([
        "arithmetic", "arithmetic", "arithmetic",
        "linear_eq", "linear_eq",
        "quadratic",
        "fraction",
        "power",
        "integral",
        "derivative",
    ])

    v = _rand_var()
    a, b, c = _rand_int(), _rand_int(), _rand_int()

    if kind == "arithmetic":
        ops = ["+", "-", "*", "/"]
        op = random.choice(ops)
        n1, n2 = _rand_int(1, 50), _rand_int(1, 50)
        if op == "/" and n2 == 0:
            n2 = 1
        return f"{n1} {op} {n2}"

    if kind == "linear_eq":
        rhs = a * _rand_int(1, 5) + b
        return f"{a}*{v} + {b} = {rhs}"

    if kind == "quadratic":
        return f"{v}**2 + {a}*{v} + {b} = 0"

    if kind == "fraction":
        return f"({a}*{v} + {b}) / {c}"

    if kind == "power":
        exp = random.choice([2, 3, 4])
        return f"{v}**{exp} + {a}"

    if kind == "integral":
        return f"integrate({v}**{random.choice([1,2,3])}, {v})"

    if kind == "derivative":
        return f"diff({v}**{random.choice([2,3,4])} + {a}*{v}, {v})"

    return f"{a} + {b}"


# ── Image rendering ───────────────────────────────────────────────────────────

def _render_expression(
    expression: str,
    width: int = 256,
    height: int = 64,
    font_size: int = 24,
    noise_level: float = 0.0,
    blur_radius: float = 0.0,
) -> Image.Image:
    """Render expression string as a grayscale PIL image."""
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuMono.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Centre text
    bbox = draw.textbbox((0, 0), expression, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(0, (width - text_w) // 2)
    y = max(0, (height - text_h) // 2)
    draw.text((x, y), expression, fill=0, font=font)

    if noise_level > 0:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, noise_level * 255, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


def _augment(img: Image.Image) -> Image.Image:
    """Random augmentation to simulate scanner / camera noise."""
    # Small rotation
    angle = random.uniform(-3, 3)
    img = img.rotate(angle, fillcolor=255, expand=False)

    # Random noise
    if random.random() < 0.5:
        arr = np.array(img, dtype=np.float32)
        arr += np.random.normal(0, random.uniform(5, 20), arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Slight blur
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    return img


# ── Main generation logic ─────────────────────────────────────────────────────

def generate_dataset(
    count: int = 5000,
    out_dir: str = "dataset",
    seed: int = 42,
) -> None:
    """
    Generate `count` images, split 80/20 into train/val.
    Saves images to out_dir/train/ and out_dir/val/,
    plus a metadata.json with labels.
    """
    random.seed(seed)
    np.random.seed(seed)

    train_dir = Path(out_dir) / "train"
    val_dir   = Path(out_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import train_test_split

    records = []
    for i in range(count):
        expr = _rand_expr()
        img  = _render_expression(expr)
        img  = _augment(img)
        records.append({"idx": i, "expr": expr, "img": img})

    indices = list(range(count))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=seed)

    metadata = {"train": [], "val": []}

    for idx in train_idx:
        r = records[idx]
        fname = f"train_{idx:06d}.png"
        r["img"].save(train_dir / fname)
        metadata["train"].append({"file": f"train/{fname}", "label": r["expr"]})

    for idx in val_idx:
        r = records[idx]
        fname = f"val_{idx:06d}.png"
        r["img"].save(val_dir / fname)
        metadata["val"].append({"file": f"val/{fname}", "label": r["expr"]})

    meta_path = Path(out_dir) / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Generated {count} images  ({len(train_idx)} train / {len(val_idx)} val)")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate printed math dataset")
    parser.add_argument("--count", type=int, default=5000, help="Total number of images")
    parser.add_argument("--out",   type=str, default="dataset", help="Output directory")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()
    generate_dataset(count=args.count, out_dir=args.out, seed=args.seed)
