"""
Synthetic dataset generator for HANDWRITTEN math symbols.

Generates individual symbol crops (28×28 grayscale), applies heavy
augmentation to simulate real handwriting, then splits 80/20.

Symbols covered:
    0-9, +, -, *, /, =, (, ), x, y, z, n, sqrt, pi, oo, int, diff

Usage:
    python -m dataset.generator.generate_handwritten --count 10000 --out dataset/symbols
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from backend.ocr.handwritten import SYMBOL_CLASSES

# ── Symbol → Unicode / text representation ────────────────────────────────────
_SYMBOL_TEXT = {
    "sqrt": "√", "pi": "π", "oo": "∞",
    "int": "∫", "diff": "d/dx",
    "*": "×", "/": "÷",
}


def _symbol_text(sym: str) -> str:
    return _SYMBOL_TEXT.get(sym, sym)


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render_symbol(
    symbol: str,
    size: int = 28,
    font_size: int = 20,
) -> Image.Image:
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    text = _symbol_text(symbol)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuMono.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = max(0, (size - tw) // 2)
    y = max(0, (size - th) // 2)
    draw.text((x, y), text, fill=0, font=font)
    return img


def _augment_symbol(img: Image.Image) -> Image.Image:
    """
    Heavy augmentation to simulate real handwritten symbols:
    - Random rotation ±15°
    - Random scale (crop/pad)
    - Gaussian noise
    - Blur
    - Random pen stroke width (erosion / dilation)
    """
    import cv2

    arr = np.array(img)

    # Rotation
    angle = random.uniform(-15, 15)
    h, w = arr.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    arr = cv2.warpAffine(arr, M, (w, h), borderValue=255)

    # Scale jitter: zoom in/out
    scale = random.uniform(0.75, 1.25)
    new_w = max(4, int(w * scale))
    new_h = max(4, int(h * scale))
    arr_resized = cv2.resize(arr, (new_w, new_h))
    # Pad or crop back to original size
    canvas = np.full((h, w), 255, dtype=np.uint8)
    ph = min(new_h, h)
    pw = min(new_w, w)
    oh = (h - ph) // 2
    ow = (w - pw) // 2
    sh = (new_h - ph) // 2
    sw = (new_w - pw) // 2
    canvas[oh:oh+ph, ow:ow+pw] = arr_resized[sh:sh+ph, sw:sw+pw]
    arr = canvas

    # Noise
    noise = np.random.normal(0, random.uniform(10, 30), arr.shape)
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Morphological: simulate thin vs thick pen
    kernel_size = random.choice([1, 2, 3])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if random.random() < 0.5:
        arr = cv2.erode(arr, kernel, iterations=1)   # thicker pen
    else:
        arr = cv2.dilate(arr, kernel, iterations=1)  # thinner pen

    # Blur
    if random.random() < 0.4:
        ksize = random.choice([3, 5])
        arr = cv2.GaussianBlur(arr, (ksize, ksize), 0)

    return Image.fromarray(arr)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_dataset(
    count: int = 10000,
    out_dir: str = "dataset/symbols",
    seed: int = 42,
    augment_per_symbol: int = 10,
) -> None:
    """
    Generate `count` symbol images, split 80/20.

    Each symbol class gets an equal share of the total count.
    Each base render is augmented `augment_per_symbol` times.
    """
    random.seed(seed)
    np.random.seed(seed)

    from sklearn.model_selection import train_test_split

    train_dir = Path(out_dir) / "train"
    val_dir   = Path(out_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    records = []
    idx = 0
    per_class = max(1, count // len(SYMBOL_CLASSES))

    for class_idx, symbol in enumerate(SYMBOL_CLASSES):
        base_img = _render_symbol(symbol)
        for _ in range(per_class):
            aug = _augment_symbol(base_img)
            records.append({"idx": idx, "symbol": symbol, "class_idx": class_idx, "img": aug})
            idx += 1

    indices = list(range(len(records)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=seed)

    metadata = {"symbols": SYMBOL_CLASSES, "train": [], "val": []}

    for i in train_idx:
        r = records[i]
        fname = f"sym_{r['idx']:07d}.png"
        r["img"].save(train_dir / fname)
        metadata["train"].append({
            "file": f"train/{fname}",
            "symbol": r["symbol"],
            "class_idx": r["class_idx"],
        })

    for i in val_idx:
        r = records[i]
        fname = f"sym_{r['idx']:07d}.png"
        r["img"].save(val_dir / fname)
        metadata["val"].append({
            "file": f"val/{fname}",
            "symbol": r["symbol"],
            "class_idx": r["class_idx"],
        })

    meta_path = Path(out_dir) / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    total = len(records)
    print(f"Generated {total} symbol images  ({len(train_idx)} train / {len(val_idx)} val)")
    print(f"Classes: {len(SYMBOL_CLASSES)}  |  Per-class: ~{per_class}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate handwritten symbol dataset")
    parser.add_argument("--count", type=int, default=10000, help="Approximate total images")
    parser.add_argument("--out",   type=str, default="dataset/symbols")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()
    generate_dataset(count=args.count, out_dir=args.out, seed=args.seed)
