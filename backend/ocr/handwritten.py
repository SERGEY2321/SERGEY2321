"""
OCR for handwritten math expressions.

Phase 1: Falls back to EasyOCR (printed pipeline) until the custom CNN
         model is trained (Phase 2).

Phase 2: Loads backend/models/symbol_clf.pth, segments the image into
         individual symbols, classifies each one, and reconstructs the
         expression string.
"""

from __future__ import annotations

import os
import re
import numpy as np

# ── Phase 2 imports (only used when the trained model exists) ──────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "symbol_clf.pth")

# Symbol classes used during training (must match training/train_ocr.py)
SYMBOL_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "*", "/", "=", "(", ")",
    "x", "y", "z", "n",
    "sqrt", "pi", "oo",
    "int", "diff",
]


def _load_model():
    """Lazy-load the PyTorch classifier. Returns None if not trained yet."""
    if not os.path.exists(_MODEL_PATH):
        return None
    try:
        import torch
        from torchvision import models, transforms

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(SYMBOL_CLASSES))
        model.load_state_dict(torch.load(_MODEL_PATH, map_location="cpu"))
        model.eval()
        return model
    except Exception:
        return None


_cnn_model = None
_cnn_transform = None


def _get_cnn():
    global _cnn_model, _cnn_transform
    if _cnn_model is None:
        _cnn_model = _load_model()
        if _cnn_model is not None:
            from torchvision import transforms
            _cnn_transform = transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
    return _cnn_model, _cnn_transform


def _segment_symbols(image: np.ndarray) -> list[np.ndarray]:
    """
    Split the image into individual symbol crops using connected components.
    Returns a list of cropped symbol images sorted left-to-right.
    """
    import cv2
    inv = cv2.bitwise_not(image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    crops = []
    for i in range(1, num_labels):  # skip background (0)
        x, y, w, h, area = stats[i]
        if area < 20:  # ignore tiny noise
            continue
        crop = image[y: y + h, x: x + w]
        crops.append((x, crop))
    crops.sort(key=lambda t: t[0])  # sort by x position
    return [c for _, c in crops]


def _classify_crop(crop: np.ndarray, model, transform) -> str:
    """Run the CNN on a single symbol crop and return the class label."""
    import torch
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(crop)
    tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        idx = logits.argmax(dim=1).item()
    return SYMBOL_CLASSES[idx]


def extract_expression(image: np.ndarray) -> str:
    """
    Extract a math expression from a handwritten image.

    If the trained CNN model exists, uses segmentation + classification.
    Otherwise falls back to EasyOCR.
    """
    model, transform = _get_cnn()

    if model is not None:
        crops = _segment_symbols(image)
        if crops:
            tokens = [_classify_crop(c, model, transform) for c in crops]
            return " ".join(tokens)

    # Fallback to printed OCR pipeline
    from .printed import extract_expression as printed_ocr
    return printed_ocr(image)
