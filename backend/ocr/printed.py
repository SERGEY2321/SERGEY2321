"""
OCR for printed (typed) math expressions using EasyOCR.
"""

import re
import easyocr
import numpy as np

# Initialise once — loading the model is expensive
_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


# Symbol normalisation table
_REPLACEMENTS = [
    (r"×", "*"),
    (r"✕", "*"),
    (r"÷", "/"),
    (r"−", "-"),
    (r"–", "-"),
    (r"—", "-"),
    (r"\u00b2", "**2"),   # ²
    (r"\u00b3", "**3"),   # ³
    (r"\u221a", "sqrt"),  # √
    (r"\u03c0", "pi"),    # π
    (r"\u221e", "oo"),    # ∞
    (r"'", ""),
    (r"`", ""),
]


def _normalise(text: str) -> str:
    for pattern, replacement in _REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_expression(image: np.ndarray) -> str:
    """
    Run EasyOCR on a preprocessed image and return the normalised
    math expression string.

    Args:
        image: grayscale/binary numpy array from preprocessing pipeline.

    Returns:
        Normalised expression string, e.g. "2*x + 3 = 7".
    """
    reader = _get_reader()
    results = reader.readtext(image, detail=0, paragraph=True)
    raw = " ".join(results)
    return _normalise(raw)
