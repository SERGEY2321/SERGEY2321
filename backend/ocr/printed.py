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


# ── Step 1: Unicode / special character replacements ─────────────────────────
_UNICODE_REPLACEMENTS = [
    (r"×",      "*"),
    (r"✕",      "*"),
    (r"÷",      "/"),
    (r"−",      "-"),
    (r"–",      "-"),
    (r"—",      "-"),
    (r"\u00b2", "**2"),   # ²
    (r"\u00b3", "**3"),   # ³
    (r"\u221a", "sqrt"),  # √
    (r"\u03c0", "pi"),    # π
    (r"\u221e", "oo"),    # ∞
    (r"'",      ""),
    (r"`",      ""),
]

# ── Step 2: Common EasyOCR misreadings for math ───────────────────────────────
# These are typical mistakes EasyOCR makes on monospace/handwritten math text
_OCR_FIXES = [
    # Power operator: EasyOCR reads * as ^
    (r"\^",         "**"),
    # Common function name typos
    (r"\bdilf\b",   "diff"),
    (r"\bdiff\b",   "diff"),
    (r"\bintegrate\b", "integrate"),
    (r"\bsqrt\b",   "sqrt"),
    # Remove characters that are never valid in math expressions
    (r"[{}|\\~`@#$%&_!?\"']", ""),
    # EasyOCR sometimes reads 'x' as capital letters in context
    # Fix isolated capital letters that look like variables
    # Remove stray lowercase letters attached to numbers (OCR noise)
    # e.g. "15w" → "15" only if w/W is at the END after a number
    (r"(\d+)[wWqQmMnNuUvV](?=\s|$|\))", r"\1"),
    # Fix "O" misread as "0" is handled contextually below
    # Incomplete expressions — trailing operators
    (r"[\+\-\*\/\^]\s*$", ""),
    # Dangling open bracket with nothing after
    (r"\(\s*$", ""),
]

# ── Step 3: Structural cleanup ────────────────────────────────────────────────

def _structural_cleanup(text: str) -> str:
    """Fix structural issues after character replacements."""
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Balance parentheses — add missing closing brackets
    open_count  = text.count("(")
    close_count = text.count(")")
    if open_count > close_count:
        text += ")" * (open_count - close_count)

    # Remove trailing operator or comma
    text = re.sub(r"[\+\-\*\/,]\s*$", "", text).strip()

    return text


def _normalise(text: str) -> str:
    # Step 1: unicode
    for pattern, replacement in _UNICODE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)

    # Step 2: OCR-specific fixes
    for pattern, replacement in _OCR_FIXES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Step 3: structural
    text = _structural_cleanup(text)

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
