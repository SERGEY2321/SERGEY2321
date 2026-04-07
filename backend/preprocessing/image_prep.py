"""
Image preprocessing pipeline.
Cleans up an input photo before passing it to OCR.
"""

import cv2
import numpy as np
from PIL import Image
import io


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Decode image bytes
    2. Convert to grayscale
    3. Denoise
    4. Binarize (Otsu threshold)
    5. Deskew

    Returns a clean binary numpy array (uint8, values 0 or 255).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    deskewed = _deskew(binary)
    return deskewed


def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct rotation by finding the dominant text angle."""
    coords = np.column_stack(np.where(image < 128))
    if len(coords) < 5:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def image_to_bytes(image: np.ndarray) -> bytes:
    """Convert numpy array back to PNG bytes (for debugging/logging)."""
    _, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()
