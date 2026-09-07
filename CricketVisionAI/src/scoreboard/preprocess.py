"""
Scoreboard image preprocessing for EasyOCR.

Steps (applied in order):
  1. Upscale small crops so glyphs are large enough for OCR
  2. Convert BGR → grayscale (drop colour noise)
  3. Contrast boost (CLAHE) — scoreboard LEDs / overlays are often low-contrast
  4. Gentle denoise (bilateral / median)
  5. Adaptive threshold OR keep grayscale (OCR often prefers clean gray)
  6. Optional morphological open to drop speckles
  7. Convert back to 3-channel RGB for EasyOCR
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def preprocess_scoreboard(
    crop_bgr: np.ndarray,
    min_width: int = 400,
    use_threshold: bool = False,
    clahe_clip: float = 2.5,
) -> np.ndarray:
    """
    Return an RGB uint8 image ready for EasyOCR.

    Parameters
    ----------
    crop_bgr : raw scoreboard crop from the detector (OpenCV BGR).
    min_width : upscale if crop is narrower than this (helps small HUD boxes).
    use_threshold : if True, apply adaptive threshold (binary look).
                    Default False — grayscale+CLAHE usually OCRs broadcast
                    graphics better than hard binarization.
    clahe_clip : CLAHE clip limit for local contrast.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Empty scoreboard crop")

    img = crop_bgr.copy()

    # 1) Upscale thin overlays
    h, w = img.shape[:2]
    if w < min_width:
        scale = min_width / float(w)
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    # 2) Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 3) CLAHE contrast
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4) Denoise while keeping edges
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # 5) Optional binarization (useful for high-contrast white-on-black boards)
    if use_threshold:
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8,
        )
        # 6) Speckle cleanup
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # 7) EasyOCR expects RGB / 3-channel
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb


def preprocess_variants(crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build two candidates; caller can OCR both and merge text.
    Returns (soft_gray_rgb, binary_rgb).
    """
    soft = preprocess_scoreboard(crop_bgr, use_threshold=False)
    hard = preprocess_scoreboard(crop_bgr, use_threshold=True)
    return soft, hard
