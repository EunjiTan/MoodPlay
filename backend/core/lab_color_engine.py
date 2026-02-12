"""
LAB Color Engine — CIELAB colour space utilities for the Instance-Guided
Semantic Video Colorization pipeline.

Provides:
  • RGB ↔ LAB conversion (via OpenCV, D65 illuminant)
  • ΔE*ab CIE-1994 perceptual distance
  • Palette validation (confirm colours are sRGB-gamut representable)
  • Palette correction protocol (Section 6.2 of the specification)
  • Gamut clamping helpers
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from backend.core.pipeline_config import CFG


# ────────────────────────────────────────────────────────────────────────
#  Conversion helpers
# ────────────────────────────────────────────────────────────────────────

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image or single colour (uint8) to CIELAB float32.

    Args:
        rgb: (H, W, 3) uint8 image  *or*  (3,) uint8 colour.

    Returns:
        Same shape, float32 LAB values.  L ∈ [0,100], a,b ∈ [-128,127].
    """
    single = rgb.ndim == 1
    if single:
        rgb = rgb.reshape(1, 1, 3)
    rgb_u8 = rgb.astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV stores L in 0-255, a/b in 0-255 centred at 128.  Normalise.
    lab[..., 0] = lab[..., 0] * (100.0 / 255.0)
    lab[..., 1] = lab[..., 1] - 128.0
    lab[..., 2] = lab[..., 2] - 128.0
    return lab.squeeze() if single else lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB float32 back to RGB uint8.

    Args:
        lab: (H, W, 3) float32 LAB  *or*  (3,) float32 LAB colour.

    Returns:
        Same shape, uint8 RGB.
    """
    single = lab.ndim == 1
    if single:
        lab = lab.reshape(1, 1, 3)
    lab_cv = lab.copy()
    lab_cv[..., 0] = lab_cv[..., 0] * (255.0 / 100.0)
    lab_cv[..., 1] = lab_cv[..., 1] + 128.0
    lab_cv[..., 2] = lab_cv[..., 2] + 128.0
    lab_cv = np.clip(lab_cv, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab_cv, cv2.COLOR_LAB2RGB)
    return rgb.squeeze() if single else rgb


# ────────────────────────────────────────────────────────────────────────
#  ΔE*ab  (CIE 1994)
# ────────────────────────────────────────────────────────────────────────

def delta_e_cie94(
    lab1: np.ndarray,
    lab2: np.ndarray,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> np.ndarray:
    """Compute ΔE*ab using the CIE-1994 formula.

    Works element-wise on arrays of any compatible shapes.

    Returns:
        Scalar or array of ΔE values (float64).
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    dL = L1 - L2
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    dC = C1 - C2

    da = a1 - a2
    db = b1 - b2
    dH_sq = np.maximum(da ** 2 + db ** 2 - dC ** 2, 0.0)

    SL = 1.0
    SC = 1.0 + 0.045 * C1
    SH = 1.0 + 0.015 * C1

    term_L = (dL / (kL * SL)) ** 2
    term_C = (dC / (kC * SC)) ** 2
    term_H = dH_sq / (kH * SH) ** 2

    return np.sqrt(np.maximum(term_L + term_C + term_H, 0.0))


def mean_delta_e(
    image_lab: np.ndarray,
    target_lab: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Mean ΔE*ab between an image region and a target colour.

    Args:
        image_lab: (H, W, 3) LAB image.
        target_lab: (3,) LAB colour.
        mask: Optional (H, W) bool mask.  Only pixels where mask==True
              are considered.

    Returns:
        Scalar mean ΔE.
    """
    target_broadcast = np.broadcast_to(target_lab, image_lab.shape)
    de = delta_e_cie94(image_lab, target_broadcast)
    if mask is not None:
        de = de[mask]
    if de.size == 0:
        return 0.0
    return float(de.mean())


# ────────────────────────────────────────────────────────────────────────
#  Palette helpers
# ────────────────────────────────────────────────────────────────────────

def validate_palette(palette_rgb: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    """Validate that every palette entry is sRGB-representable and convert to LAB.

    Args:
        palette_rgb: ``{"name": [R, G, B], ...}``  with values in 0–255.

    Returns:
        ``{"name": np.array([L, a, b]), ...}``
    """
    palette_lab: Dict[str, np.ndarray] = {}
    for name, rgb in palette_rgb.items():
        arr = np.array(rgb, dtype=np.uint8)
        if arr.shape != (3,):
            raise ValueError(f"Palette entry '{name}' must have 3 channels, got {arr.shape}")
        if not np.all((arr >= 0) & (arr <= 255)):
            raise ValueError(f"Palette entry '{name}' has out-of-range values: {rgb}")
        palette_lab[name] = rgb_to_lab(arr)
    return palette_lab


def gamut_clamp_rgb(rgb: np.ndarray) -> np.ndarray:
    """Clamp an RGB array to valid sRGB [0, 255] per channel."""
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────
#  Palette Correction Protocol  (Section 6.2)
# ────────────────────────────────────────────────────────────────────────

def identify_failing_pixels(
    output_lab: np.ndarray,
    target_lab: np.ndarray,
    mask: np.ndarray,
    threshold: float = CFG.DELTA_E_THRESHOLD,
) -> np.ndarray:
    """Return boolean mask of pixels inside *mask* whose ΔE > threshold."""
    target_broadcast = np.broadcast_to(target_lab, output_lab.shape)
    de = delta_e_cie94(output_lab, target_broadcast)
    return mask & (de > threshold)


def apply_lab_color_shift(
    output_rgb: np.ndarray,
    target_lab: np.ndarray,
    failing_mask: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """Shift failing pixels towards target colour in LAB space.

    Steps 2–4 of Section 6.2:
      2. Compute correction vector and apply.
      3. Chroma-clamp to sRGB gamut.
      4. Gaussian-blur at corrected-region boundary.

    Returns:
        Corrected RGB uint8 image.
    """
    output = output_rgb.copy()
    if failing_mask.sum() == 0:
        return output

    output_lab = rgb_to_lab(output)

    # Step 2 — compute correction vector
    mean_output_lab = output_lab[failing_mask].mean(axis=0)
    correction = (target_lab - mean_output_lab) * weight

    # Apply shift
    output_lab[failing_mask] += correction

    # Step 3 — gamut clamp via round-trip
    corrected_rgb = lab_to_rgb(output_lab)
    corrected_rgb = gamut_clamp_rgb(corrected_rgb)

    # Step 4 — smooth boundary
    # Build a smooth blending mask so correction doesn't create hard edges
    float_mask = failing_mask.astype(np.float32)
    k = CFG.CORRECTION_GAUSSIAN_KERNEL
    blurred_mask = cv2.GaussianBlur(float_mask, (k * 2 + 1, k * 2 + 1), 0)
    blurred_mask = blurred_mask[..., np.newaxis]  # (H, W, 1)

    # Blend corrected into original
    result = (corrected_rgb.astype(np.float32) * blurred_mask +
              output.astype(np.float32) * (1.0 - blurred_mask))

    return np.clip(result, 0, 255).astype(np.uint8)


def correct_instance_color(
    output_rgb: np.ndarray,
    target_lab: np.ndarray,
    mask: np.ndarray,
    max_iterations: int = CFG.CORRECTION_MAX_ITERATIONS,
) -> Tuple[np.ndarray, float]:
    """Full palette correction protocol (Section 6.2 Steps 1–5).

    Iteratively corrects until ΔE < threshold or max iterations reached.

    Returns:
        (corrected_rgb, final_delta_e)
    """
    current_rgb = output_rgb.copy()
    weight = 1.0

    for i in range(max_iterations):
        current_lab = rgb_to_lab(current_rgb)
        failing = identify_failing_pixels(current_lab, target_lab, mask)

        if failing.sum() == 0:
            break

        current_rgb = apply_lab_color_shift(current_rgb, target_lab, failing, weight)
        weight *= CFG.CORRECTION_WEIGHT_MULTIPLIER

    # Final ΔE
    final_lab = rgb_to_lab(current_rgb)
    final_de = mean_delta_e(final_lab, target_lab, mask)
    return current_rgb, final_de
