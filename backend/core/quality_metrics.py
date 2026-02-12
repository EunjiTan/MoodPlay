"""
Quality Metrics Module — Computes all evaluation metrics defined in
Section 4.2 and Section 6.1 of the Instance-Guided Video Colorization spec.

Metrics:
  TCV  — Temporal Color Variance (per-instance, sliding-window)
  ICA  — Instance-masked mean ΔE*ab vs. palette target
  BLS  — Boundary Leakage Score  (fraction of misassigned pixels)
  GPC  — Global Palette Coverage (palette entry pixel frequency)
  SSIM — Structural Similarity vs. grayscale input (optional)
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from backend.core.pipeline_config import CFG
from backend.core.lab_color_engine import (
    rgb_to_lab, delta_e_cie94, mean_delta_e,
)


# ────────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────────

@dataclass
class FrameMetrics:
    """Per-frame quality snapshot."""
    frame_idx: int
    per_instance_ica: Dict[int, float] = field(default_factory=dict)
    per_instance_tcv: Dict[int, float] = field(default_factory=dict)
    bls: float = 0.0
    gpc: Dict[str, float] = field(default_factory=dict)


@dataclass
class VideoMetrics:
    """Aggregated video-level quality report."""
    avg_ica: float = 0.0
    avg_tcv: float = 0.0
    max_bls: float = 0.0
    gpc_pass: bool = True
    total_frames: int = 0
    frames_with_bls_violations: int = 0
    per_instance_summary: Dict[int, Dict] = field(default_factory=dict)
    frame_reports: List[FrameMetrics] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────
# ICA — Instance-masked ΔE*ab
# ────────────────────────────────────────────────────────────────────────

def compute_ica(
    colorized_rgb: np.ndarray,
    target_lab: np.ndarray,
    instance_mask: np.ndarray,
) -> float:
    """Instance Color Accuracy: mean ΔE within instance mask region.

    Args:
        colorized_rgb: (H, W, 3) uint8 output frame.
        target_lab: (3,) LAB palette target for this instance.
        instance_mask: (H, W) bool mask for the instance.

    Returns:
        Mean ΔE*ab (float).  Target: < 5.0.
    """
    if instance_mask.sum() == 0:
        return 0.0
    lab = rgb_to_lab(colorized_rgb)
    return mean_delta_e(lab, target_lab, instance_mask)


# ────────────────────────────────────────────────────────────────────────
# TCV — Temporal Color Variance
# ────────────────────────────────────────────────────────────────────────

def compute_tcv(
    color_history_lab: List[np.ndarray],
    window_size: int = CFG.TCV_WINDOW_SIZE,
) -> float:
    """Temporal Color Variance for one instance over a sliding window.

    Args:
        color_history_lab: list of (3,) LAB means, most recent last.
        window_size: number of frames to consider.

    Returns:
        Mean L₂ variance of LAB values across the window.  Target: < 8.0.
    """
    if len(color_history_lab) < 2:
        return 0.0
    window = color_history_lab[-window_size:]
    arr = np.stack(window)  # (W, 3)
    variance = np.var(arr, axis=0).sum()  # sum of per-channel variance
    return float(np.sqrt(variance))


# ────────────────────────────────────────────────────────────────────────
# BLS — Boundary Leakage Score
# ────────────────────────────────────────────────────────────────────────

def compute_bls(
    colorized_rgb: np.ndarray,
    instance_masks: Dict[int, np.ndarray],
    palette_lab: Dict[int, np.ndarray],
) -> float:
    """Boundary Leakage Score — fraction of pixels whose colour belongs
    to a *different* instance than their mask assignment.

    Works by finding each pixel's nearest palette entry in LAB space and
    checking whether it matches the mask owner.

    Args:
        colorized_rgb: (H, W, 3) uint8 output.
        instance_masks: {instance_id: (H, W) bool mask}.
        palette_lab: {instance_id: (3,) LAB target}.

    Returns:
        BLS ∈ [0, 1].  Target: 0.
    """
    if not instance_masks or not palette_lab:
        return 0.0

    H, W = colorized_rgb.shape[:2]
    lab_img = rgb_to_lab(colorized_rgb)

    # Build per-pixel ownership map
    owner = np.full((H, W), -1, dtype=np.int32)
    for iid, mask in instance_masks.items():
        owner[mask] = iid

    # Determine effective coverage (pixels assigned to some instance)
    covered = owner >= 0
    total_covered = int(covered.sum())
    if total_covered == 0:
        return 0.0

    # For each covered pixel, find the closest palette entry
    ids = sorted(palette_lab.keys())
    palette_stack = np.stack([palette_lab[i] for i in ids])  # (K, 3)

    flat_lab = lab_img[covered]  # (N, 3)
    # Broadcast ΔE computation
    # flat_lab: (N, 1, 3),  palette: (1, K, 3)
    expanded_lab = flat_lab[:, None, :]
    expanded_pal = palette_stack[None, :, :]
    de = delta_e_cie94(expanded_lab, expanded_pal)  # (N, K)

    nearest_idx = de.argmin(axis=1)
    nearest_id = np.array([ids[j] for j in nearest_idx])

    assigned_id = owner[covered]
    mismatches = int((nearest_id != assigned_id).sum())

    return mismatches / total_covered


# ────────────────────────────────────────────────────────────────────────
# GPC — Global Palette Coverage
# ────────────────────────────────────────────────────────────────────────

def compute_gpc(
    colorized_rgb: np.ndarray,
    palette_lab: Dict[str, np.ndarray],
    min_coverage: float = CFG.GPC_MIN_COVERAGE,
) -> Dict[str, float]:
    """Global Palette Coverage — percentage of pixels nearest to each palette entry.

    Args:
        colorized_rgb: (H, W, 3) uint8 output.
        palette_lab: {name: (3,) LAB}.
        min_coverage: minimum fraction (default 0.5 %).

    Returns:
        {palette_name: pixel_fraction}.  All entries should be ≥ min_coverage.
    """
    if not palette_lab:
        return {}

    H, W = colorized_rgb.shape[:2]
    total = H * W
    lab_img = rgb_to_lab(colorized_rgb).reshape(-1, 3)

    names = list(palette_lab.keys())
    palette_stack = np.stack([palette_lab[n] for n in names])

    expanded_lab = lab_img[:, None, :]
    expanded_pal = palette_stack[None, :, :]
    de = delta_e_cie94(expanded_lab, expanded_pal)
    nearest = de.argmin(axis=1)

    coverage: Dict[str, float] = {}
    for idx, name in enumerate(names):
        count = int((nearest == idx).sum())
        coverage[name] = count / total

    return coverage


def gpc_passes(coverage: Dict[str, float]) -> bool:
    """Check if all palette entries meet the minimum coverage threshold."""
    return all(v >= CFG.GPC_MIN_COVERAGE for v in coverage.values())


# ────────────────────────────────────────────────────────────────────────
# SSIM — Structural Similarity (luminance channel)
# ────────────────────────────────────────────────────────────────────────

def compute_ssim(
    grayscale: np.ndarray,
    colorized_rgb: np.ndarray,
) -> float:
    """Structural Similarity between grayscale input and luminance of output.

    Uses a simple windowed SSIM computation via OpenCV.

    Args:
        grayscale: (H, W) uint8 or (H, W, 1).
        colorized_rgb: (H, W, 3) uint8 output.

    Returns:
        SSIM ∈ [0, 1].
    """
    if grayscale.ndim == 3:
        grayscale = grayscale[..., 0]

    # Extract luminance from output
    lab = cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2LAB)
    lum = lab[..., 0]

    # Ensure same size
    if grayscale.shape != lum.shape:
        lum = cv2.resize(lum, (grayscale.shape[1], grayscale.shape[0]))

    # Normalise grayscale to same range as LAB L channel (0-255 in OpenCV)
    g = grayscale.astype(np.float64)
    l = lum.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_g = cv2.GaussianBlur(g, (11, 11), 1.5)
    mu_l = cv2.GaussianBlur(l, (11, 11), 1.5)

    sigma_g2 = cv2.GaussianBlur(g ** 2, (11, 11), 1.5) - mu_g ** 2
    sigma_l2 = cv2.GaussianBlur(l ** 2, (11, 11), 1.5) - mu_l ** 2
    sigma_gl = cv2.GaussianBlur(g * l, (11, 11), 1.5) - mu_g * mu_l

    num = (2 * mu_g * mu_l + C1) * (2 * sigma_gl + C2)
    den = (mu_g ** 2 + mu_l ** 2 + C1) * (sigma_g2 + sigma_l2 + C2)

    ssim_map = num / den
    return float(ssim_map.mean())


# ────────────────────────────────────────────────────────────────────────
# Aggregate report builder
# ────────────────────────────────────────────────────────────────────────

def build_video_report(frame_reports: List[FrameMetrics]) -> VideoMetrics:
    """Aggregate per-frame metrics into a video-level quality report."""
    vm = VideoMetrics()
    vm.total_frames = len(frame_reports)
    vm.frame_reports = frame_reports

    if not frame_reports:
        return vm

    all_ica = []
    all_tcv = []
    instance_ica_acc: Dict[int, List[float]] = {}
    instance_tcv_acc: Dict[int, List[float]] = {}

    for fr in frame_reports:
        # BLS
        if fr.bls > vm.max_bls:
            vm.max_bls = fr.bls
        if fr.bls > 0:
            vm.frames_with_bls_violations += 1

        # ICA
        for iid, ica in fr.per_instance_ica.items():
            all_ica.append(ica)
            instance_ica_acc.setdefault(iid, []).append(ica)

        # TCV
        for iid, tcv in fr.per_instance_tcv.items():
            all_tcv.append(tcv)
            instance_tcv_acc.setdefault(iid, []).append(tcv)

        # GPC
        if fr.gpc:
            if not gpc_passes(fr.gpc):
                vm.gpc_pass = False

    vm.avg_ica = float(np.mean(all_ica)) if all_ica else 0.0
    vm.avg_tcv = float(np.mean(all_tcv)) if all_tcv else 0.0

    # Per-instance summary
    all_ids = set(list(instance_ica_acc.keys()) + list(instance_tcv_acc.keys()))
    for iid in all_ids:
        vm.per_instance_summary[iid] = {
            "avg_ica": float(np.mean(instance_ica_acc.get(iid, [0]))),
            "avg_tcv": float(np.mean(instance_tcv_acc.get(iid, [0]))),
        }

    return vm
