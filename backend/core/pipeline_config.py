"""
Pipeline Configuration — Central constants and thresholds for the
Instance-Guided Semantic Video Colorization system.

All numerical thresholds referenced in the specification are defined here
so that every module draws from a single source of truth.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the colorization pipeline."""

    # ── Perceptual Color Thresholds (Section 6.1) ──────────────────────
    DELTA_E_THRESHOLD: float = 5.0          # Max acceptable ΔE*ab per instance
    DELTA_E_IMPERCEPTIBLE: float = 1.0      # Ideal range
    DELTA_E_ACCEPTABLE: float = 3.0         # Perceivable to trained eye

    # ── Temporal Consistency (Section 4.2) ─────────────────────────────
    TCV_THRESHOLD: float = 8.0              # Temporal Color Variance max
    TCV_WINDOW_SIZE: int = 5                # Sliding window (frames)
    TEMPORAL_SMOOTHING_WINDOW: int = 3      # Median filter window

    # ── Boundary (Section 5) ───────────────────────────────────────────
    BLS_THRESHOLD: float = 0.001            # Frame-level re-run trigger
    BLS_HARD_ZERO: float = 0.0              # Target for hard boundaries
    SOFT_BOUNDARY_DILATION_PX: int = 3      # Fuzzy-edge dilation
    BUFFER_ZONE_PX: int = 1                 # Inter-instance contact buffer

    # ── Global Palette Coverage (Section 4.2) ──────────────────────────
    GPC_MIN_COVERAGE: float = 0.005         # 0.5 % minimum per palette entry

    # ── Detection / Segmentation ───────────────────────────────────────
    MASK_CONFIDENCE_MIN: float = 0.7        # SAM-2 re-segment threshold
    DETECTION_CONFIDENCE_MIN: float = 0.5   # YOLO suppress below this
    MASK_IOU_THRESH: float = 0.86           # SAM-2 auto-mask threshold
    MASK_STABILITY_THRESH: float = 0.92     # SAM-2 stability score min

    # ── Tracking ───────────────────────────────────────────────────────
    MOTION_MAGNITUDE_THRESHOLD: float = 30.0  # Pixels; triggers SAM re-val
    REENTRY_IOU_THRESHOLD: float = 0.3        # IoU for re-entry matching
    COTRACKER_GRID_SIZE: int = 20             # Default tracking grid

    # ── ControlNet Weight Schedule (Section 3.4) ───────────────────────
    CONTROLNET_WEIGHT_EARLY: float = 0.9    # First 60 % of steps
    CONTROLNET_WEIGHT_LATE: float = 0.4     # Last 40 % of steps
    CONTROLNET_EARLY_RATIO: float = 0.6     # Fraction of steps at high wt

    # ── LoRA (Section 3.5) ─────────────────────────────────────────────
    LORA_RANK_STANDARD: int = 16
    LORA_RANK_STYLIZED: int = 32
    LORA_DEFAULT_WEIGHT: float = 0.7

    # ── Diffusion Defaults ─────────────────────────────────────────────
    NUM_INFERENCE_STEPS: int = 20
    GUIDANCE_SCALE: float = 7.5
    DENOISE_STRENGTH: float = 0.75
    TARGET_SIZE: Tuple[int, int] = (512, 512)

    # ── Palette Correction (Section 6.2) ───────────────────────────────
    CORRECTION_GAUSSIAN_KERNEL: int = 3     # Smoothness blur kernel
    CORRECTION_MAX_ITERATIONS: int = 3      # Re-attempt ceiling
    CORRECTION_WEIGHT_MULTIPLIER: float = 1.5  # Stronger shift each retry

    # ── Stylized Palette Names (trigger higher LoRA rank) ──────────────
    STYLIZED_PALETTES: Tuple[str, ...] = (
        "neon_cyberpunk", "vintage_sepia", "retro_vhs",
        "noir", "psychedelic", "vaporwave",
    )


# Singleton — importable everywhere as `from backend.core.pipeline_config import CFG`
CFG = PipelineConfig()
