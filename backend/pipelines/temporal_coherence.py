"""
Temporal Coherence Module (Enhanced)
Ensures colour consistency across video frames.
Section 4: TCV computation, sliding-window analysis, temporal median
filtering, CoTracker-based colour warping, and per-instance LAB tracking.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple

from backend.core.pipeline_config import CFG
from backend.core.lab_color_engine import rgb_to_lab, mean_delta_e


class TemporalCoherence:
    def __init__(self):
        self.prev_frame = None
        self.prev_colorized = None
        self.flow_cache = {}
        self.color_memory: Dict[int, np.ndarray] = {}  # {track_id: avg LAB}
        self._frame_history: List[np.ndarray] = []      # ring-buffer of colorised frames

    # ── Optical flow ───────────────────────────────────────────────────

    def compute_optical_flow(self, frame1, frame2):
        """Compute dense optical flow (Farneback) between two frames."""
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1

        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        return flow

    def warp_with_optical_flow(self, image, flow):
        """Warp image using optical flow field."""
        h, w = flow.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x_new = x + flow[..., 0]
        y_new = y + flow[..., 1]
        warped = cv2.remap(
            image, x_new, y_new,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    def propagate_colors(self, current_frame, prev_colorized_frame, blend_weight=0.3):
        """Propagate colours from previous frame via optical flow."""
        if self.prev_frame is None:
            return None
        flow = self.compute_optical_flow(self.prev_frame, current_frame)
        warped_color = self.warp_with_optical_flow(prev_colorized_frame, flow)
        return warped_color

    # ── Per-instance colour memory ─────────────────────────────────────

    def update_color_memory(self, track_id: int, colorized_frame: np.ndarray, mask: np.ndarray):
        """Update colour memory (EMA) for a tracked object in LAB space."""
        if mask.sum() == 0:
            return
        lab = rgb_to_lab(colorized_frame)
        masked = lab[mask > 0]
        avg_lab = masked.mean(axis=0)

        if track_id in self.color_memory:
            self.color_memory[track_id] = 0.7 * self.color_memory[track_id] + 0.3 * avg_lab
        else:
            self.color_memory[track_id] = avg_lab

    def get_color_hint_from_memory(self, track_id: int) -> Optional[np.ndarray]:
        """Get stored LAB colour for an object."""
        return self.color_memory.get(track_id)

    # ── Temporal smoothing ─────────────────────────────────────────────

    def apply_temporal_smoothing(self, current_colorized, prev_colorized, alpha=0.2):
        """Blend current frame with previous to reduce flicker."""
        if prev_colorized is None:
            return current_colorized

        if current_colorized.shape != prev_colorized.shape:
            prev_colorized = cv2.resize(
                prev_colorized,
                (current_colorized.shape[1], current_colorized.shape[0]),
            )

        smoothed = cv2.addWeighted(
            current_colorized, 1 - alpha,
            prev_colorized, alpha, 0,
        )
        return smoothed

    def update_frame_history(self, grayscale_frame, colorized_frame):
        """Update frame history for next iteration."""
        self.prev_frame = grayscale_frame.copy()
        self.prev_colorized = colorized_frame.copy()

        # Ring-buffer for median filtering
        self._frame_history.append(colorized_frame.copy())
        max_hist = CFG.TEMPORAL_SMOOTHING_WINDOW + 2
        if len(self._frame_history) > max_hist:
            self._frame_history = self._frame_history[-max_hist:]

    # ── NEW: TCV — Temporal Color Variance (Section 4.2) ───────────────

    @staticmethod
    def compute_tcv(
        color_history_lab: List[np.ndarray],
        window_size: int = CFG.TCV_WINDOW_SIZE,
    ) -> float:
        """Temporal Color Variance for one instance over a sliding window.

        Returns sqrt of summed per-channel variance across the window.
        Target: < 8.0.
        """
        if len(color_history_lab) < 2:
            return 0.0
        window = color_history_lab[-window_size:]
        arr = np.stack(window)  # (W, 3)
        return float(np.sqrt(np.var(arr, axis=0).sum()))

    # ── NEW: Temporal median filter (Section 7.3) ──────────────────────

    def apply_temporal_median_filter(
        self,
        current_colorized: np.ndarray,
        window: int = CFG.TEMPORAL_SMOOTHING_WINDOW,
    ) -> np.ndarray:
        """Per-pixel temporal median over the last N frames.

        Reduces residual flicker in final colour values.
        """
        if len(self._frame_history) < window:
            return current_colorized

        stack = np.stack(self._frame_history[-window:])  # (W, H, Width, 3)
        median = np.median(stack, axis=0).astype(np.uint8)
        return median

    # ── NEW: Warped colour prior (Section 4.1 Step T3–T4) ─────────────

    def get_warped_color_prior(
        self,
        current_frame: np.ndarray,
        prev_colorized: np.ndarray,
        instance_masks: Dict[int, np.ndarray],
        affine_matrices: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """Warp Frame N-1 colour conditioning to Frame N.

        Uses per-instance affine transforms (from CoTracker) when available,
        otherwise falls back to dense optical flow.

        Returns:
            (warped_color_image, per_instance_tcv_flags)
        """
        H, W = current_frame.shape[:2]
        warped = np.zeros((H, W, 3), dtype=np.uint8)
        tcv_flags: Dict[int, float] = {}

        if affine_matrices:
            for iid, mask in instance_masks.items():
                M = affine_matrices.get(iid)
                if M is not None:
                    # Warp per-instance region
                    instance_color = prev_colorized.copy()
                    instance_color[~mask] = 0
                    warped_inst = cv2.warpAffine(instance_color, M, (W, H),
                                                  flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_REPLICATE)
                    warped_mask = cv2.warpAffine(mask.astype(np.uint8), M, (W, H)) > 0
                    warped[warped_mask] = warped_inst[warped_mask]
                else:
                    warped[mask] = prev_colorized[mask]
        else:
            # Fallback: dense optical flow
            flow_warped = self.propagate_colors(current_frame, prev_colorized)
            if flow_warped is not None:
                warped = flow_warped

        return warped, tcv_flags

    # ── NEW: Reset ─────────────────────────────────────────────────────

    def reset(self):
        """Clear all temporal state."""
        self.prev_frame = None
        self.prev_colorized = None
        self.flow_cache.clear()
        self.color_memory.clear()
        self._frame_history.clear()
