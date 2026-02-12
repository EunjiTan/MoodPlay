"""
Instance Hints Module (Enhanced)
Manages user-specified colour hints for specific instances.
Stores LAB palette colours per instance, generates per-instance colour
hint masks, and supports palette override propagation.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image

from backend.core.lab_color_engine import rgb_to_lab, lab_to_rgb


class InstanceHintManager:
    """
    Manages colour hints for specific instance IDs.
    Each hint binds an instance to a text prompt and/or an exact RGB colour.
    """

    def __init__(self):
        # Maps instance_id → hint dict
        self.hints: Dict[int, Dict] = {}

    def add_hint(
        self,
        instance_id: int,
        prompt: Optional[str] = None,
        color: Optional[Tuple[int, int, int]] = None,
    ):
        """Add a hint for a specific instance."""
        hint: Dict = {"prompt": prompt, "color": None, "color_lab": None}
        if color is not None:
            rgb = np.array(color, dtype=np.uint8)
            hint["color"] = rgb
            hint["color_lab"] = rgb_to_lab(rgb)
        self.hints[instance_id] = hint

    def get_hint(self, instance_id: int) -> Optional[Dict]:
        """Get hint for an instance."""
        return self.hints.get(instance_id)

    def apply_hints_to_prompt(
        self,
        base_prompt: str,
        active_instance_ids: List[int],
    ) -> str:
        """Dynamically modify prompt based on visible instances."""
        hints_text = []
        for iid in active_instance_ids:
            hint = self.get_hint(iid)
            if hint and hint.get("prompt"):
                hints_text.append(hint["prompt"])

        if not hints_text:
            return base_prompt

        return base_prompt + ", " + ", ".join(hints_text)

    # ── NEW: LAB palette colour storage ────────────────────────────────

    def set_palette_color(
        self,
        instance_id: int,
        color_rgb: Tuple[int, int, int],
    ):
        """Set or update the palette colour for an instance in both RGB and LAB."""
        rgb = np.array(color_rgb, dtype=np.uint8)
        if instance_id not in self.hints:
            self.hints[instance_id] = {"prompt": None}
        self.hints[instance_id]["color"] = rgb
        self.hints[instance_id]["color_lab"] = rgb_to_lab(rgb)

    def get_palette_lab(self, instance_id: int) -> Optional[np.ndarray]:
        """Return the LAB palette colour for an instance, or None."""
        hint = self.hints.get(instance_id)
        if hint is None:
            return None
        return hint.get("color_lab")

    # ── NEW: Per-instance colour hint mask ─────────────────────────────

    @staticmethod
    def generate_color_hint_mask(
        instance_masks: Dict[int, np.ndarray],
        palette_rgb: Dict[int, np.ndarray],
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Paint each instance mask region with its assigned palette colour.

        Creates a full-frame (H, W, 3) uint8 image where each instance
        region is filled with its palette colour.  Un-owned pixels are black.

        Args:
            instance_masks: {instance_id: (H, W) bool mask}.
            palette_rgb: {instance_id: (3,) uint8 RGB colour}.
            frame_shape: (H, W).

        Returns:
            (H, W, 3) uint8 colour hint image.
        """
        H, W = frame_shape
        hint_image = np.zeros((H, W, 3), dtype=np.uint8)

        for iid, mask in instance_masks.items():
            color = palette_rgb.get(iid)
            if color is None:
                continue
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H)) > 0
            hint_image[mask] = color

        return hint_image

    # ── NEW: Warp colour hint mask between frames ──────────────────────

    @staticmethod
    def warp_hint_mask(
        hint_mask: np.ndarray,
        affine_matrix: np.ndarray,
    ) -> np.ndarray:
        """Warp a colour hint mask using a 2×3 affine matrix.

        Used to align the colour conditioning from Frame N-1 to Frame N
        using CoTracker correspondence.

        Args:
            hint_mask: (H, W, 3) uint8.
            affine_matrix: (2, 3) float.

        Returns:
            Warped (H, W, 3) uint8.
        """
        H, W = hint_mask.shape[:2]
        warped = cv2.warpAffine(
            hint_mask, affine_matrix, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    # ── NEW: Palette override propagation ──────────────────────────────

    def propagate_override(
        self,
        instance_id: int,
        new_color_rgb: Tuple[int, int, int],
    ):
        """Override palette colour and propagate to the hint store.

        This is the user-initiated palette reassignment path.
        """
        self.set_palette_color(instance_id, new_color_rgb)


# Global instance
instance_hint_manager = InstanceHintManager()
