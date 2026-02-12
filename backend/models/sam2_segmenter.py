"""
SAM-2 Segmentation Module (Enhanced)
Wraps SAM-2 (Segment Anything Model 2) for high-quality instance segmentation
with temporal propagation, occlusion handling, mask-overlap resolution,
and soft-boundary support as required by Section 3.1 of the specification.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from backend.core.logging import get_logger, log_gpu_memory, time_execution
from backend.core.device import get_device, get_dtype, oom_safe, device_manager
from backend.core.exceptions import ModelLoadError, ResourceError
from backend.core.pipeline_config import CFG

logger = get_logger("models.sam2_segmenter")


class SAM2Segmenter:
    """
    SAM-2 wrapper for instance segmentation.
    Uses the Hiera architecture for best quality.

    Features:
    - High-quality instance masks
    - Temporal propagation via memory bank
    - Occlusion-resilient tracking
    - Zero-overlap mask enforcement
    - Soft-boundary dilation for fuzzy objects
    """

    CHECKPOINT_MAP = {
        "tiny": {
            "filename": "sam2_hiera_tiny.pt",
            "config": "sam2_hiera_t.yaml"
        },
        "small": {
            "filename": "sam2_hiera_small.pt",
            "config": "sam2_hiera_s.yaml"
        },
        "base_plus": {
            "filename": "sam2_hiera_base_plus.pt",
            "config": "sam2_hiera_b+.yaml"
        },
        "large": {
            "filename": "sam2_hiera_large.pt",
            "config": "sam2_hiera_l.yaml"
        }
    }

    # Labels that get soft-boundary treatment (hair, fur, smoke, etc.)
    SOFT_BOUNDARY_CLASSES = {
        "person", "cat", "dog", "horse", "bird",
        "teddy bear", "hair dryer",
    }

    def __init__(self, model_size: str = "tiny"):
        """
        Initialize SAM-2 segmenter.

        Args:
            model_size: One of 'tiny', 'small', 'base_plus', 'large'.
        """
        if model_size not in self.CHECKPOINT_MAP:
            raise ValueError(f"Unknown model size: {model_size}. "
                             f"Choose from: {list(self.CHECKPOINT_MAP.keys())}")

        self.model_size = model_size
        self.device = get_device()
        self.model = None
        self.mask_generator = None
        self.predictor = None

        # Temporal memory bank: {instance_id: last_mask}
        self._mask_memory: Dict[int, np.ndarray] = {}
        self._confidence_memory: Dict[int, float] = {}

        # Checkpoint paths
        self.checkpoint_dir = Path("checkpoints/sam2")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SAM-2 Segmenter initialized (size={model_size})")

    def _get_checkpoint_path(self) -> Path:
        config = self.CHECKPOINT_MAP[self.model_size]
        return self.checkpoint_dir / config["filename"]

    def _check_checkpoint_exists(self) -> bool:
        return self._get_checkpoint_path().exists()

    @time_execution(logger)
    def load_model(self):
        """Load SAM-2 model to GPU."""
        if self.model is not None:
            return

        checkpoint_path = self._get_checkpoint_path()

        if not checkpoint_path.exists():
            raise ModelLoadError(
                f"SAM-2 checkpoint not found at {checkpoint_path}. "
                f"Please download from Meta and place in {self.checkpoint_dir}"
            )

        try:
            logger.info(f"Loading SAM-2 {self.model_size}...")

            try:
                from sam2.build_sam import build_sam2
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError:
                try:
                    from segment_anything_2.build_sam import build_sam2
                    from segment_anything_2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                    from segment_anything_2.sam2_image_predictor import SAM2ImagePredictor
                except ImportError:
                    raise ModelLoadError(
                        "SAM-2 package not installed. "
                        "Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
                    )

            config = self.CHECKPOINT_MAP[self.model_size]

            self.model = build_sam2(
                config_file=config["config"],
                ckpt_path=str(checkpoint_path),
                device=str(self.device)
            )

            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.model,
                points_per_side=32,
                pred_iou_thresh=CFG.MASK_IOU_THRESH,
                stability_score_thresh=CFG.MASK_STABILITY_THRESH,
                crop_n_layers=0,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=100,
            )

            self.predictor = SAM2ImagePredictor(self.model)

            log_gpu_memory(logger, "SAM-2 Loaded")

        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM-2 model: {e}")

    # ── Core segmentation ──────────────────────────────────────────────

    @staticmethod
    def _to_rgb_np(image) -> np.ndarray:
        """Normalise input to (H, W, 3) uint8 RGB."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        return image.astype(np.uint8)

    @oom_safe(logger)
    def segment_frame(self, image) -> List[Dict]:
        """Generate automatic instance masks for a single frame."""
        if self.model is None:
            self.load_model()

        image_np = self._to_rgb_np(image)
        with torch.no_grad():
            masks = self.mask_generator.generate(image_np)

        logger.debug(f"Generated {len(masks)} masks")
        return masks

    @oom_safe(logger)
    def segment_with_points(
        self,
        image,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Segment with point prompts (interactive refinement)."""
        if self.model is None:
            self.load_model()

        image_np = self._to_rgb_np(image)
        with torch.no_grad():
            self.predictor.set_image(image_np)
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False,
            )
        return masks[0], float(scores[0])

    # ── NEW: Box-prompted segmentation ─────────────────────────────────

    @oom_safe(logger)
    def segment_frame_with_boxes(
        self,
        image,
        boxes: List[np.ndarray],
    ) -> List[Tuple[np.ndarray, float]]:
        """Segment using bounding boxes as prompts (YOLO-driven).

        Args:
            image: RGB image (PIL or ndarray).
            boxes: list of (4,) arrays [x1, y1, x2, y2].

        Returns:
            List of (mask, iou_score) per box.
        """
        if self.model is None:
            self.load_model()

        image_np = self._to_rgb_np(image)

        results: List[Tuple[np.ndarray, float]] = []
        with torch.no_grad():
            self.predictor.set_image(image_np)
            for box in boxes:
                box_arr = np.array(box).reshape(1, 4)
                masks, scores, _ = self.predictor.predict(
                    box=box_arr,
                    multimask_output=False,
                )
                results.append((masks[0], float(scores[0])))

        return results

    # ── NEW: Temporal mask propagation ─────────────────────────────────

    def propagate_masks(
        self,
        image,
        prev_masks: Dict[int, np.ndarray],
        prev_confidences: Dict[int, float],
        boxes: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[int, Tuple[np.ndarray, float]]:
        """Propagate masks from previous frame using memory.

        If the previous mask confidence is above the threshold, use the
        stored mask as a prompt instead of re-segmenting from scratch.

        Args:
            image: Current frame.
            prev_masks: {instance_id: (H, W) bool mask}.
            prev_confidences: {instance_id: iou_score}.
            boxes: Optional {instance_id: (4,) bbox} for refinement.

        Returns:
            {instance_id: (mask, confidence)}.
        """
        if self.model is None:
            self.load_model()

        image_np = self._to_rgb_np(image)
        results: Dict[int, Tuple[np.ndarray, float]] = {}

        with torch.no_grad():
            self.predictor.set_image(image_np)

            for iid, prev_mask in prev_masks.items():
                conf = prev_confidences.get(iid, 0.0)

                if conf >= CFG.MASK_CONFIDENCE_MIN:
                    # Use centre of previous mask as point prompt
                    ys, xs = np.where(prev_mask)
                    if len(ys) > 0:
                        cy, cx = int(ys.mean()), int(xs.mean())
                        points = np.array([[cx, cy]])
                        labels = np.array([1])

                        masks, scores, _ = self.predictor.predict(
                            point_coords=points,
                            point_labels=labels,
                            multimask_output=False,
                        )
                        results[iid] = (masks[0], float(scores[0]))
                        continue

                # Fallback: use box prompt if available
                if boxes and iid in boxes:
                    box_arr = np.array(boxes[iid]).reshape(1, 4)
                    masks, scores, _ = self.predictor.predict(
                        box=box_arr,
                        multimask_output=False,
                    )
                    results[iid] = (masks[0], float(scores[0]))
                else:
                    # Keep previous mask as-is
                    results[iid] = (prev_mask, conf)

        # Update internal memory
        for iid, (mask, score) in results.items():
            self._mask_memory[iid] = mask
            self._confidence_memory[iid] = score

        return results

    # ── NEW: Handle occlusion ──────────────────────────────────────────

    def handle_occlusion(self, instance_id: int) -> Optional[np.ndarray]:
        """Return the last known mask for an occluded instance."""
        return self._mask_memory.get(instance_id)

    # ── NEW: Zero-overlap enforcement ──────────────────────────────────

    @staticmethod
    def resolve_overlaps(
        masks: Dict[int, np.ndarray],
        confidences: Dict[int, float],
    ) -> Dict[int, np.ndarray]:
        """Ensure zero overlap between instance masks.

        For overlapping pixels, the higher-confidence mask wins.

        Args:
            masks: {instance_id: (H, W) bool}.
            confidences: {instance_id: float}.

        Returns:
            Corrected masks with guaranteed zero overlap.
        """
        if not masks:
            return masks

        ids = sorted(masks.keys(), key=lambda k: confidences.get(k, 0.0), reverse=True)

        ref_shape = next(iter(masks.values())).shape
        occupied = np.zeros(ref_shape, dtype=bool)
        resolved: Dict[int, np.ndarray] = {}

        for iid in ids:
            mask = masks[iid].copy()
            overlap = mask & occupied
            if overlap.any():
                mask[overlap] = False
                logger.debug(f"Clipped {overlap.sum()} px from instance {iid}")
            resolved[iid] = mask
            occupied |= mask

        return resolved

    # ── NEW: Soft boundary dilation ────────────────────────────────────

    @staticmethod
    def dilate_soft_boundaries(
        mask: np.ndarray,
        class_label: str,
        dilation_px: int = CFG.SOFT_BOUNDARY_DILATION_PX,
    ) -> np.ndarray:
        """Dilate mask edges for objects with naturally fuzzy boundaries.

        Only applied to classes in SOFT_BOUNDARY_CLASSES.

        Returns:
            Dilated bool mask (same shape as input).
        """
        if class_label.lower() not in SAM2Segmenter.SOFT_BOUNDARY_CLASSES:
            return mask

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
        )
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return dilated.astype(bool)

    # ── Memory management ──────────────────────────────────────────────

    def clear_memory(self) -> None:
        """Clear temporal mask memory."""
        self._mask_memory.clear()
        self._confidence_memory.clear()

    def unload_model(self):
        """Free GPU resources."""
        if self.model is not None:
            del self.mask_generator
            del self.predictor
            del self.model
            self.model = None
            self.mask_generator = None
            self.predictor = None
            self.clear_memory()
            device_manager.empty_cache()
            logger.info("SAM-2 model unloaded")


# Global instance (uses tiny for VRAM efficiency)
sam2_segmenter = SAM2Segmenter(model_size="tiny")
