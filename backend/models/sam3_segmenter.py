"""
SAM-3 Segmentation Module
High-quality instance segmentation using SAM-3 (Segment Anything Model 3).
ONLY SAM-3 - No SAM-1, No SAM-2.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from backend.core.logging import get_logger, log_gpu_memory
from backend.core.device import get_device, get_dtype, device_manager
from backend.core.exceptions import ModelLoadError

logger = get_logger("models.sam3_segmenter")


class SAM3Segmenter:
    """
    SAM-3 (Segment Anything Model 3) wrapper for instance segmentation.
    Uses the latest SAM version from Meta AI for best quality masks.
    
    Features:
    - Automatic mask generation
    - Point-based prompting
    - High-quality instance masks
    - VRAM-optimized loading
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "tiny": {
            "config": "sam3_hiera_t.yaml",
            "checkpoint": "sam3_hiera_tiny.pt"
        },
        "small": {
            "config": "sam3_hiera_s.yaml", 
            "checkpoint": "sam3_hiera_small.pt"
        },
        "base_plus": {
            "config": "sam3_hiera_b+.yaml",
            "checkpoint": "sam3_hiera_base_plus.pt"
        },
        "large": {
            "config": "sam3_hiera_l.yaml",
            "checkpoint": "sam3_hiera_large.pt"
        }
    }
    
    def __init__(
        self,
        model_size: str = "large",
        checkpoint_dir: str = "checkpoints/sam3",
        device: str = None
    ):
        """
        Initialize SAM-3 segmenter.
        
        Args:
            model_size: Model size ('tiny', 'small', 'base_plus', 'large')
            checkpoint_dir: Directory containing SAM-3 checkpoints
            device: Device to use (auto-detected if None)
        """
        self.model_size = model_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or get_device()
        self.dtype = get_dtype()
        
        self.model = None
        self.mask_generator = None
        self.predictor = None
        
        # Get config for this model size
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}. Use: {list(self.MODEL_CONFIGS.keys())}")
        
        self.config = self.MODEL_CONFIGS[model_size]
        logger.info(f"SAM-3 Segmenter initialized (size={model_size}, device={self.device})")
    
    def _get_checkpoint_path(self) -> Path:
        """Get full path to checkpoint file."""
        return self.checkpoint_dir / self.config["checkpoint"]
    
    def load_model(self):
        """Load SAM-3 model into memory."""
        if self.model is not None:
            return
        
        checkpoint_path = self._get_checkpoint_path()
        
        if not checkpoint_path.exists():
            raise ModelLoadError(
                f"SAM-3 checkpoint not found: {checkpoint_path}\n"
                f"Please download from Meta AI and place in {self.checkpoint_dir}"
            )
        
        try:
            logger.info(f"Loading SAM-3 ({self.model_size}) from {checkpoint_path}...")
            
            # Import SAM-3 (try multiple possible package names)
            try:
                from sam3.build_sam import build_sam3
                from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator
                from sam3.sam3_image_predictor import SAM3ImagePredictor
            except ImportError:
                try:
                    # Fallback: SAM-3 might use sam2-style imports during transition
                    from segment_anything_3.build_sam import build_sam3
                    from segment_anything_3.automatic_mask_generator import SAM3AutomaticMaskGenerator
                    from segment_anything_3.sam3_image_predictor import SAM3ImagePredictor
                except ImportError:
                    raise ModelLoadError(
                        "SAM-3 package not installed. Install with:\n"
                        "pip install git+https://github.com/facebookresearch/segment-anything-3.git"
                    )
            
            # Build model
            self.model = build_sam3(
                config_file=self.config["config"],
                ckpt_path=str(checkpoint_path),
                device=str(self.device)
            )
            
            # Create mask generator for automatic segmentation
            self.mask_generator = SAM3AutomaticMaskGenerator(
                model=self.model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100
            )
            
            # Create predictor for point-based prompts
            self.predictor = SAM3ImagePredictor(self.model)
            
            log_gpu_memory(logger, "SAM-3 Loaded")
            logger.info(f"SAM-3 ({self.model_size}) ready")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM-3: {e}")
    
    def segment_frame(
        self,
        frame: Image.Image,
        return_largest_n: int = None
    ) -> List[Dict]:
        """
        Segment a frame using automatic mask generation.
        
        Args:
            frame: PIL Image (RGB)
            return_largest_n: If set, only return N largest masks
            
        Returns:
            List of mask dictionaries with keys:
            - 'segmentation': boolean numpy array (H, W)
            - 'area': mask area in pixels
            - 'bbox': bounding box [x, y, w, h]
            - 'predicted_iou': model's IoU prediction
            - 'stability_score': mask stability score
        """
        if self.mask_generator is None:
            self.load_model()
        
        # Convert PIL to numpy
        frame_np = np.array(frame)
        
        # Generate masks
        masks = self.mask_generator.generate(frame_np)
        
        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Optionally limit count
        if return_largest_n:
            masks = masks[:return_largest_n]
        
        return masks
    
    def segment_with_points(
        self,
        frame: Image.Image,
        points: List[Tuple[int, int]],
        labels: List[int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Segment a specific region using point prompts.
        
        Args:
            frame: PIL Image (RGB)
            points: List of (x, y) coordinates
            labels: List of 1 (foreground) or 0 (background) for each point
            
        Returns:
            Tuple of (mask, score)
        """
        if self.predictor is None:
            self.load_model()
        
        frame_np = np.array(frame)
        self.predictor.set_image(frame_np)
        
        points_np = np.array(points)
        labels_np = np.array(labels) if labels else np.ones(len(points))
        
        masks, scores, _ = self.predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=False
        )
        
        return masks[0], scores[0]
    
    def create_id_map(
        self,
        masks: List[Dict],
        frame_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create a single ID map from multiple masks.
        
        Args:
            masks: List of mask dictionaries from segment_frame
            frame_size: (height, width)
            
        Returns:
            ID map where each pixel has the mask ID (0 = background)
        """
        h, w = frame_size
        id_map = np.zeros((h, w), dtype=np.uint16)
        
        # Larger masks first (so smaller masks overlay them)
        for i, mask_data in enumerate(masks, 1):
            id_map[mask_data['segmentation']] = i
        
        return id_map
    
    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.model:
            del self.model
            self.model = None
        if self.mask_generator:
            del self.mask_generator
            self.mask_generator = None
        if self.predictor:
            del self.predictor
            self.predictor = None
        
        device_manager.empty_cache()
        logger.info("SAM-3 unloaded")


# Global instance for convenience
sam3_segmenter = SAM3Segmenter(model_size="large")
