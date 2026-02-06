"""
SAM-2 Segmentation Module
Wraps SAM-2 (Segment Anything Model 2) for high-quality instance segmentation.
Replaces SAM-1 with better quality and temporal consistency.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from backend.core.logging import get_logger, log_gpu_memory, time_execution
from backend.core.device import get_device, get_dtype, oom_safe, device_manager
from backend.core.exceptions import ModelLoadError, ResourceError

logger = get_logger("models.sam2_segmenter")


class SAM2Segmenter:
    """
    SAM-2 wrapper for instance segmentation.
    Uses the Hiera architecture for best quality.
    
    Features:
    - High-quality instance masks
    - Better temporal consistency for video
    - Memory-efficient inference
    """
    
    # Checkpoint configurations
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
    
    def __init__(self, model_size: str = "tiny"):
        """
        Initialize SAM-2 segmenter.
        
        Args:
            model_size: One of 'tiny', 'small', 'base_plus', 'large'
                       Default is 'tiny' for VRAM efficiency.
        """
        if model_size not in self.CHECKPOINT_MAP:
            raise ValueError(f"Unknown model size: {model_size}. "
                           f"Choose from: {list(self.CHECKPOINT_MAP.keys())}")
        
        self.model_size = model_size
        self.device = get_device()
        self.model = None
        self.mask_generator = None
        self.predictor = None
        
        # Checkpoint paths
        self.checkpoint_dir = Path("checkpoints/sam2")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SAM-2 Segmenter initialized (size={model_size})")
    
    def _get_checkpoint_path(self) -> Path:
        """Get path to checkpoint file."""
        config = self.CHECKPOINT_MAP[self.model_size]
        return self.checkpoint_dir / config["filename"]
    
    def _check_checkpoint_exists(self) -> bool:
        """Check if checkpoint file exists."""
        checkpoint_path = self._get_checkpoint_path()
        return checkpoint_path.exists()
    
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
            
            # Try importing sam2 package
            try:
                from sam2.build_sam import build_sam2
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError:
                # Fallback to alternative package name
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
            
            # Build the model
            self.model = build_sam2(
                config_file=config["config"],
                ckpt_path=str(checkpoint_path),
                device=str(self.device)
            )
            
            # Create automatic mask generator
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=100,
            )
            
            # Create predictor for point/box prompts
            self.predictor = SAM2ImagePredictor(self.model)
            
            log_gpu_memory(logger, "SAM-2 Loaded")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM-2 model: {e}")
    
    @oom_safe(logger)
    def segment_frame(self, image: Image.Image) -> List[Dict]:
        """
        Generate instance masks for a single frame.
        
        Args:
            image: PIL Image to segment
            
        Returns:
            List of dicts with keys:
            - 'segmentation': np.ndarray boolean mask
            - 'area': int, mask area in pixels
            - 'bbox': [x, y, w, h] bounding box
            - 'predicted_iou': float, model confidence
            - 'stability_score': float, mask stability
        """
        if self.model is None:
            self.load_model()
        
        # Convert PIL -> NumPy RGB
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            # Grayscale -> RGB
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 4:
            # RGBA -> RGB
            image_np = image_np[:, :, :3]
        
        # Generate masks
        with torch.no_grad():
            masks = self.mask_generator.generate(image_np)
        
        logger.debug(f"Generated {len(masks)} masks")
        return masks
    
    @oom_safe(logger)
    def segment_with_points(
        self, 
        image: Image.Image,
        points: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Segment with point prompts (for interactive refinement).
        
        Args:
            image: PIL Image
            points: Nx2 array of point coordinates
            labels: N array of labels (1=foreground, 0=background)
            
        Returns:
            Tuple of (mask, iou_score)
        """
        if self.model is None:
            self.load_model()
        
        image_np = np.array(image)
        
        with torch.no_grad():
            self.predictor.set_image(image_np)
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False
            )
        
        return masks[0], scores[0]
    
    def unload_model(self):
        """Free GPU resources."""
        if self.model is not None:
            del self.mask_generator
            del self.predictor
            del self.model
            self.model = None
            self.mask_generator = None
            self.predictor = None
            device_manager.empty_cache()
            logger.info("SAM-2 model unloaded")


# Global instance (uses tiny for VRAM efficiency)
sam2_segmenter = SAM2Segmenter(model_size="tiny")
