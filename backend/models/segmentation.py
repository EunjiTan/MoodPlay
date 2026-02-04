"""
Segmentation Module
Wraps SAM (Segment Anything Model) for instance segmentation with OOM safety.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from huggingface_hub import hf_hub_download

from backend.core.logging import get_logger, log_gpu_memory, time_execution
from backend.core.device import get_device, get_dtype, oom_safe, device_manager
from backend.core.exceptions import ModelLoadError, ResourceError

logger = get_logger("models.segmentation")

class Segmenter:
    """Wrapper for SAM to perform instance segmentation."""
    
    def __init__(self, model_type: str = "vit_b"):
        self.model_type = model_type
        self.device = get_device()
        self.model = None
        self.mask_generator = None
        
        # Checkpoint paths
        self.checkpoint_dir = Path("checkpoints/sam")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_map = {
            "vit_b": "sam_vit_b_01ec64.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_h": "sam_vit_h_4b8939.pth"
        }
        
    def _download_checkpoint(self):
        """Download SAM checkpoint if missing."""
        filename = self.checkpoint_map.get(self.model_type)
        if not filename:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        local_path = self.checkpoint_dir / filename
        if local_path.exists():
            return local_path
            
        logger.info(f"Downloading SAM {self.model_type} checkpoint...")
        try:
            # Note: SAM checkpoints are usually manual downloads, but we can try HF mirror 
            # or raise error if not found. For now, assuming manual/script setup or HF mirror.
            # Using a known HF repo for SAM weights
            path = hf_hub_download(
                repo_id="ybelkada/segment-anything", 
                filename=f"checkpoints/{filename}",
                local_dir=str(self.checkpoint_dir)
            )
            return Path(path)
        except Exception as e:
            # Fallback to direct URL download logic could go here
            raise ModelLoadError(f"Failed to download SAM checkpoint: {e}")

    @time_execution(logger)
    def load_model(self):
        """Load SAM model to GPU."""
        if self.model is not None:
            return

        checkpoint_path = self._download_checkpoint()
        
        try:
            logger.info(f"Loading SAM {self.model_type}...")
            self.model = sam_model_registry[self.model_type](checkpoint=str(checkpoint_path))
            self.model.to(device=self.device)
            
            # Create mask generator with reasonable defaults
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=100,
            )
            log_gpu_memory(logger, "SAM Loaded")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM model: {e}")

    @oom_safe(logger)
    def segment_frame(self, image: Image.Image) -> list:
        """
        Generate masks for a single frame.
        Returns list of dicts: {'segmentation': np.array, 'area': int, 'bbox': list, ...}
        """
        if self.model is None:
            self.load_model()
            
        # Convert PIL -> NumPy
        image_np = np.array(image)
        
        # Generate masks
        with torch.no_grad():
            masks = self.mask_generator.generate(image_np)
            
        logger.debug(f"Generated {len(masks)} masks")
        return masks

    def unload_model(self):
        """Free GPU resources."""
        if self.model is not None:
            del self.mask_generator
            del self.model
            self.model = None
            self.mask_generator = None
            device_manager.empty_cache()
            logger.info("SAM model unloaded")

# Global instance
segmenter = Segmenter()
