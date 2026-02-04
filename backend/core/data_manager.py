"""
Core Data Manager
Handles all disk I/O for the staged pipeline.
Ensures frames, masks, and flow data are saved/loaded consistently.
"""

import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Optional, Union, Tuple

from backend.core.logging import get_logger

logger = get_logger("core.data")

class DataManager:
    """
    Manages the 'workspace' folder structure for a processing job.
    Structure:
      workspace/{job_id}/
          0_raw_frames/      (Full resolution)
          1_process_frames/  (Downscaled 512p)
          2_masks/           (SAM output)
          3_flow/            (CoTracker output)
          4_keyframes/       (SD1.5 colorized)
          5_final/           (Propagated & Upscaled)
    """
    
    STAGES = {
        "raw": "0_raw_frames",
        "proc": "1_process_frames",
        "masks": "2_masks",
        "flow": "3_flow",
        "keys": "4_keyframes",
        "final": "5_final"
    }

    def __init__(self, base_dir: str = "workspace", job_id: str = "default"):
        self.base_dir = Path(base_dir) / job_id
        self.dirs = {k: self.base_dir / v for k, v in self.STAGES.items()}
        self._init_dirs()

    def _init_dirs(self):
        """Create all necessary directories."""
        for p in self.dirs.values():
            p.mkdir(parents=True, exist_ok=True)
            
    def clear_workspace(self):
        """DANGER: Clear all data for this job."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        self._init_dirs()

    # --- Frame I/O ---
    
    def save_frame(self, image: Union[Image.Image, np.ndarray], idx: int, stage: str = "proc"):
        """Save a frame image."""
        path = self.dirs[stage] / f"{idx:06d}.png"
        
        if isinstance(image, Image.Image):
            image.save(path)
        else:
            # Assume BGR numpy if saving via cv2, or convert RGB
            # Standardizing on RGB PIL usage primarily, but handling numpy
            if len(image.shape) == 3 and image.shape[2] == 3:
                 # Check if likely BGR (OpenCV default) -> RGB
                 # For safety in this pipeline, let's assume inputs to save_frame are consistent.
                 # Let's enforce PIL for internal consistency if possible, or simple cv2 write
                 cv2.imwrite(str(path), image)

    def load_frame(self, idx: int, stage: str = "proc") -> Image.Image:
        """Load a frame as PIL Image."""
        path = self.dirs[stage] / f"{idx:06d}.png"
        if not path.exists():
            raise FileNotFoundError(f"Frame {idx} not found in {stage}")
        return Image.open(path).convert("RGB")

    def frame_exists(self, idx: int, stage: str = "proc") -> bool:
        return (self.dirs[stage] / f"{idx:06d}.png").exists()

    def get_frame_count(self, stage: str = "proc") -> int:
        return len(list(self.dirs[stage].glob("*.png")))

    # --- Mask I/O ---

    def save_mask(self, mask: np.ndarray, idx: int):
        """Save segmentation mask (compressed)."""
        # Save as compressed numpy to handle instance IDs (int array)
        # PNG can only do 8-bit or 16-bit easily. NPZ is safer for raw IDs.
        path = self.dirs["masks"] / f"{idx:06d}.npz"
        np.savez_compressed(path, mask=mask)

        # DEBUG: Also save a visualization png
        vis_path = self.dirs["masks"] / f"{idx:06d}_vis.png"
        # Normalize IDs to visible colors
        vis = (mask * 37 % 255).astype(np.uint8) # Pseudo-color
        cv2.imwrite(str(vis_path), vis)

    def load_mask(self, idx: int) -> np.ndarray:
        """Load mask (returns instance ID map)."""
        path = self.dirs["masks"] / f"{idx:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Mask {idx} not found")
        data = np.load(path)
        return data['mask']

    # --- Flow I/O ---

    def save_flow(self, flow: np.ndarray, idx: int):
        """Save optical flow (forward/backward)."""
        path = self.dirs["flow"] / f"{idx:06d}.npz"
        np.savez_compressed(path, flow=flow)

    def load_flow(self, idx: int) -> np.ndarray:
        path = self.dirs["flow"] / f"{idx:06d}.npz"
        if not path.exists():
             raise FileNotFoundError(f"Flow {idx} not found")
        return np.load(path)['flow']

# Global manager factory
def get_data_manager(job_id: str) -> DataManager:
    return DataManager(job_id=job_id)
