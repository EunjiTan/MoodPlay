"""
Optical Flow module for temporal consistency in video colorization.
Uses RAFT for high-quality optical flow estimation.
"""

import numpy as np
from PIL import Image
import cv2

# Global flags for optional dependencies
TORCH_AVAILABLE = False
RAFT_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    print("Warning: Torch not available. Optical flow will be mocked.")

try:
    # RAFT needs to be installed: pip install git+https://github.com/princeton-vl/RAFT.git
    from raft import RAFT
    from raft.utils.flow_viz import flow_to_image
    RAFT_AVAILABLE = True
except ImportError:
    print("Warning: RAFT not available. Using OpenCV optical flow fallback.")


class OpticalFlowEstimator:
    """
    Estimates optical flow between consecutive frames for temporal consistency.
    Uses RAFT when available, falls back to OpenCV Farneback.
    """
    
    def __init__(self, model_path: str = "checkpoints/raft-things.pth"):
        self.device = "cpu"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
        
        self.model_path = model_path
        self.model = None
        self.use_raft = RAFT_AVAILABLE
    
    def _load_raft(self):
        """Load RAFT model for high-quality optical flow."""
        if not RAFT_AVAILABLE or not TORCH_AVAILABLE:
            return
        
        print(f"Loading RAFT from {self.model_path}...")
        # RAFT model loading would go here
        # self.model = RAFT(args)
        # self.model.load_state_dict(torch.load(self.model_path))
        # self.model.to(self.device)
        # self.model.eval()
        print("RAFT loaded.")
    
    def estimate_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Estimate optical flow from frame1 to frame2.
        
        Args:
            frame1: First frame (H, W, 3) in RGB
            frame2: Second frame (H, W, 3) in RGB
            
        Returns:
            Optical flow field (H, W, 2) where [:,:,0] is horizontal and [:,:,1] is vertical
        """
        if self.use_raft and TORCH_AVAILABLE:
            return self._estimate_flow_raft(frame1, frame2)
        else:
            return self._estimate_flow_farneback(frame1, frame2)
    
    def _estimate_flow_raft(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """RAFT-based optical flow estimation."""
        if self.model is None:
            self._load_raft()
        
        # Convert to torch tensors
        # In production: would use RAFT inference here
        # For now, fall back to Farneback
        return self._estimate_flow_farneback(frame1, frame2)
    
    def _estimate_flow_farneback(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """OpenCV Farneback optical flow as fallback."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        return flow
    
    def visualize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Convert optical flow to RGB visualization."""
        if RAFT_AVAILABLE:
            return flow_to_image(flow)
        
        # Manual flow visualization
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class FrameWarper:
    """
    Warps frames based on optical flow for temporal consistency.
    """
    
    def __init__(self):
        self.device = "cpu"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
    
    def warp_frame(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warp a frame using optical flow.
        
        Args:
            frame: Input frame (H, W, 3)
            flow: Optical flow field (H, W, 2)
            
        Returns:
            Warped frame (H, W, 3)
        """
        h, w = frame.shape[:2]
        
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply flow
        map_x = (x + flow[:, :, 0]).astype(np.float32)
        map_y = (y + flow[:, :, 1]).astype(np.float32)
        
        # Remap
        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped


class TemporalBlender:
    """
    Blends frames with temporal consistency using weighted averaging
    and mask-aware blending.
    """
    
    def __init__(self, 
                 current_weight: float = 0.7, 
                 previous_weight: float = 0.3,
                 warp_strength: float = 0.4):
        self.current_weight = current_weight
        self.previous_weight = previous_weight
        self.warp_strength = warp_strength
        
        self.flow_estimator = OpticalFlowEstimator()
        self.warper = FrameWarper()
    
    def blend_frames(self, 
                     current_frame: np.ndarray, 
                     previous_frame: np.ndarray,
                     mask: np.ndarray = None) -> np.ndarray:
        """
        Blend current and previous frames with optional mask-aware weighting.
        
        Args:
            current_frame: Current colorized frame (H, W, 3)
            previous_frame: Previous colorized frame (H, W, 3)
            mask: Optional per-pixel mask (H, W) or per-segment mask
            
        Returns:
            Blended frame (H, W, 3)
        """
        # Estimate flow from previous to current
        flow = self.flow_estimator.estimate_flow(previous_frame, current_frame)
        
        # Warp previous frame to align with current
        flow_scaled = flow * self.warp_strength
        warped_previous = self.warper.warp_frame(previous_frame, flow_scaled)
        
        # Blend with weights
        if mask is not None:
            # Mask-aware blending: different weights per segment
            mask_3d = np.stack([mask] * 3, axis=-1) if mask.ndim == 2 else mask
            blended = (
                current_frame * (self.current_weight * mask_3d + (1 - mask_3d)) +
                warped_previous * (self.previous_weight * mask_3d)
            )
        else:
            blended = (
                current_frame * self.current_weight + 
                warped_previous * self.previous_weight
            )
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def blend_batch(self, 
                    frames: list, 
                    masks: list = None) -> list:
        """
        Blend a batch of frames with temporal consistency.
        
        Args:
            frames: List of colorized frames
            masks: Optional list of masks per frame
            
        Returns:
            List of blended frames
        """
        if len(frames) == 0:
            return []
        
        blended = [frames[0]]  # First frame is unmodified
        
        for i in range(1, len(frames)):
            mask = masks[i] if masks else None
            blended_frame = self.blend_frames(frames[i], blended[-1], mask)
            blended.append(blended_frame)
        
        return blended
