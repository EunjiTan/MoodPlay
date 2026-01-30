"""
Module (d-continued): Temporal Coherence Module
Ensures color consistency across video frames.
"""

import cv2
import numpy as np
import torch
from PIL import Image

class TemporalCoherence:
    def __init__(self):
        self.prev_frame = None
        self.prev_colorized = None
        self.flow_cache = {}
        self.color_memory = {}  # {track_id: average_color}
    
    def compute_optical_flow(self, frame1, frame2):
        """
        Compute optical flow between two frames.
        
        Args:
            frame1 (np.ndarray): Previous frame (grayscale or BGR)
            frame2 (np.ndarray): Current frame (grayscale or BGR)
        
        Returns:
            np.ndarray: Optical flow field
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
        
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2
        
        # Compute dense optical flow using Farneback
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
    
    def warp_with_optical_flow(self, image, flow):
        """
        Warp image using optical flow.
        
        Args:
            image (np.ndarray): Image to warp (H, W, C)
            flow (np.ndarray): Optical flow field (H, W, 2)
        
        Returns:
            np.ndarray: Warped image
        """
        h, w = flow.shape[:2]
        
        # Create coordinate grid
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Apply flow
        x_new = x + flow[..., 0]
        y_new = y + flow[..., 1]
        
        # Remap
        warped = cv2.remap(
            image,
            x_new, y_new,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return warped
    
    def propagate_colors(self, current_frame, prev_colorized_frame, blend_weight=0.3):
        """
        Propagate colors from previous frame using optical flow.
        
        Args:
            current_frame (np.ndarray): Current grayscale frame
            prev_colorized_frame (np.ndarray): Previous colorized frame
            blend_weight (float): Weight for temporal blending
        
        Returns:
            np.ndarray: Color-propagated frame (as guidance)
        """
        if self.prev_frame is None:
            return None
        
        # Compute optical flow
        flow = self.compute_optical_flow(self.prev_frame, current_frame)
        
        # Warp previous colorized frame
        warped_color = self.warp_with_optical_flow(prev_colorized_frame, flow)
        
        return warped_color
    
    def update_color_memory(self, track_id, colorized_frame, mask):
        """
        Update color memory for a tracked object.
        
        Args:
            track_id (int): Object track ID
            colorized_frame (np.ndarray): Colorized frame
            mask (np.ndarray): Object mask
        """
        # Extract average color within mask
        if mask.sum() > 0:
            masked_region = colorized_frame[mask > 0]
            avg_color = masked_region.mean(axis=0)
            
            # Update memory with exponential moving average
            if track_id in self.color_memory:
                self.color_memory[track_id] = 0.7 * self.color_memory[track_id] + 0.3 * avg_color
            else:
                self.color_memory[track_id] = avg_color
    
    def get_color_hint_from_memory(self, track_id):
        """Get stored color for an object."""
        return self.color_memory.get(track_id)
    
    def apply_temporal_smoothing(self, current_colorized, prev_colorized, alpha=0.2):
        """
        Apply temporal smoothing to reduce flicker.
        
        Args:
            current_colorized (np.ndarray): Current frame
            prev_colorized (np.ndarray): Previous frame
            alpha (float): Smoothing factor
        
        Returns:
            np.ndarray: Smoothed frame
        """
        if prev_colorized is None:
            return current_colorized
        
        # Ensure same size
        if current_colorized.shape != prev_colorized.shape:
            prev_colorized = cv2.resize(prev_colorized, 
                                       (current_colorized.shape[1], current_colorized.shape[0]))
        
        # Blend
        smoothed = cv2.addWeighted(
            current_colorized, 1 - alpha,
            prev_colorized, alpha,
            0
        )
        
        return smoothed
    
    def update_frame_history(self, grayscale_frame, colorized_frame):
        """Update frame history for next iteration."""
        self.prev_frame = grayscale_frame.copy()
        self.prev_colorized = colorized_frame.copy()
