"""
Module (e): Reconstruction + Stabilization
Handles latent decoding, chrominance preservation, and temporal reassembly.
"""

import cv2
import numpy as np
from PIL import Image
import torch

class ReconstructionStabilization:
    def __init__(self, sdxl_service):
        self.sdxl_service = sdxl_service
    
    def decode_latent(self, latent):
        """
        Sub-module (e1): Latent Decoder
        Decode latent to RGB image using SDXL VAE.
        
        Args:
            latent (torch.Tensor): Latent representation
        
        Returns:
            PIL.Image: Decoded image
        """
        return self.sdxl_service.decode_from_latent(latent)
    
    def preserve_chrominance(self, original_grayscale, colorized):
        """
        Sub-module (e2): Chrominance Preservation
        Preserve luminance from original while keeping chrominance from colorized.
        
        Args:
            original_grayscale (np.ndarray or PIL.Image): Original grayscale
            colorized (np.ndarray or PIL.Image): Colorized output
        
        Returns:
            np.ndarray: Chrominance-preserved image
        """
        # Convert to numpy if PIL
        if isinstance(original_grayscale, Image.Image):
            original_grayscale = np.array(original_grayscale)
        if isinstance(colorized, Image.Image):
            colorized = np.array(colorized)
        
        # Ensure same size
        if original_grayscale.shape[:2] != colorized.shape[:2]:
            colorized = cv2.resize(colorized, 
                                  (original_grayscale.shape[1], original_grayscale.shape[0]))
        
        # Convert to LAB color space
        if len(original_grayscale.shape) == 2:
            original_grayscale = cv2.cvtColor(original_grayscale, cv2.COLOR_GRAY2BGR)
        
        original_lab = cv2.cvtColor(original_grayscale, cv2.COLOR_BGR2LAB)
        colorized_lab = cv2.cvtColor(colorized, cv2.COLOR_RGB2LAB)
        
        # Replace L channel with original
        result_lab = colorized_lab.copy()
        result_lab[:, :, 0] = original_lab[:, :, 0]
        
        # Convert back to RGB
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def apply_color_correction(self, image):
        """
        Apply color correction and enhancement.
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            np.ndarray: Color-corrected image
        """
        # Apply CLAHE for better contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return result
    
    def temporal_reassembly(self, frames, temporal_coherence_module):
        """
        Sub-module (e3): Temporal Reassembly
        Apply final temporal smoothing across all frames.
        
        Args:
            frames (list): List of colorized frames
            temporal_coherence_module: TemporalCoherence instance
        
        Returns:
            list: Smoothed frames
        """
        if len(frames) < 2:
            return frames
        
        smoothed_frames = [frames[0]]
        
        for i in range(1, len(frames)):
            smoothed = temporal_coherence_module.apply_temporal_smoothing(
                frames[i],
                smoothed_frames[-1],
                alpha=0.15  # Light smoothing
            )
            smoothed_frames.append(smoothed)
        
        return smoothed_frames
    
    def reduce_flicker(self, frames, window_size=3):
        """
        Reduce temporal flicker using moving average.
        
        Args:
            frames (list): List of frames
            window_size (int): Temporal window size
        
        Returns:
            list: Flicker-reduced frames
        """
        if len(frames) < window_size:
            return frames
        
        filtered_frames = []
        half_window = window_size // 2
        
        for i in range(len(frames)):
            start = max(0, i - half_window)
            end = min(len(frames), i + half_window + 1)
            
            # Average frames in window
            window_frames = frames[start:end]
            averaged = np.mean(window_frames, axis=0).astype(np.uint8)
            filtered_frames.append(averaged)
        
        return filtered_frames
