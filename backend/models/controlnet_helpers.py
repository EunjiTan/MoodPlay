"""
ControlNet Helpers
Utilities for preparing ControlNet inputs (Canny, Depth, etc).
"""

import cv2
import numpy as np
from PIL import Image

def extract_canny(image: Image.Image, low_threshold=None, high_threshold=None) -> Image.Image:
    """
    Extract Canny edges from a PIL Image.
    Uses adaptive thresholding if thresholds are not provided.
    Returns RGB PIL Image suitable for ControlNet input.
    """
    image_np = np.array(image)
    
    # Convert to gray
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # Adaptive thresholding
    if low_threshold is None or high_threshold is None:
        v = np.median(gray)
        sigma = 0.33
        low_threshold = int(max(0, (1.0 - sigma) * v))
        high_threshold = int(min(255, (1.0 + sigma) * v))
    
    # Noise reduction (lighter than before)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Canny
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Convert back to RGB (ControlNet expects 3 channels)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(edges_rgb)
