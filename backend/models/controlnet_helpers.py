"""
ControlNet Helpers
Utilities for preparing ControlNet inputs (Canny, Depth, etc).
"""

import cv2
import numpy as np
from PIL import Image

def extract_canny(image: Image.Image, low_threshold=50, high_threshold=150) -> Image.Image:
    """
    Extract Canny edges from a PIL Image.
    Returns RGB PIL Image suitable for ControlNet input.
    """
    image_np = np.array(image)
    
    # Convert to gray
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # Noise reduction
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Canny
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Convert back to RGB (ControlNet expects 3 channels)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(edges_rgb)
