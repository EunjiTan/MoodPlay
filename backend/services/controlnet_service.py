"""
ControlNet Service for Structural Guidance
Extracts Canny edges and depth maps for conditioning SDXL.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from pathlib import Path

class ControlNetService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.controlnet_canny = None
        self.controlnet_depth = None
        self.model_loaded = False
        
        # Paths
        base_dir = Path(__file__).parent.parent.parent
        self.canny_path = base_dir / "checkpoints" / "controlnet" / "canny"
        self.depth_path = base_dir / "checkpoints" / "controlnet" / "depth"
    
    def load_models(self):
        """Load ControlNet models."""
        if self.model_loaded:
            return
        
        try:
            print(f"Loading ControlNet models on {self.device}...")
            
            # Load Canny ControlNet
            if self.canny_path.exists():
                print(f"Loading Canny ControlNet from {self.canny_path}...")
                self.controlnet_canny = ControlNetModel.from_pretrained(
                    str(self.canny_path),
                    torch_dtype=self.dtype
                ).to(self.device)
            
            # Load Depth ControlNet
            if self.depth_path.exists():
                print(f"Loading Depth ControlNet from {self.depth_path}...")
                self.controlnet_depth = ControlNetModel.from_pretrained(
                    str(self.depth_path),
                    torch_dtype=self.dtype
                ).to(self.device)
            
            self.model_loaded = True
            print("✓ ControlNet models loaded successfully!")
            
        except Exception as e:
            print(f"✗ Error loading ControlNet: {e}")
            raise
    
    def extract_canny(self, image, low_threshold=100, high_threshold=200):
        """
        Extract Canny edges from image.
        Args:
            image (PIL.Image or np.ndarray): Input image
            low_threshold (int): Canny low threshold
            high_threshold (int): Canny high threshold
        Returns:
            PIL.Image: Canny edge map
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Canny
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert to 3-channel for ControlNet
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(edges)
    
    def extract_depth(self, image):
        """
        Extract depth map from image using MiDaS.
        Args:
            image (PIL.Image): Input image
        Returns:
            PIL.Image: Depth map
        """
        try:
            # Use transformers MiDaS for depth estimation
            from transformers import pipeline
            
            depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
            depth = depth_estimator(image)["depth"]
            
            # Normalize to 0-255
            depth_array = np.array(depth)
            depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
            depth_array = (depth_array * 255).astype(np.uint8)
            
            # Convert to 3-channel
            depth_rgb = cv2.cvtColor(depth_array, cv2.COLOR_GRAY2RGB)
            
            return Image.fromarray(depth_rgb)
            
        except Exception as e:
            print(f"⚠ Depth estimation failed: {e}, using simple gradient")
            # Fallback: simple gradient-based depth
            img_array = np.array(image.convert('L'))
            depth = cv2.Sobel(img_array, cv2.CV_64F, 1, 1, ksize=5)
            depth = np.abs(depth)
            depth = (depth / depth.max() * 255).astype(np.uint8)
            depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(depth_rgb)
    
    def get_controlnet_conditioning(self, image, use_canny=True, use_depth=False):
        """
        Get ControlNet conditioning for an image.
        Args:
            image (PIL.Image): Input image
            use_canny (bool): Extract Canny edges
            use_depth (bool): Extract depth map
        Returns:
            dict: Conditioning maps
        """
        conditioning = {}
        
        if use_canny:
            conditioning['canny'] = self.extract_canny(image)
        
        if use_depth:
            conditioning['depth'] = self.extract_depth(image)
        
        return conditioning
    
    def unload_models(self):
        """Unload models from GPU."""
        if self.controlnet_canny:
            self.controlnet_canny.to("cpu")
        if self.controlnet_depth:
            self.controlnet_depth.to("cpu")
        torch.cuda.empty_cache()
        print("✓ ControlNet unloaded from GPU")

# Global instance
controlnet_service = ControlNetService()
