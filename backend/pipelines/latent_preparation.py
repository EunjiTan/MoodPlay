"""
Module (b): Latent Preparation Network
Handles VAE encoding and diffusion noise initialization.
"""

import torch
import numpy as np
from PIL import Image
import cv2

class LatentPreparation:
    def __init__(self, sdxl_service):
        self.sdxl_service = sdxl_service
        self.device = sdxl_service.device
        self.dtype = sdxl_service.dtype
    
    def encode_to_latent(self, grayscale_frame):
        """
        Sub-module (b1): VAE Encoder
        Encode grayscale frame to latent space.
        
        Args:
            grayscale_frame (np.ndarray or PIL.Image): Grayscale input
        
        Returns:
            torch.Tensor: Latent representation
        """
        # Convert to PIL if numpy
        if isinstance(grayscale_frame, np.ndarray):
            # If BGR, convert to RGB
            if len(grayscale_frame.shape) == 3:
                grayscale_frame = cv2.cvtColor(grayscale_frame, cv2.COLOR_BGR2RGB)
            grayscale_frame = Image.fromarray(grayscale_frame)
        
        # Convert grayscale to RGB (3 channels)
        if grayscale_frame.mode != 'RGB':
            grayscale_frame = grayscale_frame.convert('RGB')
        
        # Encode using SDXL VAE
        latent = self.sdxl_service.encode_to_latent(grayscale_frame)
        
        return latent
    
    def initialize_noise(self, latent_shape, strategy='anchored', grayscale_latent=None):
        """
        Sub-module (b2): Diffusion Initialization
        Initialize noise for diffusion process.
        
        Args:
            latent_shape (tuple): Shape of latent tensor
            strategy (str): 'anchored', 'video-first', or 'gaussian'
            grayscale_latent (torch.Tensor): Optional grayscale latent for anchoring
        
        Returns:
            torch.Tensor: Initialized noise
        """
        if strategy == 'anchored' and grayscale_latent is not None:
            # Anchored noise: blend structure from grayscale with random noise
            noise = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
            # Mix 30% structure, 70% noise
            anchored_noise = 0.3 * grayscale_latent + 0.7 * noise
            return anchored_noise
        
        elif strategy == 'video-first':
            # Temporally consistent noise (same seed for similar frames)
            # This helps with temporal coherence
            generator = torch.Generator(device=self.device).manual_seed(42)
            noise = torch.randn(
                latent_shape,
                generator=generator,
                device=self.device,
                dtype=self.dtype
            )
            return noise
        
        else:  # 'gaussian'
            # Standard Gaussian noise
            noise = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
            return noise
    
    def prepare_latent_for_diffusion(self, grayscale_frame, noise_strategy='anchored'):
        """
        Complete latent preparation pipeline.
        
        Args:
            grayscale_frame (np.ndarray or PIL.Image): Input frame
            noise_strategy (str): Noise initialization strategy
        
        Returns:
            dict: {
                'latent': Encoded latent,
                'noise': Initialized noise,
                'shape': Latent shape
            }
        """
        # Encode to latent
        latent = self.encode_to_latent(grayscale_frame)
        
        # Initialize noise
        noise = self.initialize_noise(
            latent.shape,
            strategy=noise_strategy,
            grayscale_latent=latent
        )
        
        return {
            'latent': latent,
            'noise': noise,
            'shape': latent.shape
        }
