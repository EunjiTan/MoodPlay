"""
SDXL Service for Video Colorization
Handles latent diffusion, VAE encoding/decoding, and denoising.
"""

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from pathlib import Path
import os

class SDXLService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.pipeline = None
        self.vae = None
        self.model_loaded = False
        self.load_error = None
        
        # Paths
        base_dir = Path(__file__).parent.parent.parent
        self.sdxl_path = base_dir / "checkpoints" / "sdxl"
        self.vae_path = base_dir / "checkpoints" / "vae"
    
    def load_models(self):
        """Load SDXL pipeline and VAE with memory optimizations."""
        if self.model_loaded:
            return
        
        try:
            print(f"Loading SDXL on {self.device} with {self.dtype}...")
            
            # Load VAE first (FP16 fix for better quality)
            if self.vae_path.exists():
                print(f"Loading VAE from {self.vae_path}...")
                self.vae = AutoencoderKL.from_pretrained(
                    str(self.vae_path),
                    torch_dtype=self.dtype
                )
            
            # Load SDXL pipeline
            print(f"Loading SDXL from {self.sdxl_path}...")
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                str(self.sdxl_path),
                vae=self.vae,
                torch_dtype=self.dtype,
                use_safetensors=True,
            )
            
            # Memory optimizations for RTX 3050
            self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_vae_slicing()
            
            # Enable xformers if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("✓ xformers enabled")
            except:
                print("⚠ xformers not available, using default attention")
            
            self.model_loaded = True
            print("✓ SDXL loaded successfully!")
            
        except Exception as e:
            self.load_error = str(e)
            print(f"✗ Error loading SDXL: {e}")
            raise
    
    def encode_to_latent(self, image):
        """
        Encode image to latent space.
        Args:
            image (PIL.Image or torch.Tensor): Input image
        Returns:
            torch.Tensor: Latent representation
        """
        if not self.model_loaded:
            self.load_models()
        
        # Convert PIL to tensor if needed
        if not isinstance(image, torch.Tensor):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            image = transform(image).unsqueeze(0)
        
        image = image.to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        
        return latent
    
    def decode_from_latent(self, latent):
        """
        Decode latent to image.
        Args:
            latent (torch.Tensor): Latent representation
        Returns:
            PIL.Image: Decoded image
        """
        if not self.model_loaded:
            self.load_models()
        
        latent = latent / self.vae.config.scaling_factor
        
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype("uint8")
        
        from PIL import Image
        return Image.fromarray(image)
    
    def colorize_frame(self, grayscale_image, prompt, num_inference_steps=20, guidance_scale=7.5, controlnet_conditioning=None):
        """
        Colorize a grayscale frame using SDXL.
        Args:
            grayscale_image (PIL.Image): Grayscale input
            prompt (str): Text prompt for colorization
            num_inference_steps (int): Denoising steps
            guidance_scale (float): CFG scale
            controlnet_conditioning (dict): Optional ControlNet inputs
        Returns:
            PIL.Image: Colorized image
        """
        if not self.model_loaded:
            self.load_models()
        
        # Generate
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,  # Start with 512 for RTX 3050
                width=512,
            ).images[0]
        
        return result
    
    def unload_models(self):
        """Unload models from GPU to free memory."""
        if self.pipeline:
            self.pipeline.to("cpu")
        if self.vae:
            self.vae.to("cpu")
        torch.cuda.empty_cache()
        print("✓ SDXL unloaded from GPU")

# Global instance
sdxl_service = SDXLService()
