"""
SD1.5 Diffusion Runner Module
Core generative engine combining SD1.5 + ControlNet + LoRA with OOM safety.
Updated with:
- DPMSolver scheduler for better denoising
- Built-in denoising strength parameter
- Bilateral filter post-processing
- 4-color palette guidance integration
"""

import torch
import gc
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    DPMSolverMultistepScheduler
)
from pathlib import Path
from typing import Optional, List, Dict, Union

from backend.core.logging import get_logger, log_gpu_memory, time_execution
from backend.core.device import get_device, get_dtype, oom_safe, device_manager
from backend.core.exceptions import ModelLoadError, OutOfMemoryError, ResourceError

logger = get_logger("diffusion.sd15_runner")


class SD15Runner:
    """
    Handles the Stable Diffusion 1.5 generation pipeline.
    
    Updated features:
    - DPMSolverMultistepScheduler (better denoising than UniPC)
    - Adjustable denoise_strength parameter
    - Bilateral filter post-processing
    - Optional Non-Local Means denoising
    - Tuned defaults for photorealistic output
    """
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.device = get_device()
        self.dtype = get_dtype()
        self.pipe = None
        self.controlnet = None
        
        # Paths
        self.lora_dir = Path("checkpoints/lora")
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_controlnet(self):
        """Load ControlNet model."""
        try:
            logger.info("Loading ControlNet (Canny)...")
            self.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=self.dtype
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load ControlNet: {e}")

    @time_execution(logger)
    def load_pipeline(self):
        """Load the full SD1.5 + ControlNet pipeline with DPMSolver."""
        if self.pipe is not None:
            return

        if self.controlnet is None:
            self._load_controlnet()

        try:
            logger.info(f"Loading SD1.5 Pipeline from {self.model_id}...")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=self.controlnet,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Use DPMSolver for better denoising
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                solver_order=2,
                use_karras_sigmas=True
            )
            
            # Move to device
            self.pipe.to(self.device)
            
            # Memory optimizations
            self.pipe.enable_attention_slicing()
            
            if self.device.type == "cuda":
                self.pipe.enable_model_cpu_offload()
                
            log_gpu_memory(logger, "Pipeline Loaded with DPMSolver")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load SD1.5 pipeline: {e}")

    def load_lora(self, lora_path: str, weight: float = 0.5):
        """
        Load a LoRA adapter.
        
        Args:
            lora_path: Path to LoRA .safetensors file
            weight: LoRA weight (0.5 recommended for palette guidance)
        """
        if self.pipe is None:
            self.load_pipeline()
            
        try:
            logger.info(f"Loading LoRA from {lora_path} (scale={weight})...")
            self.pipe.load_lora_weights(lora_path)
            logger.info("LoRA loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load LoRA {lora_path}: {e}")
            # Non-fatal, proceed without LoRA
    
    def _apply_bilateral_filter(
        self, 
        image: Image.Image, 
        d: int = 9, 
        sigma_color: float = 75, 
        sigma_space: float = 75
    ) -> Image.Image:
        """
        Apply bilateral filter for edge-preserving smoothing.
        Removes noise while preserving edges.
        """
        img_np = np.array(image)
        filtered = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)
        return Image.fromarray(filtered)
    
    def _apply_nlm_denoising(
        self, 
        image: Image.Image, 
        h: float = 10, 
        template_size: int = 7, 
        search_size: int = 21
    ) -> Image.Image:
        """
        Apply Non-Local Means denoising for stronger artifact removal.
        """
        img_np = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(
            img_np, None, h, h, template_size, search_size
        )
        return Image.fromarray(denoised)
    
    def _postprocess(
        self, 
        image: Image.Image, 
        denoise_strength: float
    ) -> Image.Image:
        """
        Apply post-processing based on denoise_strength.
        
        Args:
            image: Input image
            denoise_strength: 0-1 (0=none, 1=maximum)
        """
        if denoise_strength <= 0:
            return image
        
        # Always apply bilateral filter (fast, edge-preserving)
        # Scale filter strength with denoise_strength
        # REDUCED SIGMA: 75 -> 25 to preserve texture ("pixel level realism")
        sigma = 15 + (denoise_strength * 30)  # 15-45 range (was 50-150)
        image = self._apply_bilateral_filter(
            image, 
            d=5,  # Reduced diameter (was 9)
            sigma_color=sigma, 
            sigma_space=sigma
        )
        
        # For stronger denoising, also apply NLM
        if denoise_strength > 0.5:
            nlm_h = 5 + (denoise_strength - 0.5) * 20  # 5-15 range
            image = self._apply_nlm_denoising(image, h=nlm_h)
        
        return image

    @oom_safe(logger)
    def generate_frame(
        self, 
        image: Image.Image, 
        prompt: str, 
        negative_prompt: str = "", 
        num_inference_steps: int = 50,      # Increased from 40
        guidance_scale: float = 5.5,        # Lowered for realism
        controlnet_scale: float = 0.95,     # High for structure
        lora_scale: float = 0.5,            # Moderate for palette
        denoise_strength: float = 0.3,      # Post-processing denoising
        seed: int = None
    ) -> Image.Image:
        """
        Generate a single colored frame with denoising.
        
        Args:
            image: ControlNet input (Canny edge map)
            prompt: Positive prompt
            negative_prompt: Negative prompt
            num_inference_steps: Diffusion steps (50 recommended)
            guidance_scale: CFG scale (5.5 for realism)
            controlnet_scale: ControlNet conditioning (0.95)
            lora_scale: LoRA weight (0.5 for palette guidance)
            denoise_strength: Post-processing strength (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            Colorized PIL Image
        """
        if self.pipe is None:
            self.load_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(
            f"Gen: Steps={num_inference_steps}, CFG={guidance_scale}, "
            f"CN={controlnet_scale}, LoRA={lora_scale}, Denoise={denoise_strength}"
        )

        # Run inference
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            cross_attention_kwargs={"scale": lora_scale},
            generator=generator,
            output_type="pil"
        ).images[0]
        
        # Apply post-processing denoising
        if denoise_strength > 0:
            result = self._postprocess(result, denoise_strength)
        
        return result
    
    def generate_frame_with_palette(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        palette_image: Image.Image = None,
        **kwargs
    ) -> Image.Image:
        """
        Generate frame with optional palette conditioning image.
        Currently just calls generate_frame, but can be extended
        for IP-Adapter style palette conditioning in the future.
        
        Args:
            image: ControlNet input
            prompt: Positive prompt
            negative_prompt: Negative prompt
            palette_image: 4-color palette conditioning image (future use)
            **kwargs: Additional args passed to generate_frame
        """
        # For now, standard generation (LoRA handles palette guidance)
        # Future: integrate IP-Adapter for direct palette conditioning
        return self.generate_frame(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            **kwargs
        )

    def unload_pipeline(self):
        """Unload all models to free VRAM."""
        if self.pipe:
            del self.pipe
            self.pipe = None
        if self.controlnet:
            del self.controlnet
            self.controlnet = None
        
        device_manager.empty_cache()
        logger.info("SD1.5 Pipeline unloaded")


# Global instance
sd15_runner = SD15Runner()
