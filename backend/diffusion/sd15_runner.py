"""
SD1.5 Diffusion Runner Module
Core generative engine combining SD1.5 + ControlNet + LoRA with OOM safety.
"""

import torch
import gc
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from pathlib import Path
from typing import Optional, List, Dict, Union

from backend.core.logging import get_logger, log_gpu_memory, time_execution
from backend.core.device import get_device, get_dtype, oom_safe, device_manager
from backend.core.exceptions import ModelLoadError, OutOfMemoryError, ResourceError

logger = get_logger("diffusion.sd15_runner")

class SD15Runner:
    """
    Handles the Stable Diffusion 1.5 generation pipeline.
    Integrates ControlNet for structure and LoRA for style availability.
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
        """Load the full SD1.5 + ControlNet pipeline."""
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
                safety_checker=None, # Disable for speed/memory in research context
                requires_safety_checker=False
            )
            
            # Optimization: Scheduler
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            # Optimization: Move to device
            self.pipe.to(self.device)
            
            # Optimization: Attention Slicing (default on to save VRAM)
            self.pipe.enable_attention_slicing()
            
            # Optimization: Model Offloading (crucial for 4GB VRAM)
            # This automatically moves models to CPU when not in use
            if self.device.type == "cuda":
                self.pipe.enable_model_cpu_offload()
                
            log_gpu_memory(logger, "Pipeline Loaded")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load SD1.5 pipeline: {e}")

    def load_lora(self, lora_path: str, weight: float = 0.6):
        """Load a LoRA adapter."""
        if self.pipe is None:
            self.load_pipeline()
            
        try:
            logger.info(f"Loading LoRA from {lora_path} (scale={weight})...")
            # Note: diffusers LoRA loading might vary by version. 
            # safe assumption: standard .safetensors load
            
            self.pipe.load_lora_weights(lora_path)
            
            # Fuse/Unfuse can be complex. For simplicity in this architecture, 
            # we rely on cross_attention_kwargs={"scale": ...} during inference
            # which works best for dynamic weight adjustment without fusing.
            
            logger.info("LoRA loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load LoRA {lora_path}: {e}")
            # Non-fatal, proceed without LoRA

    @oom_safe(logger)
    def generate_frame(
        self, 
        image: Image.Image, 
        prompt: str, 
        negative_prompt: str = "", 
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.7,
        lora_scale: float = 0.6,
        seed: int = None
    ) -> Image.Image:
        """
        Generate a single colored frame.
        """
        if self.pipe is None:
            self.load_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed) # CPU generator for reproducibility across devices

        # Run Inference
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image, # ControlNet input (Canny edge map assumed)
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            cross_attention_kwargs={"scale": lora_scale},
            generator=generator,
            output_type="pil"
        ).images[0]

        return result

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
