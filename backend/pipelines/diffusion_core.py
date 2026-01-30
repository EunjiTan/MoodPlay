"""
Module (d): Conditional Latent Diffusion Core
Implements the iterative denoising loop with SDXL and ControlNet.
"""

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler

class DiffusionCore:
    def __init__(self, sdxl_service, controlnet_service, lora_service):
        self.sdxl_service = sdxl_service
        self.controlnet_service = controlnet_service
        self.lora_service = lora_service
        
        self.device = sdxl_service.device
        self.dtype = sdxl_service.dtype
        
        self.controlnet_pipeline = None
    
    def load_controlnet_pipeline(self):
        """Load SDXL + ControlNet integrated pipeline."""
        if self.controlnet_pipeline:
            return
        
        try:
            print("Loading SDXL + ControlNet pipeline...")
            
            # Load ControlNet models
            self.controlnet_service.load_models()
            
            # Create multi-ControlNet pipeline
            controlnets = []
            if self.controlnet_service.controlnet_canny:
                controlnets.append(self.controlnet_service.controlnet_canny)
            if self.controlnet_service.controlnet_depth:
                controlnets.append(self.controlnet_service.controlnet_depth)
            
            if not controlnets:
                print("⚠ No ControlNet models loaded, using base SDXL")
                self.controlnet_pipeline = self.sdxl_service.pipeline
                return
            
            # Create ControlNet pipeline
            from pathlib import Path
            base_dir = Path(__file__).parent.parent.parent
            sdxl_path = base_dir / "checkpoints" / "sdxl"
            
            self.controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                str(sdxl_path),
                controlnet=controlnets,
                torch_dtype=self.dtype,
                use_safetensors=True,
            )
            
            self.controlnet_pipeline.to(self.device)
            self.controlnet_pipeline.enable_attention_slicing()
            self.controlnet_pipeline.enable_vae_slicing()
            
            # Use efficient scheduler
            self.controlnet_pipeline.scheduler = DDIMScheduler.from_config(
                self.controlnet_pipeline.scheduler.config
            )
            
            print("✓ ControlNet pipeline loaded!")
            
        except Exception as e:
            print(f"✗ Error loading ControlNet pipeline: {e}")
            # Fallback to base SDXL
            self.controlnet_pipeline = self.sdxl_service.pipeline
    
    def denoise_step(self, latent, timestep, conditioning_bundle):
        """
        Single denoising step.
        
        Args:
            latent (torch.Tensor): Current noisy latent
            timestep (int): Current timestep
            conditioning_bundle (dict): Conditioning data
        
        Returns:
            torch.Tensor: Denoised latent
        """
        # This would implement a single U-Net forward pass
        # For now, we'll use the full pipeline
        return latent
    
    def colorize_frame(self, grayscale_image, conditioning_bundle, num_steps=20, guidance_scale=7.5, controlnet_scale=0.8):
        """
        Colorize a single frame using conditional latent diffusion.
        
        Args:
            grayscale_image (PIL.Image): Grayscale input
            conditioning_bundle (dict): Complete conditioning from GuidanceFusion
            num_steps (int): Number of denoising steps
            guidance_scale (float): CFG scale
            controlnet_scale (float): ControlNet conditioning strength
        
        Returns:
            PIL.Image: Colorized frame
        """
        if not self.controlnet_pipeline:
            self.load_controlnet_pipeline()
        
        # Extract conditioning components
        prompt = conditioning_bundle['text_prompt']
        controlnet_inputs = conditioning_bundle['controlnet_inputs']
        
        # Prepare ControlNet images
        control_images = []
        if controlnet_inputs.get('canny_image'):
            control_images.append(controlnet_inputs['canny_image'])
        if controlnet_inputs.get('depth_image'):
            control_images.append(controlnet_inputs['depth_image'])
        
        # Resize grayscale to target size
        target_size = (512, 512)  # Start with 512 for RTX 3050
        grayscale_resized = grayscale_image.resize(target_size)
        
        # Generate with ControlNet guidance
        with torch.no_grad():
            if control_images:
                # Resize control images
                control_images = [img.resize(target_size) for img in control_images]
                
                result = self.controlnet_pipeline(
                    prompt=prompt,
                    image=control_images,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_scale,
                    height=target_size[1],
                    width=target_size[0],
                ).images[0]
            else:
                # Fallback to base SDXL
                result = self.sdxl_service.colorize_frame(
                    grayscale_resized,
                    prompt,
                    num_steps,
                    guidance_scale
                )
        
        return result
    
    def apply_lora(self, lora_name, weight=1.0):
        """Apply LoRA to the pipeline."""
        if self.controlnet_pipeline:
            self.lora_service.load_lora(
                self.controlnet_pipeline,
                lora_name,
                weight=weight
            )
    
    def unload_models(self):
        """Unload models from GPU."""
        if self.controlnet_pipeline:
            self.controlnet_pipeline.to("cpu")
        torch.cuda.empty_cache()
