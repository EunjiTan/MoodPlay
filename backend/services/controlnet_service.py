import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import os

class ControlNetService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.controlnet_id = "lllyasviel/sd-controlnet-canny"
        self.model_loaded = False

    def load_model(self):
        """Lazy load the diffusion model."""
        if self.model_loaded:
            return

        print(f"Loading ControlNet: {self.controlnet_id}...")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        print(f"Loading SD Pipeline: {self.model_id}...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id, 
            controlnet=controlnet, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload() # Saves VRAM
            
        self.model_loaded = True
        print("ControlNet Pipeline Loaded.")

    def process_frame(self, image: Image.Image, prompt: str, canny_lower=100, canny_upper=200):
        if not self.model_loaded:
            self.load_model()
            
        # Canny edge detection
        image_np = np.array(image)
        # Convert to grayscale if needed for canny
        if len(image_np.shape) == 3:
            gray = 0.299 * image_np[:,:,0] + 0.587 * image_np[:,:,1] + 0.114 * image_np[:,:,2]
            gray = gray.astype(np.uint8)
        else:
            gray = image_np
            
        import cv2
        edges = cv2.Canny(gray, canny_lower, canny_upper)
        edges = np.stack([edges, edges, edges], axis=2)
        control_image = Image.fromarray(edges)
        
        # Generation
        output = self.pipe(
            prompt,
            image=control_image,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
        
        return output

# Global Instance
controlnet_service = ControlNetService()
