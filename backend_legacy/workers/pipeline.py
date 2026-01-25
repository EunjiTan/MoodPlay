"""
Enhanced Render Pipeline for video colorization with VideoControlNet and LoRA support.
Includes frame batching, temporal consistency, and mask-aware processing.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Generator
import tempfile

from backend.workers.colorization_config import (
    BATCH_SIZE, OVERLAP_FRAMES, CONTROLNET_WEIGHT, GUIDANCE_SCALE,
    LORA_COLORIZATION_WEIGHT, LORA_STYLE_WEIGHT,
    CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD,
    DEFAULT_PROMPT_TEMPLATE, DEFAULT_NEGATIVE_PROMPT,
    ENABLE_CPU_OFFLOAD, ENABLE_ATTENTION_SLICING, ENABLE_VAE_SLICING,
    OUTPUT_CODEC, OUTPUT_QUALITY, DEFAULT_RESOLUTION
)

# Global flags
TORCH_AVAILABLE = False
DIFFUSERS_AVAILABLE = False

import torch
TORCH_AVAILABLE = True

from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    UniPCMultistepScheduler
)
from diffusers.utils import load_image
DIFFUSERS_AVAILABLE = True


class RenderPipeline:
    """
    Advanced rendering pipeline for video colorization.
    Supports ControlNet (canny/depth), LoRA, and temporal consistency.
    """
    
    def __init__(self, 
                 sd_model_id: str = "runwayml/stable-diffusion-v1-5", 
                 controlnet_id: str = "lllyasviel/sd-controlnet-canny"):
        self.device = "cpu"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
            
        self.sd_model_id = sd_model_id
        self.controlnet_id = controlnet_id
        self.pipe = None
        self.loras_loaded = []
        
        # Import temporal modules
        try:
            from backend.processing.optical_flow import TemporalBlender
            from backend.processing.temporal import TemporalConsistencyEnforcer
            self.temporal_blender = TemporalBlender()
            self.consistency_enforcer = TemporalConsistencyEnforcer()
        except ImportError:
            self.temporal_blender = None
            self.consistency_enforcer = None

    def _load_pipeline(self):
        """Load the Stable Diffusion + ControlNet pipeline."""
        if not DIFFUSERS_AVAILABLE or not TORCH_AVAILABLE:
            print("ML libraries missing. Skipping pipeline load.")
            return

        print(f"Loading ControlNet: {self.controlnet_id}...")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id, 
            torch_dtype=torch.float16
        )
        
        print(f"Loading SD Pipeline: {self.sd_model_id}...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.sd_model_id, 
            controlnet=controlnet, 
            torch_dtype=torch.float16,
            safety_checker=None  # Disable for video processing
        )
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Apply memory optimizations
        if ENABLE_CPU_OFFLOAD:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
            
        if ENABLE_ATTENTION_SLICING:
            self.pipe.enable_attention_slicing()
            
        if ENABLE_VAE_SLICING:
            self.pipe.enable_vae_slicing()
            
        print("Pipeline Loaded.")

    def load_lora(self, lora_path: str, weight: float = 1.0, adapter_name: str = None):
        """
        Load a LoRA adapter into the pipeline.
        
        Args:
            lora_path: Path to the LoRA weights file
            weight: LoRA weight/strength
            adapter_name: Optional name for the adapter
        """
        if not DIFFUSERS_AVAILABLE or not TORCH_AVAILABLE:
            print(f"Mock: Would load LoRA from {lora_path} with weight {weight}")
            return
            
        if self.pipe is None:
            self._load_pipeline()
            
        if not os.path.exists(lora_path):
            print(f"Warning: LoRA path not found: {lora_path}")
            return
            
        print(f"Loading LoRA from {lora_path} with weight {weight}")
        
        adapter_name = adapter_name or f"lora_{len(self.loras_loaded)}"
        
        self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        self.loras_loaded.append((adapter_name, weight))
        
        # If multiple LoRAs, set adapter weights
        if len(self.loras_loaded) > 1:
            adapter_names = [name for name, _ in self.loras_loaded]
            adapter_weights = [w for _, w in self.loras_loaded]
            self.pipe.set_adapters(adapter_names, adapter_weights)

    def unload_loras(self):
        """Unload all LoRA adapters."""
        if self.pipe and hasattr(self.pipe, 'unload_lora_weights'):
            self.pipe.unload_lora_weights()
            self.loras_loaded = []

    def process_frame(self, 
                      image: Image.Image, 
                      prompt: str, 
                      negative_prompt: str = None,
                      control_image: Image.Image = None, 
                      mask: np.ndarray = None,
                      strength: float = CONTROLNET_WEIGHT,
                      guidance_scale: float = GUIDANCE_SCALE,
                      num_inference_steps: int = 20) -> Image.Image:
        """
        Process a single frame with colorization.
        
        Args:
            image: Original frame
            prompt: Colorization prompt
            negative_prompt: Negative prompt
            control_image: Pre-computed control image (canny/depth)
            mask: Optional mask for selective colorization
            strength: ControlNet conditioning strength
            guidance_scale: CFG scale
            num_inference_steps: Number of diffusion steps
            
        Returns:
            Colorized frame
        """
        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
            
        if control_image is None:
            control_image = self.preprocess_canny(np.array(image))
        
        if not DIFFUSERS_AVAILABLE or not TORCH_AVAILABLE:
            # Mock processing: apply a simple color effect
            print("Mock processing frame...")
            return self._mock_colorize(image, mask)

        if self.pipe is None:
            self._load_pipeline()
        
        # Convert mask to PIL if provided
        mask_image = None
        if mask is not None:
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            
        # Generate
        output = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=strength,
            guidance_scale=guidance_scale,
        ).images[0]
        
        # If mask provided, blend original background with colorized foreground
        if mask is not None:
            output = self._apply_mask_blend(image, output, mask)
        
        return output

    def process_batch(self,
                      frames: List[np.ndarray],
                      prompts: List[str],
                      masks: List[np.ndarray] = None,
                      progress_callback=None) -> List[np.ndarray]:
        """
        Process a batch of frames together.
        
        Args:
            frames: List of frames as numpy arrays (H, W, 3)
            prompts: List of prompts (one per frame or single prompt)
            masks: Optional list of masks
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of colorized frames
        """
        if len(frames) == 0:
            return []
            
        # Expand single prompt to all frames
        if len(prompts) == 1:
            prompts = prompts * len(frames)
            
        results = []
        
        for i, frame in enumerate(frames):
            if progress_callback:
                progress_callback(i, len(frames))
                
            pil_frame = Image.fromarray(frame)
            prompt = prompts[i] if i < len(prompts) else prompts[0]
            mask = masks[i] if masks and i < len(masks) else None
            
            colorized = self.process_frame(pil_frame, prompt, mask=mask)
            results.append(np.array(colorized))
        
        return results

    def process_video(self,
                      video_path: str,
                      prompt: str,
                      masks_by_frame: dict = None,
                      output_path: str = None,
                      lora_colorization: str = None,
                      lora_style: str = None,
                      progress_callback=None) -> str:
        """
        Process an entire video with temporal consistency.
        
        Args:
            video_path: Path to input video
            prompt: Colorization prompt
            masks_by_frame: Dict mapping frame_idx to mask array
            output_path: Path for output video
            lora_colorization: Path to colorization LoRA
            lora_style: Path to style LoRA (optional)
            progress_callback: Callback(frame_idx, total_frames, status)
            
        Returns:
            Path to the output video
        """
        # Load LoRAs if provided
        if lora_colorization:
            self.load_lora(lora_colorization, LORA_COLORIZATION_WEIGHT, "colorization")
        if lora_style:
            self.load_lora(lora_style, LORA_STYLE_WEIGHT, "style")
        
        # Format prompt
        formatted_prompt = DEFAULT_PROMPT_TEMPLATE.format(
            main_prompt=prompt,
            style_keywords=""
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare output
        if output_path is None:
            output_path = video_path.rsplit('.', 1)[0] + '_colorized.mp4'
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process in batches with overlap
        all_colorized = []
        frame_idx = 0
        batch_buffer = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for processing if needed
            target_size = DEFAULT_RESOLUTION
            frame_resized = cv2.resize(frame_rgb, target_size)
            
            batch_buffer.append({
                'frame': frame_resized,
                'original_size': (width, height),
                'mask': masks_by_frame.get(frame_idx) if masks_by_frame else None
            })
            
            # Process batch when full
            if len(batch_buffer) >= BATCH_SIZE:
                batch_results = self._process_batch_with_overlap(
                    batch_buffer, 
                    formatted_prompt,
                    all_colorized[-OVERLAP_FRAMES:] if all_colorized else []
                )
                all_colorized.extend(batch_results)
                batch_buffer = batch_buffer[-OVERLAP_FRAMES:]  # Keep overlap frames
                
                if progress_callback:
                    progress_callback(frame_idx, total_frames, "Processing batch...")
            
            frame_idx += 1
        
        # Process remaining frames
        if batch_buffer:
            batch_results = self._process_batch_with_overlap(
                batch_buffer,
                formatted_prompt, 
                all_colorized[-OVERLAP_FRAMES:] if all_colorized else []
            )
            all_colorized.extend(batch_results)
        
        # Apply temporal consistency
        if self.consistency_enforcer:
            all_colorized, scores = self.consistency_enforcer.process_video_frames(
                all_colorized,
                lambda i, t, s: progress_callback(i, t, f"Consistency check: {s:.2f}") if progress_callback else None
            )
        
        # Write output video
        for i, frame in enumerate(all_colorized):
            # Resize back to original
            frame_full = cv2.resize(frame, (width, height))
            frame_bgr = cv2.cvtColor(frame_full, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            if progress_callback and i % 10 == 0:
                progress_callback(i, len(all_colorized), "Writing output...")
        
        cap.release()
        out.release()
        
        # Unload LoRAs
        self.unload_loras()
        
        return output_path

    def _process_batch_with_overlap(self,
                                    batch: List[dict],
                                    prompt: str,
                                    previous_frames: List[np.ndarray]) -> List[np.ndarray]:
        """Process a batch with overlap blending."""
        frames = [item['frame'] for item in batch]
        masks = [item['mask'] for item in batch]
        
        # Colorize batch
        colorized = self.process_batch(frames, [prompt], masks)
        
        # Blend with previous frames using temporal blender
        if previous_frames and self.temporal_blender:
            for i in range(min(OVERLAP_FRAMES, len(colorized))):
                if i < len(previous_frames):
                    colorized[i] = self.temporal_blender.blend_frames(
                        colorized[i],
                        previous_frames[-(OVERLAP_FRAMES - i)]
                    )
        
        return colorized

    def preprocess_canny(self, image_np: np.ndarray) -> Image.Image:
        """
        Convert numpy image to canny edge map for ControlNet.
        
        Args:
            image_np: Input image (H, W, 3) in RGB
            
        Returns:
            Canny edge map as PIL Image
        """
        # Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        
        # Convert to 3-channel image
        edges_3ch = np.stack([edges, edges, edges], axis=2)
        
        return Image.fromarray(edges_3ch)

    def preprocess_depth(self, image_np: np.ndarray) -> Image.Image:
        """
        Generate depth map for ControlNet depth variant.
        Uses MiDaS or similar depth estimation.
        
        Args:
            image_np: Input image (H, W, 3)
            
        Returns:
            Depth map as PIL Image
        """
        # Placeholder: would use MiDaS or similar
        # For now, return a mock depth map
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        depth = cv2.GaussianBlur(gray, (15, 15), 0)
        depth_3ch = np.stack([depth, depth, depth], axis=2)
        return Image.fromarray(depth_3ch)

    def _apply_mask_blend(self, 
                          original: Image.Image, 
                          colorized: Image.Image, 
                          mask: np.ndarray) -> Image.Image:
        """Blend colorized foreground with original background using mask."""
        orig_np = np.array(original)
        color_np = np.array(colorized)
        
        # Ensure mask is 3D
        if mask.ndim == 2:
            mask_3d = np.stack([mask] * 3, axis=-1)
        else:
            mask_3d = mask
            
        # Blend
        blended = color_np * mask_3d + orig_np * (1 - mask_3d)
        
        return Image.fromarray(blended.astype(np.uint8))

    def _mock_colorize(self, image: Image.Image, mask: np.ndarray = None) -> Image.Image:
        """Mock colorization for testing without ML libraries."""
        img_np = np.array(image)
        
        # Apply a warm color tint to simulate colorization
        if len(img_np.shape) == 2:
            # Grayscale input
            colored = cv2.applyColorMap(img_np, cv2.COLORMAP_INFERNO)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        else:
            # Already RGB, apply color enhancement
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Boost saturation
            colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        if mask is not None:
            mask_3d = np.stack([mask] * 3, axis=-1) if mask.ndim == 2 else mask
            result = colored * mask_3d + img_np * (1 - mask_3d)
            return Image.fromarray(result.astype(np.uint8))
        
        return Image.fromarray(colored)


if __name__ == "__main__":
    # Test script
    pipeline = RenderPipeline()
    print("Pipeline ready to test.")
    
    # Test mock colorization
    test_image = Image.new('RGB', (512, 512), color='gray')
    result = pipeline.process_frame(test_image, "vibrant colorful scene")
    result.save("test_colorized.png")
    print("Test complete.")
