"""
Stage 4: Keyframe Colorization
Colorizes specific keyframes using SD1.5 + ControlNet + LoRA.
Updated with 4-color palette system and denoising.
"""

from tqdm import tqdm
import numpy as np
from PIL import Image
from backend.core.stage_runner import StageRunner
from backend.diffusion.sd15_runner import sd15_runner
from backend.models.controlnet_helpers import extract_canny
from backend.conditioning.style_conditioning import style_manager


class KeyframeColorizationStage(StageRunner):
    """
    Runs SD1.5 pipeline on keyframes (e.g. every 10th frame).
    Uses 4-color palette guidance via LoRA and DPMSolver for quality.
    """
    
    def run(
        self, 
        mood: str = "sunny_day", 
        base_prompt: str = "", 
        lora_path: str = None, 
        keyframe_interval: int = 10,
        denoise_strength: float = 0.3,
        seed: int = None
    ):
        """
        Run keyframe colorization.
        
        Args:
            mood: Mood preset name (e.g., 'sunny_day', 'winter')
            base_prompt: Additional prompt text
            lora_path: Path to 4-color LoRA (optional)
            keyframe_interval: Colorize every N frames
            denoise_strength: Post-processing denoising (0-1)
            seed: Random seed for reproducibility
        """
        self.logger.info(f"Starting Keyframe Colorization (Mood: {mood}, Interval: {keyframe_interval})")
        
        # Setup Prompt & Style from 4-color system
        prompt, negative = style_manager.build_prompt(base_prompt, mood)
        style_config = style_manager.get_style_config(mood)
        lora_scale = style_config.get("lora_scale", 0.5)
        
        self.logger.info(f"Prompt: {prompt[:100]}...")
        self.logger.info(f"LoRA Scale: {lora_scale}, Denoise: {denoise_strength}")
        
        # Load Pipeline
        sd15_runner.load_pipeline()
        if lora_path:
            sd15_runner.load_lora(lora_path, weight=lora_scale)
            
        frame_count = self.dm.get_frame_count("proc")
        
        try:
            # Process only keyframes
            indices = list(range(0, frame_count, keyframe_interval))
            
            for idx in tqdm(indices, desc="Colorizing Keyframes"):
                # Check if done
                if (self.dm.dirs["keys"] / f"{idx:06d}.png").exists():
                    continue

                # Load Input
                frame_pil = self.dm.load_frame(idx, stage="proc")
                
                # Prepare ControlNet Input (Canny edges)
                edges = extract_canny(frame_pil)
                
                # Generate with updated parameters
                colorized = sd15_runner.generate_frame(
                    image=edges,
                    prompt=prompt,
                    negative_prompt=negative,
                    num_inference_steps=50,     # Updated default
                    guidance_scale=5.5,         # Updated for realism
                    controlnet_scale=0.95,
                    lora_scale=lora_scale,
                    denoise_strength=denoise_strength,
                    seed=seed
                )
                
                # Save Keyframe
                self.dm.save_frame(colorized, idx, stage="keys")
                
        finally:
            sd15_runner.unload_pipeline()
            self.release_resources()
