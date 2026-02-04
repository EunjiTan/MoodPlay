"""
Stage 4: Keyframe Colorization
Colorizes specific keyframes using SD1.5 + ControlNext + LoRA.
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
    Runs SD1.5 pipeline on keyframes (e.g. every 5th frame).
    Strictly isolated VRAM usage.
    """
    
    def run(self, 
            style_name: str = "cinematic", 
            base_prompt: str = "", 
            lora_path: str = None, 
            keyframe_interval: int = 5):
        
        self.logger.info(f"Starting Keyframe Colorization (Interval: {keyframe_interval})")
        
        # 1. Setup Prompt & Style
        prompt, negative = style_manager.build_prompt(base_prompt, style_name)
        style_config = style_manager.get_style_config(style_name)
        lora_scale = style_config.get("lora_scale", 0.6)
        
        # 2. Load Pipeline
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
                
                # Prepare ControlNext Input (Edges)
                # Note: We extract edges here on CPU/light GPU use.
                edges = extract_canny(frame_pil)
                
                # Generate
                # Note: sd15_runner handles the generation process
                colorized = sd15_runner.generate_frame(
                    image=edges,
                    prompt=prompt,
                    negative_prompt=negative,
                    controlnet_scale=0.7,
                    lora_scale=lora_scale
                )
                
                # Save Keyframe
                self.dm.save_frame(colorized, idx, stage="keys")
                
        finally:
            # 3. Unload
            sd15_runner.unload_pipeline()
            self.release_resources()
